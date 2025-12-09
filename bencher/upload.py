import argparse
import json
import socket
import subprocess
from datetime import datetime, timedelta

import requests


def get_git_reponame():
    try:
        result = subprocess.run(
            ["git", "config", "--get", "remote.origin.url"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip().split("/")[-1].split(".")[0]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def get_git_hash():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def get_git_branch():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def add_metadata(input_file):
    # Read existing data
    with open(input_file, "r") as f:
        data = json.load(f)

    # Add metadata section
    data["metadata"] = {
        "hostname": socket.gethostname(),
        "git_reponame": get_git_reponame(),
        "git_branch": get_git_branch(),
        "git_hash": get_git_hash(),
        "timestamp": datetime.now().isoformat(),
    }

    # Write back to the same file
    with open(input_file, "w") as f:
        json.dump(data, f, indent=2)

    return data


def upload_to_bencher(
    json_data, bencher_url, project_slug, benchmark_name, branch_slug, token=None
):
    # Extract data
    timings = json_data.get("timings", {})
    parameters = json_data.get("parameters", {})
    metadata = json_data.get("metadata", {})

    # Use hostname as testbed slug
    testbed_slug = metadata.get("hostname", "unknown")

    # Combine the timings and parameters in the format that Bencher expects.
    results_dict = {benchmark_name: {}}

    # Add timing metrics to the results
    total_runtime = 0
    for metric_name, value in timings.items():
        results_dict[benchmark_name][metric_name] = {
            "value": value,
        }
        total_runtime += value
    results_dict[benchmark_name]["runtime"] = {"value": total_runtime}

    # Add parameters to the results
    for parameter_name, value in parameters.items():
        results_dict[benchmark_name][parameter_name] = {
            "value": value,
        }

    # Calculate timestamps
    now = datetime.now()

    def to_rfc3339_format(timestamp):
        return timestamp.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-4] + "Z"

    start_time = to_rfc3339_format(now)
    end_time = to_rfc3339_format(now + timedelta(milliseconds=total_runtime))

    # Construct payload
    payload = {
        "branch": branch_slug,
        "hash": metadata.get("git_hash", "unknown"),
        "testbed": testbed_slug,
        "start_time": start_time,
        "end_time": end_time,
        "results": [json.dumps(results_dict)],
        "settings": {"adapter": "json"},
    }

    # Set up headers and make request
    headers = {"Content-Type": "application/json", "User-Agent": "bencher-cli"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    url = f"{bencher_url.rstrip('/')}/v0/projects/{project_slug}/reports"

    try:
        # Send the request
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()

        # Handle the response
        response_data = response.json()
        report_uuid = response_data.get("uuid")
        project_name = response_data.get("project", {}).get("name")

        if report_uuid and project_name:
            report_url = f"https://bencher.dev/console/projects/{project_name}/reports/{report_uuid}"
        else:
            report_url = "Unable to generate URL"

        # Print success message
        nr_metrics = len(results_dict[benchmark_name])
        print(f"Successfully uploaded {nr_metrics} metrics to Bencher")
        print(
            f"Project: {project_slug}, Branch: {branch_slug}, Hash: {metadata.get('git_hash', 'unknown')[:8]}, Testbed: {testbed_slug}"
        )
        print(f"Report URL: {report_url}")

    except requests.exceptions.RequestException as e:
        print(f"Failed to upload to Bencher: {e}")
        if hasattr(e, "response") and e.response is not None:
            print(f"Response: {e.response.text}")


def main():
    parser = argparse.ArgumentParser(description="Upload timing data to Bencher")
    parser.add_argument("json_file", help="Path to JSON file with timings")
    parser.add_argument(
        "--bencher-url",
        required=False,
        default="https://api.bencher.dev",
        help="Bencher instance URL",
    )
    parser.add_argument("--project", required=True, help="Project slug")
    parser.add_argument("--benchmark", required=True, help="Benchmark name")
    parser.add_argument("--branch", required=True, help="Branch slug")
    parser.add_argument("--token", help="Bencher API token (if required)")

    args = parser.parse_args()
    json_data = add_metadata(args.json_file)

    upload_to_bencher(
        json_data=json_data,
        bencher_url=args.bencher_url,
        project_slug=args.project,
        benchmark_name=args.benchmark,
        branch_slug=args.branch,
        token=args.token,
    )


if __name__ == "__main__":
    main()
