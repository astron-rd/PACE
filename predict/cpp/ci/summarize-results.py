import json
import os
from argparse import ArgumentParser
import seaborn
import pandas
import matplotlib.pyplot as plt

seaborn.set_theme(palette="bright", style="whitegrid")


def parse_args():
    parser = ArgumentParser(
        description="Combine benchmark metrics from Google Benchmarks Framework"
    )
    parser.add_argument("files", nargs="+", help="Metrics")
    parser.add_argument("output", help="Combined metrics json")
    parser.add_argument("--filter", help="Filter tests by name")
    return parser.parse_args()


def read_and_combine_json(files, filter):
    results = []
    for f_name in files:
        basename = os.path.basename(f_name)
        with open(f_name, "r") as f_stream:
            result_obj = json.load(f_stream)
            results_normalized = result_obj["benchmarks"]
            for result_normalized in results_normalized:
                if filter and filter not in result_normalized["name"]:
                    continue
                result_normalized["context"] = result_obj["context"]
                (
                    result_normalized["compiler_version"],
                    result_normalized["architecture"],
                ) = (
                    basename.replace("results-", "").replace(".json", "").split("-")
                )
                results.append(result_normalized)
    return results


def store_combined(outfile, obj):
    with open(outfile + ".json", "w") as f_stream:
        json.dump(obj, f_stream, indent=4)


def create_summary_plot(metrics, outplot_name):
    time_unit = metrics.time_unit[0]
    metrics["test_name"] = metrics["name"].str.split("/", n=2).str[1]
    metrics["parameter"] = metrics["name"].str.split("/", n=2).str[2]

    grid = seaborn.FacetGrid(metrics, col="architecture", row="compiler_version")
    grid.map(seaborn.scatterplot, "parameter", "cpu_time", "test_name")
    grid.set_ylabels(f"CPU time({time_unit})")
    grid.set_xlabels("Parameter")
    grid.add_legend()
    plt.setp(grid._legend.get_texts(), fontsize=8)
    for ax in grid.axes.flat:
        for label in ax.get_xticklabels():
            label.set_rotation(90)
            label.set_fontsize(8)

        ax.set_title(ax.get_title(), fontsize=8)

    compilers = sorted(metrics["compiler_version"].unique())
    archs = sorted(metrics["architecture"].unique())
    title = f"Predict compilers: [{', '.join(compilers)}], architectures: [{', '.join(archs)}]"

    grid.fig.suptitle(title, fontsize=14)
    grid.fig.subplots_adjust(top=0.75)

    grid.savefig(outplot_name + ".png")


def main():
    args = parse_args()
    metrics_results = read_and_combine_json(args.files, args.filter)
    metrics_dataframe = pandas.DataFrame(metrics_results)
    print(metrics_results)
    create_summary_plot(metrics_dataframe, args.output)
    store_combined(args.output, metrics_results)


if __name__ == "__main__":
    main()
