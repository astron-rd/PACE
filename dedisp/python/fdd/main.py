import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark", action="store_true", help="Display timing results"
    )
    args = parser.parse_args()

    print(args)

    print("DEDISP MAIN")
