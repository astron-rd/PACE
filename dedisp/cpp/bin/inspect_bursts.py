import argparse

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_burst():
    pass

def main():
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot radio bursts stored in the NPY format')

    parser.add_argument('directory')
    parser.add_argument('-f', '--filterbank', action='store_true', help='Assume the source is the example filterbank')

    main()
