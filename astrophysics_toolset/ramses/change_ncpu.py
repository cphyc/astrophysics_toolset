import argparse
from .oct_handler import Octree


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Write new initial conditions with a new number of processors.')
    parser.add_argument('input', help='Folder containing the data, ex. `output_00080`.')
    parser.add_argument('-n', '--ncpu', help='Number of CPUs to use.')

    args = parser.parse_args()