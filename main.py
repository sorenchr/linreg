from numpy import matrix
import csv
import argparse


def test():
    print('test')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str, help="the CSV file containing the data")
    args = parser.parse_args()
    test()
