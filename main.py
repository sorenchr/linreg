import argparse
import numpy
import csv


def run(datafile):
    dataarray = readcsv(datafile)
    matrix = numpy.matrix(dataarray)
    print(matrix)


def readcsv(file):
    """Read a CSV file into a multidimensional array of rows and columns."""
    rows = []
    with open(file, newline='') as csvfile:
        reader = csv .reader(csvfile, delimiter=',', quotechar='"')
        next(reader, None)  # headers
        for row in reader:
            rows.append([float(x) for x in row])
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str, help="the CSV file containing the data")
    args = parser.parse_args()
    run(args.data)
