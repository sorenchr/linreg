import argparse
import numpy
import csv


def run(datafile):
    dataarray = readcsv(datafile)
    matrix = numpy.matrix(dataarray)
    matrix = numpy.insert(matrix, 0, 1, axis=1)  # Left-pad the matrix with 1's

    matrix = scalefeatures(matrix)

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


def scalefeatures(matrix):
    """Scales the features of the matrix such that they are in the range [-1;1]."""
    colindex = -1
    for column in matrix.T:  # Not costly, don't worry
        colindex += 1
        stddev = numpy.max(column) - numpy.min(column)

        if stddev == 0:  # Ignore features that don't change in value
            continue

        avg = numpy.full((matrix.shape[0], 1), numpy.average(column))
        stddev = numpy.full((matrix.shape[0], 1), stddev)

        matrix[:, colindex] = (column.T - avg) / stddev

    return matrix


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str, help="the CSV file containing the data")
    args = parser.parse_args()
    run(args.data)
