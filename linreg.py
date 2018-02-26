import argparse
import numpy as np
import csv


def _run(datafile):
    # Read CSV file into matrix and split into features and values
    headers, rows = _readcsv(datafile)
    matrix = np.matrix(rows)
    features = matrix[:, :-1]
    values = matrix[:, -1]
    features = np.insert(features, 0, 1, axis=1)  # left-pad the features with 1's

    # Scale the features for better performance
    features = scalefeatures(features)

    # Run gradient descent
    alpha = 0.01
    iterations = 1500
    history = gradientdescent(features, values, iterations, alpha)
    #costs = np.ravel(history[:, -1]).tolist()

    # Print the parameters for the features
    output = ', '.join(['%s = %s' % (key, value) for (key, value) in _mergeresult(headers, history[-1:, :-1]).items()])
    print('Found the following parameters that best fits the data:\n' + output)


def _readcsv(file):
    """Read a CSV file into a multidimensional array of rows and columns."""
    rows = []
    with open(file, newline='') as csvfile:
        reader = csv .reader(csvfile, delimiter=',', quotechar='"')
        headers = next(reader, None)  # headers
        for row in reader:
            rows.append([float(x) for x in row])
    return headers, rows


def scalefeatures(matrix):
    """Scales the features of the matrix such that they are in the range [-1;1]."""
    colindex = -1
    for column in matrix.T:  # Not costly, don't worry
        colindex += 1
        stddev = np.max(column) - np.min(column)

        if stddev == 0:  # Ignore features that don't change in value
            continue

        avg = np.full((matrix.shape[0], 1), np.average(column))
        stddev = np.full((matrix.shape[0], 1), stddev)

        matrix[:, colindex] = (column.T - avg) / stddev

    return matrix


def gradientdescent(features, values, iterations, alpha):
    """Performs gradient descent and returns the parameters associated with their cost for each iteration."""
    m = features.shape[0]  # number of training examples
    n = features.shape[1]  # number of features
    history = np.zeros((iterations, n+1))
    params = np.zeros((n, 1))

    for itr in range(iterations):
        newparams = np.zeros((n, 1))

        # Iterate through all parameters
        for j in range(n):
            # Iterate through all test sets to compute the delta sum
            deltasum = 0
            for i in range(m):
                deltasum += (params.T * features[i, :].T - values.item(i)) * features.item((i, j))

            # Compute the theta update
            newparams[j] = params.item(j) - alpha * (1/m) * deltasum

        # Update the params
        params = newparams

        # Store the parameters and their associated cost in the history matrix
        history[itr, :-1] = params.T
        history[itr, -1] = cost(features, values, params)

    return history


def cost(features, values, parameters):
    """Computes the cost of applying the parameters to the features."""
    m = features.shape[0]  # number of training examples
    quaderrs = np.square(features * parameters - values)
    return np.asscalar((1/(2*m)) * np.ones((1, m)) * quaderrs)


def _mergeresult(headers, params):
    """Merges the headers from the CSV file with the found parameters into a dictionary."""
    result = {}
    for i, header in enumerate(headers[:-1]):
        result[header] = params.item(i)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str, help="the CSV file containing the data")
    args = parser.parse_args()
    _run(args.data)
