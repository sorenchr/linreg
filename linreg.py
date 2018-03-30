import argparse
import numpy as np
import csv
import logging

# Setup logging
logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)


def _run(datafile, iterations, alpha, force, scaling):
    # Read CSV file into matrix and split into features and values
    logging.info('Reading {}'.format(datafile))
    headers, rows = _readcsv(datafile)
    headers.insert(0, 'intercept')  # add the y-intercept as a feature header itself
    matrix = np.matrix(rows)
    features = matrix[:, :-1]
    values = matrix[:, -1]
    features = np.insert(features, 0, 1, axis=1)  # left-pad the features with 1's

    # Determine if we should use the normal equation method or gradient descent
    if features.shape[0] <= 10000 and force is False:
        logging.info('Less than 10,000 features, normal equation method is used')
        params = np.linalg.inv(features.T * features) * features.T * values
    else:
        logging.info('Using gradient descent (less than 10,000 features or forced)')

        # Scale the features for better performance
        if scaling:
            logging.info('Scaling features for better performance')
            scales = scalefeatures(features)
            output = ', '.join(['%s = %s' % (key, value) for (key, value) in _mergeheaders(headers, scales).items()])
            logging.info('Scaled features with the following scales: \n' + output)

        # Run gradient descent
        history = gradientdescent(features, values, iterations, alpha)

        # Get the best parameters from the history
        params = history[-1:, :-1]

    # Print the parameters for the features
    output = ', '.join(['%s = %s' % (key, value) for (key, value) in _mergeheaders(headers, params).items()])
    logging.info('Found the following parameters that best fits the data:\n' + output)

    # Test parameters and print accuracy
    paramscost = cost(features, values, params)
    logging.info('Parameters cost: %s' % paramscost)


def _readcsv(file):
    """Read a CSV file into a multidimensional array of rows and columns."""
    rows = []
    with open(file, newline='') as csvfile:
        reader = csv .reader(csvfile, delimiter=',', quotechar='"')
        headers = next(reader, None)  # headers
        for row in reader:
            rows.append([float(x) for x in row])
    return headers, rows


def scalefeatures(features):
    """Scales the features of the matrix such that they are in the range [-1;1]."""
    colindex = -1
    n = features.shape[1]  # number of features
    scales = np.ones((n, 1))
    for column in features.T:
        colindex += 1
        stddev = np.max(column) - np.min(column)

        if stddev == 0:  # ignore features that don't change in value
            continue

        avg = np.full((features.shape[0], 1), np.average(column))
        stddev = np.full((features.shape[0], 1), stddev)

        features[:, colindex] = (column.T - avg) / stddev
        scales[colindex] = 1 / stddev.item(colindex)

    return scales


def gradientdescent(features, values, iterations, alpha):
    """Performs gradient descent and returns the parameters associated with their cost for each iteration."""
    m = features.shape[0]  # number of training examples
    n = features.shape[1]  # number of features
    history = np.zeros((iterations, n+1))
    params = np.zeros((n, 1))

    for itr in range(iterations):
        # Perform vectorized gradient descent
        gradient = (alpha / m) * features.T * (features * params - values)
        params = params - gradient

        # Store the parameters and their associated cost in the history matrix
        history[itr, :-1] = params.T
        history[itr, -1] = cost(features, values, params)

    return history


def cost(features, values, parameters):
    """Computes the cost of applying the parameters to the features."""
    m = features.shape[0]  # number of training examples
    quaderrs = np.square(features * parameters - values)
    return np.asscalar((1/(2*m)) * np.ones((1, m)) * quaderrs)


def _mergeheaders(headers, params):
    """Merges the headers from the CSV file with the found parameters into a dictionary."""
    result = {}
    for i, header in enumerate(headers[:-1]):
        result[header] = params.item(i)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=str, help='the CSV file containing the data')
    parser.add_argument('-a', '--alpha', type=float, default=0.01, help='the learning rate for gradient descent')
    parser.add_argument('-i', '--iterations', type=int, default=1500,
                        help='the number of iterations for gradient descent')
    parser.add_argument('-f', '--force', action='store_true', default=False, help='skip the normal equation fallback')
    parser.add_argument('-ns', '--noscaling', action='store_true', default=False, help='turn off feature scaling')
    args = parser.parse_args()
    _run(args.data, args.iterations, args.alpha, args.force, not args.noscaling)
