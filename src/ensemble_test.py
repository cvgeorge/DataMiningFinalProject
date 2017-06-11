import sys
import re
import math
import copy
from sklearn.neural_network import MLPClassifier
import ensemble_object

def get_random_classifier():
    pass
    # TODO: Fill in this function

def read_data(filename):
    f = open(filename, 'r')
    p = re.compile(',')
    data = []
    header = f.readline().strip()
    varnames = p.split(header)
    namehash = {}
    for l in f:
        data.append([int(x) for x in p.split(l.strip())])
    return (data, varnames)

def main(argv):
    (train, varnames) = read_data(argv[0])
    (test, testvarnames) = read_data(argv[1])

    ensemble_system = ensemble_object.EnsembleObject(train, test, 100, ensemble_object.Combiner.SVM)

    ensemble_system.train_system(.2)

    predictions = ensemble_system.test_system()

    ensemble_system.report_accuracy(predictions)


if __name__ == "__main__":
    main(sys.argv[1:])
