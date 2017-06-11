from enum import Enum
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
import random
import numpy as np
import time

TYPES_OF_CLASSIFIERS = 1

def logging(to_print):
    print(str(to_print) + "\n")

class Combiner(Enum):
    COUNT = 1
    DECISION_TREE = 2
    NEURAL_NET = 3
    SVM = 4

class EnsembleObject:

    def __init__(self, training_data, test_data, num_classifiers, combo_method):
        self.train = training_data
        self.test = test_data
        self.true_test, self.testing_labels = self.format_test_data()
        self.classifier_list = []
        for x in range(num_classifiers):
            self.classifier_list.append(self.get_random_classifier())
        self.combination = combo_method
        self.combiner = None
        if combo_method == Combiner.NEURAL_NET:
            self.combiner = MLPClassifier()
            #self.combiner = MLPClassifier(hidden_layer_sizes=(100, ), activation='relu', solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        if combo_method == Combiner.DECISION_TREE:
            self.combiner = tree.DecisionTreeClassifier()
        if combo_method == Combiner.SVM:
            self.combiner = svm.SVC()


    def get_random_classifier(self):
        TYPES_OF_CLASSIFIERS = 3
        classifier_type = random.randint(0, TYPES_OF_CLASSIFIERS - 1)

        if classifier_type == 0:
            return tree.DecisionTreeClassifier()
        if classifier_type == 1:
            return svm.SVC()
        if classifier_type == 2:
            return GaussianNB()


    def format_test_data(self):
        true_test = []
        labels = []
        for row in self.test:
            labels.append(row[-1])
            true_test.append(row[:-1])
        return true_test, labels

    def random_sample(self, training_set):
        dataset = None
        if training_set:
            dataset = self.train
        else:
            dataset = self.test
        rand_index = random.randint(0, len(dataset) - 1)

        return dataset[rand_index][:-1], dataset[rand_index][-1]

    def generate_random_subset(self, num_samples, training_set):
        data_rows = []
        data_labels = []
        for x in range(num_samples):
            data, label = self.random_sample(training_set)
            data_rows.append(data)
            data_labels.append(label)

        return (data_rows, data_labels)


    def train_system(self, rand_sample_percent):
        num_samples = int(rand_sample_percent * len(self.train))

        logging("Beginning training ->")

        for classifier in self.classifier_list:
            temp_subset = self.generate_random_subset(num_samples, training_set=True)
            classifier.fit(np.array(temp_subset[0]), np.array(temp_subset[1]))

        logging("Interior nodes trained ->")

        second_wave_subset = self.generate_random_subset(num_samples, training_set=True)
        predictions = []

        for row in second_wave_subset[0]:
            pred_row = []
            for classifier in self.classifier_list:
                pred_row.append(classifier.predict(np.array(row).reshape(1, -1))[0])
            predictions.append(row + pred_row)

        if self.combination == Combiner.COUNT:
            logging("Training not applicable for count ->")
            pass
            # TODO: Do the naieve method
        else:
            s = ""
            if self.combination == Combiner.DECISION_TREE:
                s = "decision tree"
            if self.combination == Combiner.NEURAL_NET:
                s = "neural net"
            if self.combination == Combiner.SVM:
                s = "SVM"
            logging("Training " + s + " ->")
            self.combiner.fit(np.array(predictions), np.array(second_wave_subset[1]))

        print("Training complete!")

    def test_system(self):
        test_data = np.array(self.true_test)

        interior_predictions = []

        for classifier in self.classifier_list:
            interior_predictions.append(classifier.predict(test_data))

        interior_predictions = np.array(interior_predictions)
        interior_predictions = np.transpose(interior_predictions)

        final_predictions = []

        if self.combination == Combiner.COUNT:

            for pred_row in interior_predictions:
                numOnes = 0
                numZeros = 0
                for item in pred_row:
                    if item == 1:
                        numOnes += 1
                    elif item == 0:
                        numZeros += 1
                    else:
                        print("Error! Non binary prediction was made!")

                if numOnes > numZeros:
                    final_predictions.append(1)
                else:
                    final_predictions.append(0)
        else:
            if len(interior_predictions) != len(self.true_test):
                print("Error!  Length mismatch on predictions and rows")
                return None

            lists = []
            for index in range(len(interior_predictions)):
                row_list = interior_predictions[index].tolist() + self.true_test[index]
                lists.append(row_list)

            final_predictions = self.combiner.predict(np.array(lists))

        return np.array(final_predictions)

    def report_accuracy(self, predictions):
        if len(self.testing_labels) != len(predictions):
            print(len(self.testing_labels))
            print(len(predictions))
            print("Error! Length mismatch when checking accuracy!")
            return

        total = 0
        correct = 0
        false_pos = 0
        false_neg = 0

        for index in range(len(predictions)):
            total += 1

            if predictions[index] == self.testing_labels[index]:
                correct += 1
            else:
                if predictions[index] == 1:
                    false_pos += 1
                else:
                    false_neg += 1

        print("\nEnsemble System Accuracy")
        print("------------------------")
        print("Total predictions: " + str(total))
        print("Total correct: " + str(correct))
        print("Accuracy: " + str(float(correct)/float(total)))
        print("False positives: " + str(false_pos))
        print("False positive rate: " + str(float(false_pos)/float(total)))

        print("False negatives: " + str(false_neg))
        print("False negative rate: " + str(float(false_neg)/float(total)))
        print("\n")





