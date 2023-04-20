""" Create Learning from Softconstraint evaluations"""

import argparse
from os.path import abspath, join, isfile
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder
from sklearn import linear_model
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import config as cfg
import readectt


def parse_args():
    """Parse command line arguments passed to script invocation."""
    parser = argparse.ArgumentParser(
        description='Create a machine learning model to evaluate solutions')

    parser.add_argument('filename', help='Input file containing traing data')

    parser.add_argument('mltype',
                        help="Type of learing model - Classification=0"
                        "or Regression=1")

    return parser.parse_args()


def comparenumbers(first, second):
    """Compare two numbers
    Return
          1  if first less or equal to second
          0  if first greater than second"""
    # print(type(first), type(second))
    # if np.float64(first) == np.float64(second):
    #     return 0
    if np.float64(first) <= np.float64(second):
        return 1
    return 0


def getheaders(batch):
    """Get list of variable names for data frame headers"""
    headers = []
    for i in range(len(batch[0])):
        feature = "Event" + str(i)
        if i == len(batch[0]) - 1:
            feature = "class"
        headers.append(feature)
    # print(headers)
    return headers


def createclassificationmodel(batch):
    """Create machine learning classification model from input"""

    dataset = pd.DataFrame(batch, columns=getheaders(batch))
    # print(dataset)

    # descriptions
    # print(dataset.describe())

    # class distribution
    # print(dataset.groupby('class').size())

    col = len(dataset.columns) - 1

    # get input data Y and class X from dataset
    array = dataset.values
    x_data = array[:, 0:col]
    y_data = array[:, col]

    x_train, x_validation, y_train, y_validation = train_test_split(
        x_data, y_data, test_size=0.3, random_state=8)

    # Create a random forest classifier. By convention, clf means 'classifier'
    # clf = RandomForestClassifier(n_estimators=4000, random_state=0, n_jobs=4)
    clf = SVC(C=1.0, kernel='rbf', gamma=0.001)
    clf.fit(x_train, y_train)
    predictions = clf.predict(x_validation)
    accuracy = accuracy_score(y_validation, predictions)
    precision, recall, fscore, _ = precision_recall_fscore_support(
        y_validation, predictions, average='weighted')
    accuracy = accuracy_score(y_validation, predictions)
    print("Normal - Precision: {} Recall: {} Fscore: {} Accuracy: {}"
          .format(precision, recall, fscore, accuracy))
    cfg.REPORT_NORM['precision'].append(precision)
    cfg.REPORT_NORM['recall'].append(recall)
    cfg.REPORT_NORM['fscore'].append(fscore)
    cfg.REPORT_NORM['accuracy'].append(accuracy)

    # incremental classification
    # print(type(cfg.CLF))
    if cfg.CLF is None:
        cfg.CLF = linear_model.SGDClassifier()

    cfg.CLF = cfg.CLF.partial_fit(x_train, y_train,
                                  classes=np.array(['F', 'NF']))

    predictions_incremental = cfg.CLF.predict(x_validation)
    precision, recall, fscore, _ = precision_recall_fscore_support(
        y_validation, predictions_incremental, average='weighted')
    accuracy = accuracy_score(y_validation, predictions_incremental)
    print("Incremental - Precision: {} Recall: {} Fscore: {} Accuracy: {}"
          .format(precision, recall, fscore, accuracy))
    cfg.REPORT_INCR['precision'].append(precision)
    cfg.REPORT_INCR['recall'].append(recall)
    cfg.REPORT_INCR['fscore'].append(fscore)
    cfg.REPORT_INCR['accuracy'].append(accuracy)
    print("*********************************\n")


def createregressionmodel(batch):
    """Create machine learning regression model from input"""

    dataset = pd.DataFrame(batch, columns=getheaders(batch))
    # print(dataset)

    # descriptions
    # print(dataset.describe())

    # class distribution
    # print(dataset.groupby('class').size())

    col = len(dataset.columns) - 1

    # get input data Y and class X from dataset
    array = dataset.values
    x_data = array[:, 0:col]
    y_data = array[:, col]

    # transform x_data to one hot encoding
    onehot_encoder = OneHotEncoder(n_values=len(cfg.ROOMPERIODLIST))
    onehot_encoder.fit(x_data)
    x_data = onehot_encoder.transform(x_data)
    # if cfg.CLF is None:
    #     cfg.CLF = linear_model.SGDRegressor()
    # print(onehotlabels.shape)
    # print(x_data.shape)
    # print(onehot_encoded)

    x_train, x_validation, y_train, y_validation = train_test_split(
        x_data, y_data, test_size=0.3, random_state=8)

    # supervised learning is regression
    # Fit regression model
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    # svr_rbf = KernelRidge(alpha=1.0)
    # svr_rbf = SVR(kernel='linear', C=1e3)
    # svr_lin = SVR(kernel='linear', C=1e3)
    # svr_poly = SVR(kernel='poly', C=1e3, degree=2)
    y_rbf = svr_rbf.fit(x_train, y_train).predict(x_validation)
    # y_lin = svr_lin.fit(x_train, y_train).predict(x_validation)
    # y_poly = svr_poly.fit(x_train, y_train).predict(x_validation)

    # for regression evaluation
    x_vals = [x for x in range(x_validation.shape[0])]
    # get combination of for comparison
    combination = set()
    # combination = [val for val in itertools.product(x_vals, x_vals)
    #                if list(val)[0] != list(val)[1]]
    gen_expr = (val for val in itertools.product(x_vals, x_vals)
                if list(val)[0] != list(val)[1])
    for val in gen_expr:
        # print(tuple(sorted(list(val))))
        combination.add(tuple(sorted(list(val))))
        # print("\n")

    # print(len(combination))

    # dict containing pairs for comparison
    # pairs cld be greater(1),  less(-1) or equal(0)
    # compare actual value with predicted value to
    # determine accuracy
    same = 0
    dict_comb = dict.fromkeys(combination, [])
    for key, _ in dict_comb.items():
        y_true_compare = comparenumbers(y_validation[key[0]],
                                        y_validation[key[1]])
        # check if ml predicted can be compared regular evaluated
        y_pred_compare = comparenumbers(y_validation[key[0]],
                                        y_rbf[key[1]])
        # check if two ml predicted compares like two regular evaluated
        # y_pred_compare = comparenumbers(y_rbf[key[0]],
        #                                 y_rbf[key[1]])
        dict_comb[key] = [y_true_compare, y_pred_compare]
        if y_true_compare == y_pred_compare:
            same += 1
        # print(key, dict_comb[key])
    print("Normal - Accuracy: ", same / len(dict_comb))
    accuracy = same / len(dict_comb)
    cfg.REPORT_NORM['accuracy'].append(accuracy)
    # print(dict_comb)

    # incremental regression
    if cfg.CLF is None:
        cfg.CLF = linear_model.SGDRegressor()

    y_rbf = cfg.CLF.partial_fit(x_train, y_train).predict(x_validation)
    same = 0
    dict_comb = dict.fromkeys(combination, [])
    for key, _ in dict_comb.items():
        y_true_compare = comparenumbers(y_validation[key[0]],
                                        y_validation[key[1]])
        # check if ml predicted can be compared regular evaluated
        # y_pred_compare = comparenumbers(y_validation[key[0]],
        #                                 y_rbf[key[1]])
        # check if two ml predicted compares like two regular evaluated
        y_pred_compare = comparenumbers(y_rbf[key[0]],
                                        y_rbf[key[1]])
        dict_comb[key] = [y_true_compare, y_pred_compare]
        if y_true_compare == y_pred_compare:
            same += 1
        # print(key, dict_comb[key])
    print("Incremental - Accuracy: ", same / len(dict_comb))
    accuracy = same / len(dict_comb)
    cfg.REPORT_INCR['accuracy'].append(accuracy)
    print("*********************************\n")


def createmodel(batch):
    """Create machine learning model from input"""

    dataset = pd.DataFrame(batch, columns=getheaders(batch))
    # print(dataset)

    # descriptions
    # print(dataset.describe())

    # class distribution
    # print(dataset.groupby('class').size())

    col = len(dataset.columns) - 1

    # get input data Y and class X from dataset
    array = dataset.values
    x_data = array[:, 0:col]
    y_data = array[:, col]

    # transform x_data to one hot encoding
    if cfg.MLTYPE == 1:
        onehot_encoder = OneHotEncoder(n_values=len(cfg.ROOMPERIODLIST))
        onehot_encoder.fit(x_data)
        x_data = onehot_encoder.transform(x_data)
        # if cfg.CLF is None:
        #     cfg.CLF = linear_model.SGDRegressor()
        # print(onehotlabels.shape)
        # print(x_data.shape)
        # print(onehot_encoded)

    x_train, x_validation, y_train, y_validation = train_test_split(
        x_data, y_data, test_size=0.3, random_state=8)

    if cfg.MLTYPE == 1:
        # supervised learning is regression
        # Fit regression model
        # svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
        # svr_rbf = KernelRidge(alpha=1.0)
        svr_rbf = SVR(kernel='linear', C=1e3)
        # svr_lin = SVR(kernel='linear', C=1e3)
        # svr_poly = SVR(kernel='poly', C=1e3, degree=2)
        y_rbf = svr_rbf.fit(x_train, y_train).predict(x_validation)
        # y_lin = svr_lin.fit(x_train, y_train).predict(x_validation)
        # y_poly = svr_poly.fit(x_train, y_train).predict(x_validation)

        # for regression evaluation
        x_vals = [x for x in range(x_validation.shape[0])]
        # get combination of for comparison
        combination = set()
        # combination = [val for val in itertools.product(x_vals, x_vals)
        #                if list(val)[0] != list(val)[1]]
        gen_expr = (val for val in itertools.product(x_vals, x_vals)
                    if list(val)[0] != list(val)[1])
        for val in gen_expr:
            # print(tuple(sorted(list(val))))
            combination.add(tuple(sorted(list(val))))
        # print("\n")

        # print(len(combination))

        # dict containing pairs for comparison
        # pairs cld be greater(1),  less(-1) or equal(0)
        # compare actual value with predicted value to
        # determine accuracy
        same = 0
        dict_comb = dict.fromkeys(combination, [])
        for key, _ in dict_comb.items():
            y_true_compare = comparenumbers(y_validation[key[0]],
                                            y_validation[key[1]])
            # check if ml predicted can be compared regular evaluated
            y_pred_compare = comparenumbers(y_validation[key[0]],
                                            y_rbf[key[1]])
            # check if two ml predicted compares like two regular evaluated
            # y_pred_compare = comparenumbers(y_rbf[key[0]],
            #                                 y_rbf[key[1]])
            dict_comb[key] = [y_true_compare, y_pred_compare]
            if y_true_compare == y_pred_compare:
                same += 1
            # print(key, dict_comb[key])
        print("Num identical comparison for predict and actual is: ",
              same, len(dict_comb), same / len(dict_comb))
        # print(dict_comb)

        # for x in range(x_validation.shape[0]):
        #     print(y_validation[x], "\t",  y_rbf[x])

        # linewth = 2
        # plt.scatter(x_vals,
        #             y_validation, color='darkorange', label='data')
        # plt.plot(x_vals, y_rbf, color='navy',
        #          lw=linewth, label='RBF model')
        # # plt.plot(x_vals, y_lin, color='c',
        # #          lw=linewth, label='Linear model')
        # # plt.plot(x_vals, y_poly, color='cornflowerblue',
        # #          lw=linewth, label='Polynomial model')
        # plt.xlabel('data')
        # plt.ylabel('target')
        # plt.title('Support Vector Regression')
        # plt.legend()
        # plt.show()
        return

    # get training and test splits

    # get number instances -- rows in array
    # num_instances = x_data.shape[0]
    # determine test set size as 2/3 of instances
    # num_train = int(num_instances / 10) * 9
    # first two thirds is for trainging and the remaining test
    # not using a random split as we are trying to recreate incremental
    # learning for evaluation -- first train and then use subsequent
    # for evaluation
    # train_indices = [x for x in range(num_train)]
    # test_indices = [x for x in range(num_train, num_instances)]

    # x_train = x_data[train_indices, :]
    # y_train = y_data[train_indices]
    # x_validation = x_data[test_indices, :]
    # y_validation = y_data[test_indices]

    # print(x_train.shape, y_train.shape)
    # print(x_validation.shape, y_validation.shape)

    # Create a random forest classifier. By convention, clf means 'classifier'
    clf = RandomForestClassifier(n_estimators=4000, random_state=0, n_jobs=4)
    # clf = SVC(C=1.0, kernel='rbf', gamma=0.001)
    clf.fit(x_train, y_train)
    predictions = clf.predict(x_validation)
    accuracy = accuracy_score(y_validation, predictions)
    precision, recall, fscore, _ = precision_recall_fscore_support(
        y_validation, predictions, average='weighted')
    accuracy = accuracy_score(y_validation, predictions)
    print("Normal - Precision: {} Recall: {} Fscore: {} Accuracy: {}"
          .format(precision, recall, fscore, accuracy))

    # print(type(cfg.CLF))
    if cfg.CLF is None:
        cfg.CLF = linear_model.SGDClassifier()

    cfg.CLF = cfg.CLF.partial_fit(x_train, y_train,
                                  classes=np.array(['F', 'NF']))

    predictions_incremental = cfg.CLF.predict(x_validation)
    print("\n\n")
    precision, recall, fscore, _ = precision_recall_fscore_support(
        y_validation, predictions_incremental, average='weighted')
    accuracy = accuracy_score(y_validation, predictions_incremental)
    print("Incremental - Precision: {} Recall: {} Fscore: {} Accuracy: {}"
          .format(precision, recall, fscore, accuracy))
    print("*********************************")
    # print(classification_report(y_validation, predictions))


def main():
    """Module's main entry point (zopectl.command)."""

    # Reading input file - Timetable problem and saving in global variable
    args = parse_args()
    filename = args.filename
    inputfilename = filename + '.data'
    cfg.MLTYPE = int(args.mltype)
    fullpath = join(abspath('../ML-Training-Data'), inputfilename)

    if not isfile(fullpath):
        raise Exception('Directory does not exist ({0}).'.format(fullpath))

    # get room period list to enable value for one hot encoding
    timetableinput = filename + '.ectt'
    fullpath_tt = join(abspath('../InputData/ITC-2007_ectt'), timetableinput)
    if not isfile(fullpath_tt):
        raise Exception('Directory does not exist ({0}).'.format(fullpath_tt))
    cfg.DATA = list(readectt.readecttfile(fullpath_tt))
    cfg.ROOMPERIODLIST = readectt.createroomperiodlist(
        cfg.DATA[2][0], cfg.DATA[0].getnumperiods())

    if cfg.MLTYPE == 0:
        cfg.REPORT_NORM = {'precision': [], 'recall': [],
                           'fscore': [], 'accuracy': []}
        cfg.REPORT_INCR = {'precision': [], 'recall': [],
                           'fscore': [], 'accuracy': []}
    else:
        cfg.REPORT_NORM = {'accuracy': []}
        cfg.REPORT_INCR = {'accuracy': []}

    batchsize = 10000     # Experiment with different sizes
    # batchsize = 200000   # maximum size that fits all inst.
    batch = []
    index = 0
    with open(fullpath, "r") as readfile:
        for line in readfile:
            # print("Reading batch ", index, "index ", batchsize * index)
            # if index == 3:
            #     break
            if line == '':
                # should not have an empty line
                # but added to avoid
                # Else clause on loop pylint highlight
                break
            values = [val for val in line.replace('\n', '').split(',')]
            if cfg.MLTYPE == 1:
                del values[-1]
            else:
                del values[-2]
            batch.append(values)
            if len(batch) == batchsize:
                print("Reading batch ", index, "index ", batchsize * index)
                if cfg.MLTYPE == 1:
                    createregressionmodel(batch)
                else:
                    createclassificationmodel(batch)
                batch = []
                index += 1
                # break
        else:
            # No more lines to be read from file
            # train on last set of instances less than
            # batch size
            print("Reading batch ", index, "index ", batchsize * index)
            if cfg.MLTYPE == 1:
                createregressionmodel(batch)
            else:
                createclassificationmodel(batch)
            # print(" no more lines to read", len(batch))

    # print(cfg.REPORT_NORM)
    # print(cfg.REPORT_INCR)

    # print results incremental
    print("\n\n*************\nIncremental results\n*****************")
    if 'accuracy' in cfg.REPORT_INCR:
        print("accuracy")
        print("min accuracy is {0:.2f} ".format(
            min(cfg.REPORT_INCR['accuracy'])))
        print("max accuracy is {0:.2f} ".format(
            max(cfg.REPORT_INCR['accuracy'])))
        print("average accuracy is {0:.2f} ".format(
            sum(cfg.REPORT_INCR['accuracy']) / len(
                cfg.REPORT_INCR['accuracy'])))
    if 'precision' in cfg.REPORT_INCR:
        print("\nprecision")
        print("min precision is {0:.2f} ".format(
            min(cfg.REPORT_INCR['precision'])))
        print("max precision is {0:.2f} ".format(
            max(cfg.REPORT_INCR['precision'])))
        print("average precision is {0:.2f} ".format(
            sum(cfg.REPORT_INCR['precision']) / len(
                cfg.REPORT_INCR['precision'])))
    if 'recall' in cfg.REPORT_INCR:
        print("\nrecall")
        print("min recall is {0:.2f} ".format(
            min(cfg.REPORT_INCR['recall'])))
        print("max recall is {0:.2f} ".format(
            max(cfg.REPORT_INCR['recall'])))
        print("average recall is {0:.2f} ".format(
            sum(cfg.REPORT_INCR['recall']) / len(
                cfg.REPORT_INCR['recall'])))
    if 'fscore' in cfg.REPORT_INCR:
        print("\nfscore")
        print("min fscore is {0:.2f} ".format(
            min(cfg.REPORT_INCR['fscore'])))
        print("max fscore is {0:.2f} ".format(
            max(cfg.REPORT_INCR['fscore'])))
        print("average fscore is {0:.2f} ".format(
            sum(cfg.REPORT_INCR['fscore']) / len(
                cfg.REPORT_INCR['fscore'])))

    # print results normal
    print("\n\n*************\nNormal results\n*****************")
    if 'accuracy' in cfg.REPORT_NORM:
        print("accuracy")
        print("min accuracy is {0:.2f} ".format(min(
            cfg.REPORT_NORM['accuracy'])))
        print("max accuracy is {0:.2f} ".format(
            max(cfg.REPORT_NORM['accuracy'])))
        print("average accuracy is {0:.2f} ".format(
            sum(cfg.REPORT_NORM['accuracy']) / len(
                cfg.REPORT_NORM['accuracy'])))
    if 'precision' in cfg.REPORT_NORM:
        print("\nprecision")
        print("min precision is {0:.2f} ".format(
            min(cfg.REPORT_NORM['precision'])))
        print("max precision is {0:.2f} ".format(
            max(cfg.REPORT_NORM['precision'])))
        print("average precision is {0:.2f} ".format(
            sum(cfg.REPORT_NORM['precision']) / len(
                cfg.REPORT_NORM['precision'])))
    if 'recall' in cfg.REPORT_NORM:
        print("\nrecall")
        print("min recall is {0:.2f} ".format(min(cfg.REPORT_NORM['recall'])))
        print("max recall is {0:.2f} ".format(max(cfg.REPORT_NORM['recall'])))
        print("average recall is {0:.2f} ".format(
            sum(cfg.REPORT_NORM['recall']) / len(
                cfg.REPORT_NORM['recall'])))
    if 'fscore' in cfg.REPORT_NORM:
        print("\nfscore")
        print("min fscore is {0:.2f} ".format(min(cfg.REPORT_NORM['fscore'])))
        print("max fscore is {0:.2f} ".format(max(cfg.REPORT_NORM['fscore'])))
        avg = sum(cfg.REPORT_NORM['fscore']) / len(cfg.REPORT_NORM['fscore'])
        print("average fscore is {0:.2f} ".format(avg))


if __name__ == '__main__':
    main()
