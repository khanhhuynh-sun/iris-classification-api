import base64
import logging

import numpy as np
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

matplotlib.use('Agg')
from sklearn import datasets, neighbors, metrics

from rest_framework.response import Response
from rest_framework.decorators import api_view

import io
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt

def iris_knn_classification(k, sl, pl):
    float(sl)
    float(pl)
    float(k)
    iris = datasets.load_iris()
    X = iris.data[:, [0, 2]]
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    knn = neighbors.KNeighborsClassifier(n_neighbors=k, weights='uniform')
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)
    accuracy = round(metrics.accuracy_score(y_test, predictions), 2)
    conf_matrix = confusion_matrix(y_test, predictions)

    if sl == None or pl == None:
        your_sample = {}
    else:
        sample_distances, sample_indices = knn.kneighbors([[sl, pl]], k)
        your_sample = {
            'location': [sl, pl],
            'predict_class': knn.predict([[sl, pl]])[0],
            'distances' : sample_distances[0],
            'near_sample': X_train[sample_indices][0],
            'near_class': y_train[sample_indices][0]
        }

    prediction_list = []
    distanses, indices = knn.kneighbors(X_test, k)
    for truly ,test, indice, distanse, prediction in zip(y_test, X_test, indices, distanses ,predictions):
        prediction_list.append({
            'location': test,
            'truly_sample': truly,
            'predict_sample': prediction,
            'distanse' : distanse,
            'near_sample': X_train[indice],
            'near_class': y_train[indice]
        })

    training_list = []
    for location, label in zip(X_train, y_train):
        training_list.append({
            'location': location,
            'label': label
        })

    return knn, X_test, y_test, X_train, y_train, accuracy, your_sample, conf_matrix, prediction_list, training_list

def visualization_training(clf, k, X_train, y_train):
    X = X_train
    y = y_train
    plot_decision_regions(X, y, clf=clf, legend=2)
    # plt.xlabel('Chiều dài đài hoa')
    # plt.ylabel('Chiều dài cánh hoa')
    # plt.title('Biểu đồ phân lớp Iris Dataset bằng KNN trên tập training với K = ' + str(k))

    trainIObytes = io.BytesIO()
    plt.savefig(trainIObytes, format='png')
    trainIObytes.seek(0)
    plt.clf()
    return base64.b64encode(trainIObytes.read())

def visualization_testing(clf, k, X_test, y_test):
    X = X_test
    y = y_test
    plot_decision_regions(X, y, clf=clf, legend=2)
    # plt.xlabel('Chiều dài đài hoa')
    # plt.ylabel('Chiều dài cánh hoa')
    # plt.title('Biểu đồ kiểm tra độ chính xác thuật toán KNN với K = ' + str(k))

    testIObytes = io.BytesIO()
    plt.savefig(testIObytes, format='png')
    testIObytes.seek(0)
    plt.clf()
    return base64.b64encode(testIObytes.read())


@api_view(['GET'])
def knn(request):
    k = request.GET['k']
    sl = request.GET['sl']
    pl = request.GET['pl']
    is_show_graph = request.GET['show_graph']

    knn, X_test, y_test, X_train, y_train, accuracy, your_sample, conf_matrix, prediction_list, training_list = iris_knn_classification(
        int(k), float(sl), float(pl))
    logging.warning(is_show_graph)
    test_graph=''
    train_graph=''
    if is_show_graph=='true':
        test_graph = visualization_testing(knn, k, X_test, y_test)
        train_graph = visualization_training(knn, k, X_train, y_train)

    data = {
        'accuracy': accuracy,
        'your_sample': your_sample,
        'k': k,
        'confusion_matrix': conf_matrix,
        'prediction_list': prediction_list,
        'training_list': training_list,
        'test_graph': test_graph,
        'train_graph': train_graph
    }

    return Response(data)
