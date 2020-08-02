from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm


class Experiment:
    def __init__(self, name, data_function, score_collector):
        self._name = name
        self._X_train, self._X_test, self._y_train, self._y_test = data_function()
        self.score_collector = score_collector

    def fit(self):

        svm_clf = svm.SVC()
        rf_clf = RandomForestClassifier()
        mlp_clf = MLPClassifier()
        knn_clf = KNeighborsClassifier()

        classfiers = [
            ("SVM", svm_clf),
            ("Random Forest", rf_clf),
            ("Multilayer Percepteron", mlp_clf),
            ("K Nearest Neighbours", knn_clf),
        ]

        for classifier_name, classifier in classfiers:
            classifier.fit(self._X_train, self._y_train)
            y_pred = classifier.predict(self._X_test)
            self.score_collector.collect(
                self._name, classifier_name, self._y_test, y_pred
            )

        self.score_collector.flush()
