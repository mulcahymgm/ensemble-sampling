import pandas as pd
from sklearn import metrics


class ScoreCollector:
    def __init__(self, file_name="results.csv"):
        self._scores_dict = {}
        self._file_name = file_name

    def collect(self, experiment_name, classifier_name, y_true, y_pred):
        # import classification_report, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

        scores = {}
        scores["Accuracy Score"] = metrics.accuracy_score(
            y_true, y_pred
        )
        scores["Balanced Accuracy Score"] = metrics.balanced_accuracy_score(
            y_true, y_pred
        )
        scores["F1 Score"] = metrics.f1_score(y_true, y_pred)
        scores["Precision Score"] = metrics.precision_score(y_true, y_pred)
        scores["Recall Score"] = metrics.recall_score(y_true, y_pred)
        scores["ROC AUC Score"] = metrics.roc_auc_score(y_true, y_pred)
        # binary case can access the true and false positives and negatives
        (
            scores["True Negatives"],
            scores["False Positives"],
            scores["False Negatives"],
            scores["True Positives"],
        ) = metrics.confusion_matrix(y_true, y_pred).ravel()

        if experiment_name not in self._scores_dict:
            self._scores_dict[experiment_name] = {}

        if classifier_name not in self._scores_dict[experiment_name]:
            self._scores_dict[experiment_name][classifier_name] = {}

        self._scores_dict[experiment_name][classifier_name] = scores

    def flush(self):
        dataframe = pd.DataFrame(self._scores_dict)
        dataframe.to_csv(self._file_name)
