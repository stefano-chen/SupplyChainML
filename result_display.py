import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay


def show():
    plt.show()

# Parent class to create and print plots
class ResultDisplay:

    def __init__(self):
        self.scores = []

    def add_result(self, result: dict) -> None:
        self.scores.append(result)

    def plot(self):
        pass


# This class and the MultiClassResultDisplay could be a unique class called ClassificationResultDisplay
# that uses a boolean attribute to determ if to print a binary classification plot or a multiclass one
class BinaryResultDisplay(ResultDisplay):

    def __init__(self):
        super().__init__()
        self.fig, self.axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))  # type: plt.Figure, list[plt.Axes]
        plt.gcf().canvas.manager.set_window_title('Binary Classification')
        self.axes[0].set_title('Accuracy')
        self.axes[1].set_title('ROC_AUC')

    def plot(self):
        x_labels = []
        accuracies = []
        colors = []
        fpr = []
        tpr = []
        auc = []
        for result in self.scores:
            x_labels.append(result['name'])
            accuracies.append(result['accuracy'])
            fpr.append(result['fpr'])
            tpr.append(result['tpr'])
            auc.append(result['auc'])
            colors.append(result['color'])
            RocCurveDisplay(fpr=result['fpr'], tpr=result['tpr'], roc_auc=result['auc'], estimator_name=result['name'],
                            pos_label=1).plot(self.axes[1], color=result['color'])
        self.axes[0].bar(x_labels, accuracies, color=colors)


class MultiClassResultDisplay(ResultDisplay):

    def __init__(self):
        super().__init__()
        self.fig, self.axes = plt.subplots(figsize=(10, 6))  # type: plt.Figure, plt.Axes
        plt.gcf().canvas.manager.set_window_title('Multiclass Classification')
        self.axes.set_title('Accuracy')

    def plot(self):
        x_labels = []
        accuracies = []
        colors = []
        for result in self.scores:
            x_labels.append(result['name'])
            accuracies.append(result['accuracy'])
            colors.append(result['color'])
        self.axes.bar(x_labels, accuracies, color=colors)


class RegressionResultDisplay(ResultDisplay):

    def __init__(self):
        super().__init__()
        self.fig, self.axes = plt.subplots(nrows=1, ncols=4, figsize=(16, 8))  # type: plt.Figure, list[plt.Axes]
        plt.gcf().canvas.manager.set_window_title('Regression')
        self.axes[0].set_title('MAE')
        self.axes[1].set_title('MSE')
        self.axes[2].set_title('RMSE')
        self.axes[3].set_title('MAPE')

    def plot(self):
        x_labels = []
        mae = []
        mse = []
        rmse = []
        mape = []
        colors = []
        for result in self.scores:
            x_labels.append(result['name'])
            mae.append(result['mae'])
            mse.append(result['mse'])
            rmse.append(result['rmse'])
            mape.append(result['mape'])
            colors.append(result['color'])
        self.axes[0].bar(x_labels, mae, color=colors)
        self.axes[1].bar(x_labels, mse, color=colors)
        self.axes[2].bar(x_labels, rmse, color=colors)
        self.axes[3].bar(x_labels, mape, color=colors)
