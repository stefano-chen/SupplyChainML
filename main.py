# Consider the DataCo Smart Supply Chain Dataset
# 1. Given the objective of classifying if an order is marked as late delivery,
#    design and implement an ML procedure which answers the question:
#         what is the best classification technique?
# 2. Given the objective of predicting the sales of each order,
#    design and implement an ML procedure which answers the question:
#         what is the best regression technique?
import math

import numpy as np
import pandas
import pandas as pd
from result_display import ResultDisplay, BinaryResultDisplay, MultiClassResultDisplay, RegressionResultDisplay, show
from sklearn import set_config
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

set_config(transform_output='pandas')

k_cross_validation = 3


# Function to load the data from the csv file
def data_load() -> pandas.DataFrame:
    df = pd.read_csv('./datasets/DataCoSupplyChainDataset.csv', encoding='latin-1')
    df['Customer Full Name'] = df['Customer Fname'].astype(str) + df['Customer Lname'].astype(str)
    df.drop(columns=['Customer Email', 'Product Status', 'Customer Password', 'Customer Street', 'Customer Fname',
                     'Customer Lname', 'Latitude', 'Longitude', 'Product Description', 'Product Image', 'Order Zipcode',
                     'shipping date (DateOrders)', 'Late_delivery_risk'], inplace=True)

    most_frequent_country = df['Order Country'].value_counts()[:10].keys()
    df = df[df['Order Country'].isin(most_frequent_country)]

    selected_categories = ['Outdoor', 'Fitness', 'Technology']
    df = df[df['Department Name'].isin(selected_categories)]
    df.dropna(inplace=True)
    return df


# Function to preprocess the data for a classification problem,
# I used the binary parameters to manage binary classification and multiclass classification
def classification_preprocessing(data, binary: bool):
    enc = LabelEncoder()
    scaler = StandardScaler()
    scaling_columns = list(data.select_dtypes(exclude=['object']).keys())
    if binary:
        data['Delivery Status'] = np.where(data['Delivery Status'] == 'Late delivery', 1, 0)
    object_type_columns = list(data.select_dtypes(include=['object']).keys())
    data[object_type_columns] = data[object_type_columns].apply(enc.fit_transform)
    data[scaling_columns] = scaler.fit_transform(data[scaling_columns])
    x = data.drop(['Delivery Status'], axis=1).to_numpy()
    y = data[['Delivery Status']].to_numpy().ravel()
    return train_test_split(x, y, test_size=0.2, random_state=26)


# Function to preprocess the data for a regression problem
def regression_preprocessing(data: pd.DataFrame):
    enc = LabelEncoder()
    scaler = StandardScaler()
    scaling_columns = list(data.select_dtypes(exclude=['object']).keys())
    scaling_columns.remove('Sales')
    categorical_columns = list(data.select_dtypes(include=['object']).keys())
    data[categorical_columns] = data[categorical_columns].apply(enc.fit_transform)
    data[scaling_columns] = scaler.fit_transform(data[scaling_columns])
    x = data.drop(columns=['Sales'], axis=1).to_numpy()
    y = data['Sales'].to_numpy()
    return train_test_split(x, y, test_size=0.2, random_state=26)


# Function to fit and calculate the performance indexes of a given estimator(classifier)
def classification(classifier, name: str, parameters, x_train, x_test, y_train, y_test, color, binary=True):
    cls = GridSearchCV(classifier, param_grid=parameters, scoring='accuracy', cv=k_cross_validation)
    cls.fit(x_train, y_train)
    y_pred = cls.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    if binary:
        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        roc_auc = auc(fpr, tpr)
        return {
            'name': name,
            'accuracy': accuracy,
            'fpr': fpr,
            'tpr': tpr,
            'auc': roc_auc,
            'color': color
        }
    return {'name': name, 'accuracy': accuracy, 'color': color}


# Function to fit and calculate the performance indexes of a given estimator(regressor)
def regression(regressor, name: str, x_train, x_test, y_train, y_test, *, color='blue') -> dict:
    """
        Fit and return a dictionary containing the performance indexes

        Parameters
            regressor : an instance of a regression technique
            name : the name of the regression technique. It's used in the plot

        Return
            A dictionary with the following keys:
                name, mae, mse, rmse, mape, color
    """
    rgs = regressor
    rgs.fit(x_train, y_train)
    y_pred = rgs.predict(x_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = math.sqrt(mse)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    return {
        'name': name,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'mape': mape,
        'color': color
    }


# Function to compare all the classification technique
# The binary parameter allow to decide if apply a binary classification or a multiclass one
def label_classification(responses: ResultDisplay, dataset, binary):
    x_train, x_test, y_train, y_test = classification_preprocessing(dataset, binary=binary)
    # Decision Tree
    responses.add_result(classification(DecisionTreeClassifier(), 'TREE', {
        'criterion': ['gini', 'entropy', 'log_loss']
    }, x_train, x_test, y_train, y_test, 'red', binary=binary))
    # Random Forest
    responses.add_result(classification(RandomForestClassifier(), 'RF', {
        'n_estimators': [50, 100],
        'criterion': ['gini', 'entropy', 'log_loss']
    }, x_train, x_test, y_train, y_test, 'blue', binary=binary))
    # Support Vector Machine
    responses.add_result(classification(SVC(), 'SVM', {
        'C': [10000]
    }, x_train, x_test, y_train, y_test, 'pink', binary=binary))
    # Naive Bayes
    responses.add_result(classification(GaussianNB(), 'Naive Bayes', {
    }, x_train, x_test, y_train, y_test, 'violet', binary=binary))
    # K Nearest Neighbor
    responses.add_result(classification(KNeighborsClassifier(), 'KNN', {
        'n_neighbors': [100],
        'weights': ['distance'],
        'p': [2]
    }, x_train, x_test, y_train, y_test, 'orange', binary=binary))


# Function to compare all the regression technique
def sales_regression(responses: ResultDisplay, dataset):
    x_train, x_test, y_train, y_test = regression_preprocessing(dataset)
    # Decision Tree
    responses.add_result(regression(DecisionTreeRegressor(), 'Tree', x_train, x_test, y_train, y_test, color='red'))
    # Random Forest
    responses.add_result(regression(RandomForestRegressor(), 'RF', x_train, x_test, y_train, y_test, color='blue'))
    # Support Vector Regression
    responses.add_result(regression(SVR(), 'SVR', x_train, x_test, y_train, y_test, color='pink'))
    # K Nearest Neighbors
    responses.add_result(regression(KNeighborsRegressor(), 'KNN', x_train, x_test, y_train, y_test, color='orange'))


if __name__ == '__main__':
    dataframe = data_load()
    binary_result = BinaryResultDisplay()
    multiclass_result = MultiClassResultDisplay()
    regression_result = RegressionResultDisplay()
    label_classification(binary_result, dataframe.copy(), binary=True)
    label_classification(multiclass_result, dataframe.copy(), binary=False)
    sales_regression(regression_result, dataframe.copy())
    binary_result.plot()
    multiclass_result.plot()
    regression_result.plot()
    show()
