import plotly.express as px
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import json
import sys
import os
import pandas as pd
import numpy as np


def test_melt():
    technologies = ({
        'Courses': ["Spark", "PySpark", "Hadoop", "Pandas"],
        'Fee': [22000, 25000, 30000, 35000],
        'Duration': ['30days', '50days', '40days', '35days'],
        'Discount': [1000, 2000, 2500, 1500]
    })
    df = pd.DataFrame(technologies)
    print(df)
    # Example 1: Use pandas.melt() function
    df2 = pd.melt(df, id_vars=['Courses'], value_vars=['Fee'])
    print(df2)
    # Example 2:  Using id_vars & value_vars
    # to melt() of a pandas dataframe
    df2 = pd.melt(df, id_vars=['Courses'], value_vars=['Fee', 'Discount'])
    print(df2)
    # Example 3:  Using var_name & value_name
    # to melt() of a pandas dataframe
    df2 = pd.melt(df, id_vars=['Courses'], value_vars=['Fee'],
                  var_name='Courses Fees', value_name='Courses Fee')
    print(df2)
    # Example 4:  Using ignore_index
    df2 = pd.melt(df, id_vars=['Courses'], value_vars=[
                  'Fee', 'Duration'], ignore_index=False)
    print(df2)
    # Example 5: Use multi-index columns
    df.columns = [list('ABCD'), list('EFGH')]


def change_height():
    root = "D:\\"
    path = os.path.join(root, "safe_combustion")
    print(path)
    for r, d, f in os.walk(path):
        for file in f:
            file_path = os.path.join(r, file)
            if file_path.find('.json') >= 0:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                if data['height'] != 40:
                    data['height'] = 40
                    print(file_path)
                    with open(file_path, 'w') as json_file:
                        json.dump((data), json_file)


def test_lda():
    # print(__doc__)

    iris = datasets.load_iris()

    X = iris.data
    y = iris.target
    target_names = iris.target_names
    pca = PCA(n_components=2)
    X_r = pca.fit(X).transform(X)

    lda = LDA(n_components=2)
    X_r2 = lda.fit(X, y).transform(X)

    # Percentage of variance explained for each components
    print('explained variance ratio (first two components): %s'
          % str(pca.explained_variance_ratio_))

    plt.figure()
    for c, i, target_name in zip("rgb", [0, 1, 2], target_names):
        plt.scatter(X_r[y == i, 0], X_r[y == i, 1], c=c, label=target_name)
    plt.legend()
    plt.title('PCA of IRIS dataset')

    plt.figure()
    for c, i, target_name in zip("rgb", [0, 1, 2], target_names):
        plt.scatter(X_r2[y == i, 0], X_r2[y == i, 1], c=c, label=target_name)
    plt.legend()
    plt.title('LDA of IRIS dataset')

    plt.show()


def get_key_by_value(data:dict, value:int):
        for key, val in data.items():
            print(key, val)
            if val == value:
                return val


def test_px():
    df = px.data.iris()
    df["e"] = df["sepal_width"]/100
    fig = px.scatter(df, x="sepal_width", y="sepal_length", color="species",
                    error_x="e", error_y="e")
    fig.show()


test_px()