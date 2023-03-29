from plots.plot_mult_stat import plot_components, plot_heat
from helpers.helpers import Helpers as hp
from sklearn.preprocessing import MinMaxScaler, Normalizer, MaxAbsScaler, RobustScaler, StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import pandas as pd
from sklearn.metrics import confusion_matrix
import numpy as np


def calc_lda(features: pd.DataFrame, infos: dict, properties: dict):
    print('processing lda...')
    # prepare labels
    sample_dict, sample_numbers = hp.sample_to_numbers(infos['sample'])

    # scale data
    # hat nur geringen Einfluss auf die LDA, aber visuell erkennbar. Güte noch zu evaluieren
    scalar = MinMaxScaler()
    # scalar = StandardScaler()
    # scalar = Normalizer()
    # scalar = MaxAbsScaler()
    # scalar = RobustScaler()
    scalar.fit(features)
    scaled_data = scalar.transform(features)

    # perform lda
    lda = LDA(n_components=3)
    x_lda = lda.fit(scaled_data, sample_numbers).transform(scaled_data)

    # create df
    df_x_lda = pd.DataFrame(x_lda, index=infos['sample'],
                            columns=['C1', 'C2', 'C3'])
    # plot lda
    plot_components(df_x_lda,
                    properties,
                    infos,
                    name='LDA')

    cv_data = cross_validate(lda, x_lda, sample_numbers, sample_dict)
    plot_heat(cv_data)
    # computing cor curve
    # get_roc(df_result, path, properties)


def cross_validate(function: LDA, x: np.array, y: np.array, sample_dict: dict):
    result = []
    for i in range(len(x)):
        x_test = x[i]
        y_test = y[i]
        x_train = np.delete(x, i, 0)
        y_train = np.delete(y, i, 0)
        function.fit(x_train, y_train).transform(x_train)
        prediction = function.predict([x_test])[0]
        result.append({'true': y_test, 'predict': prediction,
                      'result': y_test == prediction})
    data = pd.DataFrame(result)
    data['true_sample'] = [hp.get_key_by_value(
        sample_dict, i) for i in data['true']]
    data['predict_sample'] = [hp.get_key_by_value(
        sample_dict, i) for i in data['predict']]
    accuracy = (data['result'].value_counts().loc[True] /
                len(data['result']))*100
    print(f'accuracy of lda cross validation is {accuracy} %')
    conf_matrix = confusion_matrix(data['true_sample'], data['predict_sample'])
    sample_names = data['true_sample'].unique()
    df_conf_matrix = pd.DataFrame(
        conf_matrix, columns=sample_names, index=sample_names)
    return df_conf_matrix
