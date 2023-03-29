from plots.plot_mult_stat import plot_components, plot_loadings_heat
from helpers.helpers import Helpers as hp
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler, Normalizer, MaxAbsScaler, RobustScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import LeaveOneOut
import matplotlib.pyplot as plt
import pandas as pd
from os.path import join
import seaborn as sns


def calc_lda(features: pd.DataFrame, infos: dict, properties: dict):
    print('processing lda...')
    # prepare labels
    sample_dict, sample_numbers = hp.sample_to_numbers(infos['sample'])

    # scale data
    # hat nur geringen Einfluss auf die LDA, aber visuell erkennbar. Güte noch zu evaluieren
    scalar = MinMaxScaler()
    #scalar = StandardScaler()
    #scalar = Normalizer()
    #scalar = MaxAbsScaler()
    #scalar = RobustScaler()
    scalar.fit(features)
    scaled_data = scalar.transform(features)

    # perform lda
    lda = LinearDiscriminantAnalysis(n_components=3)
    x_lda = lda.fit(scaled_data, sample_numbers).transform(scaled_data)

    # create df
    df_x_lda = pd.DataFrame(x_lda, index=infos['sample'],
                            columns=['C1', 'C2', 'C3'])
    # plot lda
    plot_components(df_x_lda,
                    properties,
                    infos,
                    name='LDA')

    # cross_validate(lda, features, sample_numbers)


def cross_validate(function, x, y):
    df_result = pd.DataFrame()
    loo = LeaveOneOut()
    loo.get_n_splits(x)
    for train_index, test_index in loo.split(x):
        # split dataset
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        print(x_train, x_test )
        print(y_train, y_test )
        # function.fit(x_train, y_train).transform(x_train)
        # predictions = function.predict(x_test)
        # print(predictions)
        # result = pd.DataFrame({'true': y_test, 'predict': predictions})
        # result['value'] = result['predict'] == result['true']
        # df_result = df_result.append(result, ignore_index=True)
    # print('error rate: ' + str((df_result[df_result['value']
    #       == False]['value'].count()/len(df_result))*100) + '%')

    # df_conf = create_confusion(df_result)
    
def plot_confusion(data:pd.DataFrame):
    print(data)
    # fig, ax = plt.subplots(
    #     figsize=plot_properties['size'], dpi=plot_properties['dpi'])
    # count_size = plot_properties['count_size']
    # sns.heatmap(df_conf.fillna(0), linewidths=.5, annot=True, fmt='g',
    #             cbar=False, cmap="viridis", ax=ax, annot_kws={"size": count_size})
    # ax.set_ylabel('true', fontsize=plot_properties['label_size'])
    # ax.set_xlabel('predicted', fontsize=plot_properties['label_size'])
    # plt.yticks(size=plot_properties['font_size'], rotation=30)
    # plt.xticks(size=plot_properties['font_size'], rotation=30)
    # plt.tight_layout()
    # hp.save_jpeg(fig, path, 'heatmap_crossvalidation_LDA')
    # # plt.show()
    # plt.close()

    # computing cor curve
    # get_roc(df_result, path, properties)


def create_confusion(df):
    labels = df['true'].unique()
    df_conf = pd.DataFrame(columns=labels, index=labels)
    for i in df['true'].unique():
        for n in df['true'].unique():
            value = df[(df['true'] == i) & (
                df['predict'] == n)]['true'].count()
            df_conf.loc[i, n] = value  # zeilen sind true spalten predict
    return df_conf
