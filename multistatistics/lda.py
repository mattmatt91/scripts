from plots.plot_mult_stat import plot_components, plot_loadings_heat
from helpers.helpers import Helpers as hp
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import LeaveOneOut
import matplotlib.pyplot as plt
import pandas as pd
from os.path import join
import seaborn as sns


def calc_lda(df, path, properties):
    print('processing lda...')
    names = df.index

    scalar = StandardScaler()
    # scalar = MinMaxScaler()
    scalar.fit(df)
    scaled_data = scalar.transform(df)

    lda = LinearDiscriminantAnalysis(n_components=3)
    x_lda = lda.fit(scaled_data, df.index).transform(scaled_data)
    df_x_lda = pd.DataFrame(x_lda, index=df.index, columns='C1 C2 C3'.split())
    file_path = join(path, 'results', 'plots', 'statistics')
    plot_components(df_x_lda,  file_path, properties,
                    names, name='LDA', col_names=['C1', 'C2', 'C3'])

    # cross_validate(lda, scaled_data, df.index, path, properties)


def cross_validate(function, x, y, path, properties):
    plot_properties = properties['plot_properties']["confusion_matrix"]
    df_result = pd.DataFrame()
    loo = LeaveOneOut()
    loo.get_n_splits(x)
    for train_index, test_index in loo.split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        function.fit(x_train, y_train).transform(x_train)
        predictions = function.predict(x_test)
        result = pd.DataFrame({'true': y_test, 'predict': predictions})
        result['value'] = result['predict'] == result['true']
        df_result = df_result.append(result, ignore_index=True)
    print('error rate: ' + str((df_result[df_result['value']
          == False]['value'].count()/len(df_result))*100) + '%')

    df_conf = create_confusion(df_result)
    fig, ax = plt.subplots(
        figsize=plot_properties['size'], dpi=plot_properties['dpi'])
    count_size = plot_properties['count_size']
    sns.heatmap(df_conf.fillna(0), linewidths=.5, annot=True, fmt='g',
                cbar=False, cmap="viridis", ax=ax, annot_kws={"size": count_size})
    ax.set_ylabel('true', fontsize=plot_properties['label_size'])
    ax.set_xlabel('predicted', fontsize=plot_properties['label_size'])
    plt.yticks(size=plot_properties['font_size'], rotation=30)
    plt.xticks(size=plot_properties['font_size'], rotation=30)
    plt.tight_layout()
    hp.save_jpeg(fig, path, 'heatmap_crossvalidation_LDA')
    # plt.show()
    plt.close()

    # computing cor curve
    get_roc(df_result, path, properties)


def create_confusion(df):
    """
    This function creates a confusion matrix with the passed results of the cross validation.

    Args:
        df (pandas.DataFrame): DataFrame with predicted and true values from all measurements

    Returns:
        df_conf (pandas.DataFrame): DataFrame with confusion matrix. rows are true and columns predicted values
    """
    labels = df['true'].unique()
    df_conf = pd.DataFrame(columns=labels, index=labels)
    for i in df['true'].unique():
        for n in df['true'].unique():
            value = df[(df['true'] == i) & (
                df['predict'] == n)]['true'].count()
            df_conf.loc[i, n] = value  # zeilen sind true spalten predict
    return df_conf
