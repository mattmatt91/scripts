
from os.path import join
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import LeaveOneOut
from sklearn import metrics
# from roc import get_roc
from helpers.helpers import Helpers as hp
from plots.plot_mult_stat import plot_components, plot_loadings_heat


def get_statistics(df, path):
    print('processing statistics...')
    samples = df['sample'].unique().tolist()
    statistics_list = {}
    df_mean = pd.DataFrame()
    df_std = pd.DataFrame()
    path = join(path, 'results', 'statistics')

    for sample in samples:
        df_sample = df[df['sample'] == sample].describe()
        hp.save_df(df_sample, path, sample)
        statistics_list[sample] = df_sample
        df_mean[sample] = df_sample.T['mean']
        df_std[sample] = df_sample.T['std']

    hp.save_df(df_mean.T, path, 'mean')
    hp.save_df(df_std.T, path, 'std')


def calc_pca(df: pd.DataFrame, path: str, properties: dict):
    print('processing pca...')

    # df.drop(drop_list, axis=1, inplace=True)
    names = df['name']
    df.drop(['name','height'], axis=1, inplace=True) # removing hehgth ?????
    scalar = StandardScaler()
    scalar.fit(df)
    scaled_data = scalar.transform(df)
    pca = PCA(n_components=3)
    pca.fit(scaled_data)
    x_pca = pca.transform(scaled_data)
    # create df for plotting with PCs and samples as index
    df_x_pca = pd.DataFrame(x_pca, index=df.index,
                            columns='PC1 PC2 PC3'.split())
    components = pd.DataFrame(
        pca.components_, columns=df.columns, index=['PC1', 'PC2', 'PC3'])
    hp.save_df(components, join(path, 'results', 'statistics'), 'PCA_components')
    plot_components(df_x_pca, path, properties, names, name='PCA',)

    # Loadings
    process_loadings(components, path, properties)


# creates a df with the loadings and a column for sensor and feature
def process_loadings(df, path, properties):

    df_components = get_true_false_matrix(df)
    plot_loadings_heat(df_components, path, properties)
    hp.save_df(df, path, 'PCA_loadings')


def get_true_false_matrix(df):
    df = df.T
    sensors = [x[:x.find('_')] for x in df.index.tolist()]
    df['sensors'] = sensors
    features = [x[x.find('_')+1:] for x in df.index.tolist()]
    df['features'] = features
    return df


def convert_df_pd(df):
    df.reset_index(drop=True, inplace=True)
    # formt den df um sodass pc keine Spalten mehr sind
    pcs = 'PC1 PC2 PC3'.split()
    df_converted = pd.DataFrame()
    for i, m, k in zip(df['sensors'], df['features'], range(len(df['features']))):
        for n in pcs:
            df_converted = df_converted.append(
                {'sensor': i, 'feature': m, 'PC': n, 'value': df.iloc[k][n]}, ignore_index=True)
    return df_converted


def calc_lda(df, path, df_names, properties, browser=True, dimension=True, drop_keywords=[]):
    print('processing lda...')
    # drop_list = create_droplist(drop_keywords, df.columns)
    # df.drop(drop_list, axis=1, inplace=True)

    scalar = StandardScaler()
    scalar.fit(df)
    scaled_data = scalar.transform(df)
    lda = LinearDiscriminantAnalysis(n_components=3)
    x_lda = lda.fit(scaled_data, df.index).transform(scaled_data)
    df_x_lda = pd.DataFrame(x_lda, index=df.index, columns='C1 C2 C3'.split())

    axis_label = 'C'
    plot_components(df_x_lda, df_names, path, properties,
                    name='LDA', browser=browser, dimension=dimension, axis_label=axis_label)
    cross_validate(lda, scaled_data, df.index, path, properties)


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
    save_jpeg(fig, path, 'heatmap_crossvalidation_LDA')
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


def calculate(path, properties, statistic=True, pca=True, lda=True):
    # preparing result.csv for statistics
    df = pd.read_csv(join(path, 'results', 'results.csv'),
                     delimiter=';', decimal=',')
    # select sensors to drop for statisctics e.g. name
    df.fillna(0)
    df_stat = df.drop(['datetime', 'height' ,  'number', 'rate'], axis=1)
    df_stat.set_index(['name'], inplace=True)
    df_mult_stat = df.drop(['datetime', 'number', 'rate'], axis=1)
    df_mult_stat.set_index(['sample'], inplace=True)

    # do statistics
    if statistic:
        get_statistics(df_stat, path)
    if pca:
        calc_pca(df_mult_stat, path, properties)

    if lda:
        calc_lda(df, path, df_names, properties)
    exit()


if __name__ == '__main__':
    path = 'E:\\Promotion\Daten\\29.06.21_Paper_reduziert'
    calculate(path)
