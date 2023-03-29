from plots.plot_mult_stat import plot_components, plot_loadings_heat
from helpers.helpers import Helpers as hp
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, MaxAbsScaler, RobustScaler
from sklearn.decomposition import PCA
import pandas as pd
from os.path import join
import os


def calc_pca(features: pd.DataFrame, infos: dict, properties: dict):
    print('processing pca...')

    # scale data
    # Merklicher Einfluss auf die PCA, aktuell RobustScalar die schlechteste Möglichkeit visuell. Präferiert: MinMax, MaxAbs
    scalar = MinMaxScaler()
    # scalar = StandardScaler()
    # scalar = Normalizer()
    # scalar = MaxAbsScaler()
    # scalar = RobustScaler()
    scalar.fit(features)
    scaled_data = scalar.transform(features)

    # perform pca
    pca = PCA(n_components=3)
    pca.fit(scaled_data)
    x_pca = pca.transform(scaled_data)

    # create df for plotting with PCs and samples as index
    df_x_pca = pd.DataFrame(x_pca, index=infos['sample'],
                            columns=['PC1', 'PC2', 'PC3'])

    components = pd.DataFrame(
        pca.components_, columns=features.columns, index=['PC1', 'PC2', 'PC3'])

    # storing some data
    create_result(pca)

    # safe df
    file_path = join(os.getenv("DATA_PATH"), 'results', 'statistics')
    hp.save_df(components, file_path, 'PCA_components')
    # do plots

    # plotting
    plot_components(df_x_pca,
                    properties,
                    infos,
                    name='PCA')
    process_loadings(components, properties)


def create_result(pca: PCA):
    data = {'explained variance ratio':
            pca.explained_variance_ratio_,
            'singular values':
            pca.singular_values_
            }
    df = pd.DataFrame(data, index=['PC1', 'PC2', 'PC3'])
    path = join(os.getenv("DATA_PATH"), 'results', 'statistics')
    hp.save_df(df, path, 'results_pca', index=True)


# creates a df with the loadings and a column for sensor and feature
def process_loadings(df: pd.DataFrame, properties: dict):
    df_components = get_true_false_matrix(df)
    plot_loadings_heat(df_components, properties)
    path = join(os.getenv("DATA_PATH"), 'results', 'statistics')
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
