from plots.plot_mult_stat import plot_components, plot_loadings_heat
from helpers.helpers import Helpers as hp
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
from os.path import join

def calc_pca(df: pd.DataFrame, path: str, properties: dict):
    print('processing pca...')
    names = df['name']
    df.drop(['name','height'], axis=1, inplace=True) 
    
    
    scalar = StandardScaler()
    scalar = MinMaxScaler()
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
    
    
    
    file_path = join(path, 'results', 'statistics')
    hp.save_df(components, file_path, 'PCA_components')
    plot_components(df_x_pca, join(path,'plots', 'statistics'), properties, names, name='PCA', col_names=['PC1', 'PC2','PC3'])
    process_loadings(components, join(path,'plots', 'statistics'), properties)


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