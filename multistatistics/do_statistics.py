
from os.path import join, exists
import pandas as pd
from multistatistics.lda import calc_lda
from multistatistics.pca import calc_pca
from multistatistics.statistics import get_statistics


def do_statistics(path, properties, statistic=True, pca=True, lda=True):
    # preparing result.csv for statistics
    file_path = join(path, 'results', 'results.csv')
    if not exists(file_path):
        print('no results available, please run read files from main')
    else:
        df = pd.read_csv(file_path ,delimiter=';', decimal=',')
        # select sensors to drop for statisctics e.g. name
        df.fillna(0)
        df_stat = df.drop(['datetime', 'height',  'number', 'rate'], axis=1)
        df_stat.set_index(['name'], inplace=True)
        df_mult_stat = df.drop(['datetime', 'number', 'rate'], axis=1)
        df_mult_stat.set_index(['sample'], inplace=True)

        # do statistics
        if statistic:
            get_statistics(df_stat, path)
        if pca:
            calc_pca(df_mult_stat, path, properties)
        # if lda:
        #     calc_lda(df, path, df_names, properties)
        # exit()


if __name__ == '__main__':
    path = 'E:\\Promotion\Daten\\29.06.21_Paper_reduziert'
    do_statistics(path)
