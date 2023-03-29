from filereader.read_files import scan_folder
from helpers.helpers import Helpers as hp
from multistatistics.do_statistics import do_statistics
from compare.compare_measurements import compare
from plots.plot_feauters import plot_features
import os


# root_path = 'C:\\Users\\User\\Desktop\\test_dataset'  # set path to data repository
root_path = 'D:\\safe_combustion'
os.environ["DATA_PATH"] = root_path

if __name__ == '__main__':
    scan_folder()  # reads all measurement and evaluates them
    # processes statisctics
    do_statistics(statistic=True, pca=False, lda=False)
    compare()  # does some plots of sensor signals
    plot_features()  # plots feauteres with all samples
    print('finished')
