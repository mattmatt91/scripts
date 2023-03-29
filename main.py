from filereader.read_files import scan_folder
from helpers.helpers import Helpers as hp
from multistatistics.do_statistics import do_statistics
from compare.compare_measurements import compare
from plots.plot_feauters import plot_features
import os


# root_path = 'C:\\Users\\User\\Desktop\\test_dataset'  # set path to data repository
root_path = 'D:\\test_dataset'

os.environ["DATA_PATH"] = root_path


if __name__ == '__main__':  
    # scan_folder() 
    # do_statistics(statistic=True, pca=False, lda=False) 
    compare()
    # plot_features(root_path, properties)  # plots feauteres with all samples
    print('finished')
