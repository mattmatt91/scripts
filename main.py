from filereader.read_files import scan_folder
from helpers.helpers import Helpers as hp
from multistatistics.do_statistics import do_statistics
from compare.compare_measurements import compare
from plots.plot_feauters import plot_features
import os


# hello world


# root_path = 'C:\\Users\\User\\Desktop\\test_dataset'  # set path to data repository
root_path = 'C:\\Users\\User\\Desktop\\safe_combustion'
os.environ["DATA_PATH"] = root_path


if __name__ == '__main__':
    # reading properties for plots and evatluation
    properties = hp.read_json('properties', 'properties.json')
    # scan_folder(root_path, properties) # reading files and merging
    do_statistics(properties, statistic=False, pca=True,
                  lda=True)  # computing statistics
    # compare(root_path, properties)
    # plot_features(root_path, properties)  # plots feauteres with all samples
    print('finished')
