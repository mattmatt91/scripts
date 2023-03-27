from filereader.read_files import scan_folder
from helpers.helpers import Helpers as hp
from multistatistics.do_statistics import do_statistics
from compare.compare_measurements import compare
from plots.plot_feauters import plot_features


root_path = 'D:\\test_dataset'
# root_path = 'D:\\safe_combustion'


if __name__ == '__main__':
    properties = hp.read_json('properties', 'properties.json')
    # scan_folder(root_path, properties)
    # do_statistics(root_path, properties, statistic=True,pca=True, lda=True)  # computing statistics
    compare(root_path, properties)
    exit()
    plot_features(root_path, properties)  # plots feauteres with all samples
    print('finished')
