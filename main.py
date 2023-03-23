from filereader.read_files import scan_folder
from helpers.helpers import Helpers as hp
from multistatistics.do_statistics import calculate
from compare.compare_measurements import compare
from plots.plot_feauters import plot_features

if __name__ == '__main__':
    root_path = 'D:\\test_dataset'
    # root_path = 'C:\\Users\\mmuhr-adm\\Desktop\\02.03.2023'
    properties = hp.read_json('properties', 'properties.json')
    # scan_folder(root_path, properties)
    calculate(root_path, properties, statistic=False, pca=True, lda=True) # computing statistics
    # compare(root_path, properties)
    # plot_features(root_path, properties) # plots feauteres with all samples
    print('finished')
