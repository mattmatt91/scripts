from filereader.read_files import scan_folder
from hepers.helpers import read_json
from multistatistics.do_statistics import calculate
from compare.compare_measurements import compare
from plots.plot_feauters import plot_features

if __name__ == '__main__':

    # root_path = 'data\\22.09.2022'
    root_path = 'C:\\Users\\49157\\Desktop\\Paper II\\data\\22.09.2022'
    properties = read_json('', 'properties.json')

    # scan_folder(root_path, properties)
    calculate(root_path, properties, statistic=False, pca=True, lda=True) # computing statistics
    # compare(root_path, properties)
    # plot_features(root_path, properties) # plots feauteres with all samples
    print('finished')
