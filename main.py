from filereader.read_files import scan_folder
from helpers.helpers import Helpers as hp
from multistatistics.do_statistics import do_statistics
from compare.compare_measurements import compare
from plots.plot_feauters import plot_features
import os
import sys


if __name__ == '__main__':
    arg = sys.argv[1]
    root_path = sys.argv[2]

    if len(sys.argv) < 3:
        print("Usage: python main.py -a <path to data>")
        sys.exit(1)

    if arg == "-p":
        print("Argument value:", root_path)
        os.environ["DATA_PATH"] = root_path

        # readfiles and create results file
        scan_folder()

        # specify the appearance of features
        how_to_plot = {"size": "combustion",
                       "symbol": "height",
                       "color": "sample",
                       "none": "height"}

        seperation_key = 'sample'
        # seperation_key = 'height' # use this for selecting seperator for LDA

        # selector = {'combustion_bool':True} # always pass dict, if no selection use 'none' as key
        # always pass dict, if no selection use 'none' as key
        selector = {'none': True}

        # do_statistics(seperation_key,
        #               how_to_plot,
        #               selector,
        #               statistic=True,
        #               pca=True,
        #               lda=True)

        compare()  # does some plots of sensor signals
        plot_features()  # plots feauteres with all samples
        print('finished')

    else:
        print("Invalid argument:", arg)
        exit()


# new samples missung in sensor plot
