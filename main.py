from filereader.read_files import scan_folder
from helpers.helpers import Helpers as hp
from multistatistics.do_statistics import do_statistics
from compare.compare_measurements import compare
from plots.plot_feauters import plot_features
from machine_learning.machine_learning import do_machine_learning
import os


# root_path = 'C:\\Users\\User\\Desktop\\test_dataset'  # set path to data repository
root_path = 'D:\\safe_combustion'
# root_path = 'D:\\test_plot'

if __name__ == '__main__':
    os.environ["DATA_PATH"] = root_path
    scan_folder()  # reads all measurement and evaluates them
    do_statistics(statistic=True, pca=True, lda=True)# processes statisctics
    # do_machine_learning() # 
    compare()  # does some plots of sensor signals
    plot_features()  # plots feauteres with all samples
    print('finished')




# blank 3 kein signal????