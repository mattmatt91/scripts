from read_files import scan_folder, extract_properties
# from do_statistics import calculate
# from compare_measurements import compare
# from plot_feauters import  plot_features

if __name__ == '__main__':

    properties = extract_properties()
    root_path = properties['root_path']

    scan_folder(root_path, properties) # reading raw data
    exit()
    calculate(root_path, properties, statistic=True, pca=True, lda=True, browser=True, dimension=True) # computing statistics
    compare(root_path, properties) # evaluates the files with one sensor, all measurments
    plot_features(root_path, properties) # plots feauteres with all samples 
    
