import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from random import uniform, randint, random
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
from datetime import datetime, timedelta


def do_lda(x_train: pd.DataFrame, y_train: pd.Series):
    print(x_train.columns)
    lda = LDA(n_components=3)
    lda.fit(x_train, y_train)
    x_train_transform = lda.transform(x_train)
    return lda, x_train_transform


def predict(lda: LDA, x_test: pd.DataFrame, y_test: pd.Series, sample_dict: dict):
    predictions_int = lda.predict(x_test)
    predictions_propa = lda.predict_proba(x_test)

    predictions = [get_key_by_value(sample_dict, i) for i in predictions_int]

    for i, n in zip(predictions, y_test):
        print(f'pred: {i}  true: {n}')
    return lda.transform(x_test)


def get_datetime(date_string:str, minutes_to_add:int):
    input_format = "%d-%m-%Y_%H-%M-%S"
    output_format = "%d-%m-%Y_%H-%M-%S"
    datetime_obj = datetime.strptime(date_string, input_format)
    datetime_obj += timedelta(minutes=minutes_to_add)
    new_date_string = datetime_obj.strftime(output_format)
    return new_date_string

def get_number(old_number:int, counter:int):
    num = 100*old_number + counter
    return num

def get_name(old_name:str, counter:int):
    name = old_name.split('_')[0]
    try:
        number = int(old_name.split('_')[1])
    except:
        number = int(old_name.split('_')[2])

    new_number = 100*number + counter
    new_name = f'{name}_{new_number}'
    return new_name


def sample_to_numbers(samlpes: pd.Series):
    samles_unique = samlpes.unique()
    sample_dict = {}
    for sample, i in zip(samles_unique, range(len(samles_unique))):
        sample_dict[sample] = i
    numbers = [sample_dict[s] for s in samlpes]
    return sample_dict, numbers


def sample_test_to_numbers(sample_dict: dict, samples: pd.Series):
    return [sample_dict[i] for i in samples]


def get_key_by_value(data: dict, value: int):
    for key, val in data.items():
        if val == value:
            return key


def load_data(file_path: str):
    df = pd.read_csv(file_path, delimiter=';', decimal=',')
    df.fillna(0)  # features without values filled with 0.
    info_cols = ['datetime',
                 'height',
                 'number',
                 'sample',
                 'name',
                 'ball',
                 'rate',
                 'combustion',
                 'combustion_bool']
    infos = df[info_cols]
    features = df.drop(columns=info_cols)
    features.set_index(infos['sample'], inplace=True)
    return infos, features


def generate_data(infos:pd.DataFrame, features:pd.DataFrame, num_new:int):
    # do PCA
    pca = PCA(n_components=3) 
    transformed_data = pca.fit_transform(features)
    df_transformed_data = pd.DataFrame(transformed_data, index=features.index)

    # generate data
    new_data = []
    new_data_infos = []
    labels = []
    for class_label in features.index.unique():
        mean = df_transformed_data[df_transformed_data.index == class_label].mean(
        )
        std = df_transformed_data[df_transformed_data.index ==
                                  class_label].std()
        for i in range(num_new):
            new_measurement = (mean + std * \
                [randint(0, 10)/10 for i in range(3)]).tolist()
            new_data.append(new_measurement)
            labels.append(class_label)
            new_info_template = infos[infos['sample'] == class_label].iloc[0]
            for key in new_info_template.index:
                if key == 'name':
                    new_info_template['number'] = get_number(new_info_template['number'], i)
                elif key == 'number':
                    new_info_template['name'] = get_name(new_info_template['name'], i)
                elif key == 'datetime':
                    new_info_template['datetime'] = get_datetime(new_info_template['datetime'], i)
            new_data_infos.append(new_info_template.tolist())
    inverse_transformed_data = pca.inverse_transform(new_data)
    new_df = pd.DataFrame(inverse_transformed_data, columns=features.columns)
    new_df_infos = pd.DataFrame(new_data_infos, columns=infos.columns)
    merged_df = pd.concat([new_df_infos, new_df], axis=1)
    merged_df.to_csv('fake_results.csv', decimal=',', sep=';')



if __name__ == '__main__':
    # path_test = "results.csv"
    path_train = "results.csv"


    infos, features = load_data(path_train)

    generate_data(infos, features, 10)