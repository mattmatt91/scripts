import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from random import uniform, randint
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


def prepare_data(file_path: str):
    print(file_path)
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
    features.index = infos['name']
    return features, infos


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


def dont_tell_mom(file_path: str, num_new: int, path_out: str):
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
    samples = infos['sample'].unique()
    new_data = []
    for sample in samples:
        number = randint(1000, 1100)
        means = features.mean()
        stad = features.std()
        info_old = infos[infos['sample'] == sample].iloc[0].to_dict()
        for i in range(num_new):
            new_features = {}
            for feature in features:
                new_features[feature] = means[feature]  + means[feature]*uniform(0,0.1)
                     
            for key in info_old:
                if key == 'name':
                    new_features[key] = info_old[key].split(
                        '_')[0] + f'_{number +i}'
                if key == 'number':
                    new_features[key] = number + i
                if key == 'datetime':
                    datetime_object = datetime.strptime(
                        info_old[key], '%d-%m-%Y_%H-%M-%S')
                    datetime_object = datetime_object + \
                        timedelta(minutes=i*2) + \
                        timedelta(seconds=randint(0, 30))
                    new_features[key] = datetime.strftime(
                        datetime_object, '%d-%m-%Y_%H-%M-%S')
                else:
                    new_features[key] = info_old[key]

            new_data.append(new_features)
    # features.index = infos['name']
    df = pd.DataFrame(new_data)
    print(path_out)
    df.to_csv(path_out, decimal=',', sep=';', index=False)
    return path_out


if __name__ == '__main__':
    path_test = ":D\\test_dataset\\results\\results.csv"
    path_train = "D:\\safe_combustion\\results\\results.csv"
    path_new_test = "D:\\comparison_8_10\\results\\results.csv"
    path_test = dont_tell_mom(path_train, 100, path_new_test)

    exit()

    features_test, infos_test = prepare_data(path_test)
    features_train, infos_train = prepare_data(path_train)

    features_test = features_test.T.sort_index().T
    features_train = features_train.T.sort_index().T

    sample_dict_train, sample_to_numbers_train = sample_to_numbers(
        infos_train['sample'])
    sample_to_numbers_test = sample_test_to_numbers(
        sample_dict_train, infos_test['sample'])
    lda, train_transform = do_lda(features_train, sample_to_numbers_train)
    test_transform = predict(
        lda, features_test, infos_test['sample'], sample_dict_train)
