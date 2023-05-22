import pandas as pd


df = pd.read_csv('test.txt')
df.set_index('time', inplace=True)

def get_baseline(data: pd.Series, heigth: float, peak_x: float):
    data_first = data[data > heigth]
    # print(data_new.index.to_list())
    ix1 = data.index.get_loc(data_first.index[0])
    x1 = data.index[ix1-1]
    y1 = data[x1]

    data_second = data[peak_x:]
    if len(data_second[data_second < heigth]) > 0:
        ix2 = data.index.get_loc(data_second[data_second < heigth].index[0])
        x2 = data.index[ix2+1]
    else:
        ix2 = data.index.get_loc(data.index[-1])
        x2 = data.index[ix2]
    y2 = data[x2]
    width = x2-x1
    # print({'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'width': width})
    return {'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'width': width}

print(df)