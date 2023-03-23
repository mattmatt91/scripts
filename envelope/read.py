import pandas as pd
import datetime 
import numpy as np

class ReadFile:
    def read(path):
        df = pd.read_csv(path, decimal=',',delimiter=';')
        df['datatime'] = [datetime.datetime.fromtimestamp(i) for i in df['time']]
        df.set_index('time', inplace=True)
        # print(df)
        df= df.loc[1.2:]
        data = df['Mikro'].abs()
        return np.array(data.to_list()), df.index.tolist()