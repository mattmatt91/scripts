import pandas as pd


data = {'time':[1,2,3], 'data':[34,65,34]}
df = pd.DataFrame(data)
print(df)

df.to_csv('your_file.csv', mode='a', header=False, index=False)