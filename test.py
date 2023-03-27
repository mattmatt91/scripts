import json
import sys
import os
import pandas as pd
import numpy as np


def test_melt():
    technologies = ({
        'Courses': ["Spark", "PySpark", "Hadoop", "Pandas"],
        'Fee': [22000, 25000, 30000, 35000],
        'Duration': ['30days', '50days', '40days', '35days'],
        'Discount': [1000, 2000, 2500, 1500]
    })
    df = pd.DataFrame(technologies)
    print(df)
    # Example 1: Use pandas.melt() function
    df2 = pd.melt(df, id_vars=['Courses'], value_vars=['Fee'])
    print(df2)
    # Example 2:  Using id_vars & value_vars
    # to melt() of a pandas dataframe
    df2 = pd.melt(df, id_vars=['Courses'], value_vars=['Fee', 'Discount'])
    print(df2)
    # Example 3:  Using var_name & value_name
    # to melt() of a pandas dataframe
    df2 = pd.melt(df, id_vars=['Courses'], value_vars=['Fee'],
                  var_name='Courses Fees', value_name='Courses Fee')
    print(df2)
    # Example 4:  Using ignore_index
    df2 = pd.melt(df, id_vars=['Courses'], value_vars=[
                  'Fee', 'Duration'], ignore_index=False)
    print(df2)
    # Example 5: Use multi-index columns
    df.columns = [list('ABCD'), list('EFGH')]


def change_height():
    root = "D:\\"
    path = os.path.join(root, "safe_combustion")
    print(path)
    for r, d, f in os.walk(path):
        for file in f:
            file_path = os.path.join(r, file)
            if file_path.find('.json') >= 0:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                if data['height'] != 40:
                    data['height'] = 40
                    print(file_path)
                    with open(file_path, 'w') as json_file:
                        json.dump((data), json_file)


test_melt()
