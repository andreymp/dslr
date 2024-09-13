import sys
import pandas as pd
import numpy as np
from ft import count, mean, std, min, percentille_25, percentille_50, percentille_75, max

def describe(dataset: pd.DataFrame):
    dataset_funcs = [count, mean, std, min, percentille_25, percentille_50, percentille_75, max]
    dataset = dataset[dataset.select_dtypes(include=np.number).columns.array[3:]]

    description: pd.DataFrame = dataset.agg(dataset_funcs)
    description.rename(index={'percentille_25': '25%', 'percentille_50': '50%', 'percentille_75': '75%'}, inplace=True)
    description.map(lambda elem: f"{elem:0.6f}")
    print(description)

def read_dataset(filename):
    try:
        return pd.read_csv(filename)
    except IOError:
        print('Imposiible to read dataset')
        return None

if __name__ == '__main__':
    if len(sys.argv) == 2:
        if not sys.argv[1].endswith('.csv'):
            print("Wrong dataset extension")
            sys.exit()
        dataset = read_dataset(str(sys.argv[1]))
        if dataset is None:
            sys.exit()
        describe(dataset)
    else:
        print("Wrong number of arguments")