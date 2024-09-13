import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def pair_plot(dataset: pd.DataFrame):
    features = dataset.select_dtypes(include=np.number).columns

    plt.title("Pairplot for all features")
    
    sns.pairplot(dataset, x_vars=features, y_vars=features, corner=True)
    
    plt.savefig('./pair_plot.png')

def read_dataset(filename):
    try:
        dataset = pd.read_csv(filename)
        dataset.drop(['Index', 'First Name', 'Last Name', 'Birthday', 'Best Hand'], axis=1 ,inplace=True)
        return dataset
    except IOError:
        print('Imposible to read dataset')
        return None

if __name__ == '__main__':
    if len(sys.argv) == 2:
        if not sys.argv[1].endswith('.csv'):
            print("Wrong dataset extension")
            sys.exit()
        dataset = read_dataset(str(sys.argv[1]))
        if dataset is None:
            sys.exit()
        pair_plot(dataset)
    else:
        print("Wrong number of arguments")