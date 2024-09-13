import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def scatter_plot(dataset: pd.DataFrame, feature_1, feature_2):
    plt.figure()
    plt.title(f"Scatter plot for comparison scores {feature_1} vs. {feature_2}")
    
    sns.scatterplot(data=dataset, x=feature_1, y=feature_2)

    plt.show()


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
        feature_1 = input('Enter feature:\n- Arithmancy\n- Astronomy\n- Herbology\n- Defense Against the Dark Arts\n- Divination\n- Muggle Studies\n- Ancient Runes\n -History of Magic\n -Transfiguration\n -Potions\n- Care of Magical Creatures\n- Charms\n- Flying\n')
        feature_2 = input('Choose the second\n')
        scatter_plot(dataset, feature_1, feature_2)
    else:
        print("Wrong number of arguments")