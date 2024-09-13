import sys
import pandas as pd
import matplotlib.pyplot as plt

def histogram(dataset: pd.DataFrame, feature):
    grouped_dataset = dataset.groupby('Hogwarts House')[feature]
    
    grouped_dataset.plot(kind='hist', title=f"Score distribution in {feature}", x='Score', alpha=0.2, legend=True)
    
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
        feature = input('Enter feature:\n- Arithmancy\n- Astronomy\n- Herbology\n- Defense Against the Dark Arts\n- Divination\n- Muggle Studies\n- Ancient Runes\n -History of Magic\n -Transfiguration\n -Potions\n- Care of Magical Creatures\n- Charms\n- Flying\n')
        histogram(dataset, feature)
    else:
        print("Wrong number of arguments")