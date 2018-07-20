from sklearn.datasets.base import Bunch
import numpy as np

bunch = Bunch(list1=[[1],[2],[3]], list2=[[1],[2],[3]])
def shuffle_bunch(databunch):
    new_databunch = Bunch()
    index = [i for i in range(len(databunch.list1))]
    np.random.shuffle(index)
    for i in databunch.keys():

        new_databunch[i] = np.random.shuffle(databunch[i])
    print(databunch)
    return new_databunch

new_bunch = shuffle_bunch(bunch)
print(new_bunch)
