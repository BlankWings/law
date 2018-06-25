import numpy as np
import os

BASEPATH = "/home/guopp/Python_Project/law/"
BIG_DATA_JSON = os.path.join(BASEPATH, "data/raw_data/cail2018_big.json")
with open(BIG_DATA_JSON, "r", encoding="utf-8") as f:
    print(f.readlines()[:5])

lenth = 100
train_list = np.random.choice(range(lenth), int(0.9*lenth), replace=False)
test_list = [item for item in range(lenth) if item not in train_list]
print(train_list)
print(len(train_list))
print(test_list)
print(len(test_list))