import os

BASE_DIR = os.path.dirname(os.getcwd())   # 此目录为...../law
# data/raw_data文件夹下的文件
RAW_BIG_DATA = os.path.join(BASE_DIR, "data/raw_data", "big_data.json")   #  原始大数据json文件
RAW_SMALL_DATA = os.path.join(BASE_DIR, "data/raw_data", "small_data.json")   #  原始小数据json文件
ACCU_FIE = os.path.join(BASE_DIR, "data/raw_data", "accu.txt")   #  原始小数据json文件
LAW_FILE = os.path.join(BASE_DIR, "data/raw_data", "law.txt")   #  原始小数据json文件
# data/helper文件夹下的文件
STOP_WORDS_FILE = os.path.join(BASE_DIR, "data/helper", "stopwords.txt")
# data/process_data文件夹下的文件， 主要是处理好的bunch文件
SMALL_TRAINBUNCH_FILE = os.path.join(BASE_DIR, "data/process_data", "small_trainbunch.dat")
SMALL_TESTBUNCH_FILE = os.path.join(BASE_DIR, "data/process_data", "small_testbunch.dat")
BIG_TRAINBUNCH_FILE = os.path.join(BASE_DIR, "data/process_data", "big_trainbunch.dat")
BIG_TESTBUNCH_FILE = os.path.join(BASE_DIR, "data/process_data", "big_testbunch.dat")

# 提取罪名字典和，法律字典。    这里应该有更简单的构建字典的方法但是忘了
accu_file = open(ACCU_FIE, "r", encoding="utf-8")
accu_list = accu_file.read().splitlines()
accu_file = open(ACCU_FIE, "r", encoding="utf-8")
accu_index = {}; index_accu = {}   #  accu_index罪名是键，索引是值。index_accu相反。
for index, accu in enumerate(accu_file.read().splitlines()):
    accu_index[accu] = index
    index_accu[index] = accu
accu_file.close()

law_file = open(LAW_FILE, "r", encoding="utf-8")
law_list = law_file.read().splitlines()
law_file = open(LAW_FILE, "r", encoding="utf-8")
law_index = {}; index_law = {}     #  同上
for index, law in enumerate(law_file.read().splitlines()):
    law_index[law] = index
    index_law[index] = law
law_file.close()
# 提取停止词列表
stopwords_file = open(STOP_WORDS_FILE, "r", encoding="utf-8")
stopwords = stopwords_file.read().splitlines()
stopwords_file.close()
# model文件夹下的文件，训练好的模型文件。由于可能要储存好多模型文件，这里设计一个函数，由主函数输入模型文件名称。
def gen_model_file(model_name):
    model_file = os.path.join(BASE_DIR, "model", model_name)
    return model_file






# DATA_FILE = "process_data.txt"
# TRAIN_DIR = os.path.join(BASE_DIR, "data/accusation/big/train_data")
# TEST_DIR = os.path.join(BASE_DIR, "data/accusation/big/test_data")
# TRAIN_BUNCH_DIR = os.path.join(BASE_DIR, "data/accusation/big/train_bunch")
# TEST_BUNCH_DIR = os.path.join(BASE_DIR, "data/accusation/big/test_bunch")
# TRAIN_BUNCH_FILE = os.path.join(TRAIN_BUNCH_DIR,"train_bunch.dat")
# TEST_BUNCH_FILE = os.path.join(TEST_BUNCH_DIR,"test_bunch.dat")
# TRAIN_TFIDF_FILE = os.path.join(TRAIN_BUNCH_DIR,"train_ftidf.dat")
# TEST_TFIDF_FILE = os.path.join(TEST_BUNCH_DIR,"test_ftidf.dat")
# MODEL_PATH = os.path.join(BASE_DIR, "model")
# TFIDF_FILE = os.path.join(MODEL_PATH, "bigtask1tfidf.m")
#
# MODEL_FILE = os.path.join(MODEL_PATH, "multi_labels_decisiontrees_model.m")
# WORD2VECTOR_FILE = os.path.join(MODEL_PATH, "word_2vector.pkl")
#
# MULTI_MODEL_FILE = os.path.join(MODEL_PATH, "multi_labels_decision_model.m")
#
# MULTI_LABELS_PATH = os.path.join(BASE_DIR, "data/accusation/multi_labels")
#
#
# PROCESS_MULTI_LABELS_FILE = os.path.join(MULTI_LABELS_PATH, "process_multilabels.txt")
#
#
#
# MULTI_LABELS_PATH = os.path.join(BASE_DIR, "data/accusation/multi_labels")
# RAW_MULTI_LABELS_FILE = os.path.join(MULTI_LABELS_PATH, "labels_contents.txt")
# PROCESS_MULTI_LABELS_FILE_WITH1 = os.path.join(MULTI_LABELS_PATH, "process_multilabels_with1.txt")
# PROCESS_MULTI_LABELS_FILE_WITHOUT1 = os.path.join(MULTI_LABELS_PATH, "process_multilabels_without1.txt")
#
# # 相关路径和参数
# TRAIN_DATA_JSON = os.path.join(BASE_DIR, "data/raw_data/data_train.json")
# TEST_DATA_JSON = os.path.join(BASE_DIR, "data/raw_data/data_test.json")
# VALID_DATA_JSON = os.path.join(BASE_DIR, "data/raw_data/data_valid.json")
# TRAIN_DATA = os.path.join(BASE_DIR, "data/accusation/train_data")
# TEST_DATA = os.path.join(BASE_DIR, "data/accusation/test_data")
# ALL_DATA = os.path.join(BASE_DIR, "data/accusation/all_data")
# ACCU_FILE = os.path.join(BASE_DIR, "data/raw_data/accu.txt")
# STOPWORDS_FILE = os.path.join(BASE_DIR, "data/stop_words/stopwords.txt")
# SEG_NAME = "seg_data.txt"
# PROCESS_NAME = "process_data.txt"
#
# DL_MODEL_FILE = os.path.join(BASE_DIR, "model/deeplearning_text_cnn.h5")
# TOKEN_MODEL_FILE = os.path.join(BASE_DIR, "model/token_dict.pkl")
#
#
# ALL_BUNCH_FILE = os.path.join(MULTI_LABELS_PATH, "allbunch1.pkl")
# # 读取罪名列表
# ACCU_DICT = {}
# with open(ACCU_FILE, "r", encoding="utf-8") as f:
#     ACCU_LIST = f.read().splitlines()
#     for index , accu in enumerate(ACCU_LIST):
#         ACCU_DICT[accu] = index
# '''
