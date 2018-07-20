import os
# 储存文件路径，基本参数等
BASE_DIR = os.path.dirname(os.path.dirname(os.getcwd()))
DATA_FILE = "process_data.txt"
TRAIN_DIR = os.path.join(BASE_DIR, "data/accusation/big/train_data")
TEST_DIR = os.path.join(BASE_DIR, "data/accusation/big/test_data")
TRAIN_BUNCH_DIR = os.path.join(BASE_DIR, "data/accusation/big/train_bunch")
TEST_BUNCH_DIR = os.path.join(BASE_DIR, "data/accusation/big/test_bunch")
TRAIN_BUNCH_FILE = os.path.join(TRAIN_BUNCH_DIR,"train_bunch.dat")
TEST_BUNCH_FILE = os.path.join(TEST_BUNCH_DIR,"test_bunch.dat")
TRAIN_TFIDF_FILE = os.path.join(TRAIN_BUNCH_DIR,"train_ftidf.dat")
TEST_TFIDF_FILE = os.path.join(TEST_BUNCH_DIR,"test_ftidf.dat")
MODEL_PATH = os.path.join(BASE_DIR, "model")
TFIDF_FILE = os.path.join(MODEL_PATH, "bigtask1tfidf.m")

MODEL_FILE = os.path.join(MODEL_PATH, "multi_labels_decisiontrees_model.m")
WORD2VECTOR_FILE = os.path.join(MODEL_PATH, "word_2vector.pkl")

MULTI_MODEL_FILE = os.path.join(MODEL_PATH, "multi_labels_decision_model.m")

MULTI_LABELS_PATH = os.path.join(BASE_DIR, "data/accusation/multi_labels")


PROCESS_MULTI_LABELS_FILE = os.path.join(MULTI_LABELS_PATH, "process_multilabels.txt")



MULTI_LABELS_PATH = os.path.join(BASE_DIR, "data/accusation/multi_labels")
RAW_MULTI_LABELS_FILE = os.path.join(MULTI_LABELS_PATH, "labels_contents.txt")
PROCESS_MULTI_LABELS_FILE_WITH1 = os.path.join(MULTI_LABELS_PATH, "process_multilabels_with1.txt")
PROCESS_MULTI_LABELS_FILE_WITHOUT1 = os.path.join(MULTI_LABELS_PATH, "process_multilabels_without1.txt")

# 相关路径和参数
TRAIN_DATA_JSON = os.path.join(BASE_DIR, "data/raw_data/data_train.json")
TEST_DATA_JSON = os.path.join(BASE_DIR, "data/raw_data/data_test.json")
VALID_DATA_JSON = os.path.join(BASE_DIR, "data/raw_data/data_valid.json")
TRAIN_DATA = os.path.join(BASE_DIR, "data/accusation/train_data")
TEST_DATA = os.path.join(BASE_DIR, "data/accusation/test_data")
ALL_DATA = os.path.join(BASE_DIR, "data/accusation/all_data")
ACCU_FILE = os.path.join(BASE_DIR, "data/raw_data/accu.txt")
STOPWORDS_FILE = os.path.join(BASE_DIR, "data/stop_words/stopwords.txt")
SEG_NAME = "seg_data.txt"
PROCESS_NAME = "process_data.txt"

DL_MODEL_FILE = os.path.join(BASE_DIR, "model/deeplearning_text_cnn.h5")
TOKEN_MODEL_FILE = os.path.join(BASE_DIR, "model/token_dict.pkl")


ALL_BUNCH_FILE = os.path.join(MULTI_LABELS_PATH, "allbunch1.pkl")
# 读取罪名列表
ACCU_DICT = {}
with open(ACCU_FILE, "r", encoding="utf-8") as f:
    ACCU_LIST = f.read().splitlines()
    for index , accu in enumerate(ACCU_LIST):
        ACCU_DICT[accu] = index
