#程序分为三个部分，分别为两个函数（写入bunch，生成tf-idf空间 ），以及主函数的svm预测部分。
#所需的辅助函数储存在tool.py中
#导入所需的库
from sklearn.datasets.base import Bunch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.externals import joblib
from sklearn import metrics
import pickle
import os, time


def writeBunch(data_path, bunch_file):
    bunch = Bunch(label=[], contents=[])
    for label_dir in os.listdir(data_path):
        data_file = os.path.join(data_path, label_dir, DATA_NAME)
        with open(data_file, "r", encoding="utf-8") as f:
            for line in f.readlines():
                label = eval(label_dir)
                bunch.label.append(label)
                bunch.contents.append(line)
    #将读取的bunch写入到bunch——file中
    with open(bunch_file, "wb") as f:
        pickle.dump(bunch, f)

def tfidfspace(bunch_file, tfidf_file, train_bunch_file=None):
    tfidfbunch = Bunch(label=[], contents=[], tdm=[], vocabulary={})
    # 读取bunch_file中的bunch, 将label赋予tfidfbunch中的label
    with open(bunch_file, "rb") as f:
        bunch = pickle.load(f)
    tfidfbunch.label = bunch.label
    tfidfbunch.contents = bunch.contents
    if train_bunch_file is None:   # 此时对训练数据生成tfidf空间
        vectorizer = TfidfVectorizer(max_df=0.4, sublinear_tf=True)
        tfidfbunch.tdm = vectorizer.fit_transform(bunch.contents)
        tfidfbunch.vocabulary = vectorizer.vocabulary_
    else: # 对测试数据生成tfidf空间，保证与训练集的单词字典是相同的。
        with open(train_bunch_file, "rb") as f:
            train_bunch = pickle.load(f)
        tfidfbunch.vocabulary = train_bunch.vocabulary
        vectorizer = TfidfVectorizer(max_df=0.4, sublinear_tf=True, vocabulary=train_bunch.vocabulary)
        tfidfbunch.tdm = vectorizer.fit_transform(bunch.contents)
    # 将tfidfbunch写入tfidf_file
    with open(tfidf_file, "wb") as f:
        pickle.dump(tfidfbunch, f)
    #保存tfidf模型
    joblib.dump(vectorizer, TFIDF_NAME)

if __name__ == '__main__':
    # 文件路径以及参数
    BASE_DIR = "/home/guopp/Python_Project/law/"
    DATA_NAME = "process_data.txt"
    TRAIN_DIR = os.path.join(BASE_DIR, "data/relevant_articles/train_data")
    TEST_DIR = os.path.join(BASE_DIR, "data/relevant_articles/test_data")
    TRAIN_BUNCH_DIR = os.path.join(BASE_DIR, "data/relevant_articles/train_bunch")
    TEST_BUNCH_DIR = os.path.join(BASE_DIR, "data/relevant_articles/test_bunch")
    TRAIN_BUNCH_FILE = os.path.join(TRAIN_BUNCH_DIR,"train_bunch.dat")
    TEST_BUNCH_FILE = os.path.join(TEST_BUNCH_DIR,"test_bunch.dat")
    TRAIN_TFIDF_FILE = os.path.join(TRAIN_BUNCH_DIR,"train_ftidf.dat")
    TEST_TFIDF_FILE = os.path.join(TEST_BUNCH_DIR,"test_ftidf.dat")
    MODEL_PATH = os.path.join(BASE_DIR, "model/relevant_articles")
    TFIDF_NAME = os.path.join(MODEL_PATH, "task2tfidf.m")
    MODEL_NAME = os.path.join(MODEL_PATH, "task2svmmodel.m")


    # 将训练数据和测试数据写入bunch
    st = time.time()
    writeBunch(TRAIN_DIR, TRAIN_BUNCH_FILE)
    writeBunch(TEST_DIR, TEST_BUNCH_FILE)
    et = time.time()
    print("将训练数据和测试数据写入bunch用时：{:.2f}s".format(et-st))

    # 读取bunch数据生成tf-idf空间
    st = time.time()
    tfidfspace(TRAIN_BUNCH_FILE, TRAIN_TFIDF_FILE)
    tfidfspace(TEST_BUNCH_FILE, TEST_TFIDF_FILE, train_bunch_file=TRAIN_TFIDF_FILE)
    et = time.time()
    print("读取bunch数据生成tf-idf空间用时：{:.2f}s".format(et - st))

    # 读取tf-idf空间数据进行训练，使用分类器预测并保存模型。
    st = time.time()
    with open (TRAIN_TFIDF_FILE, "rb") as f:
        train_set = pickle.load(f)
    with open (TEST_TFIDF_FILE, "rb") as f:
        test_set = pickle.load(f)

    clf = SVC(kernel="linear", probability=True)
    clf.fit(train_set.tdm, train_set.label)
    predict_test = clf.predict(test_set.tdm)

    for test_label, test_predict, test_content in zip(test_set.label, predict_test, test_set.contents):
        if test_label != test_predict:
            print("文本内容：{}。\n原本分类为：{}，被预测为{}。".format(test_content,test_label,test_predict))
    accuracy = "精度：{0:.3f}".format(metrics.precision_score(test_set.label, predict_test,average='weighted'))
    recall = "召回率：{0:.3f}".format(metrics.recall_score(test_set.label, predict_test,average='weighted'))
    f1 = "f1值：{0:.3f}".format(metrics.f1_score(test_set.label, predict_test,average='weighted'))

    print(accuracy)
    print(recall)
    print(f1)

    #保存模型
    joblib.dump(clf, MODEL_NAME)
    et = time.time()
    print("读取tf-idf空间数据进行训练，使用朴素贝叶斯预测并保存模型用时：{:.2f}s".format(et - st))



