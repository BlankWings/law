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
    bunch = Bunch(labels=[], contents=[])
    for label_dir in os.listdir(data_path):
        data_file = os.path.join(data_path, label_dir, DATA_FILE)
        with open(data_file, "r", encoding="utf-8") as f:
            for line in f.readlines():
                label = [eval(label_dir)]
                bunch.labels.append(label)
                bunch.contents.append(line)
    bunch.new_labels = MultiLabelBinarizer().fit_transform(bunch.labels)
    #将读取的bunch写入到bunch——file中
    with open(bunch_file, "wb") as f:
        pickle.dump(bunch, f)

def tfidfspace(bunch_file, tfidf_file, train_bunch_file=None):
    tfidfbunch = Bunch(labels=[], contents=[], tdm=[], vocabulary={})
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
    joblib.dump(vectorizer, TFIDF_FILE)




# 使用keras进行文本分类
# 使用的网络结构包含MLP， RNN， LSTM, GRU, CNN等。
from helper import *
from sklearn.datasets.base import Bunch
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import time, re
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Input, concatenate
from keras.layers.core import Dense, Flatten, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import SimpleRNN, LSTM, GRU
from keras.layers.wrappers import Bidirectional  # 构建双向循环网络
from keras.models import Model, load_model

def TEXT_CNN_NETWORK(): # 还有些方法不是很懂
    sentence_seq = Input(shape=[max_lenth], name="X_seq")  # 输入
    embedding_layer = Embedding(input_dim=token_words, output_dim=32)(sentence_seq)  # 词嵌入层
    # 卷积层如下：
    convs = []
    filter_sizes = [2, 3, 4, 5]
    for fsz in filter_sizes:
        l_conv = Conv1D(filters=100, kernel_size=fsz, activation="relu")(embedding_layer)
        l_pool = MaxPooling1D(max_lenth-fsz+1)(l_conv)
        l_pool = Flatten()(l_pool)
        convs.append(l_pool)
    merge = concatenate(convs, axis=1)
    out = Dropout(0.5)(merge)
    output = Dense(512, activation="relu")(out)
    output = Dropout(0.5)(output)
    output = Dense(202, activation="softmax")(output)
    model = Model([sentence_seq], output)

    print("神经网络的结构如下：")
    print(model.summary())
    return model









if __name__ == '__main__':
    # 文件路径以及参数
    BASE_DIR = os.path.dirname(os.path.dirname(os.getcwd()))
    DATA_FILE = "process_data.txt"
    TRAIN_DIR = os.path.join(BASE_DIR, "data/accusation/train_data")
    TEST_DIR = os.path.join(BASE_DIR, "data/accusation/test_data")
    TRAIN_BUNCH_DIR = os.path.join(BASE_DIR, "data/accusation/train_bunch")
    TEST_BUNCH_DIR = os.path.join(BASE_DIR, "data/accusation/test_bunch")
    TRAIN_BUNCH_FILE = os.path.join(TRAIN_BUNCH_DIR,"train_bunch.dat")
    TEST_BUNCH_FILE = os.path.join(TEST_BUNCH_DIR,"test_bunch.dat")
    TRAIN_TFIDF_FILE = os.path.join(TRAIN_BUNCH_DIR,"train_ftidf.dat")
    TEST_TFIDF_FILE = os.path.join(TEST_BUNCH_DIR,"test_ftidf.dat")
    MODEL_PATH = os.path.join(BASE_DIR, "model")
    TFIDF_FILE = os.path.join(MODEL_PATH, "bigtask1tfidf.m")
    MODEL_FILE = os.path.join(MODEL_PATH, "multi_label_decision_tree.m")

    # 将训练数据和测试数据写入bunch
    st = time.time()
    writeBunch(TRAIN_DIR, TRAIN_BUNCH_FILE)
    writeBunch(TEST_DIR, TEST_BUNCH_FILE)
    et = time.time()
    print("将训练数据和测试数据写入bunch用时：{:.2f}s".format(et-st))



    trainBunch = joblib.load(TRAIN_BUNCH_FILE)
    testBunch = joblib.load(TEST_BUNCH_FILE)

    # 相关参数如下：
    token_words = 3800  # 单词字典的单词数。
    max_lenth = 200  # 选取句子的长度。
    print("正在进行数据预处理》》》》")
    print(trainBunch.labels)
    print(trainBunch.new_labels)
    print(testBunch.new_labels)
    st = time.time()
    # 建立Token词典
    token = Tokenizer(num_words=token_words)  # 设置词典规模
    token.fit_on_texts(trainBunch.contents)  # 建立字典模型
    print(token.document_count)
    print(len(token.word_index.values()))
    # 将文字列表转化为数字列表
    trainBunch.contents_seq = token.texts_to_sequences(trainBunch.contents)
    testBunch.contents_seq = token.texts_to_sequences(testBunch.contents)
    # 对数字列表进行padding，截长补短。处理后的数据输入神经网络进行训练。s
    trainBunch.contents_seq_pad = sequence.pad_sequences(trainBunch.contents_seq, maxlen=max_lenth)
    testBunch.contents_seq_pad = sequence.pad_sequences(testBunch.contents_seq, maxlen=max_lenth)
    print(trainBunch.contents_seq_pad.shape)
    print(testBunch.contents_seq_pad.shape)
    et = time.time()
    print("数据预处理完成！！！！用时：{:.3f}s".format(et - st))
    # 构建神经网络，
    # model = MLP_NETWORK()
    # model = RNN_NETWORK()
    # model = LSTM_NETWORK()
    # model = GRU_NETWORK()
    # model = BIRNN_NETWORK()
    model = TEXT_CNN_NETWORK()
    # 定义训练方法
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
    # 开始训练
    train_history = model.fit(trainBunch.contents_seq_pad, trainBunch.new_labels, batch_size=1000, epochs=10, verbose=1, validation_split=0.1)
    # 评估模型准确率
    score = model.evaluate(testBunch.contents_seq_pad, testBunch.new_labels, verbose=2)
    print("模型准确率为：{:.3f}".format(score[1]))
    # keras保存模型
    # 保存模型和Token字典
    model.save(DL_MODEL_FILE)
    joblib.dump(token, TOKEN_MODEL_FILE)
    print("保存模型完成！！！")
















    # # 读取bunch数据生成tf-idf空间
    # st = time.time()
    # tfidfspace(TRAIN_BUNCH_FILE, TRAIN_TFIDF_FILE)
    # tfidfspace(TEST_BUNCH_FILE, TEST_TFIDF_FILE, train_bunch_file=TRAIN_TFIDF_FILE)
    # et = time.time()
    # print("读取bunch数据生成tf-idf空间用时：{:.2f}s".format(et - st))
    #
    # # 读取tf-idf空间数据进行训练，使用分类器预测并保存模型。
    # st = time.time()
    # with open (TRAIN_TFIDF_FILE, "rb") as f:
    #     train_set = pickle.load(f)
    # with open (TEST_TFIDF_FILE, "rb") as f:
    #     test_set = pickle.load(f)
    #
    # clf = DecisionTreeClassifier()
    # clf.fit(train_set.tdm, train_set.label)
    # predict_test = clf.predict(test_set.tdm)
    #
    # for test_label, test_predict, test_content in zip(test_set.label, predict_test, test_set.contents):
    #     if test_label != test_predict:
    #         print("文本内容：{}。\n原本分类为：{}，被预测为{}。".format(test_content,test_label,test_predict))
    # accuracy = "精度：{0:.3f}".format(metrics.precision_score(test_set.label, predict_test,average='weighted'))
    # recall = "召回率：{0:.3f}".format(metrics.recall_score(test_set.label, predict_test,average='weighted'))
    # f1 = "f1值：{0:.3f}".format(metrics.f1_score(test_set.label, predict_test,average='weighted'))
    #
    # print(accuracy)
    # print(recall)
    # print(f1)
    #
    # #保存模型
    # joblib.dump(clf, MODEL_FILE)
    # et = time.time()
    # print("读取tf-idf空间数据进行训练，使用svm预测并保存模型用时：{:.2f}s".format(et - st))



