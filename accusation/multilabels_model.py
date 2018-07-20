# 主函数完成对预处理后文件的特征选择，特征权重计算以及训练预测
from helper import *
from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.feature_selection import SelectKBest, SelectPercentile, chi2, mutual_info_classif, RFE, RFECV, f_classif
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.datasets.base import Bunch
from sklearn.externals import joblib
import pandas as pd
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
import re,time

# 获得训练样本和测试样本内容内容以及标签，储存在bunch之中
def genBunch(processDataFile):  # processDataPath类似与"../data/processData/train"
    # 声明bunch变量， 初始化labels，multi_labels, contents, vector, selectVector, tfidf, selectTfidf
    # multi_labels用于储存多标签处理后的标签向量，vector储存词频矩阵， selectVector储存特征选择后的词频矩阵，tfidf储存的是tfidf矩阵，selectTfidf储存的是特征选择后的tfidf矩阵。
    # 此函数获得的是labels和contents
    bunch = Bunch(labels=[], multi_labels=[],contents=[], vector=[],  selectVector=[], tfidf=[], selectTfidf=[])
    with open(processDataFile, "r", encoding = "utf-8") as f:
        for line in f.readlines():
            label = line.split("]")[0] + "]" # 提取罪名标签
            label_list = eval(label)  # 将罪名标签转化为list类型
            fact = line.split(label)[-1] # 提取犯罪事实
            if len(fact.split(" ")) > 20:  # 过滤掉少于20个词的犯罪事实
                bunch.labels.append(label_list)
                bunch.contents.append(fact.strip())
    # bunch.labels = bunch.labels[:50000]
    # bunch.contents = bunch.contents[:50000]
    return bunch
# 统计bunch.contents中的不同的单词数，大约有30万左右。
def counterWord(dataBunch):  # dataBunch为生成的train_bunch,test_bunch
    contents = []
    for content in dataBunch.contents:
        lenth = len(content.split(" "))
        for word in content.split(" "):
            contents.append(word)
    print(len(contents))  # 本文单词总数
    print(len(set(contents)))  # 文本中不同单词的数目，与CountVectorizer统计出的词数有2%的差距，暂不清楚差距是怎么产生的。


# 对特征选择后的databunch进行模型训练，返回预测结果
def clf2result(trainDatabunch, testDatabunch, selection_method, clf_method):
    BtrainDatabunch, BtestDatabunch = beforeTfidf(trainDatabunch, testDatabunch, selection_method)# B表示先特征选择
    AtrainDatabunch, AtestDatabunch = afterTfidf(trainDatabunch, testDatabunch, selection_method)   # A表示后进行特征选择
    Bmodel = clf_method # 初始化分类器
    Bmodel.fit(BtrainDatabunch.selectTfidf, BtrainDatabunch.labels)  #进行模型训练
    Bpredict = Bmodel.predict(BtestDatabunch.selectTfidf)  # 样本预测
    result_before = metrics.f1_score(BtestDatabunch.labels, y_pred=Bpredict, average="weighted")  # 储存预测结果的F1值
    Amodel = clf_method
    Amodel.fit(AtrainDatabunch.selectTfidf, AtrainDatabunch.labels)
    Apredict = Amodel.predict(AtestDatabunch.selectTfidf)
    result_after = metrics.f1_score(AtestDatabunch.labels, y_pred=Apredict, average="weighted")
    print(str(selection_method))
    print(result_before)
    print(result_after)
    # with open(RESULT_FILE, "w") as f:
    #     f.readlines("选取特征数为{}, tfidf之前提取特征的结果为{:.4f}， tfidf之后提取特征的结果为{:.4f}。".format(str(selection_method),result_before,result_after) + "\n")
    # # F1_after[number] = result_after  #result_before是对词频矩阵进行特征选择，然后生成tfidf矩阵产生的预测结果， result_after是对生成的tfidf矩阵进行特征选择产生的结果

# 对混淆矩阵可视化
def plot_cfm(cm, title="Confusion matrix", cmap = plt.cm.Blues):
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(CLASS_LIST))
    plt.xticks(tick_marks, CLASS_LIST, rotation=45)
    plt.yticks(tick_marks, CLASS_LIST)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    for x in range(len(cm)):
        for y in range(len(cm)):
            plt.annotate(cm[x,y], xy=(x, y), horizontalalignment='center', verticalalignment='center')



if __name__ == '__main__':
    # 初始化特征选择方法
    # vector = CountVectorizer(max_df=0.6,ngram_range=(1,2)) # 添加了2gram特征，结果表明准确率下降了
    word2vector = CountVectorizer(max_df=0.5)  # 实例化词频统计矩阵方法
    vector2tfidf = TfidfTransformer(sublinear_tf=True)  # 实例化将词频统计矩阵转换为tfidf矩阵方法
    word2tfidf = TfidfVectorizer(max_df=0.5, sublinear_tf=True)

    # 获取bunch中的labels和contents
    allBunch = genBunch(PROCESS_MULTI_LABELS_FILE_WITHOUT1)  # Bunch中的labels储存标签，contents存储文本内容，vector储存词频矩阵，selectVector储存特征选择后的词频矩阵
    # 生成databunch中的multi_labels,vector,tfidf
    allBunch.vector = word2vector.fit_transform(allBunch.contents, allBunch.labels)
    allBunch.tfidf = vector2tfidf.fit_transform(allBunch.vector, allBunch.tfidf)
    allBunch.multi_labels = MultiLabelBinarizer(classes=ACCU_LIST).fit_transform(allBunch.labels)
    joblib.dump(allBunch, ALL_BUNCH_FILE) # 保存train_bunch和test_bunch,后面的程序可以直接读取，以节省时间。

    # allBunch = joblib.load(ALL_BUNCH_FILE) # 读取保存的allBunch
    print(allBunch.vector.shape)
    print(np.max(allBunch.vector))
    allBunch.vector = allBunch.vector / np.max(allBunch.vector)
    # 先不进行特征选择对allBunch进行交叉验证，多标签训练
    trainBunch = Bunch(labels=[], multi_labels=[],contents=[], vector=[],  selectVector=[], tfidf=[], selectTfidf=[])
    testBunch = Bunch(labels=[], multi_labels=[],contents=[], vector=[],  selectVector=[], tfidf=[], selectTfidf=[])
    trainBunch.multi_labels, testBunch.multi_labels,trainBunch.vector, testBunch.vector , trainBunch.tfidf, testBunch.tfidf\
        = train_test_split(allBunch.multi_labels, allBunch.vector,  allBunch.tfidf,test_size=0.3)
    clf = DecisionTreeClassifier()
    print("正在进行训练》》》》》》")
    st =time.time()
    clf.fit(trainBunch.vector, trainBunch.multi_labels)
    predict_y = clf.predict(testBunch.vector)
    print(len(predict_y))
    et = time.time()
    print("完成训练！！！用时：{:.3f}s".format(et-st))
    # 使用混淆矩阵和分类报告对分类结果进行分析
    print("分类结果如下》》》》》")
    sum = len(testBunch.multi_labels)
    right_num = 0
    print(sum)
    for y_ture, y_pre in zip(testBunch.multi_labels, predict_y):
        if not (y_ture-y_pre).any(): # 比较两个数组是否完全一样
            right_num += 1
    print(right_num)
    print(right_num/sum)

    # 保存模型和word2vector
    joblib.dump(clf, MODEL_FILE)
    joblib.dump(word2vector, WORD2VECTOR_FILE)


    # report = metrics.accuracy_score(testBunch.multi_labels, predict_y)
    # print(report)
    # print("混淆矩阵如下》》》》》")
    # cfm = metrics.confusion_matrix(testBunch.multi_labels, predict_y)
    # plt.figure(figsize=(8,8))
    # plot_cfm(cfm)
    # plt.show()
    # print(type(cfm))
    # print(cfm.shape)
    # print(cfm)

    # chu
    # print("正在进行模型训练。。。")
    # # clf_method = DecisionTreeClassifier(class_weight="balanced")  # 设置分类器, 加入class_weight="balanced"，减缓样本分布不平衡的情况
    # clf_method = LinearSVC()  # 设置分类器

    '''
    # 测试不同的特征选择对程序结果的影响。
    pool = multiprocessing.Pool(processes=6) # 使用多进程方法加快训练速度（同时训练多个模型）
    feature_select_method = chi2
    feature_numbers = [100, 500, 1000, 2000, 3000, 5000, 10000, 20000, 50000, 100000, 200000]  # 设置选择的特征数目
    selection_method_list = [SelectKBest(feature_select_method, number) for number in feature_numbers] # 设置特征选择方法
    for i in selection_method_list:
        pool.apply_async(clf2result, (trainBunch, testBunch, i, clf_method))
    pool.close()
    pool.join()
    '''
    '''
    pool = multiprocessing.Pool(processes=6) # 使用多进程方法加快训练速度（同时训练多个模型）
    feature_numbers_per = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]  # 设置选择的特征占原特征的百分比
    selection_method_per_list = [SelectPercentile(feature_select_method, number) for number in feature_numbers_per] # 设置特征选择方法
    for i in selection_method_per_list:
        pool.apply_async(clf2result, (trainBunch, testBunch, i, clf_method))
    pool.close()
    pool.join()
    '''



