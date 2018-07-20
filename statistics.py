#!/home/guopp/anaconda3/bin/python
#_*_ coding:utf8 _*_
import os
import json
import numpy as np
import pandas as pd

def gen_labels_contents(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        for line in f.readlines():
            line = line.strip()
            dict = json.loads(line)
            accusation = dict["meta"]["accusation"]
            fact = dict["fact"]
            yield accusation, fact

def multi_labels_number(labels_file):
    ZERO_LABEL, ONE_LABEL,TWO_LABELS,THREE_LABELS = 0,0,0,0
    with open(labels_file, "r", encoding="utf-8") as f:
        SUM = len(f.readlines())
    with open(labels_file, "r", encoding="utf-8") as f:
        for line in f.readlines():
            acc_list = eval(line.strip())
            list_lenth = len(acc_list)
            if list_lenth == 1:
                ONE_LABEL += 1
            elif list_lenth == 2:
                TWO_LABELS += 1
            elif list_lenth == 3:
                THREE_LABELS += 1
            elif list_lenth == 4:
                ZERO_LABEL += 1
    return ZERO_LABEL, ONE_LABEL, TWO_LABELS, THREE_LABELS, SUM

def gen_metrix(accu_file, labels_file):
    ACCU_DICT = {}
    cnt = 0
    # 获得罪名编号字典
    with open(accu_file, "r", encoding="utf-8") as f:
        for index, accu in enumerate(f.readlines()):
            ACCU_DICT[index] = accu.strip()
    acc_list = [accu for accu in ACCU_DICT.values()]
    acc_list1 = acc_list + ["多标签情况"]
    # 统计每一项罪名出现的总数以及每两个罪名同时出现的数量
    pro_metrix = pd.DataFrame({i: pd.Series(np.zeros(203), index=acc_list1) for i in acc_list})
    num_metrix = pd.DataFrame({i: pd.Series(np.zeros(203), index=acc_list1) for i in acc_list})
    with open(labels_file,"r",encoding="utf-8") as f:
        sum = len(f.readlines())
    with open(labels_file,"r",encoding="utf-8") as f:
        for line in f.readlines():
            cnt += 1
            print("进度：{:.8f}%".format(100.0*cnt / sum))
            list = eval(line.strip())
            for acci in list:
                for accj in list:
                    num_metrix[acci][accj] += 1
    for column in acc_list:
        for index in acc_list:
            pro_metrix[column][index] = float(num_metrix[column][index])/num_metrix[column][column]
            if column != index:
                num_metrix[column]["多标签情况"] += num_metrix[column][index]
        pro_metrix[column]["多标签情况"] = float(num_metrix[column]["多标签情况"])/num_metrix[column][column]
    return pro_metrix, num_metrix


if __name__ == '__main__':
    # 文件目录
    BASE_DIR = "/home/guopp/Python_Project/law/"
    RAW_DATA_FILE = os.path.join(BASE_DIR,"data/raw_data/data_train.json")
    RESULT_PATH = os.path.join(BASE_DIR, "data/accusation/multi_labels")
    STATISTICS_PATH = os.path.join(BASE_DIR, "data/accusation/statistics")
    LABELS_CONTENTS_NAME = "labels_contents.txt"
    LABELS_NAME = "labels.txt"
    LABELS_CONTENTS_FILE = os.path.join(RESULT_PATH, LABELS_CONTENTS_NAME)
    LABELS_FILE = os.path.join(RESULT_PATH, LABELS_NAME)
    PROBABILITY_FILE  = os.path.join(STATISTICS_PATH, "probability.npy")
    NUMBER_FILE  = os.path.join(STATISTICS_PATH, "number.npy")
    ACCU_FILE = os.path.join(BASE_DIR,"data/raw_data/accu.txt")


    # 读取accusation和fact写入LABELS_CONTENTS_FILE和LABELS_FILE
    # with open(LABELS_FILE, "w", encoding="utf-8") as l_f:
    #     with open(LABELS_CONTENTS_FILE, "w", encoding="utf-8") as lc_f:
    #         for accusation, fact in gen_labels_contents(RAW_DATA_FILE):
    #             l_f.writelines(str(accusation) + "\n")
    #             lc_f.writelines(str(accusation) + " " + str(fact) + "\n")

    # 对LABELS_FILE进行统计，功能如下：
    # 一：统计LABELS_FILE中单标签数量和多标签数量
    # zero_label, one_label, two_labels, three_labels, sum_number = multi_labels_number(LABELS_FILE)
    # print("单一标签占比：{:.3f}%".format(float(one_label)/sum_number*100))
    # print("二标签占比：{:.3f}%".format(float(two_labels)/sum_number*100))
    # print("三标签占比：{:.3f}%".format(float(three_labels)/sum_number*100))
    # print("更多标签占比：{:.3f}%".format(float(sum_number-one_label-two_labels-three_labels)/sum_number*100))
    # 二：统计两个二维矩阵，一个比例矩阵，一个数量矩阵
    # 行列为202类罪名，矩阵中的值Xij是所有第i类fact中，该fact也有第j类罪名的比例/数量。
    # pro_metrix, num_metrix = gen_metrix(ACCU_FILE, LABELS_FILE)
    # np.save(PROBABILITY_FILE, pro_metrix)
    # np.save(NUMBER_FILE, num_metrix)

    # 读取已经生成的npy文件，找到一些数量和比例比较高的类别
    pro_metrix = np.load(PROBABILITY_FILE)
    num_metrix = np.load(NUMBER_FILE)
    print(pro_metrix.shape)
    print(pro_metrix[:,0].shape)
    print(pro_metrix[:,0])
    # for i in pro_metrix[:]:
    #     for j in pro_metrix[:]:
    #         print(pro_metrix[i][j])
    #         if pro_metrix[i][j] > 0.5:
    #             print(i,j,pro_metrix[i][j])


