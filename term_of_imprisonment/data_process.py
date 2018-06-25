# 处理raw_data文件，得到可以用于后续训练的文件
# 步骤一：读取文件内容，将fact和accusation根据accu.txt分类，并写入文件夹。（将train和valid写入到一起作为训练数据）
# 统计一下不同罪名的文件数一共为多少。
# 步骤二：读取分好类的文件，将其分词得到分词后的文件写入seg_data。
# 步骤三：对于分好词的文件进行进行处理（去停用词，然后较长的截短，较短的复制几遍，长度达到128）
#导入使用的库
import os
import json
import time
import matplotlib.pyplot as plt
import multiprocessing
import jieba
import re
import numpy as np
# import tensorflow

#得到分好类的文件
def writeData(json_path, data_path):
    json_path = json_path
    data_path_list = data_path
    with open(json_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            line = json.loads(line)
            fact = line["fact"].rstrip()
            accu_list = line["meta"]["accusation"]
            for accu in accu_list:
                number = ACCU_DICT[accu]   # number为罪名对应的编号
                filename = data_path_list[number-1] + "/raw_data.txt"
                with open(filename, "a", encoding="utf-8") as o:
                    o.writelines(fact + "\n")

#统计不同文件的案件数
def number(data_path):
    data_path_list = data_path
    cnt = list(range(1,203))
    numberlist = []
    for datapath in data_path_list:
        filename = datapath + "/raw_data.txt"
        with open(filename, "r", encoding="utf-8") as f:
            number = len(f.readlines())
            numberlist.append(number)
    return cnt, numberlist

#产生分词后的文件
def segData(data_path):
    rawdata = data_path + "/raw_data.txt"
    segdata = data_path + "/seg_data.txt"
    with open(rawdata, "r", encoding="utf-8") as f:
        with open(segdata, "w", encoding="utf-8") as o:
            for line in f.readlines():
                line = line.rstrip()
                o.writelines(" ".join(jieba.cut(line)) + "\n")

#将所有句子写入到一个文件之中，为生成词向量做准备
def writeAll(all_data_path, train_data_path, test_data_path, file): #file指的是要合并的文件
    all_data = os.path.join(all_data_path,file)
    train_data = os.path.join(train_data_path,file)
    test_data = os.path.join(test_data_path,file)
    with open(all_data, "a", encoding="utf-8") as f:
        with open(train_data, "r", encoding="utf-8") as o:
            with open(test_data, "r", encoding="utf-8") as l:
                for oline in o.readlines():
                    f.writelines(oline)
                for lline in l.readlines():
                    f.writelines(lline)

#对分词后的文件进行处理，训练集和测试集的处理是不同的，先生成一个训练集和测试集一起的处理后文件，用于最后的处理
#然后训练集和测试机各生成一个处理后文件（待定）
#文件处理一：去掉中文停用词, 为了多线程加速，直接对文件进行处理
#process2_data.txt 去掉了长度为1的词
def processData1(data_path):
    segdata = data_path + "/seg_data.txt"
    process1data = data_path + "/process3_data.txt"
    with open(segdata, "r", encoding="utf-8") as f:
        with open(process1data, "w", encoding="utf-8") as o:
            for line in f.readlines():
                newline = ""
                pattern1 = re.compile(u"[\u4e00-\u9fa5]+")
                line = re.findall(pattern=pattern1, string=line)
                for word in line:
                    if word not in STOPWORDS and len(word)>1 and "县" not in word and "市" not in word and "区" not in word \
                            and "省" not in word and "自治区" not in word and "村" not in word and "镇" not in word \
                            and "营" not in word:
                         newline += word + " "
                if len(newline) > 10:
                    newline = newline + "\n"
                    o.writelines(newline)



#删去长度小于10的fact， 对于长度大于256的数据，截取至256，长度小于256的数据，重复句子并截取至256
def processData2(data_path):
    process1data = data_path + "/process3_data.txt"
    process2data = data_path + "/process"+str(LENTH)+"_2data.txt"
    with open(process1data, "r", encoding="utf-8") as f:
        with open(process2data, "w", encoding="utf-8") as o:
            for line in f.readlines():
                newline = []
                line = line.rstrip().split(" ")
                if len(line) >= LENTH:
                    newline = line[:LENTH]
                elif len(line) <= 10:
                    continue
                else:
                    while(len(newline)<LENTH):
                        newline.extend(line)
                    newline = newline[:LENTH]
                # print(len(newline))
                newline = str(newline).replace("[","").replace("]","").replace("'","").replace(",","")
                newline = newline + "\n"
                o.writelines(newline)

#暂时没想好还要怎么处理
def processData3(seg_path, process_path):
    pass

if __name__ == '__main__':
    # 相关路径和参数
    TRAIN_DATA_JSON = "../data/raw_data/data_train.json"
    TEST_DATA_JSON = "../data/raw_data/data_test.json"
    VALID_DATA_JSON = "../data/raw_data/data_valid.json"
    TRAIN_DATA = "../data/train_data"
    TEST_DATA = "../data/test_data"
    ACCU_DICT_FILE = "../data/raw_data/accu.txt"
    STOPWORDS_FILE = "../data/stop_words/stopwords.txt"
    ALL_DATA = "../data/all_data"
    SEG_FILE = "seg_data.txt"
    PROCESS1_FILE = "process1_data.txt"
    PROCESS256_FILE = "process256_data.txt"
    PROCESS128_FILE = "process128_data.txt"
    LENTH = 128  # process2处理句子的长度

    # 生成需要使用的参数ACCU_DICT（字典）, TRAIN_DATA_PATH（列表）,TEST_DATA_PATH（列表）,STOPWORDS（停止词列表）
    ACCU_DICT = {}
    cnt = 0
    with open(ACCU_DICT_FILE, "r", encoding="utf-8") as f:
        for accu in f.readlines():
            cnt += 1
            accu = accu.rstrip()
            ACCU_DICT[accu] = cnt
    # print(len(ACCU_DICT))
    # print(ACCU_DICT)

    TRAIN_DATA_PATH = []
    TEST_DATA_PATH = []
    for i in range(1, 203):
        train_DATA_PATH = os.path.join(TRAIN_DATA, str(i))
        TRAIN_DATA_PATH.append(train_DATA_PATH)
        if not os.path.exists(train_DATA_PATH):
            os.makedirs(train_DATA_PATH)
        test_DATA_PATH = os.path.join(TEST_DATA, str(i))
        TEST_DATA_PATH.append(test_DATA_PATH)
        if not os.path.exists(test_DATA_PATH):
            os.makedirs(test_DATA_PATH)

    with open(STOPWORDS_FILE, "r", encoding="utf-8") as f:
        STOPWORDS = f.read().splitlines()

    # 读取JSON文件
    # st = time.time()
    # writeData(TRAIN_DATA_JSON, TRAIN_DATA_PATH)
    # writeData(TEST_DATA_JSON, TEST_DATA_PATH)
    # et = time.time()
    # time0 = et - st
    # print("写入文件用时： {}s".format(str(time0)))

    # # 统计不同样例的数量
    # cnt, numberlist = number(TRAIN_DATA_PATH)
    # print(sum(numberlist))
    # plt.scatter(cnt, numberlist)
    # plt.ylim(0,200)
    # plt.show()

    # # 生成分词后文件
    # st = time.time()
    # pool = multiprocessing.Pool(processes=12)
    # for i in TRAIN_DATA_PATH:
    #     pool.apply_async(segData, (i,))
    # pool.close()
    # pool.join()
    # pool = multiprocessing.Pool(processes=12)
    # for i in TEST_DATA_PATH:
    #     pool.apply_async(segData, (i,))
    # pool.close()
    # pool.join()
    # et = time.time()
    # time0 = et - st
    # print("分词用时： {}s".format(str(time0)))

    # 生成process3文件,去掉停用词并且过滤掉长度为1的词, 去掉句子长度小于10的句子
    st = time.time()
    pool = multiprocessing.Pool(processes=12)
    for i in TRAIN_DATA_PATH:
        pool.apply_async(processData1, (i,))
    pool.close()
    pool.join()
    pool = multiprocessing.Pool(processes=12)
    for i in TEST_DATA_PATH:
        pool.apply_async(processData1, (i,))
    pool.close()
    pool.join()
    et = time.time()
    time0 = et - st
    print("去停用词用时： {}s".format(str(time0)))

    #生成process128文件
    # st = time.time()
    # pool = multiprocessing.Pool(processes=12)
    # for i in TRAIN_DATA_PATH:
    #     pool.apply_async(processData2, (i,))
    # pool.close()
    # pool.join()
    # pool = multiprocessing.Pool(processes=12)
    # for i in TEST_DATA_PATH:
    #     pool.apply_async(processData2, (i,))
    # pool.close()
    # pool.join()
    # et = time.time()
    # time0 = et - st
    # print("生成256句子用时： {}s".format(str(time0)))

    # 将所有句子写入一个文件
    # st = time.time()
    # pool = multiprocessing.Pool(processes=12)
    # for i,j in zip(TRAIN_DATA_PATH,TEST_DATA_PATH):
    #     pool.apply_async(writeAll, (ALL_DATA, i, j, PROCESS1_FILE))
    # pool.close()
    # pool.join()
    # pool = multiprocessing.Pool(processes=12)
    # for i,j in zip(TRAIN_DATA_PATH,TEST_DATA_PATH):
    #     pool.apply_async(writeAll, (ALL_DATA, i, j, PROCESS256_FILE))
    # pool.close()
    # pool.join()
    # pool = multiprocessing.Pool(processes=12)
    # for i, j in zip(TRAIN_DATA_PATH, TEST_DATA_PATH):
    #     pool.apply_async(writeAll, (ALL_DATA, i, j, PROCESS128_FILE))
    # pool.close()
    # pool.join()
    # et = time.time()
    # time0 = et - st
    # print("将句子写入一个文件用时： {}s".format(str(time0)))



















