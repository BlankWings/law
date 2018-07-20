# 处理raw_data文件，得到可以用于后续训练的文件
# 步骤一：读取文件内容，将fact和accusation根据对应的法律条文进行分类，并写入文件夹。（将train和valid写入到一起作为训练数据）
# 统计一下不同罪名的文件数一共为多少。
# 步骤二：读取分好类的文件，将其分词得到分词后的文件写入seg_data。
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


# 生成需要的一些参数
def gen_parameters():
    # 生成罪名字典
    accu_dict = {}; cnt = 0
    with open(ACCU_FILE, "r", encoding="utf-8") as f:
        for accu in f.readlines():
            cnt += 1
            accu_dict[accu.strip()] = cnt
    # 生成停止词列表
    with open(STOPWORDS_FILE, "r", encoding="utf-8") as f:
        stopwordslist = f.read().splitlines()
    # 生成子文件夹以及子文件夹列表
    train_data_list = []; test_data_list = []
    for i in range(1, 203):
        train_data_path = os.path.join(TRAIN_DATA, str(i))
        train_data_list.append(train_data_path)
        if not os.path.exists(train_data_path):
            os.makedirs(train_data_path)
        test_data_path = os.path.join(TEST_DATA, str(i))
        test_data_list.append(test_data_path)
        if not os.path.exists(test_data_path):
            os.makedirs(test_data_path)
    return accu_dict, stopwordslist, train_data_list, test_data_list

#得到分好类的文件
def writeData(json_path, data_path, accu_dict):
    json_path = json_path
    with open(json_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            line = json.loads(line)
            fact = line["fact"].rstrip()
            accu_list = line["meta"]["accusation"]
            for accu in accu_list:
                filename =  os.path.join(data_path, str(accu_dict[accu]))+ "/raw_data.txt"
                with open(filename, "a", encoding="utf-8") as o:
                    o.writelines(fact + "\n")

#产生分词后的文件
def segData(data_path):
    rawdata = data_path + "/raw_data.txt"
    segdata = data_path + "/seg_data.txt"
    with open(rawdata, "r", encoding="utf-8") as f:
        with open(segdata, "w", encoding="utf-8") as o:
            for line in f.readlines():
                line = line.rstrip()
                o.writelines(" ".join(jieba.cut(line)) + "\n")


#对分词后的文件进行处理，训练集和测试集的处理是不同的，先生成一个训练集和测试集一起的处理后文件，用于最后的处理
#然后训练集和测试机各生成一个处理后文件（待定）
#文件处理一：去掉中文停用词, 为了多线程加速，直接对文件进行处理
#process2_data.txt 去掉了长度为1的词
def processData(data_path):
    segdata = data_path + "/seg_data.txt"
    process1data = data_path + "/process_data.txt"
    with open(segdata, "r", encoding="utf-8") as f:
        with open(process1data, "w", encoding="utf-8") as o:
            for line in f.readlines():
                newline = ""
                pattern1 = re.compile(u"[\u4e00-\u9fa5]+")
                line = re.findall(pattern=pattern1, string=line)
                for word in line:
                    if word not in stopwordslist and len(word)>1 and "县" not in word and "市" not in word and "区" not in word \
                            and "省" not in word and "自治区" not in word and "村" not in word and "镇" not in word \
                            and "营" not in word:
                         newline += word + " "
                if len(newline) > 10:
                    newline = newline + "\n"
                    o.writelines(newline)

# # 统计不同文件的案件数
# def number(data_path):
#     data_path_list = data_path
#     cnt = list(range(1, 203))
#     numberlist = []
#     for datapath in data_path_list:
#         filename = datapath + "/raw_data.txt"
#         with open(filename, "r", encoding="utf-8") as f:
#             number = len(f.readlines())
#             numberlist.append(number)
#     return cnt, numberlist


# # 将所有句子写入到一个文件之中，为生成词向量做准备
# def writeAll(all_data_path, train_data_path, test_data_path, file):  # file指的是要合并的文件
#     all_data = os.path.join(all_data_path, file)
#     train_data = os.path.join(train_data_path, file)
#     test_data = os.path.join(test_data_path, file)
#     with open(all_data, "a", encoding="utf-8") as f:
#         with open(train_data, "r", encoding="utf-8") as o:
#             with open(test_data, "r", encoding="utf-8") as l:
#                 for oline in o.readlines():
#                     f.writelines(oline)
#                 for lline in l.readlines():
#                     f.writelines(lline)
# #删去长度小于10的fact， 对于长度大于256的数据，截取至256，长度小于256的数据，重复句子并截取至256
# def processData2(data_path):
#     process1data = data_path + "/process3_data.txt"
#     process2data = data_path + "/process"+str(LENTH)+"_2data.txt"
#     with open(process1data, "r", encoding="utf-8") as f:
#         with open(process2data, "w", encoding="utf-8") as o:
#             for line in f.readlines():
#                 newline = []
#                 line = line.rstrip().split(" ")
#                 if len(line) >= LENTH:
#                     newline = line[:LENTH]
#                 elif len(line) <= 10:
#                     continue
#                 else:
#                     while(len(newline)<LENTH):
#                         newline.extend(line)
#                     newline = newline[:LENTH]
#                 # print(len(newline))
#                 newline = str(newline).replace("[","").replace("]","").replace("'","").replace(",","")
#                 newline = newline + "\n"
#                 o.writelines(newline)

# #暂时没想好还要怎么处理
# def processData3(seg_path, process_path):
#     pass

if __name__ == '__main__':
    # 相关路径和参数
    BASEPATH = "/home/guopp/Python_Project/law/"
    TRAIN_DATA_JSON = os.path.join(BASEPATH, "data/raw_data/data_train.json")
    TEST_DATA_JSON = os.path.join(BASEPATH, "data/raw_data/data_test.json")
    VALID_DATA_JSON = os.path.join(BASEPATH, "data/raw_data/data_valid.json")
    TRAIN_DATA = os.path.join(BASEPATH, "data/accusation/train_data")
    TEST_DATA = os.path.join(BASEPATH, "data/accusation/test_data")
    ALL_DATA = os.path.join(BASEPATH, "data/accusation/all_data")
    ACCU_FILE = os.path.join(BASEPATH, "data/raw_data/accu.txt")
    STOPWORDS_FILE = os.path.join(BASEPATH, "data/stop_words/stopwords.txt")
    SEG_NAME = "seg_data.txt"
    PROCESS_NAME = "process_data.txt"

    # 生成相关参数
    accudict, stopwordslist, train_data_list, test_data_list = gen_parameters()

    # 读取JSON文件
    st = time.time()
    writeData(TRAIN_DATA_JSON, TRAIN_DATA, accudict)
    writeData(TEST_DATA_JSON, TEST_DATA, accudict)
    et = time.time()
    time0 = et - st
    print("写入文件用时： {}s".format(str(time0)))


    # 生成分词后文件
    st = time.time()
    pool = multiprocessing.Pool(processes=12)
    for i in train_data_list:
        pool.apply_async(segData, (i,))
    pool.close()
    pool.join()
    pool = multiprocessing.Pool(processes=12)
    for i in test_data_list:
        pool.apply_async(segData, (i,))
    pool.close()
    pool.join()
    et = time.time()
    time0 = et - st
    print("分词用时： {}s".format(str(time0)))

    # 生成process文件,去掉停用词并且过滤掉长度为1的词, 去掉句子长度小于10的句子
    st = time.time()
    pool = multiprocessing.Pool(processes=12)
    for i in train_data_list:
        pool.apply_async(processData, (i,))
    pool.close()
    pool.join()
    pool = multiprocessing.Pool(processes=12)
    for i in test_data_list:
        pool.apply_async(processData, (i,))
    pool.close()
    pool.join()
    et = time.time()
    time0 = et - st
    print("去停用词用时： {}s".format(str(time0)))


    # 统计不同样例的数量
    # cnt, numberlist = number(TRAIN_DATA_PATH)
    # print(sum(numberlist))
    # plt.scatter(cnt, numberlist)
    # plt.ylim(0,200)
    # plt.show()

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
