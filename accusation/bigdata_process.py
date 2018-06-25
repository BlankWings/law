# 处理big_data文件，得到可以用于后续训练的文件
# 步骤一：直接将大数据文件处理为在一起，每个罪名下包含该罪名的所有fact原文件
# 步骤二：对上面的文件进行分类，经过分词去停用词获得训练集和测试集，然后进行后续的训练
#导入使用的库
import os
import json
import time
import multiprocessing
import jieba
import re
import linecache
import numpy as np

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
    all_data_list = []; train_data_list = []; test_data_list = []
    for i in range(1, 203):
        all_data_path = os.path.join(ALL_DATA, str(i))
        all_data_list.append(all_data_path)
        if not os.path.exists(all_data_path):
            os.makedirs(all_data_path)
        train_data_path = os.path.join(TRAIN_DATA, str(i))
        train_data_list.append(train_data_path)
        if not os.path.exists(train_data_path):
            os.makedirs(train_data_path)
        test_data_path = os.path.join(TEST_DATA, str(i))
        test_data_list.append(test_data_path)
        if not os.path.exists(test_data_path):
            os.makedirs(test_data_path)
    return accu_dict, stopwordslist, all_data_list, train_data_list, test_data_list

#得到分好类的文件
def writeData(json_path, data_path, accu_dict):
    json_path = json_path
    with open(json_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            line = json.loads(line)
            fact = line["fact"].rstrip().replace("\n","").replace("\r", "").replace("\r\n", "")
            accu_list = line["meta"]["accusation"]
            for accu in accu_list:
                filename =  os.path.join(data_path, str(accu_dict[accu]))+ "/raw_data.txt"
                with open(filename, "a", encoding="utf-8") as o:
                    o.writelines(fact + "\n")

#将all_data分成训练集和测试集
def divide_train_test(all_path, train_path, test_path):
    all_file = all_path + "/raw_data.txt"
    train_file = train_path + "/raw_data.txt"
    test_file = test_path + "/raw_data.txt"
    with open(all_file, "r", encoding="utf-8") as f:
        lenth = len(f.readlines())
        train_list = np.random.choice(range(lenth), int(0.9 * lenth), replace=False)
        test_list = [item for item in range(lenth) if item not in train_list]
    with open(all_file, "r", encoding="utf-8") as f:
        with open(train_file, "w", encoding="utf-8") as o:
            with open(test_file, "w", encoding="utf-8") as l:
                for i in train_list:
                    o.writelines(linecache.getline(all_file, lineno=i))
                for j in test_list:
                    l.writelines(linecache.getline(all_file, lineno=j))

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


if __name__ == '__main__':
    # 相关路径和参数
    BASEPATH = "/home/guopp/Python_Project/law/"
    BIG_DATA_JSON = os.path.join(BASEPATH, "data/raw_data/cail2018_big.json")
    ALL_DATA = os.path.join(BASEPATH, "data/accusation/big/all")
    TRAIN_DATA = os.path.join(BASEPATH, "data/accusation/big/train_data")
    TEST_DATA = os.path.join(BASEPATH, "data/accusation/big/test_data")
    ACCU_FILE = os.path.join(BASEPATH, "data/raw_data/accu.txt")
    STOPWORDS_FILE = os.path.join(BASEPATH, "data/stop_words/stopwords.txt")
    RAW_NAME = "raw_name"
    SEG_NAME = "seg_data.txt"
    PROCESS_NAME = "process_data.txt"

    # 生成相关参数
    accudict, stopwordslist, all_data_list, train_data_list, test_data_list = gen_parameters()

    # 读取JSON文件， 这里使用的append方法，要小心。
    # st = time.time()
    # writeData(BIG_DATA_JSON, ALL_DATA, accudict)
    # et = time.time()
    # time0 = et - st
    # print("写入文件用时： {}s".format(str(time0)))

    # 从all_data中以9:1的比例分别写入训练集和测试集
    # st = time.time()
    # pool = multiprocessing.Pool(processes=12)
    # for i, j, k in zip(all_data_list, train_data_list, test_data_list):
    #     pool.apply_async(divide_train_test, (i,j,k))
    # pool.close()
    # pool.join()
    # et = time.time()
    # time0 = et - st
    # print("将数据分别写入训练集测试集用时： {}s".format(str(time0)))

    # # 生成分词后文件
    # st = time.time()
    # pool = multiprocessing.Pool(processes=12)
    # for i in train_data_list:
    #     pool.apply_async(segData, (i,))
    # pool.close()
    # pool.join()
    # pool = multiprocessing.Pool(processes=12)
    # for i in test_data_list:
    #     pool.apply_async(segData, (i,))
    # pool.close()
    # pool.join()
    # et = time.time()
    # time0 = et-st
    # print("分词用时： {}s".format(str(time0)))

    # 生成process文件,去掉停用词并且过滤掉长度为1的词, 去掉句子长度小于10的句子
    # st = time.time()
    # pool = multiprocessing.Pool(processes=12)
    # for i in train_data_list:
    #     pool.apply_async(processData, (i,))
    # pool.close()
    # pool.join()
    # pool = multiprocessing.Pool(processes=12)
    # for i in test_data_list:
    #     pool.apply_async(processData, (i,))
    # pool.close()
    # pool.join()
    # et = time.time()
    # time0 = et - st
    # print("去停用词用时： {}s".format(str(time0)))
