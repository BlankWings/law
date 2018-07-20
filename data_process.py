# 处理raw_data文件，得到可以用于后续训练的文件
# 目前是第一阶段，只预测罪名
# 导入使用的库
from helper import *
import os, json, time, jieba, re
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
from sklearn.datasets.base import Bunch
from sklearn.externals import joblib
from sklearn.preprocessing import MultiLabelBinarizer


def process(raw_data_file, trainbunch_file, testbunch_file, split_value):
    '''
    此函数为控制整个处理过程的函数
    输入为原始文件，以及处理后要储存的bunch文件，还有训练集和测试集的分割比例
    函数没有返回值，最终将处理后的bunch写入文件为止
    '''
    databunch = Bunch(contents=[], accu=[])  # databunch 储存对raw_data_file处理后的数据，然后再shuffle成训练集和测试集。
    # contents储存的是处理后的fact， accu储存的是汉字的犯罪标签，new_accu储存的二值化的犯罪标签。
    print("正在写入数据》》》》"); st = time.time()
    with open(raw_data_file, "r", encoding="utf-8") as rawdata_file:
        file_lenth = len(rawdata_file.readlines()); cnt = 0    #  用于记录数据处理进度
    rawdata_file = open(raw_data_file, "r", encoding="utf-8")
    for line_json in rawdata_file.readlines():   # line_json中包含着我们所需的所有信息
        line = json.loads(line_json)             # line是一个字典
        fact = line["fact"]
        accusation = line["meta"]["accusation"]
        new_fact = process_fact(fact)   # new_fact是处理后的犯罪事实，保留了中文, 然后用jieba分词得到最终的结果。
        databunch.contents.append(new_fact)
        databunch.accu.append(accusation)
        cnt += 1
        if cnt%1000 == 0:
            print("写入数据进度为：{:.3f}%".format(cnt/file_lenth*100))
    rawdata_file.close(); et = time.time()
    print("写入数据完毕！！！！用时：{:.3f}s".format(et-st))
    print("正在处理数据》》》》"); st = time.time()
    # new_fact和accu已经写入bunch, 将databunch打乱顺序, 分为trainbunch和testbunch
    random_seed = np.random.randint(0,100)  # 保证databunch中各个内容的shuffle次序是一样的。
    for i in databunch.keys():
        np.random.seed(random_seed)
        np.random.shuffle(databunch[i])
    trainbunch, testbunch = split_train_test(databunch, split_value)
    # 对accu二值化
    multilabelbinarizer = MultiLabelBinarizer(accu_list)
    trainbunch.new_accu = multilabelbinarizer.fit_transform(trainbunch.accu)
    testbunch.new_accu = multilabelbinarizer.transform(testbunch.accu)
    # 将trainbunch, testbunch写入trainbunch_file, testbunch_file
    joblib.dump(trainbunch, trainbunch_file)
    joblib.dump(testbunch, testbunch_file)
    et = time.time()
    print("处理数据完毕！！！！用时：{:.3f}s".format(et - st))

def process_fact(fact):
    new_fact = []
    pattern = re.compile(u"[\u4e00-\u9fa5]+")
    seg_fact = re.findall(pattern=pattern, string=fact)  # 此时的fact为去掉非汉字的列表，['昌宁县人民检察院指控', '年', '月', '日下午', '时许', '被告人段某驾拖车经过鸡飞乡澡塘街子', '时逢堵车', '段某将车停在', '冰凉一夏', '冷饮店门口', '被害人王某的侄子王', '某示意段某靠边未果', '后上前敲打车门让段某离开', '段某遂驾车离开', '但对此心生怨愤', '同年', '月', '日', '时许', '被告人段某酒后与其妻子王', '某一起准备回家', '走到鸡飞乡澡塘街富达通讯手机店门口时停下', '段某进入手机店内对被害人王某进行吼骂', '紧接着从手机店出来拿得一个石头又冲进手机店内朝王某头部打去', '致王某右额部粉碎性骨折', '右眼眶骨骨折', '经鉴定', '被害人王某此次损伤程度为轻伤一级']
    for sub_sentence in seg_fact:
        new_fact.extend([word for word in jieba.cut(sub_sentence)])
    return new_fact

def shuffle_bunch(databunch):
    new_databunch = Bunch()
    for i in databunch.keys():
        np.random.shuffle(databunch[i])
    return new_databunch

def split_train_test(new_databunch, split_value):
    lenth = len(new_databunch.contents) # 数据总量
    split_point = int((1-split_value)*lenth)
    trainbunch = Bunch(); testbunch = Bunch()
    trainbunch.contents = new_databunch.contents[:split_point]
    testbunch.contents = new_databunch.contents[split_point:]
    trainbunch.accu = new_databunch.accu[:split_point]
    testbunch.accu = new_databunch.accu[split_point:]
    return trainbunch, testbunch

if __name__ == '__main__':
    process(RAW_SMALL_DATA, SMALL_TRAINBUNCH_FILE, SMALL_TESTBUNCH_FILE, split_value=0.2)


'''
        for word in line:
            if word not in stopwordslist and len(word)>1 and "县" not in word and "市" not in word and "区" not in word \
                    and "省" not in word and "自治区" not in word and "村" not in word and "镇" not in word \
                    and "营" not in word:
                 newline += word + " "
        if len(newline) > 10:
            newline = newline + "\n"
            o.writelines(newline)
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
    '''
