# 处理多标签文件，将RAW_MULTI_LABELS_FILE处理为PROCESS_MULTI_LABELS_FILE

import re
import jieba
from helper import *

BASE_DIR = "/home/guopp/Python_Project/code/classification/law/"
MULTI_LABELS_PATH = os.path.join(BASE_DIR, "data/accusation/multi_labels")
RAW_MULTI_LABELS_FILE = os.path.join(MULTI_LABELS_PATH, "labels_contents.txt")
PROCESS_MULTI_LABELS_FILE_WITH1 = os.path.join(MULTI_LABELS_PATH, "process_multilabels_with1.txt")
PROCESS_MULTI_LABELS_FILE_WITHOUT1 = os.path.join(MULTI_LABELS_PATH, "process_multilabels_without1.txt")


if __name__ == '__main__':
    # 获取文件的长度
    with open(RAW_MULTI_LABELS_FILE, "r", encoding="utf-8") as f:
        lenth = len(f.readlines())
    # 分别对文件的每一行进行处理，并写入PROCESS_MULTI_LABELS_FILE
    with open(PROCESS_MULTI_LABELS_FILE_WITH1, "w", encoding="utf-8") as with1file:
        with open(PROCESS_MULTI_LABELS_FILE_WITHOUT1, "w", encoding="utf-8") as without1file:
            with open(RAW_MULTI_LABELS_FILE, "r", encoding="utf-8") as f:
                for line in f.readlines():  # line是['交通肇事', '故意伤害'] 二审经审理查明的事实和采信的证据与原审一致。这样的话
                    accusation = line.split("]")[0] + "]"  # 提取罪名， 类型为字符串
                    with1file.writelines(accusation + " ")
                    without1file.writelines(accusation + " ")
                    fact = line.split(accusation)[-1]  # 提取犯罪事实
                    process_fact = "" # 储存处理过的犯罪事实
                    # 对fact进行处理， 先去掉xxxx查明，xxx指控：
                    if "查明" in fact[:20]:
                        del_fact = fact.split("查明")[0] + "查明"
                        fact = fact.split(del_fact)[-1]
                    elif "指控" in fact[:20]:
                        del_fact = fact.split("指控")[0] + "指控"
                        fact = fact.split(del_fact)[-1]
                    # 去掉20xx年xx，
                    fact = re.sub("201[^，]+，","",fact)
                    # 对fact分词去停用词然后和accusation一起写入PROCESS_MULTI_LABELS_FILE
                    for word in jieba.cut(fact):
                        if re.match(u"[\u4e00-\u9fa5]+", word):
                            if  "县" not in word and "市" not in word and "区" not in word and "省" not in word and "自治区" not in word and "村" not in word and "镇" not in word and "营" not in word:
                                if len(word) > 1:
                                    with1file.writelines(word + " ")
                                    without1file.writelines(word + " ")
                                else:
                                    with1file.writelines(word + " ")
                            else:
                                continue
                        else:
                            continue
                    with1file.writelines("\n")
                    without1file.writelines("\n")
