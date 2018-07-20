# from sklearn.externals import joblib
# import jieba
# from collections import Counter
# import numpy as np
#
# TFIDF_PATH = "../model/tfidf.m"
# MODEL_PATH = "../model/svmmodel.m"
# STOP_WORDS = "../data/stop_words/stopwords.txt"
# CONTENT = "公诉 机关 指控 2016 年 被告人 高某 驾驶 摩托车 农用 三轮车 合作 市 那吾乡 早子 沟 金矿 选矿厂 矿石 堆放 处 及早 子沟 号 沟 矿石 堆放 处 聚众 哄抢 矿石 被告人 高某 家中 搜查 哄抢 所得 锑 矿石 共计 3480 千克 甘肃 天则 资产 评估 事务所 评估 每吨 矿石 价值 4049.44 元 人民币 总计 价值 14092.0512 元 人民币 提请 中华人民共和国 刑法 追究 刑事责任 "
#
# with open(STOP_WORDS, "r", encoding="utf-8") as f:
#     stopwordlist = f.read().splitlines()
# model = joblib.load(MODEL_PATH)
# tfidf = joblib.load(TFIDF_PATH)
#
# content = " ".join(word  for word in jieba.cut(CONTENT)  if word not in stopwordlist)
# content_list = []
# content_list.append(content)
# print(content_list)
# print(len(content_list))
#
# tfidf_content = tfidf.transform(content_list)
# print(tfidf_content)
# print(tfidf_content.shape)
#
# result = model.predict(tfidf_content)
#
# print(result)


# import re
# string = "溆浦县"
# content = "溆浦县 人民检察院 指控 2013 11 22 溆浦县 联村 村民 陈某 同村 POS 机上 刷卡 消费 7000 夫妇 产生矛盾 被告人 某华 某书家 吃饭 听说 此事 某书 儿子 陈某 平先 理论 被告人 某华 赶到 家中 发生争执 期间 某华 耳光 前来 劝架 王跃 耳光 被害人 某香 上前 阻止 某华 耳光 踢倒 鉴定 某香 损伤 轻伤 公诉 机关 所控 事实 法庭 提交 被害人 陈述 证人 证言 被告人 供述 相关 证据 该院 被告人 某华 故意伤害 公民 身体 致人 轻伤 被告人 某华 判决 宣告 刑罚 执行 完毕 判决 应二罪 提请 本院 中华人民共和国 刑法 被告人 定罪 处罚 "
# pattern1 = re.compile(u"[\u4e00-\u9fa5]+")
# pattern2 = re.compile(".?['市'|'区'|'县'|]")
# contents = re.findall(pattern=pattern1, string=content)
# result = re.match(pattern=pattern2, string=string)
#
# print(result)


line = "王某 酒后 滋事 殴打 谩骂 出警 公安人员 提请 本院 刑法 追究 刑事责任"
print(len(line))










