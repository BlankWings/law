# 根据处理后的数据生成词向量文件 gensim可以一次将多个词转换为词向量。
import gensim
import os,time

st = time.time()

DIRNAME1 =  'data/all_data/process1'
DIRNAME128 =  'data/all_data/process128'
DIRNAME256 =  'data/all_data/process256'
MODEL_FILE = "embedding/w2v/"

class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                yield line.split()


sentences = MySentences(DIRNAME256)  # a memory-friendly iterator
model = gensim.models.Word2Vec(sentences, min_count=3, size=256, workers = 12)
et = time.time()
print(model.similarity("重伤", "轻伤"))
model.save(MODEL_FILE+"256model256")
print("训练词向量模型用时： " + str(et-st) + "s")
