# 使用keras进行文本分类
# 使用的网络结构包含MLP， RNN， LSTM, GRU, CNN等。
from helper import *
from sklearn.externals import joblib
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.models import load_model
from deeplearning_model import dl_model
import time

if __name__ == '__main__':
    print("正在加载数据》》》》"); st = time.time()
    trainBunch = joblib.load(SMALL_TRAINBUNCH_FILE) # 读取保存的trainbunch
    testBunch = joblib.load(SMALL_TESTBUNCH_FILE)  # 读取保存的testbunch
    et = time.time()
    print("加载数据完成！！！！用时：{:.3f}s".format(et-st))
    # 相关参数如下：
    token_words = 38000   # 单词字典的单词数。
    max_lenth = 380      # 选取句子的长度。
    print("正在进行数据预处理》》》》")
    st = time.time()
    # 建立Token词典
    token = Tokenizer(num_words=token_words)  # 设置词典规模
    token.fit_on_texts(trainBunch.contents)   # 建立字典模型
    print(token.document_count)
    print(len(token.word_index.values()))
    # 将文字列表转化为数字列表
    trainBunch.contents_seq = token.texts_to_sequences(trainBunch.contents)
    testBunch.contents_seq = token.texts_to_sequences(testBunch.contents)
    # 对数字列表进行padding，截长补短。处理后的数据输入神经网络进行训练。
    trainBunch.contents_seq_pad = sequence.pad_sequences(trainBunch.contents_seq, maxlen=max_lenth)
    testBunch.contents_seq_pad = sequence.pad_sequences(testBunch.contents_seq, maxlen=max_lenth)
    print(trainBunch.contents_seq_pad.shape)
    print(testBunch.contents_seq_pad.shape)
    et = time.time()
    print("数据预处理完成！！！！用时：{:.3f}s".format(et-st))
    # 构建神经网络，
    DL_MODEL = dl_model(token_words, max_lenth)
    model = DL_MODEL.TEXT_CNN_NETWORK()
    # 定义训练方法
    model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=['categorical_accuracy'])
    # 开始训练
    train_history = model.fit(trainBunch.contents_seq_pad, trainBunch.new_accu, batch_size=1000, epochs=20, verbose=1, validation_split=0.1)
    # 评估模型准确率
    score = model.evaluate(testBunch.contents_seq_pad, testBunch.new_accu, verbose=2)
    print("模型准确率为：{:.3f}".format(score[1]))

    result = model.predict(testBunch.contents_seq_pad[3:4])
    print(testBunch.contents_seq_pad[3:4])
    print(result)
    print(testBunch.new_accu[3:4])
    # keras保存模型
    # 保存模型和Token字典
    model_file_name = gen_model_file("gru64.h5")
    token_file_name = gen_model_file("token_gru64.pkl")
    model.save(model_file_name)
    joblib.dump(token, token_file_name)
    print("保存模型完成！！！")

    # print("正在加载模型和token字典")
    # my_model = load_model(os.path.join(BASE_DIR, "model", "textcnn.h5"))
    # my_token = joblib.load(os.path.join(BASE_DIR, "model", "token.pkl"))
    # testbunch = joblib.load(SMALL_TESTBUNCH_FILE)
    #
    # sentence_seq = my_token.texts_to_sequences(testbun)
    # sentence_seq_pad = sequence.pad_sequences(sentence_seq, maxlen=100)
    #
    # y = my_model.predict_class(sentence_seq_pad)
    # print(y)


