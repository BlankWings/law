from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Input, concatenate
from keras.layers.core import Dense, Flatten, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import SimpleRNN, LSTM, GRU
from keras.layers.wrappers import Bidirectional  # 构建双向循环网络
from keras.models import Model, load_model

class dl_model:
    def __init__(self, token_words, max_lenth):
        self.token_words = token_words
        self.max_lenth = max_lenth

    def MLP_NETWORK(self):  # 多层感知机神经网络结构
        model = Sequential()  # 堆叠式模型
        # 网络的第一层为Embedding层,嵌入层将单词数字变为向量, output_dim嵌入层的输出维度，每个单词用多少维向量表示。input_length为输入句子长度，　input_dim为字典维度。
        model.add(Embedding(input_dim=self.token_words, input_length=self.max_lenth, output_dim=32))
        model.add(Dropout(0.2))  # 加入Dropout层，防止过拟合
        model.add(Flatten())  # 加入Flattern层，变为３２００个神经元
        model.add(Dense(units=256, activation="relu"))  # 隐藏层
        model.add(Dropout(0.5))
        model.add(Dense(units=20, activation="softmax"))  # 输出层
        print("神经网络的结构如下：")
        print(model.summary())
        return model

    def RNN_NETWORK(self):
        model = Sequential()  # 堆叠式模型
        # 网络的第一层为Embedding层,嵌入层将单词数字变为向量, output_dim嵌入层的输出维度，每个单词用多少维向量表示。input_length为输入句子长度，　input_dim为字典维度。
        model.add(Embedding(input_dim=self.token_words, input_length=self.max_lenth, output_dim=32))
        model.add(Dropout(0.2))  # 加入Dropout层，防止过拟合
        model.add(SimpleRNN(units=16))  # 加入Flattern层，变为３２００个神经元
        model.add(Dense(units=256, activation="relu"))  # 隐藏层
        model.add(Dropout(0.5))
        model.add(Dense(units=20, activation="softmax"))  # 输出层
        print("神经网络的结构如下：")
        print(model.summary())
        return model

    def BIRNN_NETWORK(self):
        model = Sequential()  # 堆叠式模型
        # 网络的第一层为Embedding层,嵌入层将单词数字变为向量, output_dim嵌入层的输出维度，每个单词用多少维向量表示。input_length为输入句子长度，　input_dim为字典维度。
        model.add(Embedding(input_dim=self.token_words, input_length=self.max_lenth, output_dim=32))
        model.add(Dropout(0.2))  # 加入Dropout层，防止过拟合
        model.add(
            Bidirectional(SimpleRNN(units=16, return_sequences=True), merge_mode="concat"))  # 加入Flattern层，变为３２００个神经元
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(units=256, activation="relu"))  # 隐藏层
        model.add(Dropout(0.5))
        model.add(Dense(units=20, activation="softmax"))  # 输出层
        print("神经网络的结构如下：")
        print(model.summary())
        return model

    def LSTM_NETWORK(self):
        model = Sequential()  # 堆叠式模型
        # 网络的第一层为Embedding层,嵌入层将单词数字变为向量, output_dim嵌入层的输出维度，每个单词用多少维向量表示。input_length为输入句子长度，　input_dim为字典维度。
        model.add(Embedding(input_dim=self.token_words, input_length=self.max_lenth, output_dim=32))
        model.add(Dropout(0.2))  # 加入Dropout层，防止过拟合
        model.add(LSTM(units=64))
        model.add(Dense(units=256, activation="relu"))  # 隐藏层
        model.add(Dropout(0.5))
        model.add(Dense(units=202, activation="softmax"))  # 输出层
        print("神经网络的结构如下：")
        print(model.summary())
        return model

    def GRU_NETWORK(self):
        model = Sequential()  # 堆叠式模型
        # 网络的第一层为Embedding层,嵌入层将单词数字变为向量, output_dim嵌入层的输出维度，每个单词用多少维向量表示。input_length为输入句子长度，　input_dim为字典维度。
        model.add(Embedding(input_dim=self.token_words, input_length=self.max_lenth, output_dim=64))
        model.add(Dropout(0.2))  # 加入Dropout层，防止过拟合
        model.add(GRU(units=128))
        model.add(Dense(units=1024, activation="relu"))  # 隐藏层
        model.add(Dropout(0.5))
        model.add(Dense(units=512, activation="relu"))  # 隐藏层
        model.add(Dropout(0.5))
        model.add(Dense(units=202, activation="sigmoid"))  # 输出层
        print("神经网络的结构如下：")
        print(model.summary())
        return model

    def TEXT_CNN_NETWORK(self):
        sentence_seq = Input(shape=[self.max_lenth], name="X_seq")  # 输入
        embedding_layer = Embedding(input_dim=self.token_words, output_dim=64)(sentence_seq)  # 词嵌入层
        # 卷积层如下：
        convs = []
        filter_sizes = [3, 5, 7, 9]
        for fsz in filter_sizes:
            l_conv = Conv1D(filters=100, kernel_size=fsz, activation="relu")(embedding_layer)
            l_pool = MaxPooling1D(self.max_lenth - fsz + 1)(l_conv)
            l_pool = Flatten()(l_pool)
            convs.append(l_pool)
        merge = concatenate(convs, axis=1)
        out = Dropout(0.5)(merge)
        output = Dense(512, activation="relu")(out)
        output = Dropout(0.5)(output)
        output = Dense(202, activation="sigmoid")(output)
        model = Model([sentence_seq], output)
        print("神经网络的结构如下：")
        print(model.summary())
        return model