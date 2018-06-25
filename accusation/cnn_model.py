import os, time
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras.optimizers import *
from keras.models import Sequential
from keras.layers import Merge
from gensim.models.word2vec import Word2Vec
import sys

BASE_DIR = '..'
MODEL_PATH = os.path.join(BASE_DIR, 'embedding/w2v/1model128')
TRAIN_DATA_DIR = os.path.join(BASE_DIR, "data/train_data")
TEST_DATA_DIR = os.path.join(BASE_DIR, "data/test_data")
WORD_INDXE = os.path.join(BASE_DIR,"embedding/w2v/word_index.txt")
MAX_SEQUENCE_LENGTH = 128
MAX_NB_WORDS = 20000
#EMBEDDING_DIM = 100
EMBEDDING_DIM = 128
#VALIDATION_SPLIT = 0.2
VALIDATION_SPLIT = 0.01

#获得训练和测试数据，文字内容以及标签
train_texts = []
train_labels = []
for name in sorted(os.listdir(TRAIN_DATA_DIR)):
    path = os.path.join(TRAIN_DATA_DIR, name, "process128_data.txt")
    with open(path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            line = line.rstrip()
            train_texts.append(line)
            train_labels.append(int(name))

test_texts = []
test_labels = []
for name in sorted(os.listdir(TEST_DATA_DIR)):
    path = os.path.join(TEST_DATA_DIR, name, "process128_data.txt")
    with open(path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            line = line.rstrip()
            test_texts.append(line)
            test_labels.append(int(name))
# 全部的文本数据
all_texts = train_texts + test_texts

st = time.time()
tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(all_texts)
et = time.time()
print("用时：" + str(et-st))
train_sequences = tokenizer.texts_to_sequences(train_texts)
test_sequences = tokenizer.texts_to_sequences(test_texts)

word_index = tokenizer.word_index
with open(WORD_INDXE, "w", encoding="utf-8") as f:
    f.write(str(word_index))
print('Found %s unique tokens.' % len(word_index))

train_data = pad_sequences(train_sequences, maxlen=MAX_SEQUENCE_LENGTH)
print(train_data[0])
test_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)

train_labels = to_categorical(np.asarray(train_labels))
test_labels = to_categorical(np.asarray(test_labels))
# print('Shape of train_data tensor:', train_data.shape)
# print('Shape of train_label tensor:', train_labels.shape)
# print('Shape of test_data tensor:', test_data.shape)
# print('Shape of test_label tensor:', test_labels.shape)


#词嵌入矩阵
model = Word2Vec.load(MODEL_PATH)
embedding_matrix = np.zeros((MAX_NB_WORDS+1, EMBEDDING_DIM))
for word, i in word_index.items():
    if i <= MAX_NB_WORDS:
        try:
            embedding_vector = model[word]
        except:
            continue
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector # word_index to word_embedding_vector ,<20000(nb_words)
# print(embedding_matrix.shape)
# print(embedding_matrix[0:10])

embedding_layer = Embedding(MAX_NB_WORDS + 1,
                            EMBEDDING_DIM,
                            input_length=MAX_SEQUENCE_LENGTH,
                            weights=[embedding_matrix],
                            trainable=False)

print('Training model.')


model = Sequential()
model.add(embedding_layer)
model.add(Conv1D(128, 3, activation='relu'))
model.add(MaxPooling1D(2))
model.add(Conv1D(256, 3, activation='relu'))
model.add(MaxPooling1D(2))
model.add(Conv1D(512, 3, activation='relu'))
model.add(MaxPooling1D(4))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dense(512, activation="relu"))
model.add(Dense(203, activation='softmax'))

#model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='Adam',
              metrics=['accuracy'])

print(model.summary())
# happy learning!
model.fit(train_data, train_labels, batch_size=128, epochs=20, validation_split=VALIDATION_SPLIT)
# model.fit([x_train,x_train], y_train, nb_epoch=5)


score = model.evaluate(test_data, test_labels, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
model.save("model/accu1.h5")
