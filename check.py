from helper import *
from main import *
from keras.models import load_model
from sklearn.externals import joblib
import numpy as np
import os

# 本程序用于测试模型

print("正在加载模型和token字典")
my_model = load_model(os.path.join(BASE_DIR, "model", "gru64.h5"))
my_token = joblib.load(os.path.join(BASE_DIR, "model", "token_gru64.pkl"))
testbunch = joblib.load(SMALL_TESTBUNCH_FILE)

sentence_seq = my_token.texts_to_sequences(testbunch.contents[0:100])
sentence_seq_pad = sequence.pad_sequences(sentence_seq, maxlen=380)

y = my_model.predict(sentence_seq_pad)
y = np.argmax(y,axis=1)

print(np.argmax(testbunch.new_accu[0:100],axis=1))
print(y)
