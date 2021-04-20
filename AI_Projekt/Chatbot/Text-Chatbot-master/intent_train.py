
from keras.layers import Dense, Dropout, Embedding, Input, merge
from keras.utils import np_utils, generic_utils
from keras import optimizers, metrics
from keras.layers.recurrent import LSTM, GRU
from keras.layers import Concatenate
from keras.preprocessing import sequence
from keras.models import Model
from sklearn import preprocessing, model_selection
import numpy as np
from preprocessor import Dataset, pad_sequences, labels, pad_sequence

dim = 32
max_len = 10 
classes = len(labels)

dataset = Dataset()
print("Datasets loaded.")
X_all = pad_sequences(dataset.X_all_vec_seq)
Y_all = dataset.Y_all
#print (X_all.shape)
x_train, x_test, y_train, y_test = model_selection.traisen_test_split(X_all,Y_all,test_size=0.2)
y_train = pad_sequence(y_train, classes)
y_test = pad_sequence(y_test, classes)
 
#Algorithm For Bidirectional lstm based on http://www.aclweb.org/anthology/C16-1329
sequence = Input(shape=(max_len,300), dtype='float64', name='input')
#Putting forwards_lstm lstm
forwards_lstm = LSTM(dim, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, dropout=0.1, recurrent_dropout=0.1)(sequence)
#Putting  backwards_lstm lstm
backwards_lstm = LSTM(dim, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, dropout=0.1, recurrent_dropout=0.1, go_backwards=True)(sequence)
#forwards_lstm + backwards_lstm = bidirectional lstm
combine = merge([forwards_lstm, backwards_lstm])
#Set dropout layer applied to avoid overfitting problem
after_dp = Dropout(0.6)(combine)
#Applying softmax activation layer
output = Dense(classes, activation='sigmoid', name='activation')(after_dp)

model_try = Model(inputs=sequence, outputs=output)
#model_try is a bidirectional LSTM cell + a dropout layer + an activation layer.
optimizers.Adam(lr=0.001, beta_1=0.6, beta_2=0.099, epsilon=1e-08, decay=0.005, clipnorm = 1., clipvalue = 0.5)
model_try.compile(optimizer = 'Adam', loss = 'categorical_crossentropy',metrics=['categorical_accuracy'])

batch_size = 20

epochs = 25

x_train = np.asarray(x_train)
x_train.ravel()

y_train = np.asarray(y_train)
y_train.ravel()

print("\nFitting to model_try")
my_model = model_try.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=[x_test, y_test])
print("\nModel Training complete.")
model_try.save("nishank/intent_models/smart_model_test2.h5")
print("Model saved to nishank folder.")