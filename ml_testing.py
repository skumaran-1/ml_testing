import tensorflow as tf
import numpy as np                                  #for large and multi-dimensional arrays
import pandas as pd                                 #for data manipulation and analysis
from bs4 import BeautifulSoup
import re, contractions, nltk, unicodedata, inflect, collections, random, math
from nltk.corpus import stopwords
# import for multi layers for the model
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Embedding, Conv1D, GlobalMaxPooling1D, LSTM, MaxPooling1D
################################################################################
import pickle
# pandas cvs
df = pd.read_csv(r'ITMO_raw_data.csv')
###############################################################################
# this is a function that will shorten and turn all the little steps into 1
# at first we were doing all the little things but then we combined this into 1
# giant function, now this will allow us to have this function anywhere and work
# with our model for pre-processing

def cleanedTokens(df):
    lenOfdf = len(df.index)
    pre = []
    for i in range(lenOfdf):
        text = df['Contract description'].iloc[i]
        text = contractions.fix(text)
        soup = BeautifulSoup(text, "html.parser")
        text = soup.get_text()
        text = re.sub('\[[^]]*\]', '', text)
        text = nltk.word_tokenize(text)
        new_words = []
        for word in text:
            new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
            new_word = new_word.lower()
            new_word = re.sub(r'[^\w\s]', '', new_word)
            if new_word != '':
                if new_word.isdigit():
                    continue
                elif new_word not in stopwords.words('english'):
                    new_words.append(new_word)
        pre.append(new_words)
    return pre

################################################################################
pre = cleanedTokens(df)

pol = df['Polarity'].to_numpy().tolist()

# this will build the data set into 4 things
# data, count, dictionary, reversed_dictionary
# data will be the raw data
# count will be counters
# dictionary is the thing we care about most
# reverse_dictionary is reverse_dictionary
def build_dataset(words, n_words):
    """Process raw inputs into a dataset."""
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary

# YOU CAN CHANGE THIS IF YOU GO ABOVE ITS JUST THE CAP
# AS OF 8/12/2019 COUNT REACHED 5013 SO WE ARE GOING TO KEEP
# THIS FOR NOW
vocabulary_size = 10000

#print(pre)
# takes each sen in pre and flattens
real = [item for sublist in pre for item in sublist]

#print(real)
#builds based on the flat
data, count, dictionary, reverse_dictionary = build_dataset(real, vocabulary_size)

#print(data)

#print(count)

#print(dictionary)

#print(reverse_dictionary)

#print('Most common words (+UNK)', count[:5])
#print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])
#print('data len =', len(data))

####
# turn words into numbers and gets the ultimate dataset
finalWordsNumberList = []
for list in pre:
    sentence = []
    for word in list:
        try:
            new = dictionary[word]
            word = new
            sentence.append(word)
        except:
            word = 0
            sentence.append(word)
    finalWordsNumberList.append(sentence)

################################################################################
# this is the end of the pre process
################################################################################
# this is the start the model itself
################################################################################

# start of the custom word embedding layer
# have this to closest "memory digit" ie 4,14,64,128,256 to 1/10 of the vocab size
# the embedding_size suually has no impact for the actual model, the number of
# layers and stuff actually matter, apply underfitting and overfitting here as well
EMBED_SIZE = 512 # more dimensions in the word will mean more time to train
VOCAB_LEN = len(dictionary.keys())

#print(VOCAB_LEN)

#print(reverse_dictionary.keys())

def getList(dict):
    list = []
    for key in dict.keys():
        list.append(key)
    return list

# word ids was for something else we never got a chance to use it but its still
# here if we want to use it in the future
words_ids = tf.constant(getList(reverse_dictionary))

embeddings = tf.keras.layers.Embedding(VOCAB_LEN, EMBED_SIZE, mask_zero=True)
# this is the embedding layer that is matched to our dimensions for our vocab set

# MASKING: https://www.tensorflow.org/beta/guide/keras/masking_and_padding

#Under the hood, these layers will create a mask tensor (2D tensor with shape (batch, sequence_length)),
#and attach it to the tensor output returned by the Masking or Embedding layer.
#tf.Tensor(
#[[ True  True  True  True  True  True]
# [ True  True  True  True  True False]
# [ True  True  True False False False]], shape=(3, 6), dtype=bool)

# this will then be essentially skipped in the model, when computing

# END MASKING

#a = 0
#c = []

#for i in finalWordsNumberList:
#    b = len(i)
#    print(b)
#    c.append(b)
#    if (b > a):
#        a = b
#    print(i)

#print("BIGGEST = ", a, c)

# a is all the words in the vocab set up to 10,000
# this increased customization of the model below
# preprocessing defaults to max length but we were using this loop for our
# refrence the amount of max words per sentences
finalWordsNumberList = tf.keras.preprocessing.sequence.pad_sequences(finalWordsNumberList, value=0, padding='post')

#for i in finalWordsNumberList:
#    print(len(i))
#    print(i)

# this pads the finalWordsNumberList sentences so that they are all the same length
# adding 0s after will allow it to fit in the model

#end of custom word embedding
################################################################################

# this is the real model
model = tf.keras.Sequential()
#model.add(tf.keras.layers.Embedding(10000, 16))
# the first layer of the model MUST BE THE EMBEDDING
model.add(embeddings)

################ option 1
# this was the base option we got and used at the begining
# after the model was done and shows some progress
# we upgraded to option 3
'''
model.add(tf.keras.layers.GlobalAveragePooling1D())
model.add(tf.keras.layers.Dense(16, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(1, activation=tf.nn.sigmoid))
'''
################################################################################
'''
############### option 3
# option 3 only works if unmasked, Added masking to to help with preprocessing and to account for a gigher vocab set count
# the final option is used because we wanted to have the most in the vocab set which is above
#######################
#filters = 64
#kernel_size = 5
#pool_size = 4
#lstm_output_size = 70
# multi layer model
# dropout
#model.add(Dropout(0.25))
# convo layer that helped
# MASKING THE EMBEDDING LAYER WILL NOT ALLOW YOU TO DO CONVOLUTION or MaxPooling1D
# you cannot reshape a masked layer
#model.add(Conv1D(filters,
#                 kernel_size,
#                 padding='valid',
#                 activation='relu',
#                 strides=1))
#model.add(MaxPooling1D(pool_size=pool_size))
#model.add(LSTM(lstm_output_size))
'''
################################################################################
# option 5
############ PLEASE CHECK THAT THE MASK IS ON IN THE EMBEDDING LAYER
# bc masking prevents convolutions we added more LSTM layers to improve accuracy
model.add(Dropout(0.25))
model.add(LSTM(128, return_sequences=True))
model.add(LSTM(64, return_sequences=True))
model.add(LSTM(1))
model.add(Dense(1))
model.add(Activation('sigmoid'))
'''
################################ NOTE ################################
THIS TAKES AROUND 40 MIN TO RUN (1.6 min per epoch)
HOWEVER THE ACCURACY AND EVERYTHING IS SIGNIFICANTLY HIGHER
# this can be changed below to make
sure you are not overfitting or underfitting
'''
################################################################################
# option 6
# TURN MASK LAYER OFF

# end model build
################################################################################

model.summary()

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc'])

################################################################################
# data splitting/fitting/training/ and evaluating

# technically there is already a tf data splitter function out there
# we wanted to play around with the numbers more freely
# both work

# split 80% partial train
# split 10% val train
# split 10% test

partial_x_train = finalWordsNumberList[:53]
partial_y_train = pol[:53]

x_val = finalWordsNumberList[53:60]
y_val = pol[53:60]

test_data = finalWordsNumberList[60:]
test_labels = pol[60:]

# epoch rules for over and underfitting for this specific case:
# IF YOU ADD MORE DATA PLEASE ADJUST THIS TO FIT UR CASE
# THIS IS WHERE THE Tf.split comes in
# 20 under
# 25 epochs turns into 100% without over or under fitting
# 30 over
# 40 over

# batch sizes: depends on computer specs tbh

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=25,
                    batch_size=1024,
                    validation_data=(x_val, y_val),
                    verbose=2,
                    )

results = model.evaluate(test_data, test_labels)

print(results)

###
# this is where the real data predict would go
###
#result = model.predict(finalWordsNumberList)

#print(result)

################################################################################
# end model building
################################################################################

# graph evaluation for us to adjust models which we commented out

history_dict = history.history
history_dict.keys()

import matplotlib.pyplot as plt

acc = history_dict['acc']
val_acc = history_dict['val_acc']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

plt.clf()   # clear figure

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.clf()   # clear figure

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

################################################################################
# atfer this will be to save and export the model
################################################################################

model.save("model.h5")

with open('dictionary.pickle', 'wb') as handle:
    pickle.dump(dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('reverse_dictionary.pickle', 'wb') as handle:
    pickle.dump(dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)
