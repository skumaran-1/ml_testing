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
################################################################################
# all these function will strip raw text and turn them into
# useful things
# there are other things that do this but this was easiest
# and for our purpose this was the best
def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()
def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)
def denoise_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    return text
def replace_contractions(text):
#    print(text)
    """Replace contractions in string of text"""
    return contractions.fix(text)
def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words
def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words
def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words
def replace_numbers(words):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    p = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            continue
        else:
            new_words.append(word)
    return new_words
def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    new_words = []
    for word in words:
        if word not in stopwords.words('english'):
            new_words.append(word)
    return new_words
def stem_words(words):
    """Stem words in list of tokenized words"""
    stemmer = LancasterStemmer()
    stems = []
    for word in words:
        stem = stemmer.stem(word)
        stems.append(stem)
    return stems
def lemmatize_verbs(words):
    """Lemmatize verbs in list of tokenized words"""
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas
def normalize(words):
    words = remove_non_ascii(words)
    words = to_lowercase(words)
    words = remove_punctuation(words)
    words = replace_numbers(words)
    words = remove_stopwords(words)
    return words
def stem(words):
    stems = stem_words(words)
    return stems
# end of functions
################################################################################

lenOfdf = len(df.index)

pre = []

#pol = []

# change all the raw text into useful stuff and store them in pre and pol
# runtime: O(n)
for i in range(lenOfdf):
    text = df['Contract description'].iloc[i]
    #polarity = df['Polarity'].iloc[i]
    review_cleaned = replace_contractions(text)
    review_cleaned = denoise_text(review_cleaned)
    review_cleaned = nltk.word_tokenize(review_cleaned)
    review_cleaned = normalize(review_cleaned)
    pre.append(review_cleaned)
    #pol.append(polarity)

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

#rint('finalWordsNumberList =', finalWordsNumberList)
#print('finalWordsNumberList index 0 =', finalWordsNumberList[0])
#print('finalWordsNumberList len =', len(finalWordsNumberList))

# this def is for a custom word2vec layer that we never used bc we found out keras took care of it for us
def random():
    #something
    '''
    data_index = 0
    # generate batch data
    def generate_batch(data, batch_size, num_skips, skip_window):
        global data_index
        assert batch_size % num_skips == 0
        assert num_skips <= 2 * skip_window
        batch = np.ndarray(shape=(batch_size), dtype=np.int32)
        context = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
        span = 2 * skip_window + 1  # [ skip_window input_word skip_window ]
        buffer = collections.deque(maxlen=span)
        for _ in range(span):
            buffer.append(data[data_index])
            data_index = (data_index + 1) % len(data)
        for i in range(batch_size // num_skips):
            target = skip_window  # input word at the center of the buffer
            targets_to_avoid = [skip_window]
            for j in range(num_skips):
                while target in targets_to_avoid:
                    target = random.randint(0, span - 1)
                targets_to_avoid.append(target)
                batch[i * num_skips + j] = buffer[skip_window]  # this is the input word
                context[i * num_skips + j, 0] = buffer[target]  # these are the context words
            buffer.append(data[data_index])
            data_index = (data_index + 1) % len(data)
        # Backtrack a little bit to avoid skipping words in the end of a batch
        data_index = (data_index + len(data) - span) % len(data)
        return batch, context

    batch, labels = generate_batch(data,batch_size=8, num_skips=2, skip_window=1)
    for i in range(8):
      print(batch[i], reverse_dictionary[batch[i]],
            '->', labels[i, 0], reverse_dictionary[labels[i, 0]])


    valid_size = 16     # Random set of words to evaluate similarity on.
    valid_window = 100  # Only pick dev samples in the head of the distribution.
    valid_examples = np.random.choice(valid_window, valid_size, replace=False)
    num_sampled = 64

    #####################################################################################################################################################################
    ################################## PRE process
    #####################################################################################################################################################################

    batch_size = 128
    embedding_size = 128  # Dimension of the embedding vector.
    skip_window = 1       # How many words to consider left and right.
    num_skips = 2         # How many times to reuse an input to generate a label.

    graph = tf.Graph()

    with graph.as_default():

      # Input data.
      train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
      train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
      valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

      # Ops and variables pinned to the CPU because of missing GPU implementation
      with tf.device('/cpu:0'):
        # Look up embeddings for inputs.
        embeddings = tf.Variable(
            tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)

        # Construct the variables for the NCE loss
        nce_weights = tf.Variable(
            tf.truncated_normal([vocabulary_size, embedding_size],
                                stddev=1.0 / math.sqrt(embedding_size)))
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

      # Compute the average NCE loss for the batch.
      # tf.nce_loss automatically draws a new sample of the negative labels each
      # time we evaluate the loss.
      loss = tf.reduce_mean(
          tf.nn.nce_loss(weights=nce_weights,
                         biases=nce_biases,
                         labels=train_labels,
                         inputs=embed,
                         num_sampled=num_sampled,
                         num_classes=vocabulary_size))

      # Construct the SGD optimizer using a learning rate of 1.0.
      optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

      # Compute the cosine similarity between minibatch examples and all embeddings.
      norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))

      normalized_embeddings = embeddings / norm
      valid_embeddings = tf.nn.embedding_lookup(
          normalized_embeddings, valid_dataset)
      similarity = tf.matmul(
          valid_embeddings, normalized_embeddings, transpose_b=True)

      # Add variable initializer.
      init = tf.global_variables_initializer()

    # Step 5: Begin training.
    num_steps = 100001

    with tf.Session(graph=graph) as session:
      # We must initialize all variables before we use them.
      init.run()
      print('Initialized')

      average_loss = 0
      for step in range(num_steps):
        batch_inputs, batch_labels = generate_batch(
            data, batch_size, num_skips, skip_window)
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
        # We perform one update step by evaluating the optimizer op (including it
        # in the list of returned values for session.run()
        _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += loss_val

        if step % 2000 == 0:
          if step > 0:
            average_loss /= 2000
          # The average loss is an estimate of the loss over the last 2000 batches.
          print('Average loss at step ', step, ': ', average_loss)
          average_loss = 0

        # Note that this is expensive (~20% slowdown if computed every 500 steps)
        if step % 10000 == 0:
          sim = similarity.eval()
          for i in range(valid_size):
            valid_word = reverse_dictionary[valid_examples[i]]
            top_k = 8  # number of nearest neighbors
            nearest = (-sim[i, :]).argsort()[1:top_k + 1]
            log_str = 'Nearest to %s:' % valid_word
            for k in range(top_k):
              try:
                  close_word = reverse_dictionary[nearest[k]]
                  log_str = '%s %s,' % (log_str, close_word)
              except:
                  continue
            print(log_str)
      final_embeddings = normalized_embeddings.eval()

    print(normalized_embeddings)
    print('final embedding:', final_embeddings)
    print('first index:', final_embeddings[0])
    print('len of final_embeddings', len(final_embeddings))
        ####
        # word embbeded layer
        ####
    def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
      assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
      plt.figure(figsize=(18, 18))  # in inches
      for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')

      plt.savefig(filename)

    try:
      # pylint: disable=g-import-not-at-top
      from sklearn.manifold import TSNE
      import matplotlib.pyplot as plt

      tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
      plot_only = 500
      low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
      labels = [reverse_dictionary[i] for i in range(plot_only)]
      plot_with_labels(low_dim_embs, labels)

    except ImportError:
      print('Please install sklearn, matplotlib, and scipy to show embeddings.')
    '''
    return 1

################################################################################
# this is the start the model itself
################################################################################

################################################################################
# start of the custom word embedding layer
# have this to closest "memory digit" ie 4,14,64,128,256 to 1/10 of the vocab size
EMBED_SIZE = 512 # more dimensions in the word will mean more time to train
VOCAB_LEN = len(dictionary.keys())

#print(VOCAB_LEN)

#print(reverse_dictionary.keys())

def getList(dict):
    list = []
    for key in dict.keys():
        list.append(key)
    return list

words_ids = tf.constant(getList(reverse_dictionary))

embeddings = tf.keras.layers.Embedding(VOCAB_LEN, EMBED_SIZE, mask_zero=True)
# this is the embedding layer that is matched to our dimensions for our vocab set

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
THIS TAKES AROUND 30 MIN TO RUN (1 per epoch)
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

# split 80% partial train
# split 10% val train
# split 10% test

partial_x_train = finalWordsNumberList[:53]
partial_y_train = pol[:53]

x_val = finalWordsNumberList[53:60]
y_val = pol[53:60]

test_data = finalWordsNumberList[60:]
test_labels = pol[60:]

# 20 under

# 30 over
# 40 over

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
