from tensorflow.keras.models import load_model
import pickle
import numpy as np                                  #for large and multi-dimensional arrays
import pandas as pd                                 #for data manipulation and analysis
import re, contractions, nltk, unicodedata, inflect, collections, random, math
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import tensorflow as tf
################################################################################
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
# Load saved model from files
model = load_model("model.h5")
# prints models layers so we can see how many params
model.summary()
# load our dictionaries
with open('dictionary.pickle', 'rb') as handle:
    dictionary = pickle.load(handle)
with open('reverse_dictionary.pickle', 'rb') as handle:
    reverse_dictionary = pickle.load(handle)
################################################################################
# this depended on what file i was looking at
#
#df = pd.read_csv(r'0 contracts.csv')

df = pd.read_csv(r'ITMO_raw_data.csv')

description = df['Contract description']

#description = df['tm_govsuite__description__c']
################################################################################
cleaned_descriptions = cleanedTokens(df)

finalWordsNumberList = []
# this loop is a little confusing to logically understand so lets break it down
# first we need to turn words into their numbers, but what if a word already was
# used in our old dictionary? that word would already be assigned a number
for list in cleaned_descriptions:
    sentence = []
    for word in list:
        try:
            # so we look up the word and replace it
            new = dictionary[word]
            word = new
            sentence.append(word)
        except:
            # what if the word doesnt exist in our dictionary? well in our model
            # if we try to give it a new word it vocab cap going to increased
            # our model then freaks out and crash because it doesnt know how to
            # interpret the new word/number
            # work around for this is simply assigning it to 0
            word = 0
            sentence.append(word)
    # add word to finalWordsNumberList like before
    finalWordsNumberList.append(sentence)

# padding
finalWordsNumberList = tf.keras.preprocessing.sequence.pad_sequences(finalWordsNumberList, value=0, padding='post')

results = []
# use the sentence predict based on the model and
a = model.predict(finalWordsNumberList,use_multiprocessing=True)
# results stored in a
print(a)
