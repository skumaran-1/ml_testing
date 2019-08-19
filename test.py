from tensorflow.keras.models import load_model
import pickle
import numpy as np                                  #for large and multi-dimensional arrays
import pandas as pd                                 #for data manipulation and analysis
import re, contractions, nltk, unicodedata, inflect, collections, random, math
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import tensorflow as tf
################################################################################
# functions
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
cleaned_descriptions = []
# this loop word_tokenizes the text
for i in range(len(df.index)):
    text = description.iloc[i]
    review_cleaned = replace_contractions(text)
    review_cleaned = denoise_text(review_cleaned)
    review_cleaned = nltk.word_tokenize(review_cleaned)
    review_cleaned = normalize(review_cleaned)
    cleaned_descriptions.append(review_cleaned)

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
