import tensorflow as tf
import numpy as np                                  #for large and multi-dimensional arrays
import pandas as pd                                 #for data manipulation and analysis

from tensorflow.python.keras.preprocessing.text import text_to_word_sequence

df = pd.read_csv(r'ITMO_raw_data.csv')

lenOfdf = len(df.index)

for i in range(lenOfdf):
    text = df['Contract description:'].iloc[i]
    pre = text_to_word_sequence(text,lower=True,split=' ')
    print(pre)
