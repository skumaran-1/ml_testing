import pandas as pd
import contractions, re, nltk, unicodedata
from bs4 import BeautifulSoup
from nltk.corpus import stopwords

df = pd.read_csv(r'ITMO_raw_data.csv')

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

list = cleanedTokens(df)

print(list)
