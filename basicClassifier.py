import pandas as pd
import nltk 
import string
from subprocess import call
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score,  precision_score, recall_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn import feature_extraction, model_selection, naive_bayes, metrics, svm
from collections import Counter
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
import numpy as np
from six.moves import cPickle as pickle
import os

def process_text(text):
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    clean_words = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    return clean_words

df = pd.read_csv("spamdata.csv", encoding='latin-1')
df.drop_duplicates(inplace=True)
df.dropna()
df = df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'],axis=1)

count1 = Counter(" ".join(df[df['v1']=='ham']["v2"]).split()).most_common(20)
df1 = pd.DataFrame.from_dict(count1)
df1 = df1.rename(columns={0: "words in non-spam", 1 : "count"})

count2 = Counter(" ".join(df[df['v1']=='spam']["v2"]).split()).most_common(20)
df2 = pd.DataFrame.from_dict(count2)
df2 = df2.rename(columns={0: "words in spam", 1 : "count_"})

f = CountVectorizer(analyzer=process_text, ngram_range=(1,1))
X = f.fit_transform(df["v2"])
features_names = (f.get_feature_names())

Tftrans = TfidfTransformer().fit(X)
TfIdf = Tftrans.transform(X)

df_idf = pd.DataFrame(Tftrans.idf_, index=features_names,columns=["idf_weights"])

df_idf.sort_values(by=['idf_weights'])

df["v1"]=df["v1"].map({'spam':1,'ham':0})
X_train, X_test, y_train, y_test = model_selection.train_test_split(TfIdf, df['v1'], test_size=0.30, random_state=42)
print([np.shape(X_train), np.shape(X_test)])

X_train.toarray()

X_test.toarray()

BayesModel = naive_bayes.MultinomialNB().fit(X_train, y_train)

pickle_file = 'MNB.pickle'
try:
    f = open(pickle_file, 'wb')
    pickle.dump(BayesModel, f, pickle.HIGHEST_PROTOCOL)
    f.close()
except Exception as e:
    print('Unable to save data to', pickle_file, ':', e)
    raise

statinfo = os.stat(pickle_file)
print('Compressed pickle size:', statinfo.st_size)
