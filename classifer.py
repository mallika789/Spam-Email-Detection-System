#krish naik
import pandas as pd
import os
import nltk 
nltk.download('stopwords')
import re
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score,  precision_score, recall_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import feature_extraction, model_selection 
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import stopwords 
from nltk.stem.porter import PorterStemmer
import pickle

df = pd.read_csv("spam.csv", encoding='latin-1')
ps = PorterStemmer()
corpus = []
for i in range(0, len(df)) :
    review = re.sub('[^a-zA-Z]', ' ', df['message'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
    
#creating bag of words
cv = CountVectorizer(max_features=2500)
X = cv.fit_transform(corpus).toarray()

y = pd.get_dummies(df['class'])
y = y.iloc[:,1].values

#Train Test split
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.20, random_state=0)
model = MultinomialNB().fit(X_train, y_train)

y_pred = model.predict(X_test)

confusionmatrix = confusion_matrix(y_test, y_pred)

accuracy = accuracy_score(y_test, y_pred)

#pickel files

pickle_file = 'MNB.pickle'
try:
    file = open(pickle_file, 'wb')
    pickle.dump(model, file, pickle.HIGHEST_PROTOCOL)
    file.close()
except Exception as e:
    print('Unable to save data to', pickle_file, ':', e)
    raise

statinfo = os.stat(pickle_file)
print('Compressed pickle size:', statinfo.st_size)

vect_file = 'vectMNB.pickle'

try:
    file = open(vect_file, 'wb')
    pickle.dump(cv, file, pickle.HIGHEST_PROTOCOL)
    file.close()
except Exception as e:
    print('Unable to save data to', vect_file, ':', e)
    raise
statinfo = os.stat(vect_file)
print('Compressed pickle size:', statinfo.st_size)