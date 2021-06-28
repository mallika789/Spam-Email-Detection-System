import pickle
from sklearn.feature_extraction.text import CountVectorizer
import nltk 
import string
from nltk.corpus import stopwords 
nltk.download('stopwords')
import streamlit as st
from win32com.client import Dispatch

def speak(text) :
    speak = Dispatch(("SAPI.SpVoice"))
    speak.Speak(text)

model=pickle.load(open("F:\\Spam Email Detection Project\\MNB.pickle", "rb"))
cv = pickle.load(open("F:\\Spam Email Detection Project\\vectMNB.pickle", "rb"))

def process_text(text):
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    clean_words = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    return clean_words

def main() :
    st.title("Spam Email Detection System")
    st.subheader("Hello user! You can enter your email corpus below to check whether your email is spam or not spam.")
    message = st.text_input("Enter email corpus : ")
    if(st.button("Predict")) :
        data = [message]
        #cv = CountVectorizer()
        vect = cv.transform(data).toarray()
        prediction = model.predict(vect)
        result = prediction[0]
        if result == 1 :
            st.error("This is a spam mail, Be safe")
            speak("This is a spam mail, Be safe")
        else :
            st.success("This is not a spam mail")
main()
        
