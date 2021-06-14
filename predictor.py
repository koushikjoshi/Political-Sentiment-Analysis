import tensorflow_hub as hub
from tensorflow import keras
import nltk
import re
import numpy as np
import os
from firebase import firebase

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
nltk.download('stopwords')
stop_words = nltk.corpus.stopwords.words("english")
embed = hub.load("https://tfhub.dev/google/Wiki-words-250/2")
clf = keras.models.load_model(os.path.join(THIS_FOLDER, 'model84.h5'))

print("NLTK imports DONE!\n\n")
print('===========================\n\n\n')

firebase = firebase.FirebaseApplication("https://sentimentapp-82bdf-default-rtdb.firebaseio.com/", None)
print('FIREBASE CONNECTED\n\n')
print('===========================\n\n\n')

'''
import pyrebase

config =  {"apiKey": "AIzaSyCeOF1Y3ZqOoDqjWb6JbTR1AhjaUA1Uc7k",
"authDomain": "sentimentapp-82bdf.firebase.com",
"databaseURL": "https://sentimentapp-82bdf.firebase.com/states/",  
"storageBucket": "sentimentapp-82bdf.appspot.com",  
"serviceAccount": str(os.path.join(THIS_FOLDER, 'service-account-key.json'))}
'''
def preprocess(text):
    text = re.sub("@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+", ' ', str(text).lower()).strip()
    tokens = []
    for token in text.split():
        if token not in stop_words:
          tokens.append(token)
    return " ".join(tokens)

def encoding(tweet):
    tweet = preprocess(tweet)
    tokens = tweet.split(" ")
    embedding = []
    for token in tokens:
        embedding.append(embed([token]).numpy()[0])

    embedding = np.array(embedding)
    padded_tweet = padding(embedding)
    return padded_tweet

def padding(encoded_tweet):
    count = 70 - encoded_tweet.shape[0]
    pad = np.zeros((1,250))
    for i in range(count):
        encoded_tweet = np.concatenate((pad,encoded_tweet), axis=0)
    return encoded_tweet

def processed_tweets(tweet):
    x = []
    enc = encoding(tweet)
    x.append(enc)
    x = np.array(x)
    return x

def predict(tweet, model):
    prediction = model.predict(processed_tweets(tweet))[0][0]
    result =  'P' if prediction > 0.5 else 'N'
    return result



statename = input("Enter state name\n")
tweet = input("Enter tweet\n")
prediction = predict(tweet, clf)

print("PREDICTED SENTIMENT IS: ",prediction)


data = {str(tweet) : str(prediction)}
result = firebase.patch('/states/'+statename+'/tweets/', data)
print(result)

'''
firebase_init = pyrebase.initialize_app(config)
db = firebase_init.database()
ref = db.child(statename).child("tweets").get()
for sentiment in ref.each():
	print(sentiment.val())
'''
