# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 19:56:45 2018

@author: Sameer
"""
import csv
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
import pickle
import time
import re
#from nltk.tokenize import word_tokenize

t0=time.time()

def load_data():
    
    file="../data/lite.csv"
    data=[]
   
    with open(file,encoding="utf8") as f:
        reader = list(csv.reader(f))
        reader.pop(0)
        for row in reader:
            
           
            tweet=(row[3]).lower()
            #Remove https, @users 
            tweet = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '',tweet) 
            tweet = re.sub('@[^\s]+','',tweet)
            tweet= re.sub(r'([^\s\w]|_)+',' ', tweet)
            tweet = re.sub('[\s]+', ' ', tweet)
            tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
            try:
                
                data.append((tweet.split('\'"'),row[0]))
            except:
                pass
            
    return data

def replaceTwoOrMore(s):
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
    return pattern.sub(r"\1\1", s)


def get_words_in_tweets(tweets):
    all_words = []
    wordnet_lemmatizer = nltk.stem.WordNetLemmatizer()
    
    for (tweet, sentiment) in tweets:
#        print(tweet)
        words = tweet[0].split()
        for w in words:
            w = replaceTwoOrMore(w)
            w = w.strip('\'"?,.')
            val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*$", w)
#            ignore if it is a stop word
            stopWords = set(stopwords.words('english'))
            if(w in stopWords or val is None):
                continue
            else:
                w=wordnet_lemmatizer.lemmatize(w)
                all_words.append(w)

    return all_words

def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    word_features = wordlist.keys()
    return word_features

def extract_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    return features




tweets=load_data()

word_features = get_word_features(get_words_in_tweets(tweets))
#print(list(word_features))
print("Wor Feature  in %0.3fs." % (time.time() - t0))

training_set = nltk.classify.apply_features(extract_features, tweets)


#pickle.dump(list(word_features), open("../data/word_featuress.pickle", "wb"))
classifier = nltk.NaiveBayesClassifier.train(training_set)
print("trainned  in %0.3fs." % (time.time() - t0))

tweet = 'this day sucks '
print(classifier.classify(extract_features(tweet.split())) )
print("done  in %0.3fs." % (time.time() - t0))

#print(classifier)
#for ls in tweets:
#    print(ls)
#    print()