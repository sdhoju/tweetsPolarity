# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 18:55:28 2018

@author: CREX
"""

import nltk
import time
import re
from nltk.corpus import stopwords
import csv
import pickle
import pandas as pd

#from nltk.tokenize import word_tokenize

t0=time.time()


'''Load all the tweets from Csv file'''
def load_data(filename):
    file="../data/"+filename
    data=[]

    f = pd.read_csv(file)
    tweets=list(f.text)
    for i in range(len(tweets)):
        words = list_from_tweet(tweets[i])
        data.append(words)
    return data

'''Load all the tweets with it's Polarity from Csv file'''
def load_data_polarity(filename):
    file="../data/"+filename
    data=[]
    f = pd.read_csv(file)
    tweets=list(f.text)
    polarity=list(f.polarity)
    for i in range(len(tweets)):
        words =   list_from_tweet(tweets[i])
        if polarity[i]==5:
            p="Positive"
        elif polarity[i]==1:
            p="Negative"
        data.append((words,p))
    return data

def load_polarity(filename):
    file="../data/"+filename
    f = pd.read_csv(file)
    polarity=list(f.polarity)
    return polarity

'''Convert a tweet into bag of words'''
def list_from_tweet(tweet):
    wordnet_lemmatizer = nltk.stem.WordNetLemmatizer()
    
    words=[]
    tweet=tweet.lower()
    tweet = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '',tweet) 
    tweet = re.sub('@[^\s]+','',tweet)
    tweet= re.sub(r'([^\s\w]|_)+',' ', tweet)
    tweet = re.sub('[\s]+', ' ', tweet)
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    try:
        words=[]
        tweet_split=tweet.split()
        for w in tweet_split:
            w = replaceTwoOrMore(w)
            w = w.strip('\'"?,.')
            val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*$", w)
#            ignore if it is a stop word or les than len 1
            stopWords = set(stopwords.words('english'))
            if(w in stopWords or val is None or len(w)<=1):
#            if(val is None or len(w)<=1):
                continue
            else:
                w=wordnet_lemmatizer.lemmatize(w)
                words.append(w)
    except:
        pass
    return words

'''replace duplicates at end like hungryyyyy to hungry. '''
def replaceTwoOrMore(s):
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
    return pattern.sub(r"\1\1", s)

'''Get all the words in list of tuple of list of tweets and sentiment value as postive and negative '''
def get_words_in_tweets(tweets):
    all_positive_words = []
    all_negative_words = []
    for (words, sentiment) in tweets:
        if sentiment=="Positive":
            all_positive_words.extend(words)
        elif sentiment=="Negative":
            all_negative_words.extend(words)
    return [all_positive_words,all_negative_words]


'''Get the Word distrubution from the list of words'''
def get_word_features(wordlist):
    word_features=[]
    wordlist = nltk.FreqDist(wordlist)
#    word_features = wordlist.keys()
    for i in range(len(wordlist.values())):
        if list(wordlist.values())[i]>5:
            word_features.append((list(wordlist.keys())[i],list(wordlist.values())[i]))
    return word_features    



    
'''Get the Prior probality of Negative and Positive tweets'''
def prior_probability(tweets,feature):
    count=0
    total=len(tweets)
    for (tweet, sentiment) in tweets:
        if sentiment==feature:
            count+=1
    return count/total

'''Get the condtional probality of the words in tweets'''
def conditional_probabilities(word_features,len_tweet,total,polarity):
    if polarity==1:
        filename='../data/negative-words.txt'
    else:
        filename='../data/positive-words.txt'
    f = open(filename, 'r')
    list_of_words=[]
    for w in f:
        list_of_words.append(w.strip("\n"))
    
    dict_of_con_prob={}
    for t in word_features:
        
        p_w=((t[1]+1)/(len_tweet+total))
        if p_w in list_of_words:
            p_w*2
        dict_of_con_prob[t[0]]=p_w
        print(t[0], end=":  ")
        print("(%d+1 )/(%d + %d)"%(t[1], len_tweet, total), end= "= ")
        print(p_w)
    print()
    return dict_of_con_prob  

'''Get the prediction value for a tweet'''
def predict_polarity(tweet,trained):
    words=list_from_tweet(tweet)
    positive_dict=trained[0]
    negative_dict=trained[1]
    prior_positive=trained[2]
    prior_negative=trained[3]
    len_pos=trained[4]
    len_neg=trained[5]
    total=trained[6]
    
#    print()
#    print(len_pos)
#    print(len_neg)
    
    pos=1
    neg=1
    for i in range(len(words)):
        try:
            if words[i] in positive_dict:
#                print(positive_dict[words[i]])
                pos*=((positive_dict[words[i]]+1)/(len_pos+total))
            if words[0] in positive_dict:
#                print(negative_dict[words[i]])
                neg*=((negative_dict[words[i]]+1)/(len_neg+total))
        except Exception as e:
            pass
    
    
    prob_positive_tweet=pos*prior_positive
    prob_negative_tweet=neg*prior_negative
    out=1
    if prob_negative_tweet/prob_positive_tweet < 1:
#        print("Ratio P(N|t)/P(P|t): {}".format(prob_negative_tweet/prob_positive_tweet))
#        print("Positive")
        out=5
    elif prob_positive_tweet/prob_negative_tweet <1:
#        print("Ratio P(P|t)/P(N|t): {}".format(prob_positive_tweet/prob_negative_tweet))
#        print("Negative")
        out=1
    else:
##        print("Neutral")
        out=3
#    print(out)
    return out
        
             
def accuracy(predicted,given):
    count=0
    total=len(given)
    if(len(predicted)==len(given)):
        for i in range(total):
            if predicted[i]==given[i]:
                count+=1
    accuracy=count/total
    print("Accuracy of the model is ".format(accuracy))
    return accuracy

'''Save everything that is required for calulating'''
def train_model(tweets):
    all_words=(get_words_in_tweets(tweets))
    pos_features = get_word_features(all_words[0])
    neg_features = get_word_features(all_words[1])
    
    len_pos=0
    for t in pos_features:
        len_pos+=t[1]
    
    len_neg=0
    for t in neg_features:
        len_neg+=t[1]
    
    total=(len(set(all_words[0]+all_words[1])))
    
    prior_positive=prior_probability(tweets,"Positive")
    prior_negative=1-prior_positive
    
    print(prior_positive)
    print(prior_negative)
    
    dict_con_positive_prob =conditional_probabilities(pos_features,len_pos,total,5)
    dict_con_negative_prob =conditional_probabilities(pos_features,len_neg,total,1)
    
    trained=[dict_con_positive_prob,dict_con_negative_prob,prior_positive,prior_negative,len_pos,len_neg,total]
        
    #Pickle was used to save the trainned model  

    pickle.dump(trained, open("../data/trainned.pickle", "wb"))
    return trained
    
"""Put everything together to implement Naive Bayes """
def Naive_Bayes(tweets):
    '''uncomment this to train the model'''
#    trained = train_model(tweets)
    trained=pickle.load(open("../data/trainned.pickle", "rb"))
    print()
    
#    print("Successfully get all the trainned data.\nList of Dictionary of conditional Probability for all words from trainning data, Prior Probabities ")

    #Example of a tweet getting predicted
    tweet="wow 12 am and i don't want to sleep :S the best day of my life "
    
    print("1 for 'Negative' 5 for 'Positive' 3 for 'Neutral'.")
#    print()
    print("Tweet: %s"%tweet)
    print("Polarity: ",end=": ")
    print(predict_polarity(tweet,trained))
    
    
    
    '''Predicting the output of the test files'''
    #for 
    #file="../data/evaluation_csci581.csv"
    file="../data/test.csv"
    f = pd.read_csv(file)
    tweets=list(f.text)
    output=[]
    
    for tweet in tweets:
        out=predict_polarity(tweet,trained)
        output.append(out)
    print(output)
    given_polarity=load_polarity("test.csv")
    
#    myfile = open('../data/output_witout_neutral.csv','w')
#    wr = csv.writer(myfile, delimiter=',')
#    wr.writerow(output)
#    myfile.close()
#    
    print(accuracy(output,given_polarity) )




    

def main():
    '''Remove the comment below and give the file name to generate bag of word and sentiment '''
#    tweets=load_data_polarity("lite.csv")   
#    pickle.dump(tweets, open("../data/trainning_set.pickle", "wb"))
#    print("Read all the Words")
    ###   
    
    #Pickle was used to save the data bag of words to save loading it multiple times 
    tweets = pickle.load(open("../data/trainning_set.pickle", "rb"))
    print("Successfully get the bag of words with Polarity. Example:")
    print(tweets[0])
    
    Naive_Bayes(tweets)
    #print(tweets)
    
main()
print("Done in  %0.3fs." % (time.time() - t0))













