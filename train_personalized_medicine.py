import os
import pandas as pd
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
import logging
tokenizer=nltk.data.load('tokenizers/punkt/english.pickle')
from gensim.models import word2vec
import time
from sklearn.svm import SVC#,LinearSVC,NuSVC
import numpy as np

datadir=os.getcwd()+'/data/'

#parameters for word2vec
num_features=300 #Word vector dimensionality
min_word_count=40 #minium word count
num_workers=4 #number of threads to run in parallel
context=10 #context window size
downsampling=1e-3 #downsample setting for frequent words


def text_to_wordlist(textblock,remove_stopwords=False):
	#Function to convert a document to a sequence of words,
	#optionally removing stop words. returns a list of words
	#
	#1.Remove HTML
	textblock_text=BeautifulSoup(textblock).get_text()
	#
	#2.Remove non-letters
	textblock_text=re.sub("[^a-zA-z]"," ",textblock_text)
	#
	#3.Convert words to lowercase and split them
	words=textblock_text.lower().split()
	#
	#4.Optionally remove stop words (fale by default)
	if remove_stopwords:
		stops=set(stopwords.words("english"))
		words=[w for w in words if not w in stops]
		#
		#5.Return list of words
	return(words)

def text_to_sentences(textblock,tokenizer,remove_stopwords=False):
	#Function to split a review into parsed sentences. Returns a 
	#list of sentences, where each sentence is a list of words
	#
	#1.use the NLTK tokenizer to split the paragraph into sentences
	textblock=textblock.strip()
	raw_sentences=tokenizer.tokenize(textblock.decode('utf-8'))
	#2.loop over each sentence
	sentences=[]
	for raw_sentence in raw_sentences:
		#if a sentence is empty, skip it
		if len(raw_sentence)>0:
			#otherwise, call text_to_wordlist to get a list of words
			sentences.append(text_to_wordlist(raw_sentence,remove_stopwords))
	#
	#REturn the list of sentences (each sentence is a list of words, so his returns a list of lists
	return sentences

def textblock_to_vector(textblock,model_var,num_features):
	wordvector=np.zeros((num_features,),dtype="float32")
	index2word_set=set(model_var.wv.index2word)
	nwords=0
	for word in textblock:
		if word in index2word_set:
			nwords=nwords+1
			wordvector=np.add(wordvector,model_var[word])
	#
	#Divide the result by the number of words to get the average
	wordvector=np.divide(wordvector,nwords)
	return wordvector


def normvector(textblocks,model_var,num_features):
	#given a set of reviews (each one a lit of words), calculate
	#the average feature vector for each one and return a 2D numpy array
	#
	#Preallocate a 2D numpy array for speed
	normvecs=np.zeros((len(textblocks),num_features),dtype="float32")
	#
	#Loop through the textblocks
	#Initialize a counter
	counter=0
	for textblock in textblocks:
		#
		#Print a status message every 1000th textblock
		if counter%1000==0:
			print "Textblock %d of %d" % (counter,len(textblocks))
		#
		#Call the function (defined above) that makes average feature vectors
		normvecs[counter]=textblock_to_vector(textblock,model_var,num_features)
		#Incrememnt the counter
		counter=counter+1
	return normvecs

################################################################################
######################Load, clean and parse training data#######################
################################################################################


print(':::::::Loading training set:::::::')
trainvar=open(datadir+'training_variants','rb').readlines()
traintext=open(datadir+'training_text','rb').readlines()
del(traintext[0])#delete the first entry because it's just labels
#Here we strip off the ID of each entry that is included in the text
for i in range(0,len(traintext)):#start from 1 because first line is the keys for the entries
	numberlength=len(str(i))#how long a number is when in string form
	traintext[i]=traintext[i][2+numberlength:]#use 2+numberlength because a '||' is included with each ID number


train_sentences=[]#initialize an empty list of sentences
print "parsing sentences from training set"

for text in traintext:
	train_sentences+=text_to_sentences(text,tokenizer,remove_stopwords=True)#If you are appending a list of lists to another list of lists, "append" will only append the first list; you need to use "+=" in order to join all of the lists at once.

trainvarKeys=trainvar[0].split(',')
trainvarKeys[-1]=trainvarKeys[-1].strip()#strip the /n character from the last entry
trainvardict=dict()
for i in trainvarKeys:
	trainvardict[i]=list()
keyorder=['ID','Gene','Variation','Class']
for i in trainvar[1:]:#start from index 1, because index 0 is just labels
	splittrainvar=i.split(',')
	splittrainvar[-1]=splittrainvar[-1].strip()
	for j in range(0,len(keyorder)):
		if j==0 or j==3:
			trainvardict[keyorder[j]].append(int(splittrainvar[j]))
		else:
			trainvardict[keyorder[j]].append(splittrainvar[j])

################################################################################
################################################################################
################################################################################

################################################################################
######################Load, clean and parse test data###########################
################################################################################
'''
print(':::::::Loading testing set:::::::')
testvar=open(datadir+'test_variants','rb').readlines()
testtext=open(datadir+'test_text','rb').readlines()
del(testtext[0])
#Here we strip off the ID of each entry that is included in the text
for i in range(0,len(testtext)):
	numberlength=len(str(i))#how long a number is when in string form
	testtext[i]=testtext[i][2+numberlength:]#use 2+numberlength because a '||' is included with each ID number

test_sentences=[]#initialize an empty list of sentences
print "parsing sentences from test set"

for text in testtext:
	test_sentences+=text_to_sentences(text,tokenizer,remove_stopwords=True)#If you are appending a list of lists to another list of lists, "append" will only append the first list; you need to use "+=" in order to join all of the lists at once.

testvarKeys=testvar[0].split(',')
testvarKeys[-1]=testvarKeys[-1].strip()#strip the /n character from the last entry
testvardict=dict()
for i in testvarKeys:
	testvardict[i]=list()
keyorder=['ID','Gene','Variation','Class']
for i in testvar[1:]:#start from index 1, because index 0 is just labels
	splittestvar=i.split(',')
	splittestvar[-1]=splittestvar[-1].strip()
	for j in range(0,len(keyorder)):
		if j==0 or j==3:
			testvardict[keyorder[j]].append(int(splittestvar[j]))
		else:
			testvardict[keyorder[j]].append(splittestvar[j])

'''
################################################################################
################################################################################
################################################################################

print '###################################start word2vec###################################'
startword2vec=time.time()
train_model=word2vec.Word2Vec(train_sentences,workers=num_workers,size=num_features,min_count=min_word_count,window=context,sample=downsampling)
#if you don't plan to train the model any further, calling init_sims will make the model much more memory efficient
train_model.init_sims(replace=True)
finishword2vec=time.time()
print 'total time for word2vec: '+str(finishword2vec-startword2vec)+' seconds'
print '###################################finished word2vec###################################'

starttextblockvecs=time.time()
print 'start textblockvecs'
textblockvecs=normvector(traintext,train_model,num_features)
endtextblockvecs=time.time()
print 'time to complete textblockvecs= '+str(endtextblockvecs-starttextblockvecs)

print 'shape of textblockvecs= '+str(textblockvecs.shape)


#####################Train SVM Classifier#####################
startsvm=time.time()
print ":::::::::::::::::Training SVM:::::::::::::::::"
svm_classifier=SVC()
svm_classifier.fit(textblockvecs, trainvardict['Class'])
endsvm=time.time()
print "Total time to train SVM was: "+str(endsvm-startsvm)+" seconds"
#svm_classifier.fit(X, y)
#print(svm_classifier.predict([[-0.8, -1]]))

