import re
import os
import glob
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

negreview_files = os.listdir("/Users/anishnuni/Documents/SA/aclImdb/train/neg/")

negreviews = []
os.chdir("/Users/anishnuni/Documents/SA/aclImdb/train/neg/")


for file in negreview_files:
    review = open(file,"r")
    neg=review.read()
    neg = neg.lower()
    whitelist = set('abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890')
    myStr = neg
    answer = ''.join(filter(whitelist.__contains__, myStr))
    negreviews.append(answer)


posreview_files = os.listdir("/Users/anishnuni/Documents/SA/aclImdb/train/pos/")
posreviews = []
os.chdir("/Users/anishnuni/Documents/SA/aclImdb/train/pos/")
for file in posreview_files:
    review = open(file,"r")
    pos=review.read()
    pos = pos.lower()
    whitelist = set('abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890')
    myStr = pos
    answer = ''.join(filter(whitelist.__contains__, myStr))
    posreviews.append(answer)

for i in range(0,len(negreview_files)):
    str = negreview_files[i]
    negreview_files[i] = str[(len(str) - 5)]
    str = posreview_files[i]
    posreview_files[i] = str[(len(str) - 5)]


postrainx = posreviews[0:9000]
negtrainx = negreviews[0:9000]
postrainy = posreview_files[0:9000]
negtrainy = negreview_files[0:9000]

postestx = posreviews[9000:]
negtestx = negreviews[9000:]
postesty = posreview_files[9000:]
negtesty = negreview_files[9000:]

trainx = postrainx + negtrainx

trainy = postrainy + negtrainy
Ytrain = np.ones(len(trainy))
Ytrain[9000:] = 0

Xtest = postestx + negtestx
Ytest = np.ones(len(Xtest))
Ytest[int((len(Xtest))/2):] = 0

ve = CountVectorizer(binary=False)
#can try non-binary after
ve.fit(trainx)
Xtrain= ve.transform(trainx)
Xtrain = Xtrain.toarray()

Xtest = ve.transform(Xtest)
Xtest = Xtest.toarray()

final_model = LogisticRegression(C=0.045)
final_model.fit(Xtrain, Ytrain)

print ("Final Accuracy: %s"
       % accuracy_score(Ytest, final_model.predict(Xtest)))
