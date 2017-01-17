#fetch data
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from nltk import word_tokenize          
from nltk.stem.porter import PorterStemmer
from sklearn.datasets import fetch_20newsgroups
twenty_all = fetch_20newsgroups(subset='all', shuffle=True, random_state=42)
print len(twenty_all.data)
print twenty_all.target[0]

#count term

twenty_all_X = twenty_all.data
count_vect = CountVectorizer(stop_words='english')
X_counts = count_vect.fit_transform(twenty_all_X)
X = X_counts.toarray()

vocab_new={}
vocab = count_vect.vocabulary_
stemmer = PorterStemmer()
for i in vocab:
    word = stemmer.stem(i)
    if word in vocab_new:
        vocab_new[word].append(vocab[i])
    else:
        vocab_new[word]=[vocab[i]]
term_num = len(vocab_new)
print term_num

doc_num = X.shape[0]
X_new = np.array ([ [ 0 for i in range(term_num)] for j in range(doc_num) ])
keys = vocab_new.keys()
for i in range(doc_num):
    # if i%100==0:
    #     print i
    for j in range(len(keys)):
        idxs = vocab_new[keys[j]]
        for idx in idxs:
            X_new[i][j]+=X[i][idx]
print X_new.shape
X_counts_arr = X_new

#class term
term_num = len(X_counts_arr[0])
class_counts = [[0 for i in range(term_num)] for i in range(20)]
for i in range(len(X_counts_arr)):
	class_num = twenty_all.target[i]
    class_counts[class_num] = [x + y for x, y in zip(class_counts[class_num], X_counts_arr[i])]

#tficf
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
#class_counts = np.load('/Users/Yingze/Desktop/class_counts.npy')
tfidf_transformer = TfidfTransformer()
class_tficf = tfidf_transformer.fit_transform(class_counts)
class_tficf_arr = class_tficf.toarray()
#sort
keys=vocab_new.keys()
map={}
for i in range(len(keys)):
	map[keys[i]] = class_tficf_arr[:,i]
#[3, 4, 6, 15]
for i in [3,4,6,15]:
	res = sorted(map.items(), key =lambda x:x[1][i])
    res.reverse()
	for j in range(10):
		print res[j][0],
	print "\n"

#result:
# term number = 149250
# scsi edu drive line com ide subject use organ card 

# edu line mac subject organ use appl quadra post problem 

# edu 00 line subject sale organ post univers com new 

# god christian edu church subject jesu homosexu peopl line sin 