from collections import Counter
import numpy as np

N = 0
sent_set = Counter()
adrs = open('../ADR_sents_type1.txt')
patients = []
for line in adrs:
    N += 1
    sents = line.strip().split()
    patients.append(len(sents))
    for sent in sents:
        sent_set[sent] += 1
adrs.close()
normals = open('../Normal_sents_type1.txt')
for line in normals:
    N += 1
    sents = line.strip().split()
    patients.append(len(sents))
    for sent in sents:
        sent_set[sent] += 1
normals.close()

# 1.1 Train/Test Data Split
import numpy as np
np.random.seed(42)

cut = int(4158*0.8)
shuffled = np.random.permutation(range(4158))
train_idx = shuffled[:cut]
test_idx = shuffled[cut:]
print(len(train_idx), len(test_idx))

adr_set = {}
normal_set = {}

adrs = open('../ADR_sents_type1.txt')

N = 0
for line in adrs:
    adr_set[N] = line.strip().split()
    N += 1
adrs.close()
normals = open('../Normal_sents_type1.txt')
N = 0
for line in normals:
    normal_set[N] = line.strip().split()
    N += 1
normals.close()

print(len(adr_set), len(normal_set))

def doc_feats(doc, sent_dic):
    #vec = np.zeros(N)
    temp = {k:0 for k in sent_dic}
    for token in doc:
        temp[token] += 1
    features = [temp[k] for k in sent_dic]
    return features

frozen_ks = tuple(sent_set.keys())

'''train set build-up'''
train_set = []
for idx in train_idx:
    adr_sents = adr_set[idx]
    train_set.append((doc_feats(adr_sents, frozen_ks), 'true'))
    norm_sents = normal_set[idx]
    train_set.append((doc_feats(norm_sents, frozen_ks), 'false'))

'''test set build-up'''
test_set = []
for idx in test_idx:
    adr_sents = adr_set[idx]
    test_set.append((doc_feats(adr_sents, frozen_ks), 'true'))
    norm_sents = normal_set[idx]
    test_set.append((doc_feats(norm_sents, frozen_ks), 'false'))

from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# TF-IDF Vectorizing
def doc_feats(doc, sent_dic):
    #vec = np.zeros(N)
    temp = {k:0 for k in sent_dic}
    for token in doc:
        temp[token] += 1
    features = ' '.join([str(temp[k]) for k in sent_dic])
    return features

'''train set build-up'''
train_set = []
for idx in train_idx:
    adr_sents = adr_set[idx]
    train_set.append((' '.join(adr_sents), 1))
    norm_sents = normal_set[idx]
    train_set.append((' '.join(norm_sents), 0))
'''test set build-up'''
test_set = []
for idx in test_idx:
    adr_sents = adr_set[idx]
    test_set.append((' '.join(adr_sents), 1))
    norm_sents = normal_set[idx]
    test_set.append((' '.join(norm_sents), 0))

from sklearn.feature_extraction.text import TfidfVectorizer

train_x, train_y = [t[0] for t in train_set], [t[1] for t in train_set]
val_set = test_set[:int(len(test_set)/2)]
val_x, val_y = [t[0] for t in val_set], [t[1] for t in val_set]

# SVM Linear Fine Tuning
def get_accuracy(params, train_x, train_y, val_x, val_y):
    vectorizer = TfidfVectorizer(min_df=params['min_df'],
                                max_df=params['max_df'],
                                 sublinear_tf=True,
                                 use_idf=True)
    train_vectors = vectorizer.fit_transform(train_x)
    val_vectors = vectorizer.transform(val_x)
    classifier_linear = svm.SVC(kernel='linear')
    classifier_linear.fit(train_vectors, train_y)
    prediction_linear = classifier_linear.predict(val_vectors)
    return accuracy_score(val_y, prediction_linear)

params = {
    'min_df':5,
    'max_df':0.5
}

# SVM Linear Test on Real Test Set
real_test_set = test_set[int(len(test_set)/2):]
test_x, test_y = [t[0] for t in val_set], [t[1] for t in real_test_set]
vectorizer = TfidfVectorizer(min_df=5,
                             max_df=0.5,
                             sublinear_tf=True,
                             use_idf=True)
train_vectors = vectorizer.fit_transform(train_x)
test_vectors = vectorizer.transform(test_x)

classifier_linear = svm.SVC(kernel='linear')
classifier_linear.fit(train_vectors, train_y)
prediction_linear = classifier_linear.predict(test_vectors)

print('accuracy', accuracy_score(test_y, prediction_linear)) # accuracy 0.627403846154

# SVM RBF TEST result

classifier_rbf = svm.SVC(kernel='rbf', random_state=0, gamma=0.1, C=4.0)
classifier_rbf.fit(train_vectors, train_y)
prediction_rbf = classifier_rbf.predict(test_vectors)
print('accuracy', accuracy_score(test_y, prediction_rbf)) # accuracy 0.626201923077
