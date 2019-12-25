from collections import Counter
import numpy as np
import nltk
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import collections, itertools
import nltk.classify.util, nltk.metrics
from nltk.classify import NaiveBayesClassifier
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist

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

print("the number of sentence types:", len(sent_set))
print("the number of patients:", N)
print("the max number of sents for a patient:", max(patients))
print("the min number of sents for a patient:", min(patients))
print("the mean of the sents per patient:", np.mean(patients))
print("the std of the sents per patient:", np.std(patients))

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

def doc_feats(doc):
    sent_types = set(doc)
    #vec = np.zeros(N)
    features = {}
    for sent_type in sent_types:
        features[sent_type] = True    # This is binary features
    return features

'''train set build-up'''
train_set = []
for idx in train_idx:
    adr_sents = adr_set[idx]
    train_set.append((doc_feats(adr_sents), 'true'))
    norm_sents = normal_set[idx]
    train_set.append((doc_feats(norm_sents), 'false'))

'''test set build-up'''
test_set = []
for idx in test_idx:
    adr_sents = adr_set[idx]
    test_set.append((doc_feats(adr_sents), 'true'))
    norm_sents = normal_set[idx]
    test_set.append((doc_feats(norm_sents), 'false'))

'''Validation set'''
val_set = test_set[:int(len(test_set)/2)]

# Parameter tuning
sent_fd = FreqDist()
label_sent_fd = ConditionalFreqDist()

adrs = open('../ADR_sents_type1.txt')
for line in adrs:
    for sent in line.strip().split():
        sent_fd[sent] += 1
        label_sent_fd['neg'][sent] += 1
adrs.close()

normals = open('../Normal_sents_type1.txt')
for line in normals:
    for sent in line.strip().split():
        sent_fd[sent] += 1
        label_sent_fd['pos'][sent] += 1
normals.close()

pos_sent_count = label_sent_fd['pos'].N()
neg_sent_count = label_sent_fd['neg'].N()
total_sent_count = pos_sent_count + neg_sent_count

sent_scores = {}

for sent, freq in sent_fd.items():
    pos_score = BigramAssocMeasures.chi_sq(label_sent_fd['pos'][sent],
        (freq, pos_sent_count), total_sent_count)
    neg_score = BigramAssocMeasures.chi_sq(label_sent_fd['neg'][sent],
        (freq, neg_sent_count), total_sent_count)
    sent_scores[sent] = pos_score + neg_score
cutoff = 9870
best = sorted(sent_scores.items(), key=lambda args: args[1], reverse=True)[:cutoff]   # best cut
best_sents = set([w for w, s in best])

def doc_feats(sents):
    return dict([(sent, True) for sent in sents if sent in best_sents])
'''train set build-up'''
train_set = []
for idx in train_idx:
    adr_sents = adr_set[idx]
    train_set.append((doc_feats(adr_sents), 'true'))
    norm_sents = normal_set[idx]
    train_set.append((doc_feats(norm_sents), 'false'))
'''test set build-up'''
test_set = []
for idx in test_idx:
    adr_sents = adr_set[idx]
    test_set.append((doc_feats(adr_sents), 'true'))
    norm_sents = normal_set[idx]
    test_set.append((doc_feats(norm_sents), 'false'))
'''Validation set'''
val_set = test_set[:int(len(test_set)/2)]

'''Experiment begin'''
classifier = nltk.NaiveBayesClassifier.train(train_set)
y_pred = []
y_test = [y[1] for y in val_set]
for t_set in val_set:
    y_pred.append(classifier.classify(t_set[0]))

print('=======================')
print("Record: # best:", cutoff)
print('accuracy', accuracy_score(y_test, y_pred))
'''
=======================
Record: # best: 9870
accuracy 0.572115384615
'''

# Fine tuning using grid-search
def doc_feats(sents, best_sents):
    return dict([(sent, True) for sent in sents if sent in best_sents])

def grid_search(start, stop, step):
    results = []
    for cutoff in range(start, stop, step):
        best = sorted(sent_scores.items(), key=lambda args: args[1], reverse=True)[:cutoff]   # best cut
        best_sents = set([w for w, s in best])
        '''train set build-up'''
        train_set = []
        for idx in train_idx:
            adr_sents = adr_set[idx]
            train_set.append((doc_feats(adr_sents, best_sents), 'true'))
            norm_sents = normal_set[idx]
            train_set.append((doc_feats(norm_sents, best_sents), 'false'))
        '''test set build-up'''
        test_set = []
        for idx in test_idx:
            adr_sents = adr_set[idx]
            test_set.append((doc_feats(adr_sents, best_sents), 'true'))
            norm_sents = normal_set[idx]
            test_set.append((doc_feats(norm_sents, best_sents), 'false'))
        '''Validation set'''
        val_set = test_set[:int(len(test_set)/2)]
        '''Experiment begin'''
        classifier = nltk.NaiveBayesClassifier.train(train_set)
        y_pred = []
        y_test = [y[1] for y in val_set]
        for t_set in val_set:
            y_pred.append(classifier.classify(t_set[0]))
        accr = accuracy_score(y_test, y_pred)
        print(cutoff, accr)
        results.append((cutoff, accr))
    return results

res = grid_search(2100, 4000, 100)
print("Best param:", sorted([r for r in res], key=lambda x: x[1], reverse=True)[0])   # Best param: (3700, 0.61899038461538458)


# Test on test-set
cutoff = 3700
best = sorted(sent_scores.items(), key=lambda args: args[1], reverse=True)[:cutoff]   # best cut
best_sents = set([w for w, s in best])

def doc_feats(sents):
    return dict([(sent, True) for sent in sents if sent in best_sents])

train_set = []
for idx in train_idx:
    adr_sents = adr_set[idx]
    train_set.append((doc_feats(adr_sents), 'true'))
    norm_sents = normal_set[idx]
    train_set.append((doc_feats(norm_sents), 'false'))
test_set = []
for idx in test_idx:
    adr_sents = adr_set[idx]
    test_set.append((doc_feats(adr_sents), 'true'))
    norm_sents = normal_set[idx]
    test_set.append((doc_feats(norm_sents), 'false'))

'''Test set'''
real_test_set = test_set[int(len(test_set)/2):]

'''Experiment begin'''
classifier = nltk.NaiveBayesClassifier.train(train_set)
y_pred = []
y_test = [y[1] for y in real_test_set]
for t_set in real_test_set:
    y_pred.append(classifier.classify(t_set[0]))
print('=======================')
print("Record: # best:", cutoff)
print('accuracy', accuracy_score(y_test, y_pred))
'''
=======================
Record: # best: 3700
accuracy 0.641826923077
'''
