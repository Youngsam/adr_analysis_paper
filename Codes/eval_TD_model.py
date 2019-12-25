import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, accuracy_score


i2s = pickle.load(open('int_to_posed_words.pkl', 'rb'))
v2i = pickle.load(open('vocab2int.pkl', 'rb'))

def pad_features(reviews_ints, seq_length):
    ''' Return features of review_ints, where each review is padded with 0's
        or truncated to the input seq_length.
    '''

    # getting the correct rows x cols shape
    features = np.zeros((len(reviews_ints), seq_length), dtype=int)

    # for each review, I grab that review and
    for i, row in enumerate(reviews_ints):
        features[i, -len(row):] = np.array(row)[:seq_length]

    return features

seq_length = 200

class CNN_Text(nn.Module):
    def __init__(self):
        super(CNN_Text, self).__init__()
        V = len(v2i)    # vocab size
        D = 200  # word embedding dim
        C = 7 # label class size
        Ci = 1
        Co = 100
        Ks = [3, 4, 5] # kernel sizes

        self.embed = nn.Embedding(V, D)
        # self.convs1 = [nn.Conv2d(Ci, Co, (K, D)) for K in Ks]
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(len(Ks)*Co, C)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        x = self.embed(x)  # (N, W, D)
        x = x.unsqueeze(1)  # (N, Ci, W, D)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
        x = torch.cat(x, 1)
        x = self.dropout(x)  # (N, len(Ks)*Co)
        logit = self.fc1(x)  # (N, C)
        return logit

cnn = CNN_Text()
cnn.cuda()

state_dict = torch.load('best_steps_20.pt')
cnn.load_state_dict(state_dict)

def sent2features(sent_idx, model):
    '''input: string index for a statement, cnn model
       output: integer index for a state for feature value'''
    ws = i2s[int(sent_idx)]
    sent_ints = [v2i[w.lower()] for w in ws]
    x = pad_features([sent_ints], seq_length)
    x = torch.tensor(x)
    x = x.type(torch.cuda.LongTensor)
    model.eval()
    output = model(x)
    _, predicted = torch.max(output, 1)
    #return label_feild.vocab.itos[predicted.data[0][0]+1]
    return np.array(predicted)[0]

adr_set = [(line.strip(),'adr') for line in open('ADR_data_short.csv').readlines()]
nor_set = [(line.strip(),'nor') for line in open('NOR_data_short.csv').readlines()]
total_set = adr_set + nor_set

rand_idx = np.random.randint(0, len(total_set), len(total_set))
train_idx = rand_idx[:int(len(total_set)*0.8)]
test_idx = rand_idx[int(len(total_set)*0.8):]
val_idx = test_idx[:int(len(test_idx)*0.5)]
test_idx = test_idx[int(len(test_idx)*0.5):]

from collections import Counter
adr_freq = {'adr':0, '0':[], '1':[], '2':[], '3':[], '4':[], '5':[], '6':[]}
nor_freq = {'nor':0, '0':[], '1':[], '2':[], '3':[], '4':[], '5':[], '6':[]}
for idx in train_idx:
    res = []
    label = total_set[idx][1]
    if label == 'adr':
        adr_freq['adr'] += 1
    elif label == 'nor':
        nor_freq['nor'] += 1
    for sent in total_set[idx][0].split():
        words = i2s[int(sent)]
        feats = sent2features(sent, cnn)
        res.append(str(feats))
    temp_dic = Counter(res)
    for k in temp_dic:
        if label == 'adr':
            adr_freq[k].append(temp_dic[k]/len(res))
        elif label == 'nor':
            nor_freq[k].append(temp_dic[k]/len(res))

# TD related functions
def get_state_value(state, states, model):
    idx = sent2features(state, model)
    if idx in (1, 2, 4, 5, 6):
        baseline = 0
    else:
        baseline = 0
    val = states[idx] + baseline
    if idx == None:
        return 0, val
    return idx, val

def get_reward(n_state, n_idx):
    if n_state == 'adr':
        return 1
    elif n_state == 'nor':
        return -1
    elif n_idx in (2, 3, 4, 5, 6):
        return 1
    else:
        return 0

np.seterr(invalid='raise')

state_size = 7

states = np.zeros(state_size)
E = np.zeros(state_size)
gamma = 0.1
alpha = 0.1
lam = 0.3
idx_tracker = []

episode_res = []
eps_cnt = 0
for tr_idx in train_idx:
    string, tag = total_set[tr_idx]
    episode = [int(v) for v in string.split()] + [tag]
    if len(episode)<=2: continue
    max_val = 0
    temp = ''
    for i in range(len(episode)):
        if i == 0:
            state = episode[i]
        if i+1 == len(episode)-1:
            break
        n_state = episode[i+1]
        idx, val = get_state_value(state, states, cnn)
        temp += str(idx) + ' '
        n_idx, n_val = get_state_value(n_state, states, cnn)
        reward = get_reward(n_state, n_idx)
        delta = reward + gamma * n_val - val
        E[idx] = (1-alpha)*E[idx] + 1
        one_val = val
        for j in range(len(states)):
            states[j] = states[j] + alpha*delta*E[j]
            E[j] = gamma*lam*E[j]
        if one_val > max_val:
            max_val = one_val
        state = n_state
    tag = 1 if tag=='adr' else 0
    episode_res.append((tag, max_val))
    eps_cnt += 1
    idx_tracker.append(temp)
    if (eps_cnt)%500 == 0:
        print("{0:.2f}-th episode is finished!".format(eps_cnt/len(train_idx)))
        pass

test_res = []

for v_idx in test_idx:
    string, tag = total_set[v_idx]
    tag = 1 if tag == 'adr' else 0
    episode = [int(v) for v in string.split()]
    vals = []
    max_val = 0
    if len(episode)<2: continue
    for i in range(len(episode)):
        state = episode[i]
        idx, val = get_state_value(state, states, cnn)
        if idx in (1, 2, 3, 4, 5, 6):
            vals.append(val)
        else:
            vals.append(0)
    test_res.append((np.mean(vals), tag))

x_val = np.array([c[0] for c in episode_res])
y_val = np.array([c[1] for c in episode_res])
x_val = x_val.reshape((-1, 1))

x_test = np.array([c[0] for c in test_res])
y_test = np.array([c[1] for c in test_res])
x_test = x_test.reshape((-1, 1))

regr = linear_model.LogisticRegression(solver='lbfgs', C=1e5)

# Train the model using the training sets
regr.fit(x_val, y_val)

predicts = regr.predict(x_test)

accuracy_score(y_test, predicts) # 0.6310096153846154
