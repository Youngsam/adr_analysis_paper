import numpy as np
import tensorflow as tf
import random
import pickle

adr_data_train = open('data_prep/new_ADR_data_short_train.txt').readlines()
nor_data_train = open('data_prep/new_NOR_data_short_train_rand.txt').readlines()
adr_data_test = open('data_prep/new_ADR_data_short_test.txt').readlines()
nor_data_test = open('data_prep/new_NOR_data_short_test_rand.txt').readlines()

train_data = [(s, 'pos') for s in adr_data_train] + [(s, 'neg') for s in nor_data_train]
test_data = [(s, 'pos') for s in adr_data_test] + [(s, 'neg') for s in nor_data_test]
np.random.shuffle(train_data)
np.random.shuffle(test_data)
temp_data = train_data[:int(len(train_data)*0.9)]
val_data = train_data[int(len(train_data)*0.9):]
train_data = temp_data

seq_len = 200
tag_to_ix = {'pos':1, 'neg':0}

train_x = np.zeros((len(train_data), seq_len), dtype=int)
train_y = np.zeros(len(train_data))

for i, row in enumerate(train_data):
    if len(row[0].strip()) < 1: continue
    vec = [int(v)+1 for v in row[0].strip().split()]
    train_x[i, -len(vec):] = np.array(vec)[:seq_len]
    train_y[i] = tag_to_ix[row[1]]

val_x = np.zeros((len(val_data), seq_len), dtype=int)
val_y = np.zeros(len(val_data))

for i, row in enumerate(val_data):
    if len(row[0].strip()) < 1: continue
    vec = [int(v)+1 for v in row[0].strip().split()]
    val_x[i, -len(vec):] = np.array(vec)[:seq_len]
    val_y[i] = tag_to_ix[row[1]]

test_x = np.zeros((len(test_data), seq_len), dtype=int)
test_y = np.zeros(len(test_data))

for i, row in enumerate(test_data):
    if len(row[0].strip()) < 1: continue
    vec = [int(v)+1 for v in row[0].strip().split()]
    test_x[i, -len(vec):] = np.array(vec)[:seq_len]
    test_y[i] = tag_to_ix[row[1]]

print("\t\t\tFeature Shapes:")
print("Train set: \t\t{}".format(train_x.shape),
      "\nValidation set: \t{}".format(val_x.shape),
      "\nTest set: \t\t{}".format(test_x.shape))
print("label set: \t\t{}".format(train_y.shape),
      "\nValidation label set: \t{}".format(val_y.shape),
      "\nTest label set: \t\t{}".format(test_y.shape))

lstm_size = 256
lstm_layers = 2
batch_size = 64
learning_rate = 0.01

n_words = 837293  # Add 1 for 0 added to vocab

# Create the graph object
tf.reset_default_graph()
with tf.name_scope('inputs'):
    inputs_ = tf.placeholder(tf.int32, [None, None], name="inputs")
    labels_ = tf.placeholder(tf.int32, [None, None], name="labels")
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")

# Using random embedding
embed_size = 200

with tf.name_scope("Embeddings"):
    embedding = tf.Variable(tf.random_uniform((n_words, embed_size), -1, 1))
    embed = tf.nn.embedding_lookup(embedding, inputs_)

def lstm_cell():
    # Your basic LSTM cell
    lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size, reuse=tf.get_variable_scope().reuse)
    # Add dropout to the cell
    return tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)

with tf.name_scope("RNN_layers"):
    # Stack up multiple LSTM layers, for deep learning
    cell = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(lstm_layers)])

    # Getting an initial state of all zeros
    initial_state = cell.zero_state(batch_size, tf.float32)

with tf.name_scope("RNN_forward"):
    outputs, final_state = tf.nn.dynamic_rnn(cell, embed, initial_state=initial_state)

with tf.name_scope('predictions'):
    predictions = tf.contrib.layers.fully_connected(outputs[:, -1], 1, activation_fn=tf.sigmoid)
    tf.summary.histogram('predictions', predictions)
with tf.name_scope('cost'):
    cost = tf.losses.mean_squared_error(labels_, predictions)
    tf.summary.scalar('cost', cost)

with tf.name_scope('train'):
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

merged = tf.summary.merge_all()

with tf.name_scope('validation'):
    correct_pred = tf.equal(tf.cast(tf.round(predictions), tf.int32), labels_)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

def get_batches(x, y, batch_size=100):
    n_batches = len(x)//batch_size
    x, y = x[:n_batches*batch_size], y[:n_batches*batch_size]
    for ii in range(0, len(x), batch_size):
        yield x[ii:ii+batch_size], y[ii:ii+batch_size]

epochs = 7

# with graph.as_default():
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter('./out_log/train', sess.graph)
    test_writer = tf.summary.FileWriter('./out_log/test', sess.graph)
    iteration = 1
    for e in range(epochs):
        state = sess.run(initial_state)

        for ii, (x, y) in enumerate(get_batches(train_x, train_y, batch_size), 1):
            feed = {inputs_: x,
                    labels_: y[:, None],
                    keep_prob: 0.5,
                    initial_state: state}
            summary, loss, state, _ = sess.run([merged, cost, final_state, optimizer], feed_dict=feed)

            train_writer.add_summary(summary, iteration)

            if iteration%5==0:
                print("Epoch: {}/{}".format(e, epochs),
                      "Iteration: {}".format(iteration),
                      "Train loss: {:.3f}".format(loss))

            if iteration%25==0:
                val_acc = []
                val_state = sess.run(cell.zero_state(batch_size, tf.float32))
                for x, y in get_batches(val_x, val_y, batch_size):
                    feed = {inputs_: x,
                            labels_: y[:, None],
                            keep_prob: 1,
                            initial_state: val_state}
                    summary, batch_acc, val_state = sess.run([merged, accuracy, final_state], feed_dict=feed)
                    val_acc.append(batch_acc)
                print("Val acc: {:.3f}".format(np.mean(val_acc)))
            iteration +=1
            test_writer.add_summary(summary, iteration)
            saver.save(sess, "checkpoints_noembed2/sentiment_manish.ckpt")
    saver.save(sess, "checkpoints_noembed2/sentiment_manish.ckpt")

test_acc = []
with tf.Session() as sess:
    saver.restore(sess, "checkpoints_noembed2/sentiment_manish.ckpt")
    test_state = sess.run(cell.zero_state(batch_size, tf.float32))
    for ii, (x, y) in enumerate(get_batches(test_x, test_y, batch_size), 1):
        feed = {inputs_: x,
                labels_: y[:, None],
                keep_prob: 1,
                initial_state: test_state}
        corrects, test_state = sess.run([correct_pred, final_state], feed_dict=feed)
        test_acc.append(corrects)

mytemp = []
for rs in test_acc:
    for r in rs:
        if r[0] == True:
            c = 1
        else:
            c = 0
        mytemp.append(c)
print("accuracy:", np.mean(mytemp))  # 0.6117788461538461
