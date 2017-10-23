import tensorflow as tf
import implementation as imp

batch_size = 32
hidden_size = 128
comment_len = 40
word_vect_len = 50
learning_rate = 0.001
# num_layers = 2
# num_steps = 50
# hidden_size = 128
# keep_prob = 0.5
# batch_size = 32
# vocab_size = 10000

weight = {
    'in': tf.Variable(tf.random_normal([comment_len, word_vect_len])),
    'out': tf.Variable(tf.random_normal([hidden_size, 2]))
}

bias = {
    'in': tf.Variable(tf.constant(0.1, shape=[hidden_size, ])),
    'out': tf.Variable(tf.constant(0.1, shape=[2, ]))
}

inputs = imp.load_data(imp.load_glove_embeddings())


input_data = tf.placeholder(tf.int32, [batch_size, comment_len, word_vect_len])
target = tf.placeholder(tf.float32, [batch_size, 2])


def LSTM(input_data, weight, bias):

    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=1, state_is_tuple=True)
    initial_state = lstm_cell.zero_state(batch_size, tf.float32)
    output, state = tf.nn.dynamic_rnn(lstm_cell, input_data, initial_state=initial_state)

    result = tf.matmul(state[1], weight['out']) + bias['out']
    return result

init = tf.global_variables_initializer()

logits = LSTM(inputs, weight, bias)
prediction = tf.nn.softmax(logits)

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=target))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

logits = tf.matmul(last, softmax_w) + softmax_b

cost = cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, target))
train_op = tf.train.AdamOptimizer().minimize(cost)