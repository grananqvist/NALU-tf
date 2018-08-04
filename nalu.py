import numpy as np
import tensorflow as tf

def nalu(input_layer, num_outputs):
    """ Neural Arithmetic Logic Unit tesnorflow layer

    Arguments:
    input_layer - A Tensor representing previous layer
    num_outputs - number of ouput units 

    Returns:
    A tensor representing the output of NALU
    """

    shape = (int(input_layer.shape[-1]), num_outputs)

    # define variables
    W_hat = tf.Variable(tf.truncated_normal(shape, stddev=0.02))
    M_hat = tf.Variable(tf.truncated_normal(shape, stddev=0.02))
    G = tf.Variable(tf.truncated_normal(shape, stddev=0.02))

    # operations according to paper
    W = tf.tanh(W_hat) * tf.sigmoid(M_hat)
    m = tf.exp(tf.matmul(tf.log(tf.abs(input_layer) + 1e-7), W))
    g = tf.sigmoid(tf.matmul(input_layer, G))
    a = tf.matmul(input_layer, W)
    out = g * a + (1 - g) * m

    return out

def generate_dataset(size=10000, op='sum'):
    """ Generate dataset for NALU toy problem
    
    Arguments:
    size - number of samples to generate
    op - the operation that the generated data should represent. sum | prod 
    Returns:
    X - the dataset
    Y - the dataset labels
    """
    
    X = np.random.randint(9, size=(size,2))

    if op == 'prod':
        Y = np.prod(X, axis=1, keepdims=True)
    else:
        Y = np.sum(X, axis=1, keepdims=True)

    return X, Y


if __name__ == "__main__":

    EPOCS = 200
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 10

    # create dataset
    X_data, Y_data = generate_dataset(op='prod')

    # define placeholders and network
    X = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 2])
    Y_true = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 1])
    Y_pred = nalu(X, 1)

    # loss and train operations
    loss = tf.nn.l2_loss(Y_pred - Y_true) # NALU uses mse
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
    train_op = optimizer.minimize(loss)

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    for ep in range(EPOCS):
        i = 0
        gts = 0
        
        while i < len(X_data):
            xs, ys = X_data[i:i+BATCH_SIZE], Y_data[i:i+BATCH_SIZE]

            _, ys_pred, l = sess.run([train_op, Y_pred, loss], 
                    feed_dict={X: xs, Y_true: ys})

            # calculate number of correct predictions from batch
            gts += np.sum(np.isclose(ys, ys_pred, atol=1e-4, rtol=1e-4)) 

            i += BATCH_SIZE

        acc = gts/len(Y_data)
        print('epoch {2}, loss: {0}, accuracy: {1}'.format(l, acc, ep))

