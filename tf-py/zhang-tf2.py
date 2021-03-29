#!/usr/bin/env python3
from mnist import MNIST
import numpy as np
import tensorflow as tf
import argparse
import time

def one_hot_encode(np_array, cats=10):
    return (np.arange(cats) == np_array[:,None]).astype(np.float32)

def cnn_train (training_rate, cats=10):
    graph = tf.Graph()
    with graph.as_default():
        img = tf.compat.v1.placeholder (tf.float32)
        lab = tf.compat.v1.placeholder (tf.float32)

        w1 = tf.Variable (name='w1', dtype=tf.float32, initial_value=tf.constant(1.0/25.0, shape=[5,5,1,6]))
        w2 = tf.Variable (name='w2', dtype=tf.float32, initial_value=tf.constant(1.0/150.0, shape=[5,5,6,12]))
        w3 = tf.Variable (name='w3', dtype=tf.float32, initial_value=tf.constant(1.0/192.0, shape=[192,1,1,cats]))

        b1 = tf.Variable (name='b1', dtype=tf.float32, initial_value=tf.constant(1.0/6.0, shape=[1,1,1,6]))
        b2 = tf.Variable (name='b2', dtype=tf.float32, initial_value=tf.constant(1.0/12.0, shape=[1,1,1,12]))
        b3 = tf.Variable (name='b3', dtype=tf.float32, initial_value=tf.constant(1.0/10.0, shape=[1,1,1,cats]))

        tr_input = tf.reshape (img, [-1, 28, 28, 1])
        lab_input = tf.reshape (lab, [-1, 1, 1, cats])

        # Layer 1
        pre_c1 = tf.nn.conv2d (input=tr_input, filters=w1, strides=[1,1,1,1], padding='VALID')
        add_c1 = tf.add (pre_c1, b1)
        c1 = tf.math.sigmoid (add_c1)
        s1 = tf.nn.avg_pool2d (input=c1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

        # Layer 2
        pre_c2 = tf.nn.conv2d (input=s1, filters=w2, strides=[1,1,1,1], padding='VALID')
        add_c2 = tf.add (pre_c2, b2)
        c2 = tf.math.sigmoid (add_c2)
        s2 = tf.nn.avg_pool2d (input=c2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

        # Flatten
        flat = tf.reshape (s2, [-1, 192, 1, 1])
        pre_out = tf.nn.conv2d (input=flat, filters=w3, strides=[1,1,1,1], padding='VALID')
        pre_add = tf.add (pre_out, b3)
        out = tf.math.sigmoid (pre_add)

        # Compute Loss
        #sub = tf.subtract (out, lab_input)
        #loss = tf.nn.l2_loss(sub)
        loss = tf.compat.v1.losses.mean_squared_error (out, lab_input, weights=0.5)

        # backprop and optimise
        optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=training_rate).minimize(loss)

        # compute correct
        prob = tf.nn.softmax (out)
        prediction = tf.argmax(input=prob, axis=1)
        correct_prediction = tf.equal(prediction, tf.cast(lab_input, tf.int64))
        accuracy = tf.reduce_mean(input_tensor=tf.cast(correct_prediction, tf.float32))
    return (graph, img, lab, out, accuracy, loss, optimizer)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Zhang CNN TF implementation')
    parser.add_argument('mnistpath', metavar='MNIST_PATH', type=str,
                    help='path to where mnist data is located.')
    parser.add_argument('-z', '--mnist-gzip', action='store_true', default=False,
                    help='indicate that the mnist dataset is gzipped.')
    parser.add_argument('-u', '--use-emnist', action='store_true', default=False,
                    help='use EMNIST datasets instead of MNIST.')
    parser.add_argument('-b', '--batch-size', type=int, default=100,
                    help='batch size')
    parser.add_argument('-t', '--training-size', type=int, default=10000,
                    help='training size (max 60000)')
    parser.add_argument('-e', '--evaluate-size', type=int, default=10000,
                    help='evaluation size (max 10000)')
    parser.add_argument('-p', '--epoch', type=int, default=20,
                    help='epochs to iterate')
    parser.add_argument('-r', '--training-rate', type=float, default=0.05,
                    help='rate to learn at')
    parser.add_argument('-n', '--num-thread', type=int, default=1,
                    help='number of threads to use')
    parser.add_argument('-g', '--gpu', action='store_true', default=False,
                    help='use the GPU (sets threads to 1)')

    args = parser.parse_args()
    tf.compat.v1.logging.set_verbosity (tf.compat.v1.logging.INFO)

    if not args.use_emnist:
        if args.training_size > 60000:
            parser.error("training size must be in domain (0,60000]!")
        if args.training_size > 10000:
            parser.error("evaluate size must be in domain (0,10000]!")
    else:
        tf.compat.v1.logging.info ('Please double-check that training/evaluation size are correct for EMNIST')
    if args.gpu:
        tf.compat.v1.logging.info ('Using GPU - setting threads to 1')
        args.num_thread = 1

    tf.compat.v1.logging.info ('Running with threads({})'.format(args.num_thread))
    tf.compat.v1.logging.info ('Parameters: epoch({}), batch-size({}), training-rate({:f}), training-size({}), evaluate-size({})'.format(
        args.epoch, args.batch_size, args.training_rate, args.training_size, args.evaluate_size))

    # Get data
    mndata = MNIST(args.mnistpath, return_type='numpy')
    cats = 10 # default for MNIST
    if args.use_emnist:
        mndata.select_emnist ('balanced')
        tf.compat.v1.logging.info ('Using EMNIST `balanced` dataset.')
        cats = 49
    mndata.gz = args.mnist_gzip
    train_images, train_labels_raw = mndata.load_training()
    test_images, test_labels_raw = mndata.load_testing()
    train_labels = one_hot_encode (train_labels_raw, cats)
    test_labels = one_hot_encode (test_labels_raw, cats)
    tf.compat.v1.logging.info ('Loaded data')

    # set TF settings
    sesscfg = tf.compat.v1.ConfigProto ()
    sesscfg.log_device_placement = False
    # comment out to run on all cores
    sesscfg.intra_op_parallelism_threads = args.num_thread
    sesscfg.inter_op_parallelism_threads = args.num_thread
    # disable use of GPU devices
    sesscfg.device_count['GPU'] = args.gpu

    cnn_graph, img, lab, out, pred, loss, opt = cnn_train (args.training_rate, cats)
    with tf.compat.v1.Session(config=sesscfg, graph=cnn_graph) as sess:
        # train
        tf.compat.v1.global_variables_initializer().run()
        tf.compat.v1.logging.info ('Initialized all variables')
        tf.compat.v1.logging.info ('Training the model now')
        train_start = time.perf_counter ()
        for epoch in range(args.epoch):
            ls = 0.0
            for i in range (0, args.training_size, args.batch_size):
                images = train_images[i:i+args.batch_size, :]
                labels = train_labels[i:i+args.batch_size, :]
                _, ls = sess.run ([opt, loss], feed_dict={img: images.reshape((args.batch_size, 28 * 28)), lab: labels.reshape ((args.batch_size, cats))})
            tf.compat.v1.logging.info ('Epoch: {:d}, Loss: {:.20f}'.format(epoch+1, ls/10.0/args.batch_size))
        train_stop = time.perf_counter () - train_start

        # run evaluation
        tf.compat.v1.logging.info ('Evaluating the model now')
        test_accuracy = 0.0
        test_start = time.perf_counter ()
        for i in range (0, args.evaluate_size, args.batch_size):
            images = test_images[i:i+args.batch_size, :]
            labels = test_labels[i:i+args.batch_size, :]
            _, accu = sess.run ([out, pred], feed_dict={img: images.reshape((args.batch_size, 28 * 28)), lab: labels.reshape ((args.batch_size, cats))})
            test_accuracy += accu
        test_stop = time.perf_counter () - test_start
        tf.compat.v1.logging.info ('Accuracy {:f}'.format(test_accuracy/args.batch_size))
        tf.compat.v1.logging.info ('Train: {:f}s, Test: {:f}s'.format(train_stop, test_stop))
