#! /usr/bin/python
# -*- coding: utf8 -*-
import itertools
import os
import time

import numpy as np
import tensorflow as tf

from datetime import datetime

from sklearn.metrics import confusion_matrix, f1_score, cohen_kappa_score
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.util import nest

from data_loaders.data_loader import SeqDataLoader
from probens.generic_seq import GenSleepNet
from data_loaders.sleep_stage import (NUM_CLASSES,
                                   EPOCH_SEC_LEN,
                                   SAMPLING_RATE)
from data_loaders.utils import iterate_batch_seq_minibatches


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_dir', 'data',
                           """Directory where to load training data.""")
tf.app.flags.DEFINE_string('model_dir', 'output',
                           """Directory where to load trained models.""")
tf.app.flags.DEFINE_string('output_dir', 'output',
                           """Directory where to load trained models.""")
tf.app.flags.DEFINE_integer('sr', 256,
                            """Sampling rate of the chosen dataset.""")
tf.app.flags.DEFINE_boolean('resume', False,
                            """Whether to resume the training process.""")
tf.app.flags.DEFINE_boolean('use_lstm', True,
                            """Use LSTM/GRU for RNN's. True is LSTM, false is GRU.""")
tf.app.flags.DEFINE_boolean('use_drop', False,
                            """Use dropout/zoneout for RNN's. True is drop, false is zone.""")
tf.app.flags.DEFINE_integer('conv1_s', 6,
                            """Define the size of the conv filters in the first cnn.""")
tf.app.flags.DEFINE_integer('conv2_s', 8,
                            """Define the size of the conv filters in the second cnn.""")
tf.app.flags.DEFINE_integer('conv_layers', 3,
                            """Define the number of conv filters in the cnns.""")
tf.app.flags.DEFINE_integer('hidden_rnn_cells', 512,
                            """Define the number of hidden used in the RNN's.""")
tf.app.flags.DEFINE_integer('rnn_layers', 2,
                            """Define the number of hidden used in the RNN's.""")
tf.app.flags.DEFINE_float('z_prob', 0.1,
                            """Define zoneout prob.""")

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def print_performance(sess, network_name, n_examples, duration, loss, cm, acc, f1, kappa):
    # Get regularization loss
    reg_loss = tf.add_n(tf.get_collection("losses", scope=network_name + "\/"))
    reg_loss_value = sess.run(reg_loss)

    # Print performance
    print (
        "duration={:.3f} sec, n={}, loss={:.3f} ({:.3f}), acc={:.3f}, "
        "f1={:.3f},kappa={:.3f}".format(
            duration, n_examples, loss, reg_loss_value, acc, f1, kappa
        )
    )
    print cm
    print " "

def softmax(z):
    #assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis] # necessary step to do broadcasting
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis] # dito
    return e_x / div

def ensemble(y1,y2,y3,y_true_,fold_idx,sub_idx):
    preds = []
    for i in range(len(y1)):
        c1 = np.asarray(softmax(y1[i]))
        c2 = np.asarray(softmax(y2[i]))
        c3 = np.asarray(softmax(y3[i]))
        cur = c1+c2+c3
        preds.extend(np.argmax(cur,axis=1))

    y_pred_ = preds
    cm = confusion_matrix(y_true_, y_pred_)
    acc = np.mean(y_true_ == y_pred_)
    mf1 = f1_score(y_true_, y_pred_, average="macro")
    kappa = cohen_kappa_score(y_true_, y_pred_)
    print "acc={:.3f}, f1={:.3f}, kappa={:.3f}".format(acc, mf1, kappa)
    print cm

    save_dict = {
        "y_true": y_true_,
        "y_pred": y_pred_
    }
    save_path = os.path.join(
        FLAGS.output_dir,
        "output_fold{}_subj_{}.npz".format(fold_idx,sub_idx)
    )
    np.savez(save_path, **save_dict)
    print "Saved outputs to {}".format(save_path)
    return y_pred_

def custom_run_epoch(
    sess,
    network,
    inputs,
    targets,
    train_op,
    is_train
):
    start_time = time.time()
    y = []
    y_true = []
    total_loss, n_batches = 0.0, 0
    for sub_f_idx, each_data in enumerate(itertools.izip(inputs, targets)):
        each_x, each_y = each_data

        # Store prediction and actual stages of each patient
        each_y_true = []
        each_y_pred = []

        for x_batch, y_batch in iterate_batch_seq_minibatches(inputs=each_x,
                                                              targets=each_y,
                                                              batch_size=network.batch_size,
                                                              seq_length=network.seq_length):
            feed_dict = {
                network.input_var: x_batch,
                network.target_var: y_batch
            }

            _, loss_value, y_pred = sess.run(
                [train_op, network.loss_op, network.pred_op, ],
                feed_dict=feed_dict
            )

            each_y_true.extend(y_batch)
            each_y_pred.extend(y_pred)

            total_loss += loss_value
            n_batches += 1

            # Check the loss value
            assert not np.isnan(loss_value), \
                "Model diverged with loss = NaN"

        y.append(each_y_pred)
        y_true.append(each_y_true)


    duration = time.time() - start_time
    total_loss /= n_batches
    #total_y_pred = np.hstack(y)
    total_y_pred = y
    total_y_true = np.hstack(y_true)

    return total_y_true, total_y_pred, total_loss, duration


def predict(
    data_dir,
    model_dir,
    output_dir,
    fold_idx,
    sub_idx,
    n_subjects_per_fold,
    hid_rnn_size
):
    # Ground truth and predictions
    y_true = []
    y_pred = []

    # The model will be built into the default Graph
    with tf.Graph().as_default(), tf.Session() as sess:
        # Build the network
        valid_net = GenSleepNet(
            batch_size=1,
            input_dims=EPOCH_SEC_LEN*FLAGS.sr,
            n_classes=NUM_CLASSES,
            seq_length=25,
            n_rnn_layers=FLAGS.rnn_layers,
            return_last=False,
            hidden_size_rnn=hid_rnn_size,
            use_lstm=FLAGS.use_lstm,
            is_train=False,
            reuse_params=False,
            z_prob=FLAGS.z_prob,
            n_conv_layers=FLAGS.conv_layers,
            conv1_size=FLAGS.conv1_s,
            conv2_size=FLAGS.conv2_s,
            sr=FLAGS.sr,
            use_dropout_feature=FLAGS.use_drop,
            use_dropout_sequence=FLAGS.use_drop
        )

        # Initialize parameters
        valid_net.init_ops()

        checkpoint_path = os.path.join(
            model_dir,
            "fold{}".format(fold_idx),
            "gensleepnet"
        )

        # Restore the trained model
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(checkpoint_path))
        #valid_net = tf.train.Saver.restore(sess, tf.train.latest_checkpoint(checkpoint_path)
        print "Model restored from: {}\n".format(tf.train.latest_checkpoint(checkpoint_path))

        # Load testing data
        x, y = SeqDataLoader.load_single_subject_data(
            data_dir=data_dir,
            subject_idx=(fold_idx*n_subjects_per_fold)+sub_idx
        )

        # Loop each epoch
        print "[{}] Predicting ...\n".format(datetime.now())

        # Evaluate the model on the subject data
        y_true_, y_pred_, loss, duration = \
            custom_run_epoch(
                sess=sess, network=valid_net,
                inputs=x, targets=y,
                train_op=tf.no_op(),
                is_train=False
                #output_dir=output_dir,
                #subject_idx=sub_idx,
                #fold_idx=fold_idx
            )


    return y_pred_, y_true_


def main(argv=None):
    # Output dir
    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)

    n_subjects = 25
    n_subjects_per_fold = 5
    y = []
    y_true_ = []
    for fold_idx in range(5):
        print "Starting ensemble on fold idx ".format(fold_idx)
        for i in range(n_subjects_per_fold):
            y_pred, y_true = predict(
                data_dir="/sleep-edf-data/physionet_sleep/pz-30mins",
                model_dir="/pz-25-test",
                output_dir="/pz-25-test",
                fold_idx=fold_idx,
                sub_idx = i,
                n_subjects_per_fold = n_subjects_per_fold,
                hid_rnn_size=512
            )

            y_pred2, _ = predict(
                data_dir="/sleep-edf-data/physionet_sleep/eog-30mins",
                model_dir="/eog-25-test/",
                output_dir="/eog-25-test/",
                fold_idx=fold_idx,
                sub_idx = i,
                n_subjects_per_fold = n_subjects_per_fold,
                hid_rnn_size=512
            )

            y_pred3,_ = predict(
                data_dir="/sleep-edf-data/physionet_sleep/fpz25-30mins",
                model_dir="/fpz_25_test",
                output_dir="/fpz_25_test",
                fold_idx=fold_idx,
                sub_idx = i,
                n_subjects_per_fold = n_subjects_per_fold,
                hid_rnn_size=512
            )

            # Make ensemble for each fold and report perfomance
            y_ = ensemble(y_pred,y_pred2,y_pred3,y_true,fold_idx,i)
            y_true_.extend(y_true)
            y.extend(y_)

    print "[{}] Overall prediction performance\n".format(datetime.now())
    y_true = np.asarray(y_true_)
    y_pred = np.asarray(y)
    cm = confusion_matrix(y_true, y_pred)
    acc = np.mean(y_true == y_pred)
    mf1 = f1_score(y_true, y_pred, average="macro")
    kappa = cohen_kappa_score(y_true, y_pred)
    print (
        "acc={:.3f}, f1={:.3f}, kappa={:.3f}".format(
            acc, mf1, kappa
        )
    )
    print cm

if __name__ == "__main__":
    tf.app.run()
