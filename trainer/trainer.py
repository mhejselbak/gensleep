import itertools
import os
import re
import time

from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, f1_score, cohen_kappa_score

from data_loaders.data_loader import NonSeqDataLoader, SeqDataLoader
from model.feature_rep import DeepFeatureNet
from model.mini_feat import DeepFeatNet
from model.seq_red_mod import DeepSleepNet
from model.seq_mini import ShallowSleepNet
from model.gru_net import GruSleepNet
from model.small_lstm import MediumSleepNet
from model.generic_seq import GenSleepNet
from model.generic_feat import GenFeatureNet
#from model.residual_feat import GenFeatureNet
from utils.optimize import adam, adam_clipping_list_lr
from data_loaders.utils import iterate_minibatches, iterate_batch_seq_minibatches

class Trainer(object):

    def __init__(
        self,
        interval_plot_filter=50,
        interval_save_model=100,
        interval_print_cm=10
    ):
        self.interval_plot_filter = interval_plot_filter
        self.interval_save_model = interval_save_model
        self.interval_print_cm = interval_print_cm

    def print_performance(self, sess, output_dir, network_name,
                           n_train_examples, n_valid_examples,
                           train_cm, valid_cm, epoch, n_epochs,
                           train_duration, train_loss, train_acc, train_f1,train_kappa,
                           valid_duration, valid_loss, valid_acc, valid_f1,valid_kappa):
        # Get regularization loss
        train_reg_loss = tf.add_n(tf.get_collection("losses", scope=network_name + "\/"))
        train_reg_loss_value = sess.run(train_reg_loss)
        valid_reg_loss_value = train_reg_loss_value

        # Print performance
        if ((epoch + 1) % self.interval_print_cm == 0) or ((epoch + 1) == n_epochs):
            print " "
            print "[{}] epoch {}:".format(
                datetime.now(), epoch+1
            )
            print (
                "train ({:.3f} sec): n={}, loss={:.3f} ({:.3f}), acc={:.3f}, "
                "f1={:.3f}, kappa={:.3f}".format(
                    train_duration, n_train_examples,
                    train_loss, train_reg_loss_value,
                    train_acc, train_f1, train_kappa
                )
            )
            print train_cm
            print (
                "valid ({:.3f} sec): n={}, loss={:.3f} ({:.3f}), acc={:.3f}, "
                "f1={:.3f}, kappa={:.3f}".format(
                    valid_duration, n_valid_examples,
                    valid_loss, valid_reg_loss_value,
                    valid_acc, valid_f1, valid_kappa
                )
            )
            print valid_cm
            print " "
        else:
            print (
                "epoch {}: "
                "train ({:.2f} sec): n={}, loss={:.3f} ({:.3f}), "
                "acc={:.3f}, f1={:.3f}, kappa={:.3f} | "
                "valid ({:.2f} sec): n={}, loss={:.3f} ({:.3f}), "
                "acc={:.3f}, f1={:.3f}, kappa={:.3f} ".format(
                    epoch+1,
                    train_duration, n_train_examples,
                    train_loss, train_reg_loss_value,
                    train_acc, train_f1, train_kappa,
                    valid_duration, n_valid_examples,
                    valid_loss, valid_reg_loss_value,
                    valid_acc, valid_f1, valid_kappa
                )
            )

    def print_network(self, network):
        print "inputs ({}): {}".format(
            network.inputs.name, network.inputs.get_shape()
        )
        print "targets ({}): {}".format(
            network.targets.name, network.targets.get_shape()
        )
        for name, act in network.activations:
            print "{} ({}): {}".format(name, act.name, act.get_shape())
        print " "

    def plot_filters(self, sess, epoch, reg_exp, output_dir, n_viz_filters):
        conv_weight = re.compile(reg_exp)
        for v in tf.trainable_variables():
            value = sess.run(v)
            if conv_weight.match(v.name):
                weights = np.squeeze(value)
                # Only plot conv that has one channel
                if len(weights.shape) > 2:
                    continue
                weights = weights.T
                plt.figure(figsize=(18, 10))
                plt.title(v.name)
                for w_idx in xrange(n_viz_filters):
                    plt.subplot(4, 4, w_idx+1)
                    plt.plot(weights[w_idx])
                    plt.axis("tight")
                plt.savefig(os.path.join(
                    output_dir, "{}_{}.png".format(
                        v.name.replace("/", "_").replace(":0", ""),
                        epoch+1
                    )
                ))
                plt.close("all")
