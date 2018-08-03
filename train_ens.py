#! /usr/bin/python
# -*- coding: utf8 -*-
import os

import numpy as np
import tensorflow as tf

from trainer.gen_seq_trainer import GenSleepNetTrainer
from trainer.gen_feat_trainer import GenFeatureNetTrainer
from data_loaders.sleep_stage import (NUM_CLASSES,
                                      EPOCH_SEC_LEN,
                                      SAMPLING_RATE)


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_dir', '/home/mhejsel/deepsleepnet/data/physionet_sleep/mix-signals',
                           """Directory where to load training data.""")
tf.app.flags.DEFINE_string('output_dir', '/media/sune/Data/Glostrup/mix-signals-2-',
                           """Directory where to save trained models """
                           """and outputs.""")
tf.app.flags.DEFINE_integer('n_folds', 5,
                           """Number of cross-validation folds.""")
tf.app.flags.DEFINE_integer('fold_idx', 0,
                            """Index of cross-validation fold to train.""")
tf.app.flags.DEFINE_integer('pretrain_epochs', 100,
                            """Number of epochs for pretraining DeepFeatureNet.""")
tf.app.flags.DEFINE_integer('finetune_epochs', 40,
                            """Number of epochs for fine-tuning DeepSleepNet.""")
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

def pretrain(n_epochs, fold_idx, output_dir, data_dir):
    trainer = GenFeatureNetTrainer(
        data_dir=data_dir,
        output_dir=output_dir,
        n_folds=FLAGS.n_folds,
        fold_idx=fold_idx,
        batch_size=100,
        input_dims=EPOCH_SEC_LEN*FLAGS.sr,
        n_classes=NUM_CLASSES,
        n_conv_layers=FLAGS.conv_layers,
        conv1_size=FLAGS.conv1_s,
        conv2_size=FLAGS.conv2_s,
        sr=FLAGS.sr,
        interval_plot_filter=50,
        interval_save_model=100,
        interval_print_cm=10
    )
    pretrained_model_path = trainer.train(
        n_epochs=n_epochs,
        resume=FLAGS.resume
    )
    return pretrained_model_path

def finetune(model_path, n_epochs,fold_idx,output_dir,data_dir):
    trainer = GenSleepNetTrainer(
        data_dir=data_dir,
        output_dir=output_dir,
        n_folds=FLAGS.n_folds,
        fold_idx=fold_idx,
        batch_size=10,
        input_dims=EPOCH_SEC_LEN*FLAGS.sr,
        n_classes=NUM_CLASSES,
        seq_length=25,
        n_rnn_layers=FLAGS.rnn_layers,
        return_last=False,
        hidden_size_rnn=FLAGS.hidden_rnn_cells,
        use_lstm=FLAGS.use_lstm,
        z_prob=FLAGS.z_prob,
        n_conv_layers=FLAGS.conv_layers,
        conv1_size=FLAGS.conv1_s,
        conv2_size=FLAGS.conv2_s,
        sr=FLAGS.sr,
        use_dropout_feature=FLAGS.use_drop,
        use_dropout_sequence=FLAGS.use_drop,
        interval_plot_filter=50,
        interval_save_model=100,
        interval_print_cm=10
    )
    finetuned_model_path = trainer.finetune(
        pretrained_model_path=model_path,
        n_epochs=n_epochs,
        resume=FLAGS.resume
    )
    return finetuned_model_path

def main(argv=None):
    # Output dir
    #output_dir = os.path.join(FLAGS.output_dir, "fold{}".format(FLAGS.fold_idx))
    #if not FLAGS.resume:
    #    if tf.gfile.Exists(output_dir):
    #        tf.gfile.DeleteRecursively(output_dir)
    #    tf.gfile.MakeDirs(output_dir)

    for i in range(FLAGS.n_folds):
        #C3-M2
        pretrained_model_path = pretrain(
            n_epochs=FLAGS.pretrain_epochs,
            fold_idx=i,
            output_dir="/media/sune/Data/Glostrup/C3-M2-model-250",
            data_dir="/media/sune/Data/Glostrup/C3-M2-data-250"
        )
        _ = finetune(
            model_path=pretrained_model_path,
            n_epochs=FLAGS.finetune_epochs,
            fold_idx=i,
            output_dir="/media/sune/Data/Glostrup/C3-M2-model-250",
            data_dir="/media/sune/Data/Glostrup/C3-M2-data-250"
        )

        #01-M2
        pretrained_model_path = pretrain(
            n_epochs=FLAGS.pretrain_epochs,
            fold_idx=i,
            output_dir="/media/sune/Data/Glostrup/O1-M2-model-250",
            data_dir="/media/sune/Data/Glostrup/O1-M2-data-250"
        )
        _ = finetune(
            model_path=pretrained_model_path,
            n_epochs=FLAGS.finetune_epochs,
            fold_idx=i,
            output_dir="/media/sune/Data/Glostrup/O1-M2-model-250",
            data_dir="/media/sune/Data/Glostrup/O1-M2-data-250"
        )
        #F3-M2
        pretrained_model_path = pretrain(
            n_epochs=FLAGS.pretrain_epochs,
            fold_idx=i,
            output_dir="/media/sune/Data/Glostrup/F3-M2-model-250",
            data_dir="/media/sune/Data/Glostrup/F3-M2-data-250"
        )
        _ = finetune(
            model_path=pretrained_model_path,
            n_epochs=FLAGS.finetune_epochs,
            fold_idx=i,
            output_dir="/media/sune/Data/Glostrup/F3-M2-model-250",
            data_dir="/media/sune/Data/Glostrup/F3-M2-data-250"
        )

if __name__ == "__main__":
    tf.app.run()
