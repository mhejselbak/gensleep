import tensorflow as tf
import numpy as np
from generic_feat import GenFeatureNet
#from residual_feat import GenFeatureNet
from functools import partial
from utils.new_zone import ZoneoutWrapper

from nn_helpers import *

class GenSleepNet(GenFeatureNet):

    def __init__(
        self,
        batch_size,
        input_dims,
        n_classes,
        seq_length,
        n_rnn_layers,
        hidden_size_rnn,
        use_lstm,
        return_last,
        is_train,
        reuse_params,
        z_prob,
        n_conv_layers,
        conv1_size,
        conv2_size,
        sr,
        use_dropout_feature,
        use_dropout_sequence,
        name="gensleepnet"
    ):
        super(self.__class__, self).__init__(
            batch_size=batch_size,
            input_dims=input_dims,
            n_classes=n_classes,
            is_train=is_train,
            n_conv_layers=n_conv_layers,
            conv1_size=conv1_size,
            conv2_size=conv2_size,
            sr=sr,
            reuse_params=reuse_params,
            use_dropout=use_dropout_feature,
            name=name
        )

        self.seq_length = seq_length
        self.n_rnn_layers = n_rnn_layers
        self.return_last = return_last

        self.use_dropout_sequence = use_dropout_sequence
        self.hidden_size_rnn = hidden_size_rnn
        self.use_lstm = use_lstm
        self.z_prob = z_prob

    def _build_placeholder(self):
        # Input
        name = "x_train" if self.is_train else "x_valid"
        self.input_var = tf.placeholder(
            tf.float32,
            shape=[self.batch_size*self.seq_length, self.input_dims, 1, 1],
            name=name + "_inputs"
        )
        # Target
        self.target_var = tf.placeholder(
            tf.int32,
            shape=[self.batch_size*self.seq_length, ],
            name=name + "_targets"
        )

    def _variable_with_weight_decay(self, name, shape, wd=None):
        # Get the number of input and output parameters
        if len(shape) == 2:
            fan_in = shape[0]
            fan_out = shape[1]
        elif len(shape) == 4:
            receptive_field_size = np.prod(shape[:2])
            fan_in = shape[-2] * receptive_field_size
            fan_out = shape[-1] * receptive_field_size
        else:
            # no specific assumptions
            fan_in = np.sqrt(np.prod(shape))
            fan_out = np.sqrt(np.prod(shape))

        # He et al. 2015 - http://arxiv.org/abs/1502.01852
        stddev = np.sqrt(2.0 / fan_in)
        initializer = tf.truncated_normal_initializer(stddev=stddev)

        # # Xavier
        # initializer = tf.contrib.layers.xavier_initializer()

        # Create or get the existing variable
        var = tf.get_variable(name,shape,initializer=initializer)

        # L2 weight decay
        if wd is not None:
            weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name="weight_loss")
            tf.add_to_collection("losses", weight_decay)

        return var

    def _my_batch_norm(self, name, input_var, is_train, decay=0.999,epsilon=1e-5):
        inputs_shape = input_var.get_shape()
        axis=list(range(len(inputs_shape)-1))
        params_shape = inputs_shape[-1:]
        with tf.variable_scope(name) as scope:
            normed = tf.layers.batch_normalization(input_var,
                momentum=decay,
                #axis=axis,
                axis=-1,
                epsilon=epsilon,
                beta_initializer=tf.zeros_initializer(),
                gamma_initializer=tf.random_normal_initializer(mean=1.0,stddev=0.002),
                training=is_train,
                moving_mean_initializer=tf.zeros_initializer(),
                moving_variance_initializer=tf.ones_initializer(),
                name=name,
                renorm=True,
                renorm_momentum=decay,
                fused=False
                )
        return normed

    def _fc(self, name, input_var, n_hiddens, bias=None, wd=None):
        with tf.variable_scope(name) as scope:
            # Get input dimension
            input_dim = input_var.get_shape()[-1].value

            # Trainable parameters
            weights = self._variable_with_weight_decay(
                "weights",
                shape=[input_dim, n_hiddens],
                wd=wd
            )

            # Multiply weights
            output_var = tf.matmul(input_var, weights)

            # Bias
            if bias is not None:
                biases = self._create_variable(
                    "biases",
                    [n_hiddens],
                    tf.constant_initializer(bias)
                )
                output_var = tf.add(output_var, biases)

            return output_var

    def build_model(self, input_var):
        # Create a network with superclass method
        network = super(self.__class__, self).build_model(
            input_var=self.input_var
        )

        # Residual (or shortcut) connection
        output_conns = []

        # Fully-connected to select some part of the output to add with the output from bi-directional LSTM
        name = "l{}_fc".format(self.layer_idx)
        with tf.variable_scope(name) as scope:
            output_tmp = self._fc(name="fc", input_var=network, n_hiddens=self.hidden_size_rnn*2, bias=None, wd=0)
            output_tmp = self._my_batch_norm(name="bn", input_var=output_tmp, is_train=self.is_train)
            output_tmp = tf.nn.relu(output_tmp, name="relu")
        self.activations.append((name, output_tmp))
        self.layer_idx += 1
        output_conns.append(output_tmp)

        ######################################################################

        # Reshape the input from (batch_size * seq_length, input_dim) to
        # (batch_size, seq_length, input_dim)
        name = "l{}_reshape_seq".format(self.layer_idx)
        input_dim = network.get_shape()[-1].value
        seq_input = tf.reshape(network,
                               shape=[-1, self.seq_length, input_dim],
                               name=name)
        assert self.batch_size == seq_input.get_shape()[0].value
        self.activations.append((name, seq_input))
        self.layer_idx += 1

        # Bidirectional LSTM network
        if self.use_lstm:
            name = "l{}_bi_lstm".format(self.layer_idx)
        else:
            name = "l{}_bi_gru".format(self.layer_idx)
        with tf.variable_scope(name) as scope:
            if self.use_lstm:
                rnn_cell = partial(tf.contrib.rnn.LSTMCell,
                                                       use_peepholes=True,
                                                       state_is_tuple=True)
            else:
                rnn_cell = partial(tf.contrib.rnn.GRUCell)
            ## Dropout
            if (self.use_dropout_sequence):
                keep_prob = 0.5 if self.is_train else 1.0
                drop_wrap = partial(tf.contrib.rnn.DropoutWrapper,
                                    output_keep_prob=keep_prob)
                fw_cell = tf.contrib.rnn.MultiRNNCell([drop_wrap(rnn_cell(self.hidden_size_rnn)) for _ in range(self.n_rnn_layers)])
                bw_cell = tf.contrib.rnn.MultiRNNCell([drop_wrap(rnn_cell(self.hidden_size_rnn)) for _ in range(self.n_rnn_layers)])
            ## Zoneout
            else:
                z_prob_cells = self.z_prob
                drop_wrap = partial(ZoneoutWrapper,is_training=self.is_train,zoneout_drop_prob=z_prob_cells)
                if self.n_rnn_layers > 1:
                    cells = [ZoneoutWrapper(drop_wrap(rnn_cell(self.hidden_size_rnn)),zoneout_drop_prob=z_prob_cells)]
                    cells += [ZoneoutWrapper(drop_wrap(rnn_cell(self.hidden_size_rnn)),zoneout_drop_prob=z_prob_cells) for _ in range(self.n_rnn_layers - 1)]
                    fw_cell = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)
                    bw_cell = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)
                else:
                    fw_cell = ZoneoutWrapper(drop_wrap(rnn_cell(self.hidden_size_rnn)),zoneout_drop_prob=z_prob_cells)
                    bw_cell = ZoneoutWrapper(drop_wrap(rnn_cell(self.hidden_size_rnn)),zoneout_drop_prob=z_prob_cells)

            # Feedforward to MultiRNNCell
            outputs, (fw_state, bw_state) = tf.nn.bidirectional_dynamic_rnn(
                dtype=tf.float32,
                cell_fw=fw_cell,
                cell_bw=bw_cell,
                inputs=seq_input
            )

            if self.return_last:
                network = outputs[-1]
            else:
                network = tf.reshape(tf.concat(outputs,1), [-1, self.hidden_size_rnn*2],
                                     name=name)
            self.activations.append((name, network))
            self.layer_idx +=1

        # Append output
        output_conns.append(network)

        ######################################################################

        # Add
        name = "l{}_add".format(self.layer_idx)
        network = tf.add_n(output_conns, name=name)
        self.activations.append((name, network))
        self.layer_idx += 1

        # Dropout
        if self.use_dropout_sequence:
            name = "l{}_dropout".format(self.layer_idx)
            if self.is_train:
                network = tf.nn.dropout(network, keep_prob=0.5, name=name)
            else:
                network = tf.nn.dropout(network, keep_prob=1.0, name=name)
            self.activations.append((name, network))
        self.layer_idx += 1

        return network

    def init_ops(self):
        self._build_placeholder()

        # Get loss and prediction operations
        with tf.variable_scope(self.name) as scope:

            # Reuse variables for validation
            if self.reuse_params:
                scope.reuse_variables()

            # Build model
            network = self.build_model(input_var=self.input_var)

            # Softmax linear
            name = "l{}_softmax_linear".format(self.layer_idx)
            network = fc(name=name, input_var=network, n_hiddens=self.n_classes, bias=0.0, wd=0)
            self.activations.append((name, network))
            self.layer_idx += 1

            # Outputs of softmax linear are logits
            self.logits = network

            ######### Compute loss #########

            # Weighted cross-entropy loss for a sequence of logits (per example)

            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits = [self.logits],
                labels = [self.target_var],
                name="sequence_loss_by_example"
            )
            #loss = tf.nn.seq2seq.sequence_loss_by_example(
            #    [self.logits],
            #    [self.target_var],
            #    [tf.ones([self.batch_size * self.seq_length])],
            #    name="sequence_loss_by_example"
            #)
            loss = tf.reduce_sum(loss) / self.batch_size

            # Regularization loss
            regular_loss = tf.add_n(
                tf.get_collection("losses", scope=scope.name + "\/"),
                name="regular_loss"
            )

            # print " "
            # print "Params to compute regularization loss:"
            # for p in tf.get_collection("losses", scope=scope.name + "\/"):
            #     print p.name
            # print " "

            # Total loss
            self.loss_op = tf.add(loss, regular_loss)

            # Predictions
            self.pred_op = tf.argmax(self.logits, 1)
