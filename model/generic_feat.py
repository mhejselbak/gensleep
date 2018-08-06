import tensorflow as tf
import numpy as np

class GenFeatureNet(object):

    def __init__(
        self,
        batch_size,
        input_dims,
        n_classes,
        is_train,
        n_conv_layers,
        conv1_size,
        conv2_size,
        sr,
        reuse_params,
        use_dropout,
        name="genfeaturenet"
    ):
        self.batch_size = batch_size
        self.input_dims = input_dims
        self.n_classes = n_classes
        self.is_train = is_train
        self.reuse_params = reuse_params
        self.use_dropout = use_dropout
        self.name = name
        self.n_conv_layers = n_conv_layers
        self.conv1_size = conv1_size
        self.conv2_size = conv2_size
        self.sr = sr

        self.activations = []
        self.layer_idx = 1
        self.monitor_vars = []

    def _build_placeholder(self):
        # Input
        name = "x_train" if self.is_train else "x_valid"
        self.input_var = tf.placeholder(
            tf.float32,
            shape=[self.batch_size, self.input_dims, 1, 1],
            name=name + "_inputs"
        )
        # Target
        self.target_var = tf.placeholder(
            tf.int32,
            shape=[self.batch_size, ],
            name=name + "_targets"
        )
    def _create_variable(self, name, shape, initializer):
        var = tf.get_variable(name, shape, initializer=initializer)
        return var

    def _flatten(self, name, input_var):
        dim = 1
        for d in input_var.get_shape()[1:].as_list():
            dim *= d
        output_var = tf.reshape(input_var,
                                shape=[-1, dim],
                                name=name)

        return output_var

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

    def _conv_1d(self, name, input_var, filter_shape, stride, padding="SAME",
                bias=None, wd=None):
        with tf.variable_scope(name) as scope:
            # Trainable parameters
            kernel = self._variable_with_weight_decay(
                "weights",
                shape=filter_shape,
                wd=wd
            )

            # Convolution
            output_var = tf.nn.conv2d(
                input_var,
                kernel,
                [1, stride, 1, 1],
                padding=padding
            )

            # Bias
            if bias is not None:
                biases = _create_variable(
                    "biases",
                    [filter_shape[-1]],
                    tf.constant_initializer(bias)
                )
                output_var = tf.nn.bias_add(output_var, biases)

            return output_var

    def _conv1d_layer(self, input_var, filter_size, n_filters, stride, wd=0):
        input_shape = input_var.get_shape()
        n_batches = input_shape[0].value
        input_dims = input_shape[1].value
        n_in_filters = input_shape[3].value
        name = "l{}_conv".format(self.layer_idx)
        with tf.variable_scope(name) as scope:
            output = self._conv_1d(name="conv1d", input_var=input_var,
                    filter_shape=[filter_size, 1, n_in_filters, n_filters],
                    stride=stride, bias=None, wd=wd)
            # # MONITORING
            # self.monitor_vars.append(("{}_before_bn".format(name), output))
            output = self._my_batch_norm(name="bn", input_var=output,is_train=self.is_train)
            # # MONITORING
            # self.monitor_vars.append(("{}_after_bn".format(name), output))
            output = tf.nn.relu(output, name="relu")
        self.activations.append((name, output))
        self.layer_idx += 1
        return output

    def _max_pool_1d(self, name, input_var, pool_size, stride, padding="SAME"):
        output_var = tf.nn.max_pool(
            input_var,
            ksize=[1, pool_size, 1, 1],
            strides=[1, stride, 1, 1],
            padding=padding,
            name=name
        )

        return output_var

    """ Using tensorflows own batch normalization instead of the modified version
    from deep sleep, might need to use the following in train steps

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss))))
    """
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
        # List to store the output of each CNNs
        output_conns = []

        ######### CNNs with small filter size at the first layer #########

        # Convolution
        if self.sr==256:
            network = self._conv1d_layer(input_var=input_var, filter_size=128, n_filters=64, stride=16, wd=1e-3)
        else:
            network = self._conv1d_layer(input_var=input_var, filter_size=50, n_filters=64, stride=6, wd=1e-3)

        # Max pooling
        name = "l{}_pool".format(self.layer_idx)
        network = self._max_pool_1d(name=name, input_var=network, pool_size=8, stride=8)
        self.activations.append((name, network))
        self.layer_idx += 1

        # Dropout
        if self.use_dropout:
            name = "l{}_dropout".format(self.layer_idx)
            if self.is_train:
                network = tf.nn.dropout(network, keep_prob=0.5, name=name)
            else:
                network = tf.nn.dropout(network, keep_prob=1.0, name=name)
            self.activations.append((name, network))
        self.layer_idx += 1

        # Convolution
        networki = self._conv1d_layer(input_var=network, filter_size=self.conv1_size, n_filters=128, stride=1)
        for i in range(1,self.n_conv_layers):
            if i==1:
                network = self._conv1d_layer(input_var=networki, filter_size=self.conv1_size, n_filters=128, stride=1)
            else:
                network = self._conv1d_layer(input_var=network, filter_size=self.conv1_size, n_filters=128, stride=1)
        #network = network + networki

        # Max pooling
        name = "l{}_pool".format(self.layer_idx)
        network = self._max_pool_1d(name=name, input_var=network, pool_size=4, stride=4)
        self.activations.append((name, network))
        self.layer_idx += 1

        # Flatten
        name = "l{}_flat".format(self.layer_idx)
        network = self._flatten(name=name, input_var=network)
        self.activations.append((name, network))
        self.layer_idx += 1

        output_conns.append(network)

        ######### CNNs with large filter size at the first layer #########

        # Convolution
        if self.sr == 256:
            network = self._conv1d_layer(input_var=input_var, filter_size=1024, n_filters=64, stride=128)
        else:
            network = self._conv1d_layer(input_var=input_var, filter_size=400, n_filters=64, stride=50)

        # Max pooling
        name = "l{}_pool".format(self.layer_idx)
        network = self._max_pool_1d(name=name, input_var=network, pool_size=4, stride=4)
        self.activations.append((name, network))
        self.layer_idx += 1
        # test Residual connection
        #name = "l{}_fc".format(self.layer_idx)
        #output_tmp = self._fc(name="fc", input_var=network, n_hiddens=256, bias=None, wd=0)
        #output_tmp = tf.nn.relu(output_tmp, name="relu")
        #self.layer_idx += 1

        # Dropout
        if self.use_dropout:
            name = "l{}_dropout".format(self.layer_idx)
            if self.is_train:
                network = tf.nn.dropout(network, keep_prob=0.5, name=name)
            else:
                network = tf.nn.dropout(network, keep_prob=1.0, name=name)
            self.activations.append((name, network))
        self.layer_idx += 1

        # Convolution
        networks = self._conv1d_layer(input_var=network, filter_size=self.conv2_size, n_filters=128, stride=1)
        for i in range(1,self.n_conv_layers):
            if i==1:
                network = self._conv1d_layer(input_var=networks, filter_size=self.conv2_size, n_filters=128, stride=1)
            else:
                network = self._conv1d_layer(input_var=network, filter_size=self.conv2_size, n_filters=128, stride=1)
        #network = network + networks
        # Max pooling
        name = "l{}_pool".format(self.layer_idx)
        network = self._max_pool_1d(name=name, input_var=network, pool_size=2, stride=2)
        self.activations.append((name, network))
        self.layer_idx += 1

        # Flatten
        name = "l{}_flat".format(self.layer_idx)
        network = self._flatten(name=name, input_var=network)
        self.activations.append((name, network))
        self.layer_idx += 1

        output_conns.append(network)

        ######### Aggregate and link two CNNs #########

        # Concat
        name = "l{}_concat".format(self.layer_idx)
        #old command, new version has swapped
        #network = tf.concat(1, output_conns, name=name)
        network = tf.concat(output_conns,1, name=name)
        self.activations.append((name, network))
        self.layer_idx += 1

        # Dropout
        if self.use_dropout:
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
            network = self._fc(name=name, input_var=network, n_hiddens=self.n_classes, bias=0.0, wd=0)
            self.activations.append((name, network))
            self.layer_idx += 1

            # Outputs of softmax linear are logits
            self.logits = network

            ######### Compute loss #########

            # Cross-entropy loss
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.logits,
                labels=self.target_var,
                name="sparse_softmax_cross_entropy_with_logits"
            )
            loss = tf.reduce_mean(loss, name="cross_entropy")

            # Regularization loss
            regular_loss = tf.add_n(
                tf.get_collection("losses", scope=scope.name + "\/"),
                name="regular_loss"
            )

            #print " "
            #print "Params to compute regularization loss:"
            #for p in tf.get_collection("losses", scope=scope.name + "\/"):
            #    print p.name
            #print " "

            # Total loss
            self.loss_op = tf.add(loss, regular_loss)

            # Predictions
            self.pred_op = tf.argmax(self.logits, axis=1)
