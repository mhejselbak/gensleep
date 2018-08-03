from trainer import *

class DeepFeatNetTrainer(Trainer):

    def __init__(
        self,
        data_dir,
        output_dir,
        n_folds,
        fold_idx,
        batch_size,
        input_dims,
        n_classes,
        interval_plot_filter=50,
        interval_save_model=100,
        interval_print_cm=10
    ):
        super(self.__class__, self).__init__(
            interval_plot_filter=interval_plot_filter,
            interval_save_model=interval_save_model,
            interval_print_cm=interval_print_cm
        )

        self.data_dir = data_dir
        self.output_dir = output_dir
        self.n_folds = n_folds
        self.fold_idx = fold_idx
        self.batch_size = batch_size
        self.input_dims = input_dims
        self.n_classes = n_classes

    def _run_epoch(self, sess, network, inputs, targets, train_op, is_train):
        start_time = time.time()
        y = []
        y_true = []
        total_loss, n_batches = 0.0, 0
        is_shuffle = True if is_train else False
        for x_batch, y_batch in iterate_minibatches(inputs,
                                                    targets,
                                                    self.batch_size,
                                                    shuffle=is_shuffle):
            feed_dict = {
                network.input_var: x_batch,
                network.target_var: y_batch
            }

            # # MONITORING
            # if n_batches == 0:
            #     print "BEFORE UPDATE [is_train={}]".format(is_train)
            #     for n, v in network.monitor_vars[:2]:
            #         val = sess.run(v, feed_dict=feed_dict)
            #         val = np.transpose(val, axes=(3, 0, 1, 2)).reshape((64, -1))
            #         mean_val = np.mean(val, axis=1)
            #         var_val = np.var(val, axis=1)
            #         print "{}: {}\nmean_shape={}, mean_val={}\nvar_shape={}, var_val={}".format(
            #             n, val.shape, mean_val.shape, mean_val[:5], var_val.shape, var_val[:5]
            #         )

            #update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            #print update_ops
#            with tf.control_dependencies(extra_update_ops):
#                ex_op = adam(loss),
            #print train_op
            #print train_op.dtype
            _, loss_value, y_pred = sess.run(
                [train_op, network.loss_op, network.pred_op],
                feed_dict=feed_dict
            )

            # # MONITORING
            # if n_batches == 0:
            #     print "AFTER UPDATE [is_train={}]".format(is_train)
            #     for n, v in network.monitor_vars[:2]:
            #         val = sess.run(v, feed_dict=feed_dict)
            #         val = np.transpose(val, axes=(3, 0, 1, 2)).reshape((64, -1))
            #         mean_val = np.mean(val, axis=1)
            #         var_val = np.var(val, axis=1)
            #         print "{}: {}\nmean_shape={}, mean_val={}\nvar_shape={}, var_val={}".format(
            #             n, val.shape, mean_val.shape, mean_val[:5], var_val.shape, var_val[:5]
            #         )

            total_loss += loss_value
            n_batches += 1
            y.append(y_pred)
            y_true.append(y_batch)

            # Check the loss value
            assert not np.isnan(loss_value), \
                "Model diverged with loss = NaN"

        duration = time.time() - start_time
        total_loss /= n_batches
        total_y_pred = np.hstack(y)
        total_y_true = np.hstack(y_true)

        return total_y_true, total_y_pred, total_loss, duration

    def train(self, n_epochs, resume):
        with tf.Graph().as_default(), tf.Session() as sess:
            # Build training and validation networks
            train_net = DeepFeatNet(
                batch_size=self.batch_size,
                input_dims=self.input_dims,
                n_classes=self.n_classes,
                is_train=True,
                reuse_params=False,
                use_dropout=True
            )
            valid_net = DeepFeatNet(
                batch_size=self.batch_size,
                input_dims=self.input_dims,
                n_classes=self.n_classes,
                is_train=False,
                reuse_params=True,
                # standard is true
                use_dropout=False
            )

            # Initialize parameters
            train_net.init_ops()
            valid_net.init_ops()

            print "Network (layers={})".format(len(train_net.activations))
            print "inputs ({}): {}".format(
                train_net.input_var.name, train_net.input_var.get_shape()
            )
            print "targets ({}): {}".format(
                train_net.target_var.name, train_net.target_var.get_shape()
            )
            for name, act in train_net.activations:
                print "{} ({}): {}".format(name, act.name, act.get_shape())
            print " "

            # Define optimization operations
            # Adding udate ops to run, slight increase but still worse than their bn
            # https://stackoverflow.com/questions/43234667/tf-layers-batch-normalization-large-test-error
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op, grads_and_vars_op = adam(
                    loss=train_net.loss_op,
                    lr=1e-4,
                    train_vars=tf.trainable_variables()
                )

            # Make subdirectory for pretraining
            output_dir = os.path.join(self.output_dir, "fold{}".format(self.fold_idx), train_net.name)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Global step for resume training
            with tf.variable_scope(train_net.name) as scope:
                global_step = tf.Variable(0, name="global_step", trainable=False)

            #print "Trainable Variables:"
            #for v in tf.trainable_variables():
            #    print v.name, v.get_shape()
            #print " "

            # print "All Variables:"
            # for v in tf.global_variables():
            #     print v.name, v.get_shape()
            # print " "

            # Create a saver
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=0)

            # Initialize variables in the graph
            sess.run(tf.global_variables_initializer())

            # Add the graph structure into the Tensorboard writer
            train_summary_wrt = tf.summary.FileWriter(
                os.path.join(output_dir, "train_summary"),
                sess.graph
            )

            # Resume the training if applicable
            if resume:
                if os.path.exists(output_dir):
                    if os.path.isfile(os.path.join(output_dir, "checkpoint")):
                        # Restore the last checkpoint
                        saver.restore(sess, tf.train.latest_checkpoint(output_dir))
                        print "Model restored"
                        print "[{}] Resume pre-training ...\n".format(datetime.now())
                    else:
                        print "[{}] Start pre-training ...\n".format(datetime.now())
            else:
                print "[{}] Start pre-training ...\n".format(datetime.now())

            # Load data
            if sess.run(global_step) < n_epochs:
                data_loader = NonSeqDataLoader(
                    data_dir=self.data_dir,
                    n_folds=self.n_folds,
                    fold_idx=self.fold_idx
                )
                x_train, y_train, x_valid, y_valid = data_loader.load_train_data()

                # Performance history
                all_train_loss = np.zeros(n_epochs)
                all_train_acc = np.zeros(n_epochs)
                all_train_f1 = np.zeros(n_epochs)
                all_train_kappa = np.zeros(n_epochs)
                all_valid_loss = np.zeros(n_epochs)
                all_valid_acc = np.zeros(n_epochs)
                all_valid_f1 = np.zeros(n_epochs)
                all_valid_kappa = np.zeros(n_epochs)

            # Loop each epoch
            for epoch in xrange(sess.run(global_step), n_epochs):
                # # MONITORING
                # print "BEFORE TRAINING"
                # monitor_vars = [
                #     "deepfeaturenet/l1_conv/bn/moving_mean:0",
                #     "deepfeaturenet/l1_conv/bn/moving_variance:0"
                # ]
                # for n in monitor_vars:
                #     v = tf.get_default_graph().get_tensor_by_name(n)
                #     val = sess.run(v)
                #     print "{}: {}, {}".format(n, val.shape, val[:5])

                # Update parameters and compute loss of training set


                y_true_train, y_pred_train, train_loss, train_duration = \
                    self._run_epoch(
                        sess=sess, network=train_net,
                        inputs=x_train, targets=y_train,
                        train_op=train_op,
                        is_train=True
                    )
                n_train_examples = len(y_true_train)
                train_cm = confusion_matrix(y_true_train, y_pred_train)
                train_acc = np.mean(y_true_train == y_pred_train)
                train_f1 = f1_score(y_true_train, y_pred_train, average="macro")
                train_kappa = cohen_kappa_score(y_true_train, y_pred_train)

                #update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                #with tf.control_dependencies(update_ops):
                #    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(train_loss)
                # # MONITORING
                # print "AFTER TRAINING and BEFORE VALID"
                # for n in monitor_vars:
                #     v = tf.get_default_graph().get_tensor_by_name(n)
                #     val = sess.run(v)
                #     print "{}: {}, {}".format(n, val.shape, val[:5])

                # Evaluate the model on the validation set
                y_true_val, y_pred_val, valid_loss, valid_duration = \
                    self._run_epoch(
                        sess=sess, network=valid_net,
                        inputs=x_valid, targets=y_valid,
                        train_op=tf.no_op(),
                        is_train=False
                    )
                n_valid_examples = len(y_true_val)
                valid_cm = confusion_matrix(y_true_val, y_pred_val)
                valid_acc = np.mean(y_true_val == y_pred_val)
                valid_f1 = f1_score(y_true_val, y_pred_val, average="macro")
                valid_kappa = cohen_kappa_score(y_true_val, y_pred_val)

                # db.train_log(args={
                #     "n_folds": self.n_folds,
                #     "fold_idx": self.fold_idx,
                #     "epoch": epoch,
                #     "train_step": "pretraining",
                #     "datetime": datetime.utcnow(),
                #     "model": train_net.name,
                #     "n_train_examples": n_train_examples,
                #     "n_valid_examples": n_valid_examples,
                #     "train_loss": train_loss,
                #     "train_acc": train_acc,
                #     "train_f1": train_f1,
                #     "train_duration": train_duration,
                #     "valid_loss": valid_loss,
                #     "valid_acc": valid_acc,
                #     "valid_f1": valid_f1,
                #     "valid_duration": valid_duration,
                # })

                all_train_loss[epoch] = train_loss
                all_train_acc[epoch] = train_acc
                all_train_f1[epoch] = train_f1
                all_train_kappa[epoch] = train_kappa
                all_valid_loss[epoch] = valid_loss
                all_valid_acc[epoch] = valid_acc
                all_valid_f1[epoch] = valid_f1
                all_valid_kappa[epoch] = valid_kappa

                # Report performance
                self.print_performance(
                    sess, output_dir, train_net.name,
                    n_train_examples, n_valid_examples,
                    train_cm, valid_cm, epoch, n_epochs,
                    train_duration, train_loss, train_acc, train_f1, train_kappa,
                    valid_duration, valid_loss, valid_acc, valid_f1, valid_kappa
                )

                # Save performance history
                np.savez(
                    os.path.join(output_dir, "perf_fold{}.npz".format(self.fold_idx)),
                    train_loss=all_train_loss, valid_loss=all_valid_loss,
                    train_acc=all_train_acc, valid_acc=all_valid_acc,
                    train_f1=all_train_f1, valid_f1=all_valid_f1,
                    train_kappa=all_train_kappa, valid_kappa=all_valid_kappa,
                    y_true_val=np.asarray(y_true_val),
                    y_pred_val=np.asarray(y_pred_val)
                )

                # Visualize weights from convolutional layers
                if ((epoch + 1) % self.interval_plot_filter == 0) or ((epoch + 1) == n_epochs):
                    self.plot_filters(sess, epoch, train_net.name + "(_[0-9])?\/l[0-9]+_conv\/(weights)", output_dir, 16)
                    self.plot_filters(sess, epoch, train_net.name + "(_[0-9])?/l[0-9]+_conv\/conv1d\/(weights)", output_dir, 16)

                # Save checkpoint
                sess.run(tf.assign(global_step, epoch+1))
                if ((epoch + 1) % self.interval_save_model == 0) or ((epoch + 1) == n_epochs):
                    start_time = time.time()
                    save_path = os.path.join(
                        output_dir, "model_fold{}.ckpt".format(self.fold_idx)
                    )
                    saver.save(sess, save_path, global_step=global_step)
                    duration = time.time() - start_time
                    print "Saved model checkpoint ({:.3f} sec)".format(duration)

                # Save paramaters
                if ((epoch + 1) % self.interval_save_model == 0) or ((epoch + 1) == n_epochs):
                    start_time = time.time()
                    save_dict = {}
                    for v in tf.global_variables():
                        save_dict[v.name] = sess.run(v)
                    np.savez(
                        os.path.join(
                            output_dir,
                            "params_fold{}.npz".format(self.fold_idx)),
                        **save_dict
                    )
                    duration = time.time() - start_time
                    print "Saved trained parameters ({:.3f} sec)".format(duration)

        print "Finish pre-training"
        return os.path.join(output_dir, "params_fold{}.npz".format(self.fold_idx))
