
# load data and split data
# adj, features, y_train, y_val, y_trainval, train_mask, val_mask, trainval_mask = \
#         load_data_vis_multi(FLAGS.dataset, use_trainval, feat_suffix)

# define the placeholder for everything we don't know during the graph construction
# placeholders = {
#     'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
#     'features': tf.placeholder(tf.float32, shape=(features.shape[0], features.shape[1])),  # sparse_placeholder
#     'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
#     'labels_mask': tf.placeholder(tf.int32),
#     'dropout': tf.placeholder_with_default(0., shape=()),
#     'num_features_nonzero': tf.placeholder(tf.int32),  # helper variable for sparse dropout
#     'learning_rate': tf.placeholder(tf.float32, shape=())
# }

# Create model
# Attention! must be complete for the graph, every intermidiate tensor unsured must be the result of a tensorflow manipulation
# tf.shape(x), tf.print(), for example 
# model = model_func(placeholders, input_dim=features.shape[1], logging=True)

# sess = tf.Session(config=create_config_proto())
# Init variables
# sess.run(tf.global_variables_initializer())

# Train model
# now_lr = FLAGS.learning_rate
# for epoch in range(FLAGS.epochs):
#     t = time.time()
#     # Construct feed dictionary
#     feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
#     feed_dict.update({placeholders['learning_rate']: now_lr})

#     # Training step
      # most important
      # out will be a result list
#     outs = sess.run([model.opt_op, model.loss, model.accuracy, model.optimizer._lr], feed_dict=feed_dict)

#     if epoch % 20 == 0:
#         print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
#               "train_loss_nol2=", "{:.5f}".format(outs[2]),
#               "time=", "{:.5f}".format(time.time() - t),
#               "lr=", "{:.5f}".format(float(outs[3])))

#     flag = 0
#     for k in range(len(save_epochs)):
#         if save_epochs[k] == epoch:
#             flag = 1

#     if flag == 1 or epoch % 500 == 0:
#         outs = sess.run(model.outputs, feed_dict=feed_dict)
#         filename = savepath + '/feat_' + os.path.basename(FLAGS.dataset) + '_' + str(epoch)
#         print(time.strftime('[%X %x %Z]\t') + 'save to: ' + filename)

#         filehandler = open(filename, 'wb')
#         pkl.dump(outs, filehandler)
#         filehandler.close()

# print("Optimization Finished!")

# sess.close()