from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os
import cPickle as pkl
import sys
import parser
sys.path.append('/home/hjz/hjzfolder/vrd-master')
from net.gcn import Graph
from net.vtranse_vgg import VTranse
from model.config import cfg
from model.ass_fun import *
from utils.utilities import *
#set the random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)
#gpu used
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# placeholder = {
#     'num_nodes' : tf.placeholder(dtype=tf.int32),
#     'feature' : tf.placeholder(dtype=tf.float32, shape=[None, hidden_dim])
# }
#
# #init knowledg
# knowledge = tf.expand_dims(knowledge, axis=2)
# knowledge = tf.tile(knowledge, multiples=[1, 1, layer_num])
#
#
# #init relation embedding
# rel_emb = np.array(rel_emb)

def train_graph():
    input_hyper = init_hyper()
    input_pls = {
        'feature': tf.placeholder(dtype=tf.float32, shape=[None, input_hyper['hidden_dim']])
    }

    vnet = VTranse()
    vnet.create_graph(input_hyper['N_each_batch'], input_hyper['index_sp'], input_hyper['index_cls'], input_hyper['N_cls'], input_hyper['N_rela'])

    graph_visual = Graph(input_hyper['layer_num'], input_hyper['hidden_dim'], input_hyper['num_cls'], input_pls, input_hyper['knowledge'])

    graph_text = Graph(input_hyper['layer_num'], input_hyper['hidden_dim'], input_hyper['num_cls'], input_pls,
                         input_hyper['knowledge'])

    text_layer_out = graph_text.get_layer_out()
    optimizer = tf.train.AdamOptimizer(learning_rate=input_hyper['lr_rate'])
    train_var = tf.trainable_variables()
    restore_var = [var for var in train_var if 'vgg_16' in var.name or 'RD' in var.name]
    saver_res = tf.train.Saver(restore_var)

    with tf.Session() as sess:
        # init
        init = tf.global_variables_initializer()
        sess.run(init)
        saver_res.restore(sess, input_hyper['model_path'])
        roidb_read = read_roidb(input_hyper['roidb_path'])
        train_roidb = roidb_read['train_roidb']
        test_roidb = roidb_read['test_roidb']
        N_train = len(train_roidb)
        N_test = len(test_roidb)

        for epoch in range(input_hyper['num_epoch']):
            for roidb_id in range(N_train):
                roidb_use = train_roidb[roidb_id]
            if len(roidb_use['rela_gt']) == 0:
                continue
            rd_loss_temp, acc_temp, diff = vnet.train_predicate(sess, roidb_use, None)
            diff = np.array(diff)
            print(diff.shape)
            vf = []
            print(np.array(input_hyper['rel_emb']).shape)
            num_batch = diff.shape[0]
            # for i in range(num_batch):
            #     num_nodes = diff[i][0]
            #
            #     vf.append(visual_feature)
            feed_dict = {}
            feed_dict.update({input_pls['feature']: input_hyper['rel_emb']})
            text_out = sess.run(text_layer_out, feed_dict = feed_dict)
            print(text_out)


if __name__ == '__main__':
    train_graph()