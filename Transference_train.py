# -*- coding: utf-8 -*-
# /usr/bin/python2
from __future__ import print_function
import tensorflow as tf

from hyperparameter import Hyperparameters as hp
from hyperparameter_local import HyperparametersLocal as hpl
from data_loadMS import *
from utils import *
# from hyperparams import Hyperparams as hp
# from data_load import get_batch_data, load_de_vocab, load_en_vocab, load_train_data
# from modules import *
import os, codecs
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import beam_search

class Graph():
    def __init__(self, is_training=True):
        # Load vocabulary

        self.graph = tf.Graph()

        with self.graph.as_default():
            if is_training:
                 self.x1, self.x2, self.y, self.num_batch = get_batch_data() # (N, T)
            else: # inference
                self.x1 = tf.placeholder(tf.int32, shape=(None, hp.maxlen))
                self.x2 = tf.placeholder(tf.int32, shape=(None, hp.maxlen))
                self.y = tf.placeholder(tf.int32, shape=(None, hp.maxlen))

            # define decoder inputs

            self.decoder_inputs = tf.concat((tf.ones_like(self.y[:, :1]) * 2, self.y[:, :-1]), -1)  # 2:<S>

            src2idx1, idx2src1 = load_src1_vocab()
            src2idx2, idx2src2 = load_src2_vocab()
            tgt2idx, idx2tgt = load_tgt_vocab()
            print('Vocab loaded')

            # Encoder1
            with tf.variable_scope("encoder1"):
                ## Embedding
                self.enc1 = embedding(self.x1,
                                      vocab_size=len(src2idx1),
                                      num_units=hp.hidden_units,
                                      scale=True,
                                      scope="enc1_embed")

                ## Positional Encoding
                if hp.sinusoid:
                    self.enc1 += positional_encoding(self.x1,
                                                     num_units=hp.hidden_units,
                                                     zero_pad=False,
                                                     scale=False,
                                                     scope="enc1_pe")
                else:
                    self.enc1 += embedding(
                        tf.tile(tf.expand_dims(tf.range(tf.shape(self.x1)[1]), 0), [tf.shape(self.x1)[0], 1]),
                        vocab_size=hp.maxlen,
                        num_units=hp.hidden_units,
                        zero_pad=False,
                        scale=False,
                        scope="enc1_pe")

                ## Dropout
                self.enc1 = tf.layers.dropout(self.enc1,
                                              rate=hp.dropout_rate,
                                              training=tf.convert_to_tensor(is_training))


                ## Blocks
                for i in range(hp.num_blocks):
                    with tf.variable_scope("enc1_num_blocks_{}".format(i)):
                        ### Multihead Attention
                        self.enc1, _ = multihead_attention(queries=self.enc1,
                                                           keys=self.enc1,
                                                           num_units=hp.hidden_units,
                                                           num_heads=hp.num_heads,
                                                           dropout_rate=hp.dropout_rate,
                                                           is_training=is_training,
                                                           causality=False)

                        ### Feed Forward
                        self.enc1 = feedforward(self.enc1, num_units=[4 * hp.hidden_units, hp.hidden_units])

            # Encoder2
            with tf.variable_scope("encoder2"):
                ## Embedding
                self.enc2 = embedding(self.x2,
                                      vocab_size=len(src2idx2),
                                      num_units=hp.hidden_units,
                                      scale=True,
                                      scope="enc2_embed")

                ## Positional Encoding
                if hp.sinusoid:
                    self.enc2 += positional_encoding(self.x2,
                                                     num_units=hp.hidden_units,
                                                     zero_pad=False,
                                                     scale=False,
                                                     scope="enc2_pe")
                else:
                    self.enc2 += embedding(
                        tf.tile(tf.expand_dims(tf.range(tf.shape(self.x2)[1]), 0), [tf.shape(self.x2)[0], 1]),
                        vocab_size=hp.maxlen,
                        num_units=hp.hidden_units,
                        zero_pad=False,
                        scale=False,
                        scope="enc2_pe")

                ## Dropout
                self.enc2 = tf.layers.dropout(self.enc2,
                                              rate=hp.dropout_rate,
                                              training=tf.convert_to_tensor(is_training))


                ## Blocks
                for i in range(hp.num_blocks):
                    with tf.variable_scope("enc2_num_blocks_{}".format(i)):
                        ### Multihead Attention
                        self.enc2, _ = multihead_attention(queries=self.enc2,
                                                           keys=self.enc2,
                                                           num_units=hp.hidden_units,
                                                           num_heads=hp.num_heads,
                                                           dropout_rate=hp.dropout_rate,
                                                           is_training=is_training,
                                                           causality=False)


                        ### Multihead Attention
                        self.enc, _ = multihead_attention(queries=self.enc2,
                                                          keys=self.enc1,
                                                          num_units=hp.hidden_units,
                                                          num_heads=hp.num_heads,
                                                          dropout_rate=hp.dropout_rate,
                                                          is_training=is_training,
                                                          causality=False,
                                                          scope="vanilla_attention_enc12")

                        ### Feed Forward
                        self.enc = feedforward(self.enc, num_units=[4 * hp.hidden_units, hp.hidden_units])
            # Decoder
            with tf.variable_scope("decoder"):
                ## Embedding
                self.dec = embedding(self.decoder_inputs,
                                     vocab_size=len(tgt2idx),
                                     num_units=hp.hidden_units,
                                     scale=True,
                                     scope="dec_embed")

                ## Positional Encoding
                if hp.sinusoid:
                    self.dec += positional_encoding(self.decoder_inputs,
                                                    num_units=hp.hidden_units,
                                                    zero_pad=False,
                                                    scale=False,
                                                    scope="dec_pe")
                else:
                    self.dec += embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(self.decoder_inputs)[1]), 0),
                                                  [tf.shape(self.decoder_inputs)[0], 1]),
                                          vocab_size=hp.maxlen,
                                          num_units=hp.hidden_units,
                                          zero_pad=False,
                                          scale=False,
                                          scope="dec_pe")

                ## Dropout
                self.dec = tf.layers.dropout(self.dec,
                                             rate=hp.dropout_rate,
                                             training=tf.convert_to_tensor(is_training))

                ## Blocks
                for i in range(hp.num_blocks):
                    with tf.variable_scope("num_blocks_{}".format(i)):
                        ## Multihead Attention ( self-attention)
                        self.dec, _ = multihead_attention(queries=self.dec,
                                                          keys=self.dec,
                                                          num_units=hp.hidden_units,
                                                          num_heads=hp.num_heads,
                                                          dropout_rate=hp.dropout_rate,
                                                          is_training=is_training,
                                                          causality=True,
                                                          scope="self_attention")

                        ## Multihead Attention ( vanilla attention)
                        self.dec, self.alignments = multihead_attention(queries=self.dec,
                                                                        keys=self.enc,
                                                                        num_units=hp.hidden_units,
                                                                        num_heads=hp.num_heads,
                                                                        dropout_rate=hp.dropout_rate,
                                                                        is_training=is_training,
                                                                        causality=False,
                                                                        scope="vanilla_attention")

                        ## Feed Forward
                        self.dec = feedforward(self.dec, num_units=[4 * hp.hidden_units, hp.hidden_units])

            # Final linear projection
            self.logits = tf.layers.dense(self.dec, len(tgt2idx))

            #self.logits = tf.squeeze(self.logits, axis=[1])


            if is_training:
                self.preds = tf.to_int32(tf.argmax(self.logits, axis=-1))
            else:
                self.preds = tf.to_int32(tf.argmax(self.logits, axis=-1))
            self.istarget = tf.to_float(tf.not_equal(self.y, 0))
            self.acc = tf.reduce_sum(tf.to_float(tf.equal(self.preds, self.y)) * self.istarget) / (
                tf.reduce_sum(self.istarget))
            tf.summary.scalar('acc', self.acc)

            if is_training:
                # Loss
                self.y_smoothed = label_smoothing(tf.one_hot(self.y, depth=len(tgt2idx)))
                self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y_smoothed)
                self.mean_loss = tf.reduce_sum(self.loss * self.istarget) / (tf.reduce_sum(self.istarget))
                #self.mean_loss = tf.reduce_mean(self.mean_loss)
                self.perplexity = np.power(2, self.mean_loss)

                # Training Scheme
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                step = tf.to_float(self.global_step)
                warmup_steps = tf.to_float(4000)
                multiplier = hp.hidden_units ** -0.5
                decay = multiplier * tf.minimum((step + 1) * (warmup_steps ** -1.5),
                                                (step + 1) ** -0.5)

                learning_rate = hp.lr * decay
                learning_rate = tf.convert_to_tensor(learning_rate, dtype=tf.float32)
                tf.summary.scalar("learning_rate", learning_rate)
                learning_rate = tf.train.exponential_decay(learning_rate, self.global_step, 100000, 0.96, staircase=True)
                print(learning_rate)
                # self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.98, epsilon=1e-8)
                self.optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9,
                                                            use_locking=False, name='Momentum', use_nesterov=True)
                self.train_op = self.optimizer.minimize(self.mean_loss, global_step=self.global_step)

                # Summary
                tf.summary.scalar('mean_loss', self.mean_loss)
                tf.summary.scalar('perplexity', self.perplexity)
                self.merged = tf.summary.merge_all()


def plot_alignment(alignment, gs):
    """
    Plots the alignment
    alignment: (numpy) matrix of shape (encoder_steps,decoder_steps)
    gs : (int) global step
    """
    fig, ax = plt.subplots()
    im = ax.imshow(alignment, cmap="Greys", interpolation='none')
    fig.colorbar(im, ax=ax)
    plt.xlabel('Decoder timestep')
    plt.ylabel('Encoder timestep')
    plt.savefig(hpl.logdir + '/alignment_%d' % gs, format='png')




if __name__ == '__main__':
    # Load vocabulary
    src2idx1, idx2src1 = load_src1_vocab()
    src2idx2, idx2src2 = load_src2_vocab()
    tgt2idx, idx2tgt = load_tgt_vocab()

    # Construct graph
    g = Graph("train")
    print("Graph loaded")

    X1, X2, Y = load_train_data()

    # calc total batch count
    num_batch = len(X1) // hp.batch_size
    print(X1.shape)
    g.num_batch = num_batch
    # Start session
    sv = tf.train.Supervisor(graph=g.graph,
                             logdir=hpl.logdir,
                             summary_op=None,
                             save_model_secs=0)
    with sv.managed_session() as sess:
        i = 0
        for epoch in range(1, hp.num_epochs + 1):
            if sv.should_stop(): break
            for step in tqdm(range(g.num_batch), total=g.num_batch, ncols=70, leave=False, unit='b'):
                x1 = X1[step * hp.batch_size:(step + 1) * hp.batch_size]
                x2 = X2[step * hp.batch_size:(step + 1) * hp.batch_size]

                y = Y[step * hp.batch_size:(step + 1) * hp.batch_size]
                x1 = np.array(x1, dtype=np.int32)
                x2 = np.array(x2, dtype=np.int32)
                y = np.array(y, dtype=np.int32)
                sess.run([g.train_op, g.merged], {g.x1: x1, g.x2: x2, g.y: y})
                i += 1
                if step % 100 == 0:
                    sv.summary_computed(sess, sess.run(g.merged, {g.x1: x1, g.x2: x2, g.y: y}))
                    _preds, _alignments, _gs = sess.run([g.preds, g.alignments, g.global_step],
                                                        {g.x1: x1, g.x2: x2, g.y: y})
                    # print("\ninput=", " ".join(idx2src[idx] for idx in x[0]))
                    # print("expected=", " ".join(idx2tgt[idx] for idx in y[0]))
                    # print("got=", " ".join(idx2tgt[idx] for idx in _preds[0]))
                    # gs = _gs
                    # plot_alignment(_alignments[0], _gs)

            # gs, _ = sess.run([g.global_step, g.merged], {g.x: x, g.y:y})
            sv.saver.save(sess, hpl.logdir + '/model_epoch_%02d_gs_%d' % (epoch, i))

    print("Done")
