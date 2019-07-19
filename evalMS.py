
from __future__ import print_function
import codecs
import os

import tensorflow as tf
import numpy as np

from hyperparameter import Hyperparameters as hp
from hyperparameter_local import HyperparametersLocal as hpl
from data_loadMS import *

from train_multisource import Graph
from nltk.translate.bleu_score import corpus_bleu


def eval():
    # Load graph
    g = Graph(is_training=False)
    print("Graph loaded")

    # Load data
    X1, X2, Sources1, Sources2, Targets = load_test_data()
    #src2idx1, idx2src1 = load_src1_vocab()
    #src2idx2, idx2src2 = load_src2_vocab()
    tgt2idx, idx2tgt = load_tgt_vocab()

    logdir = hpl.logdir
    result_dir = hpl.result_dir

    # Start session         
    with g.graph.as_default():
        sv = tf.train.Supervisor()
        with sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            ## Restore parameters
            sv.saver.restore(sess, tf.train.latest_checkpoint(logdir))
            print(logdir+ " Restored!")

            ## Get model name
            mname = open(logdir + '/checkpoint', 'r').read().split('"')[1]  # model name

            ## Inference
            if not os.path.exists(result_dir): os.mkdir(result_dir)
            print(result_dir + " is the output result directory.")
            with codecs.open(result_dir + mname, "w", "utf-8") as fout, codecs.open(result_dir + "reference", "w",
                                                                                    "utf-8") as fout1:
                list_of_refs, hypotheses = [], []

                for i in range(len(X1) // hp.batch_size):
                    flag = 0
                    ### Get mini-batches
                    x1 = X1[i * hp.batch_size: (i + 1) * hp.batch_size]
                    sources1 = Sources1[i * hp.batch_size: (i + 1) * hp.batch_size]

                    x2 = X2[i * hp.batch_size: (i + 1) * hp.batch_size]
                    sources2 = Sources2[i * hp.batch_size: (i + 1) * hp.batch_size]

                    targets = Targets[i * hp.batch_size: (i + 1) * hp.batch_size]

                    ### Autoregressive inference
                    preds = np.zeros((hp.batch_size, hp.maxlen), np.int32)
                    for j in range(hp.maxlen):
                        _preds = sess.run(g.preds, {g.x1: x1, g.x2: x2, g.y: preds})
                        preds[:, j] = _preds[:, j]


                    ### Write to file
                    for source1, source2, target, pred in zip(sources1, sources2, targets, preds):  # sentence-wise
                        got = " ".join(idx2tgt[idx] for idx in pred).split("</S>")[0].strip()
                        print("H:\t" + got)
                        fout.write(got + "\n")
                        fout.flush()
                        flag = 1
                        fout1.write(target + "\n")
                        fout1.flush()
                        # bleu score
                        ref = target.split()
                        hypothesis = got.split()
                        if len(ref) > 3 and len(hypothesis) > 3:
                            list_of_refs.append([ref])
                            hypotheses.append(hypothesis)
                    if flag == 0:
                        print(source2)

                ## Calculate bleu score
                score = corpus_bleu(list_of_refs, hypotheses)
                print("Bleu Score = " + str(100 * score))



if __name__ == '__main__':
    eval()
    print("Done")
