#! /usr/bin/python
# -*- coding: utf-8 -*-

#from time import localtime, strftime

import os, pickle, random
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


from datetime import datetime
import numpy as np
import logging, scipy

import tensorflow as tf
import tensorlayer as tl
from model import SRGAN_g, SRGAN_d, Vgg19_simple_api
from utils import *
from config import config, log_config

import time


###====================== HYPER-PARAMETERS ===========================###
## Adam
batch_size = config.TRAIN.batch_size
lr_init = config.TRAIN.lr_init
beta1 = config.TRAIN.beta1
## initialize G
n_epoch_init = config.TRAIN.n_epoch_init
## adversarial learning (SRGAN)
n_epoch = config.TRAIN.n_epoch
lr_decay = config.TRAIN.lr_decay
decay_every = config.TRAIN.decay_every

ni = int(np.sqrt(batch_size))



def evaluate(args):
    ## create folders to save result images
    save_dir = "samples/{}".format(tl.global_flag['mode'])
    tl.files.exists_or_mkdir(save_dir)

    valid_lr_img = scipy.misc.imread(args.input, mode='RGB')
    #valid_lr_img = tl.vis.read_image(os.path.basename(args.input), os.path.dirname(args.input))


    ###========================== DEFINE MODEL ============================###
    valid_lr_img = (valid_lr_img / 127.5) - 1  # rescale to ［－1, 1]

    size = valid_lr_img.shape
    print("Inpu image size: " + str(size) )
    # t_image = tf.placeholder('float32', [None, size[0], size[1], size[2]], name='input_image') # the old version of TL need to specify the image size
    t_image = tf.placeholder('float32', [1, None, None, 3], name='input_image')

    net_g = SRGAN_g(t_image, is_train=False, reuse=False)

    ###========================== RESTORE G =============================###
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    tl.layers.initialize_global_variables(sess)
    tl.files.load_and_assign_npz(sess=sess, name='checkpoint/g_srgan.npz', network=net_g)

    ###======================= EVALUATION =============================###
    start_time = time.time()
    out = sess.run(net_g.outputs, {t_image: [valid_lr_img]})
    print("took: %4.4fs" % (time.time() - start_time))

    print("LR size: %s /  generated HR size: %s" % (size, out.shape))  # LR size: (339, 510, 3) /  gen HR size: (1, 1356, 2040, 3)
    tl.vis.save_image(out[0], args.output)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', type=str, required=True, help='input image')
    parser.add_argument('--output', type=str, required=True, help='output image')

    args = parser.parse_args()

    tl.global_flag['mode'] = 'evaluate'

    if not os.path.isfile(args.input):
        print("input file not found!") 
        sys.exit(0)

    evaluate(args)


