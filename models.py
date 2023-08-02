from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from functools import partial

import os
import numpy as np
import tensorflow as tf
from ops import lrelu, linear, conv2d, deconv2d
from utils import make_batches, Prior, conv_out_size_same, create_image_grid

batch_norm = partial(tf.keras.layers.BatchNormalization())

class MGAN(object):
    """Mixture Generative Adversial Nets"""
    
    def __init__(self, mode_name='MGAN', beta=1.0, num_z=128, num_gens=4, d_batch_size=64, g_batch_size=32, z_prior="uniform", same_input=True, learning_rate=0.0002, img_size=(32, 32, 3), # (height, width, channels)
    num_conv_layers=3,
    num_gen_feature_maps=128,
    num_dis_feature_maps=128,
    sample_fp = None,
    sample_by_gen_fp=None,
    num_epochs=25000,
    random_seed=6789):
        self.beta = beta
        self.num_z = num_z
        self.num_gens = num_gens
        self.d_batch_size = d_batch_size
        self.g_batch_size = g_batch_size
        self.z_prior = Prior(z_prior)
        self.same_input = same_input
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.img_size = img_size
        self.num_conv_layers = num_conv_layers
        self.num_gen_feature_maps = num_gen_feature_maps
        self.num_dis_feature_maps = num_dis_feature_maps
        self.sample_fp = sample_fp
        self.sample_by_gen_fp = sample_by_gen_fp
        self.random_seed = random_seed
        
    
    