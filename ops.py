from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import tensorflow as tf

def lrelu(x, alpha=0.2):
    return tf.maximum(x, alpha*x)

def linear(input, output_dim, scope='linear', stddev=0.01):
    norm = tf.random_normal_initializer(stddev=stddev)
    const = tf.constant_initializer(0.0)
    with tf.name_scope(scope):
        w = tf.Variable('weights', [input.get_shape()[1], output_dim], initial_value=norm)
        b = tf.Variable('biases', [output_dim], initial_value=const)
        return tf.matmul(input, w) + b
    
def conv2d(input_, output_dim, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name="conv2d"):
    with tf.name_scope(name):
        w = tf.Variable('weights', [k_h, k_w, input_.get_shape()[-1], output_dim],
                        initial_value=tf.random_normal_initializer(stddev=stddev))
        
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w], padding='SAME')
        
        biases = tf.Variable('biases', [output_dim], initial_value=tf.constant_initializer(0.0))
        return tf.nn.bias_add(conv, biases)
    
def deconv2d(input_, output_shape, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name="deconv2d", with_w=False):
    with tf.name_scope(name):
        w = tf.Variable('weights', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]], initial_value=tf.random_normal_initializer(stddev=stddev))
        
        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape, strides=[1, d_h, d_w, 1])
 
        biases = tf.Variable('biases', [output_shape[-1]], initial_value=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
        
        if with_w:
            return deconv, w, biases
        
        else:
            return deconv
        
def gmm_sample(num_samples, mix_coeffs, mean, cov):
    z = np.random.multinomial(num_samples, mix_coeffs)
    samples = np.zeros(shape=[num_samples, len(mean[0])])
    i_start = 0
    for i in range(len(mix_coeffs)):
        i_end = i_start + z[i]
        samples[i_start:i_end, :] = np.random.multivariate_normal(mean=np.array(mean)[i, :], cov=np.diag(np.array(cov)[i, :]),
                                                                  size=z[i])
        i_start = i_end
    return samples