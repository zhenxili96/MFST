#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2017 bily     Huazhong University of Science and Technology
#
# Distributed under terms of the MIT license.

"""Contains definitions of the network in [1].

  [1] Bertinetto, L., et al. (2016).
      "Fully-Convolutional Siamese Networks for Object Tracking."
      arXiv preprint arXiv:1606.09549.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import tensorflow as tf
import numpy as np
import scipy.io as sio

from utils.misc_utils import get

slim = tf.contrib.slim

from tensorflow.python import pywrap_tensorflow

def parse_tf_model(file_name):
    result_dict = {}
    try:
        reader = pywrap_tensorflow.NewCheckpointReader(file_name)
        var_to_shape_map = reader.get_variable_to_shape_map()
        for key in sorted(var_to_shape_map):
            result_dict[key] = reader.get_tensor(key)
    except Exception as e:
        print(str(e))
    return result_dict

AlexNet_Model_Path = "/home/travail/dev/GitRepo/pretrained_model/bvlc_alexnet.npy"

def convolutional_alexnet_arg_scope(embed_config,
                                    trainable=True,
                                    is_training=False):
  """Defines the default arg scope.

  Args:
    embed_config: A dictionary which contains configurations for the embedding function.
    trainable: If the weights in the embedding function is trainable.
    is_training: If the embedding function is built for training.

  Returns:
    An `arg_scope` to use for the convolutional_alexnet models.
  """
  # Only consider the model to be in training mode if it's trainable.
  # This is vital for batch_norm since moving_mean and moving_variance
  # will get updated even if not trainable.
  is_model_training = trainable and is_training

  if get(embed_config, 'use_bn', True):
    batch_norm_scale = get(embed_config, 'bn_scale', True)
    batch_norm_decay = 1 - get(embed_config, 'bn_momentum', 3e-4)
    batch_norm_epsilon = get(embed_config, 'bn_epsilon', 1e-6)
    batch_norm_params = {
      "scale": batch_norm_scale,
      # Decay for the moving averages.
      "decay": batch_norm_decay,
      # Epsilon to prevent 0s in variance.
      "epsilon": batch_norm_epsilon,
      "trainable": trainable,
      "is_training": is_model_training,
      # Collection containing the moving mean and moving variance.
      "variables_collections": {
        "beta": None,
        "gamma": None,
        "moving_mean": ["moving_vars"],
        "moving_variance": ["moving_vars"],
      },
      'updates_collections': None,  # Ensure that updates are done within a frame
    }
    normalizer_fn = slim.batch_norm
  else:
    batch_norm_params = {}
    normalizer_fn = None

  weight_decay = get(embed_config, 'weight_decay', 5e-4)
  if trainable:
    weights_regularizer = slim.l2_regularizer(weight_decay)
  else:
    weights_regularizer = None

  init_method = get(embed_config, 'init_method', 'kaiming_normal')
  if is_model_training:
    logging.info('embedding init method -- {}'.format(init_method))
  if init_method == 'kaiming_normal':
    # The same setting as siamese-fc
    initializer = slim.variance_scaling_initializer(factor=2.0, mode='FAN_OUT', uniform=False)
  else:
    initializer = slim.xavier_initializer()

  with slim.arg_scope(
      [slim.conv2d],
      weights_regularizer=weights_regularizer,
      weights_initializer=initializer,
      padding='VALID',
      trainable=trainable,
      activation_fn=tf.nn.relu,
      normalizer_fn=normalizer_fn,
      normalizer_params=batch_norm_params):
    with slim.arg_scope([slim.batch_norm], **batch_norm_params):
      with slim.arg_scope([slim.batch_norm], is_training=is_model_training) as arg_sc:
        return arg_sc


def convolutional_alexnet(inputs, reuse=None, scope='convolutional_alexnet'):
  """Defines the feature extractor of SiamFC.

  Args:
    inputs: a Tensor of shape [batch, h, w, c].
    reuse: if the weights in the embedding function are reused.
    scope: the variable scope of the computational graph.

  Returns:
    net: the computed features of the inputs.
    end_points: the intermediate outputs of the embedding function.
  """
  with tf.variable_scope(scope, 'convolutional_alexnet', [inputs], reuse=reuse) as sc:
    end_points_collection = sc.name + '_end_points'
    with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                        outputs_collections=end_points_collection):
      net = inputs
      net = slim.conv2d(net, 96, [11, 11], 2, scope='conv1')
      net = slim.max_pool2d(net, [3, 3], 2, scope='pool1')
      with tf.variable_scope('conv2'):
        b1, b2 = tf.split(net, 2, 3)
        b1 = slim.conv2d(b1, 128, [5, 5], scope='b1')
        # The original implementation has bias terms for all convolution, but
        # it actually isn't necessary if the convolution layer is followed by a batch
        # normalization layer since batch norm will subtract the mean.
        b2 = slim.conv2d(b2, 128, [5, 5], scope='b2')
        net = tf.concat([b1, b2], 3)
      net = slim.max_pool2d(net, [3, 3], 2, scope='pool2')
      net_c3 = slim.conv2d(net, 384, [3, 3], 1, scope='conv3')
      #print(net_c3)
      
      with tf.variable_scope('conv4'):
        b1, b2 = tf.split(net_c3, 2, 3)
        b1 = slim.conv2d(b1, 192, [3, 3], 1, scope='b1')
        b2 = slim.conv2d(b2, 192, [3, 3], 1, scope='b2')
        net_c4 = tf.concat([b1, b2], 3)
        #print(net_c4)
        
      # Conv 5 with only convolution, has bias
      with tf.variable_scope('conv5'):
        with slim.arg_scope([slim.conv2d],
                            activation_fn=None, normalizer_fn=None):
          b1, b2 = tf.split(net_c4, 2, 3)
          b1 = slim.conv2d(b1, 128, [3, 3], 1, scope='b1')
          b2 = slim.conv2d(b2, 128, [3, 3], 1, scope='b2')
        net_c5 = tf.concat([b1, b2], 3)
        #print(net_c5)
        
  with tf.variable_scope("siamfc_branch_se", reuse=tf.AUTO_REUSE):
      conv3_shape = net_c3.get_shape().as_list()[1:3]
      conv3_avg = tf.layers.average_pooling2d(net_c3, conv3_shape, [1,1]) #(B, 1, 1, 384)
      conv3_sq_W = tf.get_variable("conv3_sq_W", [1,1,384,96], dtype=tf.float32, 
                                 initializer=tf.contrib.layers.xavier_initializer(),
                                 trainable=True)
      conv3_sq_b = tf.get_variable("conv3_sq_b", [96], dtype=tf.float32, 
                                 initializer=tf.constant_initializer(0.0, dtype=tf.float32),
                                 trainable=True)
      conv3_sq = tf.nn.conv2d(conv3_avg, conv3_sq_W, [1,1,1,1], padding='VALID')
      conv3_sq = tf.nn.bias_add(conv3_sq, conv3_sq_b)
      conv3_sq = tf.nn.relu(conv3_sq)
      conv3_ex_W = tf.get_variable("conv3_ex_W", [1,1,96,384], dtype=tf.float32, 
                                 initializer=tf.contrib.layers.xavier_initializer(),
                                 trainable=True)
      conv3_ex_b = tf.get_variable("conv3_ex_b", [384], dtype=tf.float32, 
                                 initializer=tf.constant_initializer(0.0, dtype=tf.float32),
                                 trainable=True)
      conv3_ex = tf.nn.conv2d(conv3_sq, conv3_ex_W, [1,1,1,1], padding='VALID')
      conv3_ex = tf.nn.bias_add(conv3_ex, conv3_ex_b)
      conv3_ex = tf.nn.sigmoid(conv3_ex)
      
      conv4_shape = net_c4.get_shape().as_list()[1:3]
      conv4_avg = tf.layers.average_pooling2d(net_c4, conv4_shape, [1,1]) #(B, 1, 1, 384)
      conv4_sq_W = tf.get_variable("conv4_sq_W", [1,1,384,96], dtype=tf.float32, 
                                 initializer=tf.contrib.layers.xavier_initializer(),
                                 trainable=True)
      conv4_sq_b = tf.get_variable("conv4_sq_b", [96], dtype=tf.float32, 
                                 initializer=tf.constant_initializer(0.0, dtype=tf.float32),
                                 trainable=True)
      conv4_sq = tf.nn.conv2d(conv4_avg, conv4_sq_W, [1,1,1,1], padding='VALID')
      conv4_sq = tf.nn.bias_add(conv4_sq, conv4_sq_b)
      conv4_sq = tf.nn.relu(conv4_sq)
      conv4_ex_W = tf.get_variable("conv4_ex_W", [1,1,96,384], dtype=tf.float32, 
                                 initializer=tf.contrib.layers.xavier_initializer(),
                                 trainable=True)
      conv4_ex_b = tf.get_variable("conv4_ex_b", [384], dtype=tf.float32, 
                                 initializer=tf.constant_initializer(0.0, dtype=tf.float32),
                                 trainable=True)
      conv4_ex = tf.nn.conv2d(conv4_sq, conv4_ex_W, [1,1,1,1], padding='VALID')
      conv4_ex = tf.nn.bias_add(conv4_ex, conv4_ex_b)
      conv4_ex = tf.nn.sigmoid(conv4_ex)
      
      conv5_shape = net_c5.get_shape().as_list()[1:3]
      conv5_avg = tf.layers.average_pooling2d(net_c5, conv5_shape, [1,1]) #(B, 1, 1, 256)
      conv5_sq_W = tf.get_variable("conv5_sq_W", [1,1,256,64], dtype=tf.float32, 
                                 initializer=tf.contrib.layers.xavier_initializer(),
                                 trainable=True)
      conv5_sq_b = tf.get_variable("conv5_sq_b", [64], dtype=tf.float32, 
                                 initializer=tf.constant_initializer(0.0, dtype=tf.float32),
                                 trainable=True)
      conv5_sq = tf.nn.conv2d(conv5_avg, conv5_sq_W, [1,1,1,1], padding='VALID')
      conv5_sq = tf.nn.bias_add(conv5_sq, conv5_sq_b)
      conv5_sq = tf.nn.relu(conv5_sq)
      conv5_ex_W = tf.get_variable("conv5_ex_W", [1,1,64,256], dtype=tf.float32, 
                                 initializer=tf.contrib.layers.xavier_initializer(),
                                 trainable=True)
      conv5_ex_b = tf.get_variable("conv5_ex_b", [256], dtype=tf.float32, 
                                 initializer=tf.constant_initializer(0.0, dtype=tf.float32),
                                 trainable=True)
      conv5_ex = tf.nn.conv2d(conv5_sq, conv5_ex_W, [1,1,1,1], padding='VALID')
      conv5_ex = tf.nn.bias_add(conv5_ex, conv5_ex_b)
      conv5_ex = tf.nn.sigmoid(conv5_ex)
        
      conv3_se = net_c3*conv3_ex
      conv4_se = net_c4*conv4_ex
      conv5_se = net_c5*conv5_ex
      # Convert end_points_collection into a dictionary of end_points.
      end_points = slim.utils.convert_collection_to_dict(end_points_collection)
# =============================================================================
#       return net_c5, net_c4, net_c3, end_points
# =============================================================================
      return conv5_se, conv4_se, conv3_se, end_points


convolutional_alexnet.stride = 8



def get_pretrained_alexnet():
    all_params = np.load(AlexNet_Model_Path).item()
    return all_params

def embed_alexnet(images):
## refer to following links and SA-Net
# https://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
# https://medium.com/@smallfishbigsea/a-walk-through-of-alexnet-6cbd137a5637
# https://www.cs.toronto.edu/~guerzhoy/tf_alexnet/myalexnet_forward.py
    with tf.variable_scope("alex_branch", reuse=tf.AUTO_REUSE):
        net_data = get_pretrained_alexnet()
        def conv(data, kernel, biases, k_h, k_w, c_o, s_h, s_w,  padding="VALID", group=1):
            '''From https://github.com/ethereon/caffe-tensorflow
            '''
            c_i = data.get_shape()[-1]
            assert c_i%group==0
            assert c_o%group==0
            convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
            
            
            if group==1:
                conv = convolve(data, kernel)
            else:
                input_groups = tf.split(data, group, 3)
                kernel_groups = tf.split(kernel, group, 3)
                output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
                conv = tf.concat(output_groups, 3)
            return  tf.reshape(tf.nn.bias_add(conv, biases), [-1]+conv.get_shape().as_list()[1:])
            
    
        #conv1
        #conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
        k_h = 11; k_w = 11; c_o = 96; s_h = 2; s_w = 2
        conv1W = tf.get_variable("conv1W", dtype=tf.float32, 
                                 initializer=tf.constant(net_data["conv1"][0]),
                                 trainable=False)
        conv1b = tf.get_variable("conv1b", dtype=tf.float32,
                                 initializer=tf.constant(net_data["conv1"][1]), 
                                 trainable=False)
        conv1_in = conv(images, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="VALID", group=1)
        conv1 = tf.nn.relu(conv1_in)
        
        #lrn1
        #lrn(2, 2e-05, 0.75, name='norm1')
        radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
        lrn1 = tf.nn.local_response_normalization(conv1,
                                                          depth_radius=radius,
                                                          alpha=alpha,
                                                          beta=beta,
                                                          bias=bias)
        
        #maxpool1
        #max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
        k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
        maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)
        
        
        #conv2
        #conv(5, 5, 256, 1, 1, group=2, name='conv2')
        k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group = 2
        conv2W = tf.get_variable("conv2W", dtype=tf.float32,
                                 initializer=tf.constant(net_data["conv2"][0]), 
                                 trainable=False)
        conv2b = tf.get_variable("conv2b", dtype=tf.float32,
                                 initializer=tf.constant(net_data["conv2"][1]),
                                 trainable=False)
        conv2_in = conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="VALID", group=group)
        conv2 = tf.nn.relu(conv2_in)
        
        
        #lrn2
        #lrn(2, 2e-05, 0.75, name='norm2')
        radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
        lrn2 = tf.nn.local_response_normalization(conv2,
                                                          depth_radius=radius,
                                                          alpha=alpha,
                                                          beta=beta,
                                                          bias=bias)
        
        #maxpool2
        #max_pool(3, 3, 2, 2, padding='VALID', name='pool2')                                                  
        k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
        maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)
        
        #conv3
        #conv(3, 3, 384, 1, 1, name='conv3')
        k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1
        conv3W = tf.get_variable("conv3W", dtype=tf.float32,
                                 initializer=tf.constant(net_data["conv3"][0]),
                                 trainable=False)
        conv3b = tf.get_variable("conv3b", dtype=tf.float32,
                                 initializer=tf.constant(net_data["conv3"][1]),
                                 trainable=False)
        conv3_in = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="VALID", group=group)
        conv3 = tf.nn.relu(conv3_in)
        #print(conv3)
        
        #conv4
        #conv(3, 3, 384, 1, 1, group=2, name='conv4')
        k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
        conv4W = tf.get_variable("conv4W", dtype=tf.float32,
                                 initializer=tf.constant(net_data["conv4"][0]),
                                 trainable=False)
        conv4b = tf.get_variable("conv4b", dtype=tf.float32,
                                 initializer=tf.constant(net_data["conv4"][1]),
                                 trainable=False)
        conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="VALID", group=group)
        conv4 = tf.nn.relu(conv4_in)
        #print(conv4)
        
        #conv5
        #conv(3, 3, 256, 1, 1, group=2, name='conv5')
        k_h = 3; k_w = 3; c_o = 256; s_h = 1; s_w = 1; group = 2
        conv5W = tf.get_variable("conv5W", dtype=tf.float32,
                                 initializer=tf.constant(net_data["conv5"][0]),
                                 trainable=False)
        conv5b = tf.get_variable("conv5b", dtype=tf.float32,
                                 initializer=tf.constant(net_data["conv5"][1]),
                                 trainable=False)
        conv5_in = conv(conv4, conv5W, conv5b, k_h, k_w, c_o, s_h, s_w, padding="VALID", group=group)
        conv5 = tf.nn.relu(conv5_in)
        #print(conv5)
        
        
    with tf.variable_scope("alex_branch_se", reuse=tf.AUTO_REUSE):
        #ALEX_DICT = parse_tf_model("/home/travail/dev/GitRepo/CBSiamFC/Logs/SiamFC/track_model_checkpoints/Alex_v1/model.ckpt-332499")
        conv3_shape = conv3.get_shape().as_list()[1:3]
        conv3_avg = tf.layers.average_pooling2d(conv3, conv3_shape, [1,1]) #(B, 1, 1, 384)
        conv3_sq_W = tf.get_variable("conv3_sq_W", [1,1,384,96], dtype=tf.float32, 
                                     #initializer=tf.constant_initializer(ALEX_DICT["alex_branch/conv2d/kernel"]),
                                     initializer=tf.contrib.layers.xavier_initializer(),
                                     trainable=True)
        conv3_sq_b = tf.get_variable("conv3_sq_b", [96], dtype=tf.float32, 
                                     #initializer=tf.constant_initializer(ALEX_DICT["alex_branch/conv2d/bias"]),
                                     initializer=tf.constant_initializer(0.0),
                                     trainable=True)
        conv3_sq = tf.nn.conv2d(conv3_avg, conv3_sq_W, [1,1,1,1], padding='VALID')
        conv3_sq = tf.nn.bias_add(conv3_sq, conv3_sq_b)
        #conv3_sq = tf.layers.conv2d(conv3_avg, 96, [1,1]) #(B, 1, 1, 96)
        conv3_sq = tf.nn.relu(conv3_sq)
        conv3_ex_W = tf.get_variable("conv3_ex_W", [1,1,96,384], dtype=tf.float32, 
                                     #initializer=tf.constant_initializer(ALEX_DICT["alex_branch/conv2d_1/kernel"]),
                                     initializer=tf.contrib.layers.xavier_initializer(),
                                     trainable=True)
        conv3_ex_b = tf.get_variable("conv3_ex_b", [384], dtype=tf.float32, 
                                     #initializer=tf.constant_initializer(ALEX_DICT["alex_branch/conv2d_1/bias"]),
                                     initializer=tf.constant_initializer(0.0),
                                     trainable=True)
        conv3_ex = tf.nn.conv2d(conv3_sq, conv3_ex_W, [1,1,1,1], padding='VALID')
        conv3_ex = tf.nn.bias_add(conv3_ex, conv3_ex_b)
        #conv3_ex = tf.layers.conv2d(conv3_sq, 384, [1,1]) #(B, 1, ,1 384)
        conv3_ex = tf.nn.sigmoid(conv3_ex)


        conv4_shape = conv4.get_shape().as_list()[1:3]
        conv4_avg = tf.layers.average_pooling2d(conv4, conv4_shape, [1,1]) #(B, 1, 1, 384)
        conv4_sq_W = tf.get_variable("conv4_sq_W", [1,1,384,96], dtype=tf.float32, 
                                     #initializer=tf.constant_initializer(ALEX_DICT["alex_branch/conv2d_2/kernel"]),
                                     initializer=tf.contrib.layers.xavier_initializer(),
                                     trainable=True)
        conv4_sq_b = tf.get_variable("conv4_sq_b", [96], dtype=tf.float32, 
                                     #initializer=tf.constant_initializer(ALEX_DICT["alex_branch/conv2d_2/bias"]),
                                     initializer=tf.constant_initializer(0.0),
                                     trainable=True)
        conv4_sq = tf.nn.conv2d(conv4_avg, conv4_sq_W, [1,1,1,1], padding='VALID')
        conv4_sq = tf.nn.bias_add(conv4_sq, conv4_sq_b)
        #conv4_sq = tf.layers.conv2d(conv4_avg, 96, [1,1]) #(B, 1, 1, 96)
        conv4_sq = tf.nn.relu(conv4_sq)
        conv4_ex_W = tf.get_variable("conv4_ex_W", [1,1,96,384], dtype=tf.float32, 
                                     #initializer=tf.constant_initializer(ALEX_DICT["alex_branch/conv2d_3/kernel"]),
                                     initializer=tf.contrib.layers.xavier_initializer(),
                                     trainable=True)
        conv4_ex_b = tf.get_variable("conv4_ex_b", [384], dtype=tf.float32, 
                                     #initializer=tf.constant_initializer(ALEX_DICT["alex_branch/conv2d_3/bias"]),
                                     initializer=tf.constant_initializer(0.0),
                                     trainable=True)
        conv4_ex = tf.nn.conv2d(conv4_sq, conv4_ex_W, [1,1,1,1], padding='VALID')
        conv4_ex = tf.nn.bias_add(conv4_ex, conv4_ex_b)
        #conv4_ex = tf.layers.conv2d(conv4_sq, 384, [1,1]) #(B, 1, ,1 384)
        conv4_ex = tf.nn.sigmoid(conv4_ex)
        
        
        conv5_shape = conv5.get_shape().as_list()[1:3]
        conv5_avg = tf.layers.average_pooling2d(conv5, conv5_shape, [1,1]) #(B, 1, 1, 384)
        conv5_sq_W = tf.get_variable("conv5_sq_W", [1,1,256,64], dtype=tf.float32, 
                                     #initializer=tf.constant_initializer(ALEX_DICT["alex_branch/conv2d_4/kernel"]),
                                     initializer=tf.contrib.layers.xavier_initializer(),
                                     trainable=True)
        conv5_sq_b = tf.get_variable("conv5_sq_b", [64], dtype=tf.float32, 
                                     #initializer=tf.constant_initializer(ALEX_DICT["alex_branch/conv2d_4/bias"]),
                                     initializer=tf.constant_initializer(0.0),
                                     trainable=True)
        conv5_sq = tf.nn.conv2d(conv5_avg, conv5_sq_W, [1,1,1,1], padding='VALID')
        conv5_sq = tf.nn.bias_add(conv5_sq, conv5_sq_b)
        #conv5_sq = tf.layers.conv2d(conv5_avg, 64, [1,1]) #(B, 1, 1, 64)
        conv5_sq = tf.nn.relu(conv5_sq)
        conv5_ex_W = tf.get_variable("conv5_ex_W", [1,1,64,256], dtype=tf.float32, 
                                     #initializer=tf.constant_initializer(ALEX_DICT["alex_branch/conv2d_5/kernel"]),
                                     initializer=tf.contrib.layers.xavier_initializer(),
                                     trainable=True)
        conv5_ex_b = tf.get_variable("conv5_ex_b", [256], dtype=tf.float32, 
                                     #initializer=tf.constant_initializer(ALEX_DICT["alex_branch/conv2d_5/bias"]),
                                     initializer=tf.constant_initializer(0.0),
                                     trainable=True)
        conv5_ex = tf.nn.conv2d(conv5_sq, conv5_ex_W, [1,1,1,1], padding='VALID')
        conv5_ex = tf.nn.bias_add(conv5_ex, conv5_ex_b)
        #conv5_ex = tf.layers.conv2d(conv5_sq, 256, [1,1]) #(B, 1, ,1 384)
        conv5_ex = tf.nn.sigmoid(conv5_ex)
        
        conv3_se = conv3*conv3_ex
        conv4_se = conv4*conv4_ex
        conv5_se = conv5*conv5_ex
        
# =============================================================================
#         ####
#         conv3_sa_f_W = tf.get_variable("conv3_sa_f_W", [1,1,384,384], dtype=tf.float32,
#                                        #initializer=tf.constant_initializer(ALEX_DICT["alex_branch/conv3_sa_f_W"]),
#                                        initializer=tf.contrib.layers.xavier_initializer(),
#                                        trainable=True)
#         conv3_sa_f_b = tf.get_variable("conv3_sa_f_b", [384], dtype=tf.float32,
#                                        #initializer=tf.constant_initializer(ALEX_DICT["alex_branch/conv3_sa_f_b"]),
#                                        initializer=tf.constant_initializer(0.0),
#                                        trainable=True)
#         conv3_sa_f = tf.nn.conv2d(conv3_se, conv3_sa_f_W, [1,1,1,1], padding="VALID")
#         conv3_sa_f = tf.nn.bias_add(conv3_sa_f, conv3_sa_f_b)
#         
#         conv3_sa_g_W = tf.get_variable("conv3_sa_g_W", [1,1,384,384], dtype=tf.float32,
#                                        #initializer=tf.constant_initializer(ALEX_DICT["alex_branch/conv3_sa_g_W"]),
#                                        initializer=tf.contrib.layers.xavier_initializer(),
#                                        trainable=True)
#         conv3_sa_g_b = tf.get_variable("conv3_sa_g_b", [384], dtype=tf.float32,
#                                        #initializer=tf.constant_initializer(ALEX_DICT["alex_branch/conv3_sa_g_b"]),
#                                        initializer=tf.constant_initializer(0.0),
#                                        trainable=True)
#         conv3_sa_g = tf.nn.conv2d(conv3_se, conv3_sa_g_W, [1,1,1,1], padding="VALID")
#         conv3_sa_g = tf.nn.bias_add(conv3_sa_g, conv3_sa_g_b)
#         
#         conv3_sa_h_W = tf.get_variable("conv3_sa_h_W", [1,1,384,384], dtype=tf.float32,
#                                        #initializer=tf.constant_initializer(ALEX_DICT["alex_branch/conv3_sa_h_W"]),
#                                        initializer=tf.contrib.layers.xavier_initializer(),
#                                        trainable=True)
#         conv3_sa_h_b = tf.get_variable("conv3_sa_h_b", [384], dtype=tf.float32,
#                                        #initializer=tf.constant_initializer(ALEX_DICT["alex_branch/conv3_sa_h_b"]),
#                                        initializer=tf.constant_initializer(0.0),
#                                        trainable=True)
#         conv3_sa_h = tf.nn.conv2d(conv3_se, conv3_sa_h_W, [1,1,1,1], padding="VALID")
#         conv3_sa_h = tf.nn.bias_add(conv3_sa_h, conv3_sa_h_b)
#         conv3_sa_ft = tf.transpose(conv3_sa_f, perm=[0,2,1,3])
#         conv3_sa_ft = tf.transpose(conv3_sa_ft, perm=[0,3,1,2])
#         conv3_sa_g = tf.transpose(conv3_sa_g, perm=[0,3,1,2])
#         conv3_sa_h = tf.transpose(conv3_sa_h, perm=[0,3,1,2])
#         conv3_sa_ftg = tf.matmul(conv3_sa_ft, conv3_sa_g)
#         conv3_sa_ftg = tf.nn.softmax(conv3_sa_ftg)
#         conv3_sa_o = tf.matmul(conv3_sa_ftg, conv3_sa_h)
#         conv3_sa_o = tf.transpose(conv3_sa_o, perm=[0,2,3,1])
#         conv3_sa_gamma = tf.get_variable("conv3_sa_gamma", [1], dtype=tf.float32,
#                                          #initializer=tf.constant_initializer(ALEX_DICT["alex_branch/conv3_sa_gamma"]),
#                                          initializer=tf.constant_initializer(0.0),
#                                          trainable=True)
#         conv3_sa = tf.multiply(conv3_sa_o, conv3_sa_gamma)+conv3_se
# =============================================================================
        
        
# =============================================================================
#         ####
#         conv4_sa_f_W = tf.get_variable("conv4_sa_f_W", [1,1,384,384], dtype=tf.float32,
#                                        #initializer=tf.constant_initializer(ALEX_DICT["alex_branch/conv4_sa_f_W"]),
#                                        initializer=tf.contrib.layers.xavier_initializer(),
#                                        trainable=True)
#         conv4_sa_f_b = tf.get_variable("conv4_sa_f_b", [384], dtype=tf.float32,
#                                        #initializer=tf.constant_initializer(ALEX_DICT["alex_branch/conv4_sa_f_b"]),
#                                        initializer=tf.constant_initializer(0.0),
#                                        trainable=True)
#         conv4_sa_f = tf.nn.conv2d(conv4_se, conv4_sa_f_W, [1,1,1,1], padding="VALID")
#         conv4_sa_f = tf.nn.bias_add(conv4_sa_f, conv4_sa_f_b)
#     
#         conv4_sa_g_W = tf.get_variable("conv4_sa_g_W", [1,1,384,384], dtype=tf.float32,
#                                        #initializer=tf.constant_initializer(ALEX_DICT["alex_branch/conv4_sa_g_W"]),
#                                        initializer=tf.contrib.layers.xavier_initializer(),
#                                        trainable=True)
#         conv4_sa_g_b = tf.get_variable("conv4_sa_g_b", [384], dtype=tf.float32,
#                                        #initializer=tf.constant_initializer(ALEX_DICT["alex_branch/conv4_sa_g_b"]),
#                                        initializer=tf.constant_initializer(0.0),
#                                        trainable=True)
#         conv4_sa_g = tf.nn.conv2d(conv4_se, conv4_sa_g_W, [1,1,1,1], padding="VALID")
#         conv4_sa_g = tf.nn.bias_add(conv4_sa_g, conv4_sa_g_b)
#         
#         conv4_sa_h_W = tf.get_variable("conv4_sa_h_W", [1,1,384,384], dtype=tf.float32,
#                                        #initializer=tf.constant_initializer(ALEX_DICT["alex_branch/conv4_sa_h_W"]),
#                                        initializer=tf.contrib.layers.xavier_initializer(),
#                                        trainable=True)
#         conv4_sa_h_b = tf.get_variable("conv4_sa_h_b", [384], dtype=tf.float32,
#                                        #initializer=tf.constant_initializer(ALEX_DICT["alex_branch/conv4_sa_h_b"]),
#                                        initializer=tf.constant_initializer(0.0),
#                                        trainable=True)
#         conv4_sa_h = tf.nn.conv2d(conv4_se, conv4_sa_h_W, [1,1,1,1], padding="VALID")
#         conv4_sa_h = tf.nn.bias_add(conv4_sa_h, conv4_sa_h_b)
#         conv4_sa_ft = tf.transpose(conv4_sa_f, perm=[0,2,1,3])
#         conv4_sa_ft = tf.transpose(conv4_sa_ft, perm=[0,3,1,2])
#         conv4_sa_g = tf.transpose(conv4_sa_g, perm=[0,3,1,2])
#         conv4_sa_h = tf.transpose(conv4_sa_h, perm=[0,3,1,2])
#         conv4_sa_ftg = tf.matmul(conv4_sa_ft, conv4_sa_g)
#         conv4_sa_ftg = tf.nn.softmax(conv4_sa_ftg)
#         conv4_sa_o = tf.matmul(conv4_sa_ftg, conv4_sa_h)
#         conv4_sa_o = tf.transpose(conv4_sa_o, perm=[0,2,3,1])
#         conv4_sa_gamma = tf.get_variable("conv4_sa_gamma", [1], dtype=tf.float32,
#                                          #initializer=tf.constant_initializer(ALEX_DICT["alex_branch/conv4_sa_gamma"]),
#                                          initializer=tf.constant_initializer(0.0),
#                                          trainable=True)
#         conv4_sa = tf.multiply(conv4_sa_o, conv4_sa_gamma)+conv4_se
# =============================================================================
        
        
# =============================================================================
#         ####
#         conv5_sa_f_W = tf.get_variable("conv5_sa_f_W", [1,1,256,256], dtype=tf.float32,
#                                        #initializer=tf.constant_initializer(ALEX_DICT["alex_branch/conv5_sa_f_W"]),
#                                        initializer=tf.contrib.layers.xavier_initializer(),
#                                        trainable=True)
#         conv5_sa_f_b = tf.get_variable("conv5_sa_f_b", [256], dtype=tf.float32,
#                                        #initializer=tf.constant_initializer(ALEX_DICT["alex_branch/conv5_sa_f_b"]),
#                                        initializer=tf.constant_initializer(0.0),
#                                        trainable=True)
#         conv5_sa_f = tf.nn.conv2d(conv5_se, conv5_sa_f_W, [1,1,1,1], padding="VALID")
#         conv5_sa_f = tf.nn.bias_add(conv5_sa_f, conv5_sa_f_b)
#     
#         conv5_sa_g_W = tf.get_variable("conv5_sa_g_W", [1,1,256,256], dtype=tf.float32,
#                                        #initializer=tf.constant_initializer(ALEX_DICT["alex_branch/conv5_sa_g_W"]),
#                                        initializer=tf.contrib.layers.xavier_initializer(),
#                                        trainable=True)
#         conv5_sa_g_b = tf.get_variable("conv5_sa_g_b", [256], dtype=tf.float32,
#                                        #initializer=tf.constant_initializer(ALEX_DICT["alex_branch/conv5_sa_g_b"]),
#                                        initializer=tf.constant_initializer(0.0),
#                                        trainable=True)
#         conv5_sa_g = tf.nn.conv2d(conv5_se, conv5_sa_g_W, [1,1,1,1], padding="VALID")
#         conv5_sa_g = tf.nn.bias_add(conv5_sa_g, conv5_sa_g_b)
#         
#         conv5_sa_h_W = tf.get_variable("conv5_sa_h_W", [1,1,256,256], dtype=tf.float32,
#                                        #initializer=tf.constant_initializer(ALEX_DICT["alex_branch/conv5_sa_h_W"]),
#                                        initializer=tf.contrib.layers.xavier_initializer(),
#                                        trainable=True)
#         conv5_sa_h_b = tf.get_variable("conv5_sa_h_b", [256], dtype=tf.float32,
#                                        #initializer=tf.constant_initializer(ALEX_DICT["alex_branch/conv5_sa_h_b"]),
#                                        initializer=tf.constant_initializer(0.0),
#                                        trainable=True)
#         conv5_sa_h = tf.nn.conv2d(conv5_se, conv5_sa_h_W, [1,1,1,1], padding="VALID")
#         conv5_sa_h = tf.nn.bias_add(conv5_sa_h, conv5_sa_h_b)
#         conv5_sa_ft = tf.transpose(conv5_sa_f, perm=[0,2,1,3])
#         conv5_sa_ft = tf.transpose(conv5_sa_ft, perm=[0,3,1,2])
#         conv5_sa_g = tf.transpose(conv5_sa_g, perm=[0,3,1,2])
#         conv5_sa_h = tf.transpose(conv5_sa_h, perm=[0,3,1,2])
#         conv5_sa_ftg = tf.matmul(conv5_sa_ft, conv5_sa_g)
#         conv5_sa_ftg = tf.nn.softmax(conv5_sa_ftg)
#         conv5_sa_o = tf.matmul(conv5_sa_ftg, conv5_sa_h)
#         conv5_sa_o = tf.transpose(conv5_sa_o, perm=[0,2,3,1])
#         conv5_sa_gamma = tf.get_variable("conv5_sa_gamma", [1], dtype=tf.float32,
#                                          #initializer=tf.constant_initializer(ALEX_DICT["alex_branch/conv5_sa_gamma"]),
#                                          initializer=tf.constant_initializer(0.0),
#                                          trainable=True)
#         conv5_sa = tf.multiply(conv5_sa_o, conv5_sa_gamma)+conv5_se
# =============================================================================
        
        return conv5_se, conv4_se, conv3_se
# =============================================================================
#         return conv5, conv4, conv3
# =============================================================================
# =============================================================================
#         return conv5_sa, conv4_sa, conv3_sa
# =============================================================================
    
    
def get_pretrained_vgg16():
    VGG16_PATH = "/home/travail/dev/GitRepo/pretrained_model/vgg16_weights.npz"
    weights = np.load(VGG16_PATH)
    #keys = sorted(weights.keys)
    #weights['conv1_1_W']
    return weights

def embed_vgg16(images):
# http://www.cs.toronto.edu/~frossard/post/vgg16/
# https://www.cs.toronto.edu/~frossard/vgg16/vgg16.py
    with tf.variable_scope("vgg16_branch", reuse=tf.AUTO_REUSE):
        with tf.name_scope('preprocess') as scope:
            mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1,1,1,3],name="img_mean")
            images=images-mean
        net_data = get_pretrained_vgg16()
        
        # conv1_1
        conv1_1_W = tf.get_variable("conv1_1_W", [3,3,3,64], dtype=tf.float32,
                                    initializer=tf.constant_initializer(net_data['conv1_1_W']),
                                    trainable=False)
        conv1_1_b = tf.get_variable("conv1_1_b", [64], dtype=tf.float32,
                                    initializer=tf.constant_initializer(net_data['conv1_1_b']),
                                    trainable=False)
        conv1_1 = tf.nn.conv2d(images, conv1_1_W, [1,1,1,1], padding="SAME")
        conv1_1 = tf.nn.bias_add(conv1_1, conv1_1_b)
        conv1_1 = tf.nn.relu(conv1_1)
        
        # conv1_2
        conv1_2_W = tf.get_variable("conv1_2_W", [3,3,64,64], dtype=tf.float32,
                                    initializer=tf.constant_initializer(net_data['conv1_2_W']),
                                    trainable=False)
        conv1_2_b = tf.get_variable("conv1_2_b", [64], dtype=tf.float32,
                                    initializer=tf.constant_initializer(net_data['conv1_2_b']),
                                    trainable=False)
        conv1_2 = tf.nn.conv2d(conv1_1, conv1_2_W, [1,1,1,1], padding="SAME")
        conv1_2 = tf.nn.bias_add(conv1_2, conv1_2_b)
        conv1_2 = tf.nn.relu(conv1_2)
        
        # pool1
        pool1 = tf.nn.max_pool(conv1_2, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME", name="pool1")
        
        # conv2_1
        conv2_1_W = tf.get_variable("conv2_1_W", [3,3,64,128], dtype=tf.float32,
                                    initializer=tf.constant_initializer(net_data['conv2_1_W']),
                                    trainable=False)
        conv2_1_b = tf.get_variable("conv2_1_b", [128], dtype=tf.float32,
                                    initializer=tf.constant_initializer(net_data['conv2_1_b']),
                                    trainable=False)
        conv2_1 = tf.nn.conv2d(pool1, conv2_1_W, [1,1,1,1], padding="SAME")
        conv2_1 = tf.nn.bias_add(conv2_1, conv2_1_b)
        conv2_1 = tf.nn.relu(conv2_1)
        
        # conv2_2
        conv2_2_W = tf.get_variable("conv2_2_W", [3,3,128,128], dtype=tf.float32,
                                    initializer=tf.constant_initializer(net_data['conv2_2_W']),
                                    trainable=False)
        conv2_2_b = tf.get_variable("conv2_2_b", [128], dtype=tf.float32,
                                    initializer=tf.constant_initializer(net_data['conv2_2_b']),
                                    trainable=False)
        conv2_2 = tf.nn.conv2d(conv2_1, conv2_2_W, [1,1,1,1], padding="SAME")
        conv2_2 = tf.nn.bias_add(conv2_2, conv2_2_b)
        conv2_2 = tf.nn.relu(conv2_2)
        
        # pool2
        pool2 = tf.nn.max_pool(conv2_2, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME", name="pool2")
        
        # conv3_1
        conv3_1_W = tf.get_variable("conv3_1_W", [3,3,128,256], dtype=tf.float32,
                                    initializer=tf.constant_initializer(net_data['conv3_1_W']),
                                    trainable=False)
        conv3_1_b = tf.get_variable("conv3_1_b", [256], dtype=tf.float32,
                                    initializer=tf.constant_initializer(net_data['conv3_1_b']),
                                    trainable=False)
        conv3_1 = tf.nn.conv2d(pool2, conv3_1_W, [1,1,1,1], padding="SAME")
        conv3_1 = tf.nn.bias_add(conv3_1, conv3_1_b)
        conv3_1 = tf.nn.relu(conv3_1)
        
        # conv3_2
        conv3_2_W = tf.get_variable("conv3_2_W", [3,3,256,256], dtype=tf.float32,
                                    initializer=tf.constant_initializer(net_data['conv3_2_W']),
                                    trainable=False)
        conv3_2_b = tf.get_variable("conv3_2_b", [256], dtype=tf.float32,
                                    initializer=tf.constant_initializer(net_data['conv3_2_b']),
                                    trainable=False)
        conv3_2 = tf.nn.conv2d(conv3_1, conv3_2_W, [1,1,1,1], padding="SAME")
        conv3_2 = tf.nn.bias_add(conv3_2, conv3_2_b)
        conv3_2 = tf.nn.relu(conv3_2)
        
        # conv3_3
        conv3_3_W = tf.get_variable("conv3_3_W", [3,3,256,256], dtype=tf.float32,
                                    initializer=tf.constant_initializer(net_data['conv3_3_W']),
                                    trainable=False)
        conv3_3_b = tf.get_variable("conv3_3_b", [256], dtype=tf.float32,
                                    initializer=tf.constant_initializer(net_data['conv3_3_b']),
                                    trainable=False)
        conv3_3 = tf.nn.conv2d(conv3_2, conv3_3_W, [1,1,1,1], padding="SAME")
        conv3_3 = tf.nn.bias_add(conv3_3, conv3_3_b)
        conv3_3 = tf.nn.relu(conv3_3)
        
        # pool3
        pool3 = tf.nn.max_pool(conv3_3, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME", name="pool3")
        
        # conv4_1
        conv4_1_W = tf.get_variable("conv4_1_W", [3,3,256,512], dtype=tf.float32,
                                    initializer=tf.constant_initializer(net_data['conv4_1_W']),
                                    trainable=False)
        conv4_1_b = tf.get_variable("conv4_1_b", [512], dtype=tf.float32,
                                    initializer=tf.constant_initializer(net_data['conv4_1_b']),
                                    trainable=False)
        conv4_1 = tf.nn.conv2d(pool3, conv4_1_W, [1,1,1,1], padding="SAME")
        conv4_1 = tf.nn.bias_add(conv4_1, conv4_1_b)
        conv4_1 = tf.nn.relu(conv4_1)
        
        # conv4_2
        conv4_2_W = tf.get_variable("conv4_2_W", [3,3,512,512], dtype=tf.float32,
                                    initializer=tf.constant_initializer(net_data['conv4_2_W']),
                                    trainable=False)
        conv4_2_b = tf.get_variable("conv4_2_b", [512], dtype=tf.float32,
                                    initializer=tf.constant_initializer(net_data['conv4_2_b']),
                                    trainable=False)
        conv4_2 = tf.nn.conv2d(conv4_1, conv4_2_W, [1,1,1,1], padding="SAME")
        conv4_2 = tf.nn.bias_add(conv4_2, conv4_2_b)
        conv4_2 = tf.nn.relu(conv4_2)
        
        # conv4_3
        conv4_3_W = tf.get_variable("conv4_3_W", [3,3,512,512], dtype=tf.float32,
                                    initializer=tf.constant_initializer(net_data['conv4_3_W']),
                                    trainable=False)
        conv4_3_b = tf.get_variable("conv4_3_b", [512], dtype=tf.float32,
                                    initializer=tf.constant_initializer(net_data['conv4_3_b']),
                                    trainable=False)
        conv4_3 = tf.nn.conv2d(conv4_2, conv4_3_W, [1,1,1,1], padding="SAME")
        conv4_3 = tf.nn.bias_add(conv4_3, conv4_3_b)
        conv4_3 = tf.nn.relu(conv4_3)
        
        # pool4
        pool4 = tf.nn.max_pool(conv4_3, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME", name="pool4")
        
        # conv5_1
        conv5_1_W = tf.get_variable("conv5_1_W", [3,3,512,512], dtype=tf.float32,
                                    initializer=tf.constant_initializer(net_data['conv5_1_W']),
                                    trainable=False)
        conv5_1_b = tf.get_variable("conv5_1_b", [512], dtype=tf.float32,
                                    initializer=tf.constant_initializer(net_data['conv5_1_b']),
                                    trainable=False)
        conv5_1 = tf.nn.conv2d(pool4, conv5_1_W, [1,1,1,1], padding="SAME")
        conv5_1 = tf.nn.bias_add(conv5_1, conv5_1_b)
        conv5_1 = tf.nn.relu(conv5_1)
        
        # conv5_2
        conv5_2_W = tf.get_variable("conv5_2_W", [3,3,512,512], dtype=tf.float32,
                                    initializer=tf.constant_initializer(net_data['conv5_2_W']),
                                    trainable=False)
        conv5_2_b = tf.get_variable("conv5_2_b", [512], dtype=tf.float32,
                                    initializer=tf.constant_initializer(net_data['conv5_2_b']),
                                    trainable=False)
        conv5_2 = tf.nn.conv2d(conv5_1, conv5_2_W, [1,1,1,1], padding="SAME")
        conv5_2 = tf.nn.bias_add(conv5_2, conv5_2_b)
        conv5_2 = tf.nn.relu(conv5_2)
        
        # conv5_3
        conv5_3_W = tf.get_variable("conv5_3_W", [3,3,512,512], dtype=tf.float32,
                                    initializer=tf.constant_initializer(net_data['conv5_3_W']),
                                    trainable=False)
        conv5_3_b = tf.get_variable("conv5_3_b", [512], dtype=tf.float32,
                                    initializer=tf.constant_initializer(net_data['conv5_3_b']),
                                    trainable=False)
        conv5_3 = tf.nn.conv2d(conv5_2, conv5_3_W, [1,1,1,1], padding="SAME")
        conv5_3 = tf.nn.bias_add(conv5_3, conv5_3_b)
        conv5_3 = tf.nn.relu(conv5_3)
        
        # pool5
        pool5 = tf.nn.max_pool(conv5_3, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME", name="pool5")
        
        
        return conv5_3, conv5_2, conv5_1
        
        
        
        
        
        
        
        
        
        
    
