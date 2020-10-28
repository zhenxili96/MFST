#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2017 bily     Huazhong University of Science and Technology
#
# Distributed under terms of the MIT license.

"""Construct the computational graph of siamese model for training. """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import tensorflow as tf

from datasets.dataloader import DataLoader
from embeddings.convolutional_alexnet import convolutional_alexnet_arg_scope, convolutional_alexnet, embed_alexnet
from metrics.track_metrics import center_dist_error, center_score_error
from utils.train_utils import construct_gt_score_maps, load_mat_model

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


class SiameseModel:
  def __init__(self, model_config, train_config, mode='train'):
    self.model_config = model_config
    self.train_config = train_config
    self.mode = mode
    assert mode in ['train', 'validation', 'inference']

    if self.mode == 'train':
      self.data_config = self.train_config['train_data_config']
    elif self.mode == 'validation':
      self.data_config = self.train_config['validation_data_config']

    self.dataloader = None
    self.exemplars = None
    self.instances = None
    self.response = None
    self.batch_loss = None
    self.total_loss = None
    self.init_fn = None
    self.global_step = None

  def is_training(self):
    """Returns true if the model is built for training mode"""
    return self.mode == 'train'

  def build_inputs(self):
    """Input fetching and batching

    Outputs:
      self.exemplars: image batch of shape [batch, hz, wz, 3]
      self.instances: image batch of shape [batch, hx, wx, 3]
    """
    if self.mode in ['train', 'validation']:
      with tf.device("/cpu:0"):  # Put data loading and preprocessing in CPU is substantially faster
        self.dataloader = DataLoader(self.data_config, self.is_training())
        self.dataloader.build()
        exemplars, instances = self.dataloader.get_one_batch()

        exemplars = tf.to_float(exemplars)
        instances = tf.to_float(instances)
    else:
      self.examplar_feed = tf.placeholder(shape=[None, None, None, 3],
                                          dtype=tf.uint8,
                                          name='examplar_input')
      self.instance_feed = tf.placeholder(shape=[None, None, None, 3],
                                          dtype=tf.uint8,
                                          name='instance_input')
      exemplars = tf.to_float(self.examplar_feed)
      instances = tf.to_float(self.instance_feed)

    self.exemplars = exemplars
    self.instances = instances

  def build_image_embeddings(self, reuse=False):
    """Builds the image model subgraph and generates image embeddings

    Inputs:
      self.exemplars: A tensor of shape [batch, hz, wz, 3]
      self.instances: A tensor of shape [batch, hx, wx, 3]

    Outputs:
      self.exemplar_embeds: A Tensor of shape [batch, hz_embed, wz_embed, embed_dim]
      self.instance_embeds: A Tensor of shape [batch, hx_embed, wx_embed, embed_dim]
    """
    config = self.model_config['embed_config']
# =============================================================================
#     arg_scope = convolutional_alexnet_arg_scope(config,
#                                                 trainable=config['train_embedding'],
#                                                 is_training=self.is_training())
# =============================================================================
    arg_scope = convolutional_alexnet_arg_scope(config,
                                                trainable=False,
                                                is_training=False)

    @functools.wraps(convolutional_alexnet)
    def embedding_fn(images, reuse=False):
      with slim.arg_scope(arg_scope):
        return convolutional_alexnet(images, reuse=reuse)

    self.exemplar_embeds_c5, self.exemplar_embeds_c4, self.exemplar_embeds_c3, _ = embedding_fn(self.exemplars, reuse=reuse)
    self.instance_embeds_c5, self.instance_embeds_c4, self.instance_embeds_c3, _ = embedding_fn(self.instances, reuse=True)
    
# =============================================================================
#     self.exemplar_embeds_c5, self.exemplar_embeds_c4, self.exemplar_embeds_c3 = embed_alexnet(self.exemplars)
#     self.instance_embeds_c5, self.instance_embeds_c4, self.instance_embeds_c3 = embed_alexnet(self.instances)
# =============================================================================
    

  def build_template(self):
    # The template is simply the feature of the exemplar image in SiamFC.
    self.templates_c5 = self.exemplar_embeds_c5
    self.templates_c4 = self.exemplar_embeds_c4
    self.templates_c3 = self.exemplar_embeds_c3

  def build_detection(self, reuse=False):
    with tf.variable_scope('detection', reuse=reuse):
      def _translation_match(x, z):  # translation match for one example within a batch
        x = tf.expand_dims(x, 0)  # [1, in_height, in_width, in_channels]
        z = tf.expand_dims(z, -1)  # [filter_height, filter_width, in_channels, 1]
        return tf.nn.conv2d(x, z, strides=[1, 1, 1, 1], padding='VALID', name='translation_match')

      output_c5 = tf.map_fn(lambda x: _translation_match(x[0], x[1]),
                         (self.instance_embeds_c5, self.templates_c5),
                         dtype=self.instance_embeds_c5.dtype)
      output_c4 = tf.map_fn(lambda x: _translation_match(x[0], x[1]),
                         (self.instance_embeds_c4, self.templates_c4),
                         dtype=self.instance_embeds_c4.dtype)
      output_c3 = tf.map_fn(lambda x: _translation_match(x[0], x[1]),
                         (self.instance_embeds_c3, self.templates_c3),
                         dtype=self.instance_embeds_c3.dtype)
      output_c5 = tf.squeeze(output_c5, [1, 4])  # of shape e.g., [8, 15, 15]
      output_c4 = tf.squeeze(output_c4, [1, 4])  # of shape e.g., [8, 15, 15]
      output_c3 = tf.squeeze(output_c3, [1, 4])  # of shape e.g., [8, 15, 15]
      print_op_c5 = tf.Print(output_c5,[output_c5])
      print_op_c4 = tf.Print(output_c4,[output_c4])
      print_op_c3 = tf.Print(output_c3,[output_c3])

      #ALEX_DICT = parse_tf_model("/home/travail/dev/GitRepo/CBSiamFC/Logs/SiamFC/track_model_checkpoints/Alex_v1/model.ckpt-332499")
      # Adjust score, this is required to make training possible.
      config = self.model_config['adjust_response_config']
# =============================================================================
#       self.bias_c5 = tf.get_variable('biases_a_c5', [1],    #[-9.63422871]
#                              dtype=tf.float32,
#                              initializer=tf.constant_initializer(0.0),#=tf.constant_initializer(ALEX_DICT['detection/biases_c5']),
#                              trainable=config['train_bias'])
#       self.bias_c4 = tf.get_variable('biases_a_c4', [1],    #[-5.29178524]
#                              dtype=tf.float32,
#                              initializer=tf.constant_initializer(0.0),#tf.constant_initializer(ALEX_DICT['detection/biases_c4']),
#                              trainable=config['train_bias'])
#       self.bias_c3 = tf.get_variable('biases_a_c3', [1],    #[-4.51134348]
#                              dtype=tf.float32,
#                              initializer=tf.constant_initializer(0.0),#tf.constant_initializer(ALEX_DICT['detection/biases_c3']),
#                              trainable=config['train_bias'])
# =============================================================================
      self.bias_c5 = tf.get_variable('biases_s_c5', [1],    #[-9.63422871]
                             dtype=tf.float32,
                             initializer=tf.constant_initializer(0.0),#=tf.constant_initializer(ALEX_DICT['detection/biases_c5']),
                             trainable=config['train_bias'])
      self.bias_c4 = tf.get_variable('biases_s_c4', [1],    #[-5.29178524]
                             dtype=tf.float32,
                             initializer=tf.constant_initializer(0.0),#tf.constant_initializer(ALEX_DICT['detection/biases_c4']),
                             trainable=config['train_bias'])
      self.bias_c3 = tf.get_variable('biases_s_c3', [1],    #[-4.51134348]
                             dtype=tf.float32,
                             initializer=tf.constant_initializer(0.0),#tf.constant_initializer(ALEX_DICT['detection/biases_c3']),
                             trainable=config['train_bias'])
# =============================================================================
#       with tf.control_dependencies([print_op_c3]):#455
# =============================================================================
# =============================================================================
#       response_c5 = 1e-2*output_c5 + self.bias_c5
#       response_c4 = 1e-3*output_c4 + self.bias_c4
#       response_c3 = 1e-3*output_c3 + self.bias_c3
# =============================================================================
# =============================================================================
#       ## for training alexnet se
#       response_c5 = 1e-3*output_c5 + self.bias_c5
#       response_c4 = 1e-4*output_c4 + self.bias_c4
#       response_c3 = 1e-5*output_c3 + self.bias_c3
# =============================================================================
      ## for training siamfc se
      response_c5 = 1e-3 * output_c5 + self.bias_c5
      response_c4 = 1e-3 * output_c4 + self.bias_c4
      response_c3 = 1e-3 * output_c3 + self.bias_c3
# =============================================================================
#       response_c5 = config['scale'] * output_c5 -4.52620411
#       response_c4 = config['scale'] * output_c4 -0.03678114
#       response_c3 = config['scale'] * output_c3 -0.49341503
# =============================================================================
      
      # weight maps for response from each layer
# =============================================================================
#       response_size = response_c5.get_shape().as_list()[1:3]
#       map_c5 = tf.get_variable('map_c5', response_size,
#                                     dtype=tf.float32,
#                                     initializer=tf.constant_initializer(0.5, dtype=tf.float32),
#                                     trainable=True)
#       map_c4 = tf.get_variable('map_c4', response_size,
#                                     dtype=tf.float32,
#                                     initializer=tf.constant_initializer(0.5, dtype=tf.float32),
#                                     trainable=True)
#       map_c3 = tf.get_variable('map_c3', response_size,
#                                     dtype=tf.float32,
#                                     initializer=tf.constant_initializer(0.5, dtype=tf.float32),
#                                     trainable=True)
# =============================================================================
# =============================================================================
#       self.weight_c5 = tf.get_variable('weight_a_c5', [1], #[ 0.71658146]
#                              dtype=tf.float32,
#                              initializer=tf.constant_initializer(0.5, dtype=tf.float32),
#                              trainable=True)
#       self.weight_c4 = tf.get_variable('weight_a_c4', [1], #[ 0.04511292]
#                              dtype=tf.float32,
#                              initializer=tf.constant_initializer(0.5, dtype=tf.float32),
#                              trainable=True)
#       self.weight_c3 = tf.get_variable('weight_a_c3', [1], #[ 0.0067619]
#                              dtype=tf.float32,
#                              initializer=tf.constant_initializer(0.5, dtype=tf.float32),
#                              trainable=True)
#       response_c5 = tf.multiply(response_c5, self.weight_c5)
#       response_c4 = tf.multiply(response_c4, self.weight_c4)
#       response_c3 = tf.multiply(response_c3, self.weight_c3)
# =============================================================================
# =============================================================================
#       response_c5_max = tf.reduce_max(response_c5)
#       response_c4_max = tf.reduce_max(response_c4)
#       response_c3_max = tf.reduce_max(response_c3)
#       self.response_c5 = tf.div(response_c5, response_c5_max)
#       self.response_c4 = tf.div(response_c4, response_c4_max)
#       self.response_c3 = tf.div(response_c3, response_c3_max)
#       self.response = 0.3*response_c5+0.6*response_c4+0.1*response_c3
# =============================================================================
# =============================================================================
#       self.response = response_c5*0.6+response_c4*0.3+response_c3*0.1
# =============================================================================
      self.response_c5 = response_c5
      self.response_c4 = response_c4
      self.response_c3 = response_c3
      

  def build_loss(self):
    response_c5 = self.response_c5
    response_c4 = self.response_c4
    response_c3 = self.response_c3
    response_size = response_c5.get_shape().as_list()[1:3]  # [height, width]

    gt = construct_gt_score_maps(response_size,
                                 self.data_config['batch_size'],
                                 self.model_config['embed_config']['stride'],
                                 self.train_config['gt_config'])

    with tf.name_scope('Loss'):
      loss_c5 = tf.nn.sigmoid_cross_entropy_with_logits(logits=response_c5,
                                                     labels=gt)
      loss_c4 = tf.nn.sigmoid_cross_entropy_with_logits(logits=response_c4,
                                                     labels=gt)
      loss_c3 = tf.nn.sigmoid_cross_entropy_with_logits(logits=response_c3,
                                                     labels=gt)

      with tf.name_scope('Balance_weights'):
        n_pos = tf.reduce_sum(tf.to_float(tf.equal(gt[0], 1)))
        n_neg = tf.reduce_sum(tf.to_float(tf.equal(gt[0], 0)))
        w_pos = 0.5 / n_pos
        w_neg = 0.5 / n_neg
        class_weights = tf.where(tf.equal(gt, 1),
                                 w_pos * tf.ones_like(gt),
                                 tf.ones_like(gt))
        class_weights = tf.where(tf.equal(gt, 0),
                                 w_neg * tf.ones_like(gt),
                                 class_weights)
        loss_c5 = loss_c5 * class_weights
        loss_c4 = loss_c4 * class_weights
        loss_c3 = loss_c3 * class_weights

      # Note that we use reduce_sum instead of reduce_mean since the loss has
      # already been normalized by class_weights in spatial dimension.
      loss_c5 = tf.reduce_sum(loss_c5, [1, 2])
      loss_c4 = tf.reduce_sum(loss_c4, [1, 2])
      loss_c3 = tf.reduce_sum(loss_c3, [1, 2])

      batch_loss_c5 = tf.reduce_mean(loss_c5, name='batch_loss_c5')
      batch_loss_c4 = tf.reduce_mean(loss_c4, name='batch_loss_c4')
      batch_loss_c3 = tf.reduce_mean(loss_c3, name='batch_loss_c3')
      tf.losses.add_loss(batch_loss_c5)
      tf.losses.add_loss(batch_loss_c4)
      tf.losses.add_loss(batch_loss_c3)

      total_loss = tf.losses.get_total_loss()
      self.batch_loss_c5 = batch_loss_c5
      self.batch_loss_c4 = batch_loss_c4
      self.batch_loss_c3 = batch_loss_c3
      self.total_loss = total_loss

      tf.summary.image('exemplar', self.exemplars, family=self.mode)
      tf.summary.image('instance', self.instances, family=self.mode)

      mean_batch_loss_c5, update_op1_c5 = tf.metrics.mean(batch_loss_c5)
      mean_batch_loss_c4, update_op1_c4 = tf.metrics.mean(batch_loss_c4)
      mean_batch_loss_c3, update_op1_c3 = tf.metrics.mean(batch_loss_c3)
      mean_total_loss, update_op2 = tf.metrics.mean(total_loss)
      with tf.control_dependencies([update_op1_c5, update_op1_c4, update_op1_c3, update_op2]):
        tf.summary.scalar('batch_loss_c5', mean_batch_loss_c5, family=self.mode)
        tf.summary.scalar('batch_loss_c4', mean_batch_loss_c4, family=self.mode)
        tf.summary.scalar('batch_loss_c3', mean_batch_loss_c3, family=self.mode)
        tf.summary.scalar('total_loss', mean_total_loss, family=self.mode)

      if self.mode == 'train':
        tf.summary.image('GT', tf.reshape(gt[0], [1] + response_size + [1]), family='GT')
      tf.summary.image('Response_c5', tf.expand_dims(tf.sigmoid(response_c5), -1), family=self.mode)
      tf.summary.histogram('Response_c5', self.response_c5, family=self.mode)
      tf.summary.image('Response_c4', tf.expand_dims(tf.sigmoid(response_c4), -1), family=self.mode)
      tf.summary.histogram('Response_c4', self.response_c4, family=self.mode)
      tf.summary.image('Response_c3', tf.expand_dims(tf.sigmoid(response_c3), -1), family=self.mode)
      tf.summary.histogram('Response_c3', self.response_c3, family=self.mode)

# =============================================================================
#       # Two more metrics to monitor the performance of training
#       tf.summary.scalar('center_score_error', center_score_error(response), family=self.mode)
#       tf.summary.scalar('center_dist_error', center_dist_error(response), family=self.mode)
# =============================================================================

  def setup_global_step(self):
    global_step = tf.Variable(
      initial_value=0,
      name='global_step',
      trainable=False,
      collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])

    self.global_step = global_step

  def setup_embedding_initializer(self):
    """Sets up the function to restore embedding variables from checkpoint."""
    embed_config = self.model_config['embed_config']
    if embed_config['embedding_checkpoint_file']:
      # Restore Siamese FC models from .mat model files
      initialize = load_mat_model(embed_config['embedding_checkpoint_file'],
                                  'convolutional_alexnet/', 'detection/')

      def restore_fn(sess):
        tf.logging.info("Restoring embedding variables from checkpoint file %s",
                        embed_config['embedding_checkpoint_file'])
        sess.run([initialize])

      self.init_fn = restore_fn

  def build(self, reuse=False):
    """Creates all ops for training and evaluation"""
    with tf.name_scope(self.mode):
      self.build_inputs()
      self.build_image_embeddings(reuse=reuse)
      self.build_template()
      self.build_detection(reuse=reuse)
      self.setup_embedding_initializer()

      if self.mode in ['train', 'validation']:
        self.build_loss()

      if self.is_training():
        self.setup_global_step()
