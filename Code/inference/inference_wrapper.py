#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2017 bily     Huazhong University of Science and Technology
#
# Distributed under terms of the MIT license.

"""Model Wrapper class for performing inference with a SiameseModel"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import logging
import os
import os.path as osp

import numpy as np
import tensorflow as tf

from embeddings.convolutional_alexnet import convolutional_alexnet_arg_scope, convolutional_alexnet, embed_alexnet, embed_vgg16
from utils.infer_utils import get_exemplar_images
from utils.misc_utils import get_center

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

class InferenceWrapper():
  """Model wrapper class for performing inference with a siamese model."""

  def __init__(self):
    self.image = None
    self.target_bbox_feed = None
    self.search_images = None
    self.embeds = None
    self.templates = None
    self.init = None
    self.model_config = None
    self.track_config = None
    self.response_up = None

  def build_graph_from_config(self, model_config, track_config, checkpoint_path):
    """Build the inference graph and return a restore function."""
    self.build_model(model_config, track_config)
    ema = tf.train.ExponentialMovingAverage(0)
    variables_to_restore = ema.variables_to_restore(moving_avg_variables=[])

    # Filter out State variables
    variables_to_restore_filterd_siamfc = {}
    variables_to_restore_filterd_alex = {}
    for key, value in variables_to_restore.items():
      if key.split('/')[1] != 'State':
          if "_a_" in key:
              continue
          if "alex_branch" in key:
              continue
          logging.info(key)
          print(key)
          variables_to_restore_filterd_siamfc[key] = value
          
    for key, value in variables_to_restore.items():
      if key.split('/')[1] != 'State':
          if "vgg16_branch" in key:
              continue
          if "convolutional_alexnet" in key:
              continue
          if "_s_" in key:
              continue
          if "siamfc_branch_se" in key:
              continue
          if "alex_branch/conv1W" in key:
              continue
          if "alex_branch/conv1b" in key:
              continue
          if "alex_branch/conv2W" in key:
              continue
          if "alex_branch/conv2b" in key:
              continue
          if "alex_branch/conv3W" in key:
              continue
          if "alex_branch/conv3b" in key:
              continue
          if "alex_branch/conv4W" in key:
              continue
          if "alex_branch/conv4b" in key:
              continue
          if "alex_branch/conv5W" in key:
              continue
          if "alex_branch/conv5b" in key:
              continue
          if "alex_branch_1/conv1W" in key:
              continue
          if "alex_branch_1/conv1b" in key:
              continue
          if "alex_branch_1/conv2W" in key:
              continue
          if "alex_branch_1/conv2b" in key:
              continue
          if "alex_branch_1/conv3W" in key:
              continue
          if "alex_branch_1/conv3b" in key:
              continue
          if "alex_branch_1/conv4W" in key:
              continue
          if "alex_branch_1/conv4b" in key:
              continue
          if "alex_branch_1/conv5W" in key:
              continue
          if "alex_branch_1/conv5b" in key:
              continue
# =============================================================================
#           if 'detection/biases_c3' in key:
#               continue
#           if 'detection/biases_c4' in key:
#               continue
#           if 'detection/biases_c5' in key:
#               continue
#           if 'detection/weight_c3' in key:
#               continue
#           if 'detection/weight_c4' in key:
#               continue
#           if 'detection/weight_c5' in key:
#               continue
# =============================================================================
          
          logging.info(key)
          print(key)
          variables_to_restore_filterd_alex[key] = value
          
    saver_alex = None
    saver_siamfc = None
    if variables_to_restore_filterd_alex=={}:
        saver_alex = None
    else:
        saver_alex = tf.train.Saver(variables_to_restore_filterd_alex)
    if variables_to_restore_filterd_siamfc=={}:
        saver_siamfc = None
    else:
        saver_siamfc = tf.train.Saver(variables_to_restore_filterd_siamfc)

    checkpoint_path_alex = "/home/travail/dev/GitRepo/MFST/Logs/SiamFC/track_model_checkpoints/alex_se/model.ckpt-332499"
    checkpoint_path_siamfc = "/home/travail/dev/GitRepo/MFST/Logs/SiamFC/track_model_checkpoints/siamfc_se/model.ckpt-332499"
# =============================================================================
#     checkpoint_path_siamfc = "/home/travail/dev/GitRepo/MFST/Logs/SiamFC/track_model_checkpoints/SiamFC-3s-color-pretrained/model.ckpt-0"
# =============================================================================

# =============================================================================
#     if osp.isdir(checkpoint_path):
#       checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
#       if not checkpoint_path:
#         raise ValueError("No checkpoint file found in: {}".format(checkpoint_path))
# =============================================================================

    def _restore_fn(sess):
      logging.info("Loading model from checkpoint: %s", checkpoint_path_alex)
      if saver_alex==None:
          logging.info("Alex No restore,,,,,")
      else:
          logging.info("Alex Restore,,,,,")
          saver_alex.restore(sess, checkpoint_path_alex)
      logging.info("Successfully loaded checkpoint: %s", os.path.basename(checkpoint_path_alex))
      
      logging.info("Loading model from checkpoint: %s", checkpoint_path_siamfc)
      if saver_siamfc==None:
          logging.info("SiamFC No restore,,,,,")
      else:
          logging.info("SiamFC Restore,,,,,")
          saver_siamfc.restore(sess, checkpoint_path_siamfc)
      logging.info("Successfully loaded checkpoint: %s", os.path.basename(checkpoint_path_siamfc))

    return _restore_fn

  def build_model(self, model_config, track_config):
    self.model_config = model_config
    self.track_config = track_config

    self.build_inputs()
    self.build_search_images()
    self.build_template()
    self.build_detection()
    self.build_upsample()
    self.dumb_op = tf.no_op('dumb_operation')

  def build_inputs(self):
    filename = tf.placeholder(tf.string, [], name='filename')
    image_file = tf.read_file(filename)
    image = tf.image.decode_jpeg(image_file, channels=3, dct_method="INTEGER_ACCURATE")
    image = tf.to_float(image)
    self.image = image
    self.target_bbox_feed = tf.placeholder(dtype=tf.float32,
                                           shape=[4],
                                           name='target_bbox_feed')  # center's y, x, height, width

  def build_search_images(self):
    """Crop search images from the input image based on the last target position

    1. The input image is scaled such that the area of target&context takes up to (scale_factor * z_image_size) ^ 2
    2. Crop an image patch as large as x_image_size centered at the target center.
    3. If the cropped image region is beyond the boundary of the input image, mean values are padded.
    """
    model_config = self.model_config
    track_config = self.track_config

    size_z = model_config['z_image_size']
    size_x = track_config['x_image_size']
    context_amount = 0.5

    num_scales = track_config['num_scales']
    scales = np.arange(num_scales) - get_center(num_scales)
    assert np.sum(scales) == 0, 'scales should be symmetric'
    search_factors = [track_config['scale_step'] ** x for x in scales]

    frame_sz = tf.shape(self.image)
    target_yx = self.target_bbox_feed[0:2]
    target_size = self.target_bbox_feed[2:4]
    avg_chan = tf.reduce_mean(self.image, axis=(0, 1), name='avg_chan')

    # Compute base values
    base_z_size = target_size
    base_z_context_size = base_z_size + context_amount * tf.reduce_sum(base_z_size)
    base_s_z = tf.sqrt(tf.reduce_prod(base_z_context_size))  # Canonical size
    base_scale_z = tf.div(tf.to_float(size_z), base_s_z)
    d_search = (size_x - size_z) / 2.0
    base_pad = tf.div(d_search, base_scale_z)
    base_s_x = base_s_z + 2 * base_pad
    base_scale_x = tf.div(tf.to_float(size_x), base_s_x)

    boxes = []
    for factor in search_factors:
      s_x = factor * base_s_x
      frame_sz_1 = tf.to_float(frame_sz[0:2] - 1)
      topleft = tf.div(target_yx - get_center(s_x), frame_sz_1)
      bottomright = tf.div(target_yx + get_center(s_x), frame_sz_1)
      box = tf.concat([topleft, bottomright], axis=0)
      boxes.append(box)
    boxes = tf.stack(boxes)

    scale_xs = []
    for factor in search_factors:
      scale_x = base_scale_x / factor
      scale_xs.append(scale_x)
    self.scale_xs = tf.stack(scale_xs)

    # Note we use different padding values for each image
    # while the original implementation uses only the average value
    # of the first image for all images.
    image_minus_avg = tf.expand_dims(self.image - avg_chan, 0)
    image_cropped = tf.image.crop_and_resize(image_minus_avg, boxes,
                                             box_ind=tf.zeros((track_config['num_scales']), tf.int32),
                                             crop_size=[size_x, size_x])
    self.search_images = image_cropped + avg_chan

  def get_image_embedding(self, images, reuse=None):
    config = self.model_config['embed_config']
    arg_scope = convolutional_alexnet_arg_scope(config,
                                                trainable=config['train_embedding'],
                                                is_training=False)

    @functools.wraps(convolutional_alexnet)
    def embedding_fn(images, reuse=False):
      with slim.arg_scope(arg_scope):
        return convolutional_alexnet(images, reuse=reuse)

    embed_s_c5, embed_s_c4, embed_s_c3, _ = embedding_fn(images, reuse)
    embed_a_c5, embed_a_c4, embed_a_c3 = embed_alexnet(images)
# =============================================================================
#     embed_v_43, embed_v_42, embed_v_41 = embed_vgg16(images)
# =============================================================================

# =============================================================================
#     return embed_v_43, embed_v_42, embed_v_41
# =============================================================================
# =============================================================================
#     return embed_a_c5, embed_a_c4, embed_a_c3 
# =============================================================================
# =============================================================================
#     return embed_s_c5, embed_s_c4, embed_s_c3
# =============================================================================
    return embed_s_c5, embed_s_c4, embed_s_c3, embed_a_c5, embed_a_c4, embed_a_c3

  def build_template(self):
    model_config = self.model_config
    track_config = self.track_config

    # Exemplar image lies at the center of the search image in the first frame
    exemplar_images = get_exemplar_images(self.search_images, [model_config['z_image_size'],
                                                               model_config['z_image_size']])
    templates_s_c5, templates_s_c4, templates_s_c3, templates_a_c5, templates_a_c4, templates_a_c3 = self.get_image_embedding(exemplar_images)
# =============================================================================
#     templates_s_c5, templates_s_c4, templates_s_c3 = self.get_image_embedding(exemplar_images)
# =============================================================================
# =============================================================================
#     templates_a_c5, templates_a_c4, templates_a_c3 = self.get_image_embedding(exemplar_images)
# =============================================================================
    center_scale = int(get_center(track_config['num_scales']))
    
    center_template_s_c5 = tf.identity(templates_s_c5[center_scale])
    center_template_s_c4 = tf.identity(templates_s_c4[center_scale])
    center_template_s_c3 = tf.identity(templates_s_c3[center_scale])
    templates_s_c5 = tf.stack([center_template_s_c5 for _ in range(track_config['num_scales'])])
    templates_s_c4 = tf.stack([center_template_s_c4 for _ in range(track_config['num_scales'])])
    templates_s_c3 = tf.stack([center_template_s_c3 for _ in range(track_config['num_scales'])])
    
    center_template_a_c5 = tf.identity(templates_a_c5[center_scale])
    center_template_a_c4 = tf.identity(templates_a_c4[center_scale])
    center_template_a_c3 = tf.identity(templates_a_c3[center_scale])
    templates_a_c5 = tf.stack([center_template_a_c5 for _ in range(track_config['num_scales'])])
    templates_a_c4 = tf.stack([center_template_a_c4 for _ in range(track_config['num_scales'])])
    templates_a_c3 = tf.stack([center_template_a_c3 for _ in range(track_config['num_scales'])])

    with tf.variable_scope('target_template'):
      # Store template in Variable such that we don't have to feed this template every time.
      with tf.variable_scope('State'):
        state_s_c5 = tf.get_variable('exemplar_s_c5',
                                initializer=tf.zeros(templates_s_c5.get_shape().as_list(), dtype=templates_s_c5.dtype),
                                trainable=False)
        state_s_c4 = tf.get_variable('exemplar_s_c4',
                                initializer=tf.zeros(templates_s_c4.get_shape().as_list(), dtype=templates_s_c4.dtype),
                                trainable=False)
        state_s_c3 = tf.get_variable('exemplar_s_c3',
                                initializer=tf.zeros(templates_s_c3.get_shape().as_list(), dtype=templates_s_c3.dtype),
                                trainable=False)
        
        state_a_c5 = tf.get_variable('exemplar_a_c5',
                                initializer=tf.zeros(templates_a_c5.get_shape().as_list(), dtype=templates_a_c5.dtype),
                                trainable=False)
        state_a_c4 = tf.get_variable('exemplar_a_c4',
                                initializer=tf.zeros(templates_a_c4.get_shape().as_list(), dtype=templates_a_c4.dtype),
                                trainable=False)
        state_a_c3 = tf.get_variable('exemplar_a_c3',
                                initializer=tf.zeros(templates_a_c3.get_shape().as_list(), dtype=templates_a_c3.dtype),
                                trainable=False)

        with tf.control_dependencies([templates_s_c5]):
          self.init_s_c5 = tf.assign(state_s_c5, templates_s_c5, validate_shape=True)
        with tf.control_dependencies([templates_s_c4]):
          self.init_s_c4 = tf.assign(state_s_c4, templates_s_c4, validate_shape=True)
        with tf.control_dependencies([templates_s_c3]):
          self.init_s_c3 = tf.assign(state_s_c3, templates_s_c3, validate_shape=True)
          
        with tf.control_dependencies([templates_a_c5]):
          self.init_a_c5 = tf.assign(state_a_c5, templates_a_c5, validate_shape=True)
        with tf.control_dependencies([templates_a_c4]):
          self.init_a_c4 = tf.assign(state_a_c4, templates_a_c4, validate_shape=True)
        with tf.control_dependencies([templates_a_c3]):
          self.init_a_c3 = tf.assign(state_a_c3, templates_a_c3, validate_shape=True)
          
        self.templates_s_c5 = state_s_c5
        self.templates_s_c4 = state_s_c4
        self.templates_s_c3 = state_s_c3
        
        self.templates_a_c5 = state_a_c5
        self.templates_a_c4 = state_a_c4
        self.templates_a_c3 = state_a_c3

  def build_detection(self):
    self.embeds_s_c5, self.embeds_s_c4, self.embeds_s_c3, self.embeds_a_c5, self.embeds_a_c4, self.embeds_a_c3 = self.get_image_embedding(self.search_images, reuse=True)
# =============================================================================
#     self.embeds_s_c5, self.embeds_s_c4, self.embeds_s_c3 = self.get_image_embedding(self.search_images, reuse=True)
# =============================================================================
# =============================================================================
#     self.embeds_a_c5, self.embeds_a_c4, self.embeds_a_c3 = self.get_image_embedding(self.search_images, reuse=True)
# =============================================================================
    with tf.variable_scope('detection'):
      def _translation_match(x, z):
        x = tf.expand_dims(x, 0)  # [batch, in_height, in_width, in_channels]
        z = tf.expand_dims(z, -1)  # [filter_height, filter_width, in_channels, out_channels]
        return tf.nn.conv2d(x, z, strides=[1, 1, 1, 1], padding='VALID', name='translation_match')

      output_s_c5 = tf.map_fn(
        lambda x: _translation_match(x[0], x[1]),
        (self.embeds_s_c5, self.templates_s_c5), dtype=self.embeds_s_c5.dtype)  # of shape [16, 1, 17, 17, 1]
      output_s_c4 = tf.map_fn(
        lambda x: _translation_match(x[0], x[1]),
        (self.embeds_s_c4, self.templates_s_c4), dtype=self.embeds_s_c4.dtype)  # of shape [16, 1, 17, 17, 1]
      output_s_c3 = tf.map_fn(
        lambda x: _translation_match(x[0], x[1]),
        (self.embeds_s_c3, self.templates_s_c3), dtype=self.embeds_s_c3.dtype)  # of shape [16, 1, 17, 17, 1]
      
      output_a_c5 = tf.map_fn(
        lambda x: _translation_match(x[0], x[1]),
        (self.embeds_a_c5, self.templates_a_c5), dtype=self.embeds_a_c5.dtype)  # of shape [16, 1, 17, 17, 1]
      output_a_c4 = tf.map_fn(
        lambda x: _translation_match(x[0], x[1]),
        (self.embeds_a_c4, self.templates_a_c4), dtype=self.embeds_a_c5.dtype)  # of shape [16, 1, 17, 17, 1]
      output_a_c3 = tf.map_fn(
        lambda x: _translation_match(x[0], x[1]),
        (self.embeds_a_c3, self.templates_a_c3), dtype=self.embeds_a_c5.dtype)  # of shape [16, 1, 17, 17, 1]
      
      output_s_c5 = tf.squeeze(output_s_c5, [1, 4])  # of shape e.g. [16, 17, 17]
      output_s_c4 = tf.squeeze(output_s_c4, [1, 4])  # of shape e.g. [16, 17, 17]
      output_s_c3 = tf.squeeze(output_s_c3, [1, 4])  # of shape e.g. [16, 17, 17]
      
      output_a_c5 = tf.squeeze(output_a_c5, [1, 4])  # of shape e.g. [16, 17, 17]
      output_a_c4 = tf.squeeze(output_a_c4, [1, 4])  # of shape e.g. [16, 17, 17]
      output_a_c3 = tf.squeeze(output_a_c3, [1, 4])  # of shape e.g. [16, 17, 17]
      
      ## for siamfc_se model
      bias_s_c5 = tf.get_variable('biases_s_c5', [1],
                             dtype=tf.float32,
                             initializer=tf.constant_initializer(0.0, dtype=tf.float32),
                             trainable=False)
      bias_s_c4 = tf.get_variable('biases_s_c4', [1],
                             dtype=tf.float32,
                             initializer=tf.constant_initializer(0.0, dtype=tf.float32),
                             trainable=False)
      bias_s_c3 = tf.get_variable('biases_s_c3', [1],
                             dtype=tf.float32,
                             initializer=tf.constant_initializer(0.0, dtype=tf.float32),
                             trainable=False)
      response_s_c5 = self.model_config['adjust_response_config']['scale'] * output_s_c5 + bias_s_c5
      response_s_c4 = self.model_config['adjust_response_config']['scale'] * output_s_c4 + bias_s_c4
      response_s_c3 = self.model_config['adjust_response_config']['scale'] * output_s_c3 + bias_s_c3

# =============================================================================
#       bias_s = tf.get_variable("biases", [1],
#                                dtype=tf.float32,
#                                initializer=tf.constant_initializer(0.0, dtype=tf.float32),
#                                trainable=False)
#       response_s_c5 = self.model_config['adjust_response_config']['scale'] * output_s_c5 + bias_s
#       response_s_c4 = self.model_config['adjust_response_config']['scale'] * output_s_c4 + bias_s
#       response_s_c3 = self.model_config['adjust_response_config']['scale'] * output_s_c3 + bias_s
# =============================================================================
      


      ## for alex_se model
      bias_c5 = tf.get_variable('biases_a_c5', [1],
                             dtype=tf.float32,
                             #initializer=tf.constant_initializer(ALEX_DICT['detection/biases_c5']),
                             initializer=tf.constant_initializer(0.0, dtype=tf.float32),
                             trainable=False)
      bias_c4 = tf.get_variable('biases_a_c4', [1],
                             dtype=tf.float32,
                             #initializer=tf.constant_initializer(ALEX_DICT['detection/biases_c4']),
                             initializer=tf.constant_initializer(0.0, dtype=tf.float32),
                             trainable=False)
      bias_c3 = tf.get_variable('biases_a_c3', [1],
                             dtype=tf.float32,
                             #initializer=tf.constant_initializer(ALEX_DICT['detection/biases_c3']),
                             initializer=tf.constant_initializer(0.0, dtype=tf.float32),
                             trainable=False)
      response_a_c5 = 1e-3 * output_a_c5 + bias_c5
      response_a_c4 = 1e-4 * output_a_c4 + bias_c4
      response_a_c3 = 1e-5 * output_a_c3 + bias_c3
      
# =============================================================================
#       response_a_c5 = self.model_config['adjust_response_config']['scale'] * output_a_c5 -4.52620411# + bias_c5
#       response_a_c4 = self.model_config['adjust_response_config']['scale'] * output_a_c4 -0.03678114#+ bias_c4
#       response_a_c3 = self.model_config['adjust_response_config']['scale'] * output_a_c3 -0.49341503#+ bias_c3
# =============================================================================
# =============================================================================
#       response_a_c5 = 1e-4 * output_a_c5 + bias_c5
#       response_a_c4 = 1e-5 * output_a_c4 + bias_c4
#       response_a_c3 = 1e-5 * output_a_c3 + bias_c3
# =============================================================================
      
      # weight maps for response from each layer
# =============================================================================
#       response_size = response_a_c5.get_shape().as_list()[1:3]
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
#       self.weight_s_c5 = tf.get_variable('weight_s_c5', [1],
#                              dtype=tf.float32,
#                              initializer=tf.constant_initializer(0.5, dtype=tf.float32),
#                              trainable=False)
#       self.weight_s_c4 = tf.get_variable('weight_s_c4', [1],
#                              dtype=tf.float32,
#                              initializer=tf.constant_initializer(0.5, dtype=tf.float32),
#                              trainable=False)
#       self.weight_s_c3 = tf.get_variable('weight_s_c3', [1],
#                              dtype=tf.float32,
#                              initializer=tf.constant_initializer(0.5, dtype=tf.float32),
#                              trainable=False)
# =============================================================================
# =============================================================================
#       self.weight_c5 = tf.get_variable('weight_a_c5', [1],
#                              dtype=tf.float32,
#                              #initializer=tf.constant_initializer(ALEX_DICT['detection/weight_c5']),
#                              initializer=tf.constant_initializer(0.5, dtype=tf.float32),
#                              trainable=False)
#       self.weight_c4 = tf.get_variable('weight_a_c4', [1],
#                              dtype=tf.float32,
#                              #initializer=tf.constant_initializer(ALEX_DICT['detection/weight_c4']),
#                              initializer=tf.constant_initializer(0.5, dtype=tf.float32),
#                              trainable=False)
#       self.weight_c3 = tf.get_variable('weight_a_c3', [1],
#                              dtype=tf.float32,
#                              #initializer=tf.constant_initializer(ALEX_DICT['detection/weight_c3']),
#                              initializer=tf.constant_initializer(0.5, dtype=tf.float32),
#                              trainable=False)
# =============================================================================
# =============================================================================
#       response_s_c5 = tf.multiply(response_s_c5, self.weight_s_c5)
#       response_s_c4 = tf.multiply(response_s_c4, self.weight_s_c4)
#       response_s_c3 = tf.multiply(response_s_c3, self.weight_s_c3)
# =============================================================================
# =============================================================================
#       response_a_c5 = tf.multiply(response_a_c5, self.weight_c5)
#       response_a_c4 = tf.multiply(response_a_c4, self.weight_c4)
#       response_a_c3 = tf.multiply(response_a_c3, self.weight_c3)
# =============================================================================
# =============================================================================
#       self.response_s = 0.6*response_s_c5+0.3*response_s_c4+0.1*response_s_c3
# =============================================================================
# =============================================================================
#       self.response_a = 0.6*response_a_c5+0.3*response_a_c4+0.1*response_a_c3
# =============================================================================
      self.response_a_c5 = response_a_c5
      self.response_a_c4 = response_a_c4
      self.response_a_c3 = response_a_c3
      
      self.response_s_c5 = response_s_c5
      self.response_s_c4 = response_s_c4
      self.response_s_c3 = response_s_c3
      
      

  def build_upsample(self):
    """Upsample response to obtain finer target position"""
    with tf.variable_scope('upsample'):
      response_s_c5 = tf.expand_dims(self.response_s_c5, 3)
      response_s_c4 = tf.expand_dims(self.response_s_c4, 3)
      response_s_c3 = tf.expand_dims(self.response_s_c3, 3)
      
      response_a_c5 = tf.expand_dims(self.response_a_c5, 3)
      response_a_c4 = tf.expand_dims(self.response_a_c4, 3)
      response_a_c3 = tf.expand_dims(self.response_a_c3, 3)
      
      up_method = self.track_config['upsample_method']
      methods = {'bilinear': tf.image.ResizeMethod.BILINEAR,
                 'bicubic': tf.image.ResizeMethod.BICUBIC}
      up_method = methods[up_method]
      response_spatial_size = self.response_s_c5.get_shape().as_list()[1:3]
      up_size = [s * self.track_config['upsample_factor'] for s in response_spatial_size]
      print(up_size)
      response_up_s_c5 = tf.image.resize_images(response_s_c5,
                                           up_size,
                                           method=up_method,
                                           align_corners=True)
      response_up_s_c4 = tf.image.resize_images(response_s_c4,
                                           up_size,
                                           method=up_method,
                                           align_corners=True)
      response_up_s_c3 = tf.image.resize_images(response_s_c3,
                                           up_size,
                                           method=up_method,
                                           align_corners=True)
      
      response_up_a_c5 = tf.image.resize_images(response_a_c5,
                                           up_size,
                                           method=up_method,
                                           align_corners=True)
      response_up_a_c4 = tf.image.resize_images(response_a_c4,
                                           up_size,
                                           method=up_method,
                                           align_corners=True)
      response_up_a_c3 = tf.image.resize_images(response_a_c3,
                                           up_size,
                                           method=up_method,
                                           align_corners=True)
      response_up_s_c5 = tf.squeeze(response_up_s_c5, [3])
      response_up_s_c4 = tf.squeeze(response_up_s_c4, [3])
      response_up_s_c3 = tf.squeeze(response_up_s_c3, [3])
      self.response_up_s_c5 = response_up_s_c5
      self.response_up_s_c4 = response_up_s_c4
      self.response_up_s_c3 = response_up_s_c3
      
      response_up_a_c5 = tf.squeeze(response_up_a_c5, [3])
      response_up_a_c4 = tf.squeeze(response_up_a_c4, [3])
      response_up_a_c3 = tf.squeeze(response_up_a_c3, [3])
      self.response_up_a_c5 = response_up_a_c5
      self.response_up_a_c4 = response_up_a_c4
      self.response_up_a_c3 = response_up_a_c3
      

  def initialize(self, sess, input_feed):
    image_path, target_bbox = input_feed
# =============================================================================
#     scale_xs, _, _, _ = sess.run([self.scale_xs, self.init_s_c5, self.init_s_c4, self.init_s_c3],
#                            feed_dict={'filename:0': image_path,
#                                       "target_bbox_feed:0": target_bbox, })
# =============================================================================
    scale_xs, _, _, _, _, _, _ = sess.run([self.scale_xs, self.init_s_c5, self.init_s_c4, self.init_s_c3, self.init_a_c5, self.init_a_c4, self.init_a_c3],
                           feed_dict={'filename:0': image_path,
                                      "target_bbox_feed:0": target_bbox, })
    return scale_xs

  def inference_step(self, sess, input_feed):
    image_path, target_bbox = input_feed
    log_level = self.track_config['log_level']
    image_cropped_op = self.search_images if log_level > 0 else self.dumb_op
    image_cropped, scale_xs, response_output_a_c5, response_output_a_c4, response_output_a_c3, response_output_s_c5, response_output_s_c4, response_output_s_c3 = sess.run(
      fetches=[image_cropped_op, self.scale_xs, \
               self.response_up_a_c5, self.response_up_a_c4, self.response_up_a_c3, \
               self.response_up_s_c5, self.response_up_s_c4, self.response_up_s_c3
               ],
      feed_dict={
        "filename:0": image_path,
        "target_bbox_feed:0": target_bbox, })
# =============================================================================
#     image_cropped, scale_xs, response_output_s_c5, response_output_s_c4, response_output_s_c3 = sess.run(
#       fetches=[image_cropped_op, self.scale_xs, \
#                self.response_up_s_c5, self.response_up_s_c4, self.response_up_s_c3
#                ],
#       feed_dict={
#         "filename:0": image_path,
#         "target_bbox_feed:0": target_bbox, })
# =============================================================================
# =============================================================================
#     image_cropped, scale_xs, response_output_a_c5, response_output_a_c4, response_output_a_c3 = sess.run(
#       fetches=[image_cropped_op, self.scale_xs, \
#                self.response_up_a_c5, self.response_up_a_c4, self.response_up_a_c3
#                ],
#       feed_dict={
#         "filename:0": image_path,
#         "target_bbox_feed:0": target_bbox, })
# =============================================================================

    output = {
      'image_cropped': image_cropped,
      'scale_xs': scale_xs,
      'response_s_c5': response_output_s_c5,
      'response_s_c4': response_output_s_c4,
      'response_s_c3': response_output_s_c3,
      'response_a_c5': response_output_a_c5,
      'response_a_c4': response_output_a_c4,
      'response_a_c3': response_output_a_c3
      }
    return output, None
