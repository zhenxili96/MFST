#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 11:21:21 2019

@author: zhlixc
"""

import vot
import sys
import time
import cv2
import numpy
import collections
import logging

import os

import tensorflow as tf

# Code root absolute path
CODE_ROOT = '/home/travail/dev/GitRepo/MFST'
CHECKPOINT = '/home/travail/dev/GitRepo/MFST/Logs/SiamFC/track_model_checkpoints/siamfc_se'

sys.path.insert(0, CODE_ROOT)

from utils.misc_utils import auto_select_gpu, load_cfgs
from inference import inference_wrapper
from inference.tracker import Tracker

# Set GPU
os.environ['CUDA_VISIBLE_DEVICES'] = auto_select_gpu()


checkpoint_path = CHECKPOINT

# Read configurations from json
model_config, _, track_config = load_cfgs(checkpoint_path)

track_config['log_level'] = 0  # Skip verbose logging for speed

# Build the inference graph.
g = tf.Graph()
with g.as_default():
  model = inference_wrapper.InferenceWrapper()
  restore_fn = model.build_graph_from_config(model_config, track_config, checkpoint_path)
#g.finalize()

gpu_options = tf.GPUOptions(allow_growth=True)
sess_config = tf.ConfigProto(gpu_options=gpu_options)

with tf.Session(graph=g, config=sess_config) as sess:
  ## used for initializing alexnet parameters
  init_global = tf.global_variables_initializer()
  sess.run(init_global)
    
  ## global initalizer must be run before restore
  # Load the model from checkpoint.
  restore_fn(sess)

  tracker = Tracker(model, model_config, track_config)
  handle = vot.VOT("rectangle")
  selection = handle.region()
  imagefile = handle.frame()
  tracker.track_vot_init(sess, selection, imagefile)
  
  while True:
      imagefile = handle.frame()
      if not imagefile:
          break
      region = tracker.track_vot(sess, imagefile)
      handle.report(region)
