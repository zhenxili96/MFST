#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 14:25:30 2018

@author: zhlixc
"""


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