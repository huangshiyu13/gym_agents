#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Shiyu Huang 
@contact: huangsy13@gmail.com
@file: model.py
"""

import init_dirs
from TART.model import Model
import tensorflow as tf


class DQN_Model(Model):
    def build_forward(self, inputs, batch_size, action_size, train=False):

        fc1 = self.dense(inputs, 'fc1',
                         in_size=inputs.get_shape().as_list()[-1],
                         out_size=200, use_relu=True)

        Qs = self.dense(fc1, 'f2',
                        in_size=fc1.get_shape().as_list()[-1],
                        out_size=action_size, use_relu=False)

        return Qs

    def build_actor(self, inputs, action_size, batch_size=1):
        if self.name:
            with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
                Qs = self.build_forward(inputs, batch_size, action_size, train=False)
                self.Qs = Qs

