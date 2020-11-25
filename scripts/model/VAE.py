# -*- coding:utf-8 -*-

import os
import sys
import numpy as np
import tensorflow as tf
import cv2
import pickle

from .tf_util import Layers, NetworkCreater, EncoderNetworkCreater, cross_entropy, KL

class _encoder_network(Layers):
    def __init__(self, name_scopes, config):
        assert(len(name_scopes) == 1)
        super().__init__(name_scopes)

        self._config = config
        self._network_creater = EncoderNetworkCreater(self._config, name_scopes[0]) 

    def set_model(self, inputs, is_training=True, reuse=False):
        return self._network_creater.create(inputs, self._config, is_training, reuse)


class _decoder_network(Layers):
    def __init__(self, name_scopes, config):
        assert(len(name_scopes) == 1)
        super().__init__(name_scopes)

        self._config = config
        self._network_creater = NetworkCreater(self._config, name_scopes[0]) 

    def set_model(self, inputs, is_training=True, reuse=False):
        return self._network_creater.create(inputs, self._config, is_training, reuse)


class VAE(object):
    
    def __init__(self, param, config, image_info):
        self._lr = param["lr"]
        self._image_width, self._image_height, self._image_channels = image_info

        self._network_encoder = _encoder_network([config["Encoder"]["network"]["name"]], config["Encoder"])
        self._network_decoder = _decoder_network([config["Decoder"]["network"]["name"]], config["Decoder"])


    def set_model(self):
        self._set_network()
        self._set_loss()
        self._set_optimizer()


    def _set_network(self):
        self.input = tf.compat.v1.placeholder(tf.float32, [None, self._image_height, self._image_width, self._image_channels])

        # For train
        self._mu, self._log_sigma = self._network_encoder.set_model(self.input, is_training=True, reuse=False)
        eps = tf.random.normal([tf.shape(self.input)[0], tf.shape(self._mu)[-1]])
        z = self._mu + eps*tf.exp(self._log_sigma)
        self._logits = self._network_decoder.set_model(z, is_training=True, reuse=False)

        # For test
        self._mu_wo, self._log_sigma_wo = self._network_encoder.set_model(self.input, is_training=False, reuse=True) 
        eps_wo = tf.random.normal([tf.shape(self.input)[0], tf.shape(self._mu_wo)[-1]])
        z_wo =  self._mu_wo + eps_wo * tf.exp(self._log_sigma_wo)
        self._logits_wo = self._network_decoder.set_model(z_wo, is_training=False, reuse=True) 


    def _set_loss(self):
        self._loss_cross_entropy = cross_entropy(tf.reshape(self.input, [-1, self._image_width*self._image_height*self._image_channels]), self._logits)
        self._loss_KL = KL(self._mu, self._log_sigma)
        self._loss_op = 1.0*tf.reduce_mean(self._loss_cross_entropy+self._loss_KL)
        #self._loss_op = tf.reduce_mean(-self._loss_KL) 


    def _set_optimizer(self):
        #self._train_op = tf.compat.v1.train.RMSPropOptimizer(self._lr).minimize(self._loss_op, var_list=self._network.get_variables())
        self._train_op = tf.compat.v1.train.AdamOptimizer(self._lr).minimize(self._loss_op)


    def train(self, sess, input_images):
        feed_dict = {self.input: input_images}
        loss, _, loss_entropy, loss_kl, mu, log_sigma = sess.run([self._loss_op, self._train_op, self._loss_cross_entropy, self._loss_KL, self._mu, self._log_sigma], feed_dict=feed_dict)
        print("loss_entropy:{}, loss_KL:{}".format(np.array(loss_entropy).mean(), np.array(loss_kl).mean()))
        #print("mu:{}, log_sigma:{}".format(mu, log_sigma))
        return loss, _

    def test(self, sess, input_images):
        feed_dict = {self.input: input_images}
        logits = sess.run([self._logits_wo], feed_dict=feed_dict)
        return np.squeeze(logits)  # remove extra dimension