# -*- coding:utf-8 -*-

import sys
import tensorflow as tf

from .network_creater import NetworkCreater
from .network import conv2d

class EncoderNetworkCreater(NetworkCreater):
    def __init__(self, config, name_scope):
        super().__init__(config, name_scope)

    def create(self, inputs, config, is_training=True, reuse=False):
        h_in = inputs
        for layer in list(config.keys())[self._model_start_key:]:
            h_out = self._creater[config[layer]["type"]](inputs=h_in,
                                                    data=config[layer],
                                                    is_training=is_training,
                                                    reuse=reuse)
        
            if config[layer]["type"] in ["conv2d", "fc"]:
                if config[layer]["name"]=="fc_mu":
                    mu = h_out
                elif config[layer]["name"]=="fc_sigma":
                    sigma = h_out
                else:
                    h_in = h_out
            else:
                h_in = h_out
            
        return mu, sigma
    
    def get_extra_feature(self):
        return self._extra_feature