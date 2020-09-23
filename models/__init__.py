from __future__ import absolute_import

from .ResNet import *
from .ResNet_hypergraphsage_part import ResNet50GRAPHPOOLPARTHyper

__factory = {
    'resnet50graphpoolparthyper': ResNet50GRAPHPOOLPARTHyper,
}

def get_names():
    return __factory.keys()

def init_model(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown model: {}".format(name))
    return __factory[name](*args, **kwargs)
