# basic RNN

import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, \
                    optimizers, serialziers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

# set data

vocab = {}

def load_data(filename):

