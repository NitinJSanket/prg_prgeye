#!/usr/bin/env python

import tensorflow as tf
import sys
import numpy as np
import inspect
from functools import wraps
from tensorflow.contrib.framework import add_arg_scope
from tensorflow.contrib.framework import arg_scope

# Don't generate pyc codes
sys.dont_write_bytecode = True

def Count(func):
    @wraps(func)
    def wrapped(self, *args, **kwargs):
        self.CurrBlock += 1
        return func(self, *args, **kwargs)
    return wrapped

def CountAndScope(func):
    @wraps(func)
    def wrapped(self, *args, **kwargs):
        with tf.variable_scope(func.__name__ + str(self.CurrBlock)):
            self.CurrBlock += 1
            return func(self, *args, **kwargs)
    return wrapped
