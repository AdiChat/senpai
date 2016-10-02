# -*- coding: utf8 -*-

from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals


from collections import namedtuple
from operator import attrgetter

SentenceNode = namedtuple("SentenceNode", ("sentence", "rating", "order",))


class AbstractSummarizer(object):
    def __init__(self):
        self.count = 5
        
    def __call__(self):
        raise NotImplementedError("This method should be overriden in subclass")

    def stem_word(self, word):
        return word
        
    def _get_best_sentences(self, sentences, count):
        return sentences
        