#!/usr/bin/python

# *****************************************************************************
#
# Author: Aditya Chatterjee
#
# Interweb/ contacts: GitHub.com/AdiChat
#                     Email: aditianhacker@gmail.com
#
# Implementation of the Random Summarization Algorithm
#
# MIT License
#
# To keep up with the latest version, consult repository: GitHub.com/AdiChat/Read-Less
#
# To get an overview of the TextTiling Algorithm, consult wiki: Github.com/AdiChat/Read-Less/wiki
#
# *****************************************************************************
import io
import nltk
import itertools
from operator import itemgetter
import networkx as nx
import os
import random as rr
from ..Segmentation import texttiling
from ..Parse import parse


class Random():

	def __init__(self):
		print("Random: Summarizing textual data")

	def random(self, firstString, secondString):
	    return rr.randrange(0, 101, 2)

	def buildGraph(self, nodes):
	    gr = nx.Graph()
	    gr.add_nodes_from(nodes)
	    nodePairs = list(itertools.combinations(nodes, 2))

	    for pair in nodePairs:
	        firstString = pair[0]
	        secondString = pair[1]
	        levDistance = self.random(firstString, secondString)
	        gr.add_edge(firstString, secondString, weight=levDistance)

	    return gr

	def extractSentences(self, text, size=201):

	    sentenceTokens = text
	    graph = self.buildGraph(sentenceTokens)
	    calculated_page_rank = nx.pagerank(graph, weight='weight')
	    sentences = sorted(calculated_page_rank,
	                       key=calculated_page_rank.get, reverse=True)
	    summary = ' '.join(sentences)
	    summaryWords = summary.split()
	    summaryWords = summaryWords[0:size]
	    summary = ' '.join(summaryWords)

	    return summary

	def summarize(self, data):
		t = texttiling.TextTiling()
		text = t.run(data)
		return self.extractSentences(text)

	def summarizeFile(self, pathToFile):
		p = parse.Parse()
		t = texttiling.TextTiling()
		data = p.dataFromFile(pathToFile)
		text = t.run(data)
		return self.extractSentences(text)