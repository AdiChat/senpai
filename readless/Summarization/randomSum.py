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
		print "Random: Summarizing textual data"

	def random(self, firstString, secondString):
		'''
		Returns a random similarity score between 2 strings
		Arguments:
			firstString: first input string
			secondString: second input string
		Returns:
			A random integer
		Raises:
			None
		'''
	    return rr.randrange(0, 101, 2)

	def buildGraph(self, nodes):
		'''
		Builds the graph with token of words as a node
		Arguments:
			nodes: list of token of words
		Returns:
			the graph
		Raises:
			None
		'''
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
		'''
		Finds out top token of words from the corpus
		Arguments:
			text: list of token of words
			size: size of summary
		Returns:
			The summary of size (size)
		Raises:
			None
		'''
	    sentenceTokens = text
	    graph = self.buildGraph(sentenceTokens)
	    calculated_page_rank = nx.pagerank(graph, weight='weight')
	    sentences = sorted(calculated_page_rank, key=calculated_page_rank.get, reverse=True)
	    summary = ' '.join(sentences)
	    summaryWords = summary.split()
	    summaryWords = summaryWords[0:size]
	    summary = ' '.join(summaryWords)

	    return summary

	def summarize(self, data):
		'''
		Summarizes a text data
		Arguments:
			data: input textual data
		Returns:
			The summary of input file
		Raises:
			None
		'''
		t = texttiling.TextTiling()
		text = t.run(data)
		return self.extractSentences(text)

	def summarizeFile(self, pathToFile):
		'''
		Summarizes a document
		Arguments:
			pathToFile: path to the file to be summarized
		Returns:
			The summary of the input file
		Raises:
			None
		'''
		p = parse.Parse()
		t = texttiling.TextTiling()
		data = p.dataFromFile(pathToFile)
		text = t.run(data)
		return self.extractSentences(text)