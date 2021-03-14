#!/usr/bin/python

# *****************************************************************************
#
# Author: Aditya Chatterjee
#
# Interweb/ contacts: GitHub.com/AdiChat
#                     Email: aditianhacker@gmail.com
#
# Implementation of the TextRank Algorithm
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
from ..Segmentation import texttiling
from ..Parse import parse


class TextRank():

	def __init__(self):
		print("TextRank: Summarizing textual data")

	def lDistance(self, firstString, secondString):
	    if len(firstString) > len(secondString):
	        firstString, secondString = secondString, firstString
	    distances = range(len(firstString) + 1)
	    for index2, char2 in enumerate(secondString):
	        newDistances = [index2 + 1]
	        for index1, char1 in enumerate(firstString):
	            if char1 == char2:
	                newDistances.append(distances[index1])
	            else:
	                newDistances.append(
	                	1 + min((distances[index1], distances[index1+1], newDistances[-1])))
	        distances = newDistances
	    return distances[-1]

	def buildGraph(self, nodes):
	    gr = nx.Graph()
	    gr.add_nodes_from(nodes)
	    nodePairs = list(itertools.combinations(nodes, 2))

	    for pair in nodePairs:
	        firstString = pair[0]
	        secondString = pair[1]
	        levDistance = self.lDistance(firstString, secondString)
	        gr.add_edge(firstString, secondString, weight=levDistance)

	    return gr

	def extractSentences(self, text):

	    sentenceTokens = text
	    graph = self.buildGraph(sentenceTokens)
	    calculated_page_rank = nx.pagerank(graph, weight='weight')
	    sentences = sorted(calculated_page_rank,
	                       key=calculated_page_rank.get, reverse=True)
	    summary = ' '.join(sentences)
	    summaryWords = summary.split()
	    summaryWords = summaryWords[0:201]
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