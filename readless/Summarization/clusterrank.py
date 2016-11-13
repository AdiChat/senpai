#!/usr/bin/python
# *****************************************************************************
#
# Author: Aditya Chatterjee
#
# Interweb/ contacts: GitHub.com/AdiChat
#                     Email: aditianhacker@gmail.com
#
# Implementation of the ClusterRank Algorithm
#
# MIT License
#
# To keep up with the latest version, consult repository: GitHub.com/AdiChat/Read-Less
#
# To get an overview of the ClusterRank Algorithm, consult wiki: Github.com/AdiChat/Read-Less/wiki/ClusterRank
#
# *****************************************************************************
import io
import nltk
import itertools
from operator import itemgetter
import networkx as nx
import os
import texttiling
import parse

class ClusterRank():

	def __init__(self):
		print "Cluster Rank"

	def lDistance(self, firstString, secondString):
		'''
		Finds the levenshtein distance between 2 strings
		Arguments:
			firstString: first input string
			secondString: second input string
		Returns:
			the levenshtein distance between the two input strings
		Raises:
			None
		'''
	    if len(firstString) > len(secondString):
	        firstString, secondString = secondString, firstString
	    distances = range(len(firstString) + 1)
	    for index2, char2 in enumerate(secondString):
	        newDistances = [index2 + 1]
	        for index1, char1 in enumerate(firstString):
	            if char1 == char2:
	                newDistances.append(distances[index1])
	            else:
	                newDistances.append(1 + min((distances[index1], distances[index1+1], newDistances[-1])))
	        distances = newDistances
	    return distances[-1]

	def buildGraph(self, nodes):
		'''
		Builds the graph with a token of words as a node
		Arguments:
			nodes: list of nodes/ token of words
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
	        levDistance = self.lDistance(firstString, secondString)
	        gr.add_edge(firstString, secondString, weight=levDistance)

	    return gr

	def extractSentences(self, text):
		'''
		Extracts sentences from the graph using pagerank
		Arguments:
			text: input textual data
		Returns:
			summary: a bunch of sentences
		Raises:
			None
		'''
	    sentenceTokens = text
	    print "Building graph"
	    graph = self.buildGraph(sentenceTokens)

	    print "Computing page rank"
	    calculated_page_rank = nx.pagerank(graph, weight='weight')

	    #most important sentences in ascending order of importance
	    print "Assigning score to sentences"
	    sentences = sorted(calculated_page_rank, key=calculated_page_rank.get, reverse=True)

	    #return a 100 word summary
	    print "Generating summary"
	    summary = ' '.join(sentences)
	    summaryWords = summary.split()
	    summaryWords = summaryWords[0:201]
	    summary = ' '.join(summaryWords)

	    print "Operation completed"
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