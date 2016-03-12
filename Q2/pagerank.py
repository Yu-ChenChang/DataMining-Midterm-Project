import sys
import scipy
import csv
import operator
from scipy import sparse
import numpy as np 
import networkx as nx

class PageRanker():
	def __init__(self, filename,outname):
		self.nodeCandidate = []
		self.graph = self.readFromFile(filename)
		#self.personalizedPageRank()
		Result = ''
		for i,n in enumerate(self.nodeCandidate):
			if i%100 == 0:
				print i
			Result += self.personalizedPageRank(n)
		self.writeToFile(outname,Result)

	def personalizedPageRank(self,rootID):
		personalize = dict((n, 0) for n in self.graph)
		personalize[rootID] =1 
		x = nx.pagerank_scipy(self.graph, alpha=0.15, tol=1.e-05, personalization=personalize)
		sorted_x = sorted(x.items(), key=operator.itemgetter(1),reverse=True)
		count = 0
		result = ''
		for key in sorted_x:
			if not self.graph.has_edge(rootID,key[0]) and rootID != key[0]:
				count += 1
				result += rootID+','+key[0]+'\n'
			if count == 5:
				break
		return result

	def writeToFile(self, filename, result):
		with open(filename,'w') as out:
			out.write(result)
		
	def readFromFile(self, filename):
		edgeCount = 0
		leftNode = 0
		graph = nx.Graph()
		with open(filename, 'rb') as fd:
			reader = csv.reader(fd, delimiter=',')
			for row in reader:
				if leftNode == row[0]:
					edgeCount = edgeCount + 1
					if edgeCount >= 10 and row[0] not in self.nodeCandidate:
							self.nodeCandidate += [row[0]]
					else:
						graph.add_edge(row[0],row[1])
				else:
					edgeCount = 1;
					leftNode = row[0]
					graph.add_edge(row[0],row[1])
		return graph

if __name__ == '__main__':
	ranker = PageRanker('edges.csv','out.csv')
