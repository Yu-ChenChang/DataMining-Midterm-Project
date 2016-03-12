import os
import csv
import re
import sys
import random
import math
from collections import defaultdict
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import svm, cross_validation
from sklearn.metrics import f1_score, roc_curve, auc
from scipy import interp, stats
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class DataPreprocessor:
	def __init__(self,trainingFileName, testingFileName):
		self.trainData = pd.read_csv(trainingFileName)
		self.testData = pd.read_csv(testingFileName)
		self.oneOfKEncoding('FICO Range')
		self.oneOfKEncoding('Loan Purpose')

	def oneOfKEncoding(self,ColName):
		encoded = pd.get_dummies(pd.concat([self.trainData[ColName],self.testData[ColName]], axis=0),\
				prefix=ColName, dummy_na=True)
		train_rows = self.trainData.shape[0]
		self.trainData[ColName] = encoded.iloc[:train_rows, :]
		self.testData[ColName] = encoded.iloc[train_rows:, :]

	def getTrainLabel(self):
		return self.trainData['Status']

	def getFeature(self,flag):
		#FEATURE = ['Amount Requested','Interest Rate Percentage','Loan Length in Months','Loan Purpose','Monthly PAYMENT','Total Amount Funded','Debt-To-Income Ratio Percentage','FICO Range']
		FEATURE = ['Interest Rate Percentage','Loan Purpose','Monthly PAYMENT','Debt-To-Income Ratio Percentage','FICO Range']
		if flag == 'train':
			return np.array(self.trainData[FEATURE])
		elif flag == 'test':
			return np.array(self.testData[FEATURE])

def crossValidate(data,foldNum,model):
	f1 = cross_validation.cross_val_score(model, data.getFeature('train'), data.getTrainLabel(),
	 cv=foldNum, scoring='f1_weighted')
	print np.mean(f1)
	return f1
	

def classify(data,foldNum):
	print '=== %d Fold Cross Validation ===' %foldNum
	kf = cross_validation.KFold(len(data.getFeature('train')), n_folds = foldNum, shuffle  = False, random_state = True)
	print '--- NBC ---'
	gnb = GaussianNB()
	gnbF1 = crossValidate(data,foldNum,gnb)
	print '--- Logistic Regression ---'
	lgr = LogisticRegression()
	lgrF1 = crossValidate(data,foldNum,lgr)
	print '--- Linear SVM ---'
	svmL = svm.SVC(C = 1, kernel = 'linear',max_iter=10000)
	svmLF1 =crossValidate(data,foldNum,svmL)
	print '--- Gaussian SVM ---'
	svmR = svm.SVC(C = 1, kernel = 'rbf')
	svmRF1 = crossValidate(data,foldNum,svmR)
	return gnbF1,lgrF1,svmLF1,svmRF1

def pltSetUp(plt):
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.0])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('ROC curve for Logistic Regression')
	plt.legend(loc="lower right")
	plt.show()		

def plotROC(foldNum,data,kf):
	i = 0
	tprMean, fprMean = 0.0, np.linspace(0, 1, 100)
	tprAll = []
	log = LogisticRegression()
	for trainInd, testInd in kf:
		i = i + 1
		proba = log.fit(data.getFeature('train')[trainInd], data.getTrainLabel()[trainInd]).predict_proba(data.getFeature('train')[testInd])
		fpr, tpr, thresholds = roc_curve(data.getTrainLabel()[testInd], proba[:, 1])
		roc_auc = auc(fpr, tpr)
		plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
		tprMean += interp(fprMean, fpr, tpr)
		tprMean[0] = 0.0
	plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Rand')
	tprMean /= len(kf)
	tprMean[-1] = 1.0
	mean_auc = auc(fprMean, tprMean)
	plt.plot(fprMean, tprMean, 'k--', label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)
	pltSetUp(plt)

def bestResult(data,kf,outname):
	log = LogisticRegression()
	[log.fit(data.getFeature('train'), data.getTrainLabel()).score(data.getFeature('train'), data.getTrainLabel()) for train, test in kf]
	result = log.predict(data.getFeature('test'))
	f1 = pd.DataFrame({'Status': result,'Record Number':data.testData['Record Number']})
	f1.to_csv(outname, sep=',',index = False)
	
def t_test(data):
	gnbF1,lgrF1,svmLF1,svmRF1  = classify(data,50)
	print '----t-test----'
	print stats.ttest_rel(lgrF1, gnbF1)
	print stats.ttest_rel(lgrF1, svmLF1)
	print stats.ttest_rel(lgrF1, svmRF1)
	print '--------------'

if __name__ == '__main__':
	data = DataPreprocessor('Bank_Data_Train.csv', 'Bank_Data_Test.csv')
	## 5-fold ##
	classify(data,5)

	## 10-fold ##
	classify(data,10)

	## plot ROC ##
	kf = cross_validation.KFold(len(data.getFeature('train')), n_folds = 5, shuffle  = False, random_state = True)
	plotROC(5,data,kf)	

	## write result ##
	kf = cross_validation.KFold(len(data.getFeature('train')), n_folds = 10, shuffle  = False, random_state = True)
	bestResult(data,kf,'Result.csv')	

	## t-test ##
	t_test(data)

	## For Libsvm ##
	'''
	result = []
	with open('out') as fd:
		for line in fd:
			result += [line.split()[0]]
	'''
		
	
	'''
	X = data.getFeature('train')
	Y = data.getTrainLabel()
	with open('test_mid_1.txt','w') as f:
		for j in range(X.shape[0]):
			f.write(" ".join(
					  [str(int(Y[j]))] + ["{}:{}".format(i+1,X[j][i]) 
					  for i in range(X.shape[1]) if X[j][i] != 0])+'\n')
		
	X = data.getFeature('test')
	with open('test_mid_test.txt','w') as f:
		for j in range(X.shape[0]):
			f.write(" ".join(
					  [str(0)] + ["{}:{}".format(i+1,X[j][i]) 
					  for i in range(X.shape[1]) if X[j][i] != 0])+'\n')
	'''
