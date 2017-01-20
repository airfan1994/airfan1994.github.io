# -*-coding:utf-8 -*-
import sys
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split 
reload(sys)
sys.setdefaultencoding('utf-8')

def totalScore(pred,y_test):
    numr = 0
    for i in range(len(pred)):
        if y_test[i] == pred[i]:
            numr+=1
    
    print 'ratio'
    print numr*1.0/len(pred)

def lrClassifier_pred(trainDatafile,trainLabelfile,testDatafile,testlabelfile):
    test_label = open(testlabelfile,'w')
    df_train_feature = pd.read_csv(trainDatafile,header=None)
    train_feature = np.array(df_train_feature,dtype=float)
    scale_train_feature = preprocessing.scale(train_feature)
    df_train_label = pd.read_csv(trainLabelfile,header=None)
    train_label = np.array(df_train_label,dtype=int)
    clf = LogisticRegression()
    clf.fit(scale_train_feature, train_label)
    df_test_feature = pd.read_csv(testDatafile,header=None)
    test_feature = np.array(df_test_feature,dtype=float)
    scale_test_feature = preprocessing.scale(test_feature)
    predict_labels = clf.predict(scale_test_feature)
    for i in xrange(len(predict_labels)):
            test_label.write(str(predict_labels[i])+"\n")
    
def lrClassifier_test(trainDatafile,trainLabelfile):
	df_train_feature = pd.read_csv(trainDatafile,header=None)
	train_feature = np.array(df_train_feature,dtype=float)
	scale_train_feature = preprocessing.scale(train_feature)
	df_train_label = pd.read_csv(trainLabelfile,header=None)
	train_label = np.array(df_train_label,dtype=int)
	
	trainData, testData, trainLabel, testLabel = train_test_split(scale_train_feature, train_label, test_size = 0.2) 
	
        clf = LogisticRegression()
	clf.fit(trainData, trainLabel)

	predict_labels = clf.predict(testData)
	totalScore(predict_labels,testLabel)
    
    
    #clf = LinearSVC( C= 0.8)
    #clf.fit(tfidf_train,np.array(trainLabel))  
    #pred = clf.predict(tfidf_test) 
    #clf = LogisticRegression()
    #clf = SVC()
    #scale_train_feature = preprocessing.scale(tfidf_train)
    #scale_test_feature = preprocessing.scale(tfidf_test)
    #clf.fit(tfidf_train, trainLabel)
    #pred = clf.predict(tfidf_test)
    
    #totalScore(pred,testData,testlabel)
	
def LinearSVCClassifier_pred(trainDatafile,trainLabelfile,testDatafile,testlabelfile):
    test_label = open(testlabelfile,'w')
    df_train_feature = pd.read_csv(trainDatafile,header=None)
    train_feature = np.array(df_train_feature,dtype=float)
    scale_train_feature = preprocessing.scale(train_feature)
    df_train_label = pd.read_csv(trainLabelfile,header=None)
    train_label = np.array(df_train_label,dtype=int)
    clf = LinearSVC( C= 0.8)
    clf.fit(scale_train_feature, train_label)
    df_test_feature = pd.read_csv(testDatafile,header=None)
    test_feature = np.array(df_test_feature,dtype=float)
    scale_test_feature = preprocessing.scale(test_feature)
    predict_labels = clf.predict(scale_test_feature)
    for i in xrange(len(predict_labels)):
            test_label.write(str(predict_labels[i])+"\n")
    
def LinearSVCClassifier_test(trainDatafile,trainLabelfile):
	df_train_feature = pd.read_csv(trainDatafile,header=None)
	train_feature = np.array(df_train_feature,dtype=float)
	scale_train_feature = preprocessing.scale(train_feature)
	df_train_label = pd.read_csv(trainLabelfile,header=None)
	train_label = np.array(df_train_label,dtype=int)
	
	trainData, testData, trainLabel, testLabel = train_test_split(scale_train_feature, train_label, test_size = 0.2) 
	
	clf = LinearSVC( C= 0.8)
	clf.fit(trainData, trainLabel)

	predict_labels = clf.predict(testData)
	totalScore(predict_labels,testLabel)

def SVMClassifier_pred(trainDatafile,trainLabelfile,testDatafile,testlabelfile):
    test_label = open(testlabelfile,'w')
    df_train_feature = pd.read_csv(trainDatafile,header=None)
    train_feature = np.array(df_train_feature,dtype=float)
    scale_train_feature = preprocessing.scale(train_feature)
    df_train_label = pd.read_csv(trainLabelfile,header=None)
    train_label = np.array(df_train_label,dtype=int)
    clf = SVC()
    clf.fit(scale_train_feature, train_label)
    df_test_feature = pd.read_csv(testDatafile,header=None)
    test_feature = np.array(df_test_feature,dtype=float)
    scale_test_feature = preprocessing.scale(test_feature)
    predict_labels = clf.predict(scale_test_feature)
    for i in xrange(len(predict_labels)):
            test_label.write(str(predict_labels[i])+"\n")
    
def SVMClassifier_test(trainDatafile,trainLabelfile):
	df_train_feature = pd.read_csv(trainDatafile,header=None)
	train_feature = np.array(df_train_feature,dtype=float)
	scale_train_feature = preprocessing.scale(train_feature)
	df_train_label = pd.read_csv(trainLabelfile,header=None)
	train_label = np.array(df_train_label,dtype=int)
	
	trainData, testData, trainLabel, testLabel = train_test_split(scale_train_feature, train_label, test_size = 0.2) 
	
	clf = SVC()
	clf.fit(trainData, trainLabel)

	predict_labels = clf.predict(testData)
	totalScore(predict_labels,testLabel)

def rfClassifier_pred(trainDatafile,trainLabelfile,testDatafile,testlabelfile):
    test_label = open(testlabelfile,'w')
    df_train_feature = pd.read_csv(trainDatafile,header=None)
    train_feature = np.array(df_train_feature,dtype=float)
    scale_train_feature = preprocessing.scale(train_feature)
    df_train_label = pd.read_csv(trainLabelfile,header=None)
    train_label = np.array(df_train_label,dtype=int)
    clf = SVC()
    clf.fit(scale_train_feature, train_label)
    df_test_feature = pd.read_csv(testDatafile,header=None)
    test_feature = np.array(df_test_feature,dtype=float)
    scale_test_feature = preprocessing.scale(test_feature)
    predict_labels = clf.predict(scale_test_feature)
    for i in xrange(len(predict_labels)):
            test_label.write(str(predict_labels[i])+"\n")
    
def rfClassifier_test(trainDatafile,trainLabelfile):
	df_train_feature = pd.read_csv(trainDatafile,header=None)
	train_feature = np.array(df_train_feature,dtype=float)
	scale_train_feature = preprocessing.scale(train_feature)
	df_train_label = pd.read_csv(trainLabelfile,header=None)
	train_label = np.array(df_train_label,dtype=int)
	
	trainData, testData, trainLabel, testLabel = train_test_split(scale_train_feature, train_label, test_size = 0.2) 
	
	clf = SVC()
	clf.fit(trainData, trainLabel)

	predict_labels = clf.predict(testData)
	totalScore(predict_labels,testLabel)
	
def gbdtClassifier_pred(trainDatafile,trainLabelfile,testDatafile,testlabelfile):
    test_label = open(testlabelfile,'w')
    df_train_feature = pd.read_csv(trainDatafile,header=None)
    train_feature = np.array(df_train_feature,dtype=float)
    scale_train_feature = preprocessing.scale(train_feature)
    df_train_label = pd.read_csv(trainLabelfile,header=None)
    train_label = np.array(df_train_label,dtype=int)
    clf = SVC()
    clf.fit(scale_train_feature, train_label)
    df_test_feature = pd.read_csv(testDatafile,header=None)
    test_feature = np.array(df_test_feature,dtype=float)
    scale_test_feature = preprocessing.scale(test_feature)
    predict_labels = clf.predict(scale_test_feature)
    for i in xrange(len(predict_labels)):
            test_label.write(str(predict_labels[i])+"\n")
    
def gbdtClassifier_test(trainDatafile,trainLabelfile):
	df_train_feature = pd.read_csv(trainDatafile,header=None)
	train_feature = np.array(df_train_feature,dtype=float)
	scale_train_feature = preprocessing.scale(train_feature)
	df_train_label = pd.read_csv(trainLabelfile,header=None)
	train_label = np.array(df_train_label,dtype=int)
	
	trainData, testData, trainLabel, testLabel = train_test_split(scale_train_feature, train_label, test_size = 0.2) 
	
	clf = SVC()
	clf.fit(trainData, trainLabel)

	predict_labels = clf.predict(testData)
	totalScore(predict_labels,testLabel)
	
def main():
	#time python predict.py pred vec_train.txt label_A_train.txt vec_test.txt label_A_test2000.txt svm
	#time python predict.py test vec_train.txt label_A_train.txt svm
	if sys.argv[1]=='pred':
		trainDatafile,trainLabelfile,testDatafile,testlabelfile = sys.argv[2:6]
		if sys.argv[6]=='svm':
			SVMClassifier_pred(trainDatafile,trainLabelfile,testDatafile,testlabelfile)
		elif sys.argv[6]=='lr':
			lrClassifier_pred(trainDatafile,trainLabelfile,testDatafile,testlabelfile)
		elif sys.argv[6]=='linearsvm':
			LinearSVCClassifier_pred(trainDatafile,trainLabelfile,testDatafile,testlabelfile)
		else:
			print "not supported"
	elif sys.argv[1]=='test':
		trainDatafile,trainLabelfile= sys.argv[2:4]
		if sys.argv[6]=='svm':
			SVMClassifier_test(trainDatafile,trainLabelfile)
		elif sys.argv[6]=='lr':
			lrClassifier_test(trainDatafile,trainLabelfile)
		elif sys.argv[6]=='linearsvm':
			LinearSVCClassifier_test(trainDatafile,trainLabelfile)
		else:
			print "not supported"
		lrClassifier_test(trainDatafile,trainLabelfile)
	else:
		"serious error!"
	
if __name__=='__main__':
	main()
