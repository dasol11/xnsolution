# -*- coding: utf-8 -*-
"""
Created on FRI Sep 16 12:31:29 2022

@author: SuHong

2.0 : DT 결과 text로 변환
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix,f1_score,roc_curve
from sklearn import tree

from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.datasets import load_iris
# 결정 나무 시각화
# plt.figure(figsize=(10,8))
# tree.plot_tree(model, 
#                class_names = iris.target_names,
#                feature_names = iris.feature_names,
#                impurity = True, filled = True,
#                rounded = True)

def plot_roc_curve(fprs, tprs):
    plt.plot(fprs, tprs, color='red', label='ROC')
    plt.plot([0, 1], [0, 1], color='black', linestyle='--', linewidth=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend()
    plt.show()

def ROCcurve_plot(model, testX, testY) :
    probs = model.predict_proba(testX)
    prob = probs[:, 1]
    fper, tper, thresholds = roc_curve(testY, prob)
    
    return plot_roc_curve(fper, tper)

# ROCcurve_plot(d['model'], testX, testY)


def classification_decisiontree(trainX, testX, trainY, testY, criterion = 'gini', max_depth = None, min_samples_leaf = 1) :
    """

    Parameters
    ----------
    trainX : array
        Train X.
    testX : array
        Test X.
    trainY : array
        Train Y.
    testY : array
        Test Y.
    criterion : {'gini', 'entropy', 'log_loss'}, default = "gini"
        의사 결정 나무의 분기 기준. 
    max_depth : int, optional, default = None
        의사 결정 나무의 최대 깊이. 의사 결정 나무가 최대 깊이에 도달하면 더 이상 분기하지 않음.
    min_samples_leaf : int or float, optional, default = 2
        노드 별 최소 샘플 갯수, 터미널 노드안에 샘플 수가 min_samples_split과 같아지면 더 이상 분기하지 않음.

    Returns
    -------
    model :
        분류 의사결정나무 모델.
    confusionMatrix : array
        분류 성능 평가를 위한 Confusion Matrix
    fpr : array
        False Positive Rate
    tpr : array
        True Positive Rate
    auc_score : float
        fpr, tpr의 AUC(Area Under Cruve) scroe (0~1). 1에 가까울 수록 분류 성능이 좋음을 의미함
    accuracy : float
        accuracy.
    precision : float
        precision
    recall : float
        recall
    f1 : float
        f1 score
    tp : int
        True positive
    fp : int
        False positive
    fn : int
        False negative
    tn : int
        True negative
    """
    
    if isinstance(trainX, pd.DataFrame) :
        trainX = trainX.values
    if isinstance(testX, pd.DataFrame) :
        testX = testX.values
    if isinstance(trainY, pd.DataFrame) :
        trainY = trainY.values
    if isinstance(testY, pd.DataFrame) :
        testY = testY.values
    
    # Decision Tree Model
    model = DecisionTreeClassifier(criterion=criterion, max_depth = max_depth, min_samples_leaf = min_samples_leaf)
    model.fit(trainX, trainY)
    
    # 예측
    pred = model.predict(testX) # 예측 분류 값
    
    # 분류 평가 지표
    cofMat = confusion_matrix(testY, pred, labels=[1, 0])  # 1: 불량, 0: 정상
    # confusion matrix의 각 요소
    tp = cofMat[1, 1]
    tn = cofMat[0, 0]
    fn = cofMat[1, 0]
    fp = cofMat[0, 1]
    
    accuracy = accuracy_score(testY, pred)
    precision = precision_score(testY, pred)
    recall = recall_score(testY, pred)
    f1 = f1_score(testY, pred)

    # calculate AUC of model
    auc = roc_auc_score(testY, pred)
    fpr, tpr, thresholds = roc_curve(testY, pred)
    
    return ({"model" : model, "confusionMatrix": cofMat, "fpr": fpr, "tpr": tpr, "auc_score":auc,
             "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn,
             "tn": tn})  # type : int32

# ## Data : Iris

iris = load_iris()
idx = iris['target'] != 2
x = iris['data'][idx,:]
y = iris['target'][idx]

trainX, testX, trainY, testY = train_test_split(x, y, test_size=0.3, random_state=11)

d = classification_decisiontree(trainX, testX, trainY, testY)

plot_confusion_matrix(d['model'],testX, testY, cmap = plt.cm.Blues, labels = [1, 0])
d

from sklearn.tree import export_text

d['model']

r = export_text(d['model'], feature_names=iris['feature_names'])
r
