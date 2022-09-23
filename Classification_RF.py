# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 13:54:17 2022

@author: suhon
"""


import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, roc_auc_score, f1_score,roc_curve

# ROC 시각화
from sklearn.metrics import RocCurveDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, plot_confusion_matrix

fpr_array = np.array([], dtype=np.int32)
tpr_array = np.array([], dtype=np.int32)

def classification_randomforest(trainX, testX, trainY, testY, n_estimators = 100, max_depth = None, criterion = 'gini') :
    """

    Parameters
    ----------
    trainX : array
        Train X.
    testX : array
        Test X.
    trainY : array
        Train Y. 이진(binary) 범주형 데이터
    testY : array
        Test Y. 이진(binary) 범저형 데이터
    n_estimators : int, optional, default = 100
        의사 결정 나무의 갯수
    max_depth : int, optional, default = None
        의사 결정 나무의 최대 깊이. 의사 결정 나무가 최대 깊이에 도달하면 더 이상 분기하지 않음.
        None 입력 시 각 노드에 1개의 샘플이 남을 때까지 분기함
    criterion : {'gini', 'entropy', 'log_loss'}, default = "gini"
        의사 결정 나무의 분기 기준. 

    Returns
    -------
    model :
        RandomForest 모델.
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
    # 데이터가 DataFrame이면 array타입으로 변경
    if isinstance(trainX, pd.DataFrame) :
        trainX = trainX.values
    if isinstance(testX, pd.DataFrame) :
        testX = testX.values
    if isinstance(trainY, pd.DataFrame) :
        trainY = trainY.values
    if isinstance(testY, pd.DataFrame) :
        testY = testY.values
        
    # 모델링
    model = RandomForestClassifier(n_estimators = n_estimators, max_depth = max_depth, criterion = criterion)
    model.fit(trainX, trainY)
    
    # 예측 값
    pred = model.predict(testX)
    
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


if __name__ == '__main__' :
    # Data Load
    shot6_1 = pd.read_csv('D:\\Google 드라이브\\데이터 분석\\프로젝트\\엑센솔루션\\data\\shot6_3D365-48910.csv')
    shot6_2 = pd.read_csv('D:\\Google 드라이브\\데이터 분석\\프로젝트\\엑센솔루션\\data\\shot6_3D365-48921.csv')
    shot6_3 = pd.read_csv('D:\\Google 드라이브\\데이터 분석\\프로젝트\\엑센솔루션\\data\\shot6_3D365-48922.csv')
    shot8_1 = pd.read_csv('D:\\Google 드라이브\\데이터 분석\\프로젝트\\엑센솔루션\\data\\shot8_32780-L2100.csv')
    
# index : 0~32 실제 값
# index : 33~ 세팅 값

# train 정상, test 정상

df = shot6_1
shot6_3['판정코드'].value_counts()
trainX, testX, trainY, testY = train_test_split(df.drop(['판정코드', '순번'], axis = 1), df['판정코드'], test_size=0.3, random_state=11, shuffle = False)

d = classification_randomforest(trainX, testX, trainY, testY)
plot_confusion_matrix(d['model'],testX, testY, cmap = plt.cm.Blues, labels = [1, 0])
d

pred_prob = d['model'].predict_proba(testX)
pred = d['model'].predict(testX)


# fpr_array = np.array([],dtype=np.int32)
# tpr_array = np.array([],dtype=np.int32)

# for i in range(1, 100):
#     predd = np.array([], dtype=np.int32)
#     for j in range(len(pred_prob)):
#        decision_boundary = i / 100
#        if (pred_prob[j][1] > decision_boundary) == True:
#             predd = np.append(predd, 1)
#        else:
#             predd = np.append(predd, 0)
    
#     cofMatt = confusion_matrix(testY, predd, labels=[1, 0])  # 1: 불량, 0: 정상
#     ##confusion matrix의 각 요소
#     tpp = cofMatt[1, 1]
#     tnn = cofMatt[0, 0]
#     fnn = cofMatt[1, 0]
#     fpp = cofMatt[0, 1]
#     tpr = tpp / (tpp + fnn)
#     fpr = fpp / (tnn + fpp)
    
#     fpr_array = np.append(fpr_array, fpr)
#     tpr_array = np.append(tpr_array, tpr)

plt.plot(tpr_array, fpr_array)

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
