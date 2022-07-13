# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 14:29:39 2022

@author: suhon
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import pickle
from scipy import linalg # covaraince sclaer

def bootstrap_limit(stat, alpha=0.05, bootstrap=100, upper = True):
    '''
        @Description
            Bootstrap sampling을 활용한 Control Limit 산출 기법

        @Parameter
            stat : 통계량 (정상상태의 데이터 입력)
            alpha : Control Limit을 정하기 위한 유의수준 (0~1)
            bootstrap : 샘플링 횟수
        @Return
            limit : 임계값 (CL : Control Limit)
    '''
    
    alpha = alpha * 100
    if(upper) : alpha = 100 - alpha
    samsize = max(100, len(stat))
    
    stat = stat.reshape(len(stat)) # 2차원 array를 1차원 array로 변환
    
    # bootstrap 수 만큼 다음 작업을 반복 : samsize(최소 10000번)만큼 정상상태 데이터를 유의수준 만큼 복원 추출 후 평균 값 사용 
    limit = np.mean(list(map(lambda x:np.percentile(np.random.choice(stat,samsize,replace=True),alpha), range(0,bootstrap))))
    
    return limit

def L2norm(stat):
    return(np.sqrt(stat**2))

def matrix_inv(matrix):
    return linalg.pinv(matrix,cond=1.490116e-08)

class msetRF_covariance_scaler() :
    
    def __init__(self) :
        
        self.cov_inv_matrix = None
            
    def fit(self, trdat):
        if isinstance(trdat, pd.DataFrame) :
            trdat = trdat.values
        cov_mat = np.cov(trdat)
        self.cov_inv_matrix = matrix_inv(cov_mat)
        
    def transform(self, tsdat) :
        if isinstance(tsdat, pd.DataFrame) :
            tsdat = tsdat.values
        if isinstance(tsdat, list) :
            tsdat = np.array(tsdat)
            
        transformed_data = np.dot(np.array(tsdat).transpose(), self.cov_inv_matrix)
        
        return transformed_data

class mset_randomforest() :
    
    def __init__(self) :
        self.forests = [] # 각 변수 별 RandomForest 모델 저장
        self.UCL = None # 전체 UCL
        self.LCL = None # 전체 LCL
        self.varucl = [] # 각 변수 별 UCL
        self.varlcl = [] # 각 변수 별 LCL
        self.ncol_range = None
        self.cov_scaler = msetRF_covariance_scaler()
        
    def fit(self, trdat, ntree = 100, alpha = 0.01) :
        '''

        Parameters
        ----------
        trdat : array
            학습 데이터
        alpha : float, 0~1, default = 0.05
            Control limit의 유의 수준
        ntree : int, optional, default = 100
            RandomForest의 의사 결정 나무의 수

        Returns
        -------
        trScore : array
            Mset RandomForest의 Train 잔차 (이상감지 통계량)
        varTrScore :
            변수 별 trScore
        
        '''

        tr_resi = []
        # dataframe to np.array
        if isinstance(trdat, pd.DataFrame) :
            trdat = trdat.values
        self.ncol_range = range(0, np.shape(trdat)[1])
        
        # 변수 별 RandomForest Regression
        for i in self.ncol_range :
            rf = RandomForestRegressor(n_estimators = ntree)
            train = np.delete(trdat, i, 1) # i번째 column제거
            forest = rf.fit(train, trdat[:,i]) # 나머지 변수로 i번째 column 예측

            tr_pred = forest.predict(train)
            tr_score = trdat[:, i] - tr_pred

            self.forests.append(forest)
            tr_resi.append(tr_score)
            
        tr_resi = np.array(tr_resi)

        # trScore(resi) rowsum
        self.cov_scaler.fit(tr_resi) # cov scaler 학습
        scaled_varTrScore = self.cov_scaler.transform(tr_resi) # 변환
        trScore = L2norm(scaled_varTrScore.sum(axis=1)) # L2norm
        
        ## CL
        # 각 변수 별 CL
        for i in self.ncol_range :
            self.varucl.append(bootstrap_limit(scaled_varTrScore[:, i], alpha=alpha/2))
            self.varlcl.append(bootstrap_limit(scaled_varTrScore[:, i], alpha=alpha/2, upper=False))
        
        # 합산 CL
        self.UCL = bootstrap_limit(trScore, alpha = alpha)
        self.LCL = self.UCL
        
        return {'trScore' : trScore, 'varTrScore' : tr_resi}
    
    def CL_printor(self) :
        """
        Returns
        -------
        UCL, LCL: float
            upper, lower Control Limit,
        varUCL, varLCL :
            변수 별  UCL, LCL
        """
        return {'UCL' : self.UCL, 'LCL' : self.LCL, 'varUCL' : self.varucl, 'varLCL' : self.varlcl}
    
    def predict(self, tsdat) :
        '''

        Parameters
        ----------
        tsdat : array
            예측 데이터
        
        Returns
        -------
        tsScore : array
            Mset RandomForest의 Test 잔차 (이상감지 통계량)
        varTsScore :
            변수 별 tsScore
        
        '''        
        ts_resi = []
        # dataframe to np.array
        if isinstance(tsdat, pd.DataFrame) :
            tsdat = tsdat.values
        
        if tsdat.ndim == 1 :
            tsdat = tsdat.reshape(1,-1)
            delete_axis = None
        
            for i in self.ncol_range :
                test = np.delete(tsdat, i, 1)
                ts_pred = self.forests[i].predict(test)
                ts_score = tsdat[:, i] - ts_pred
            
                ts_resi.append(ts_score)
        else :
            delete_axis = 1
            
            for i in self.ncol_range :
                test = np.delete(tsdat, i, delete_axis)
                ts_pred = self.forests[i].predict(test)
                ts_score = tsdat[:, i] - ts_pred
                
                ts_resi.append(ts_score)
            
        ts_scaled_resi = self.cov_scaler.transform(ts_resi)
        tsScore = L2norm(ts_scaled_resi.sum(axis=1))
        
        return {'tsScore' : tsScore, 'varTsScore' : ts_resi}


def mset_RF(trdat, tsdat, alpha = 0.05, ntree = 100) :
    '''

    Parameters
    ----------
    trdat : array
        학습 데이터
    tsdat : array
        평가 데이터, 예측 데이터
    alpha : float, 0~1, default = 0.05
        Control limit의 유의 수준
    ntree : int, optional, default = 100
        RandomForest의 의사 결정 나무의 수

    Returns
    -------
    trScore : array
        Mset RandomForest의 Train 잔차 (이상감지 통계량)
    tsScore : array
        Mset RandomForest의 Test 잔차 (이상감지 통계량)
    UCL, LCL: float
        upper, lower Control Limit,
    var ... :
        변수 별 trScore, tsScore, UCL, LCL
        

    '''
    
    model = mset_randomforest() # 인스턴스
    fit = model.fit(trdat, alpha = alpha, ntree = ntree) # 모델 fitting
    CL = model.CL_printor()
    pred = model.predict(tsdat)
    
    # model pickle 저장
    saved_model = joblib.dump(model, 'mset_RF.pkl')
    return {'trScore' : fit['trScore'], 'tsScore' : pred['tsScore'], 'UCL' : CL['UCL'], 'LCL' : CL['LCL'],
            'varTrScore' : fit['varTrScore'], 'varTsScore' : pred['varTsScore'], 'varUCL' : CL['varUCL'], 'varLCL' : CL['varLCL']}

# 예제
if __name__ == '__main__' :
    df = pd.read_csv('test_data.csv', encoding='euc-kr')
    
    trdat = df.iloc[0:600, :]
    tsdat = df.iloc[600:650, :]
    
    RF_model = mset_RF(trdat, tsdat, ntree = 10)
    #print(RF_model['trScore'])
    print(RF_model['tsScore'])
    #print(RF_model['UCL'])
    
    # pickle 저장 예시
    mset_rf_form_joblib = joblib.load('mset_RF.pkl') 
    print(mset_rf_form_joblib.predict(tsdat)['tsScore'])
    
    
    # Testing Model load
    def mset_model_loader(model, tsdat) :
        """
        저장한 모델을 로드한 후, 로드한 모델과 데이터를 활용해 분석 결과 리턴
        
        Parameters
        ----------
        model : ?
            로드한 모델
        tsdat : array
            예측 데이터

        Returns
        -------
        모델 리턴과 동일

        """
        CL = model.CL_printor()
        pred = model.predict(tsdat)
        
        return {'tsScore' : pred['tsScore'], 'UCL' : CL['UCL'], 'LCL' : CL['LCL'],
                'varTsScore' : pred['varTsScore'], 'varUCL' : CL['varUCL'], 'varLCL' : CL['varLCL']}
        
    mset_model_loader(mset_rf_form_joblib, tsdat)


