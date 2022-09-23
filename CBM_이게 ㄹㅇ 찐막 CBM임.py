"""
Created on Thur Jul  7 16:20:39 2022
Revised on Thur Jul  13 15:20:39 2022

@author: Junhyun
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import joblib

def bootstrap_limit(stat, alpha=0.05, bootstrap=100):
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
    alpha = alpha*100
    alpha = 100 - alpha
    samsize = max(100, len(stat))
    
    stat = stat.reshape(len(stat)) # 2차원 array를 1차원 array로 변환
    
    # bootstrap 수 만큼 다음 작업을 반복 : samsize(최소 10000번)만큼 정상상태 데이터를 유의수준 만큼 복원 추출 후 평균 값 사용 
    limit = np.mean(list(map(lambda x:np.percentile(np.random.choice(stat,samsize,replace=True),alpha), range(0,bootstrap))))
    
    return limit

class CBM():
    """
    CBM : Clustering Base Mahalanobis distance
    """
    
    def __init__(self):
        
        self.tr_k_mu = None
        self.tr_k_cov = None
        self.cl = None
        self.n_clusters = None
        
    def fit(self, trdat, n_clusters=3, max_iter=100, alpha=0.05):
        """

        Parameters
        ----------
        trdat : array
            Train data
        n_clusters : int
            cluster의 수
        max_iter : int
            kmeans clustering 학습 횟수        
        alpha : int, 0~1
            Bootstrap Limit value. The default is 0.05.

        Returns
        -------
        trScore : array
            Train Score, 이상치 점수를 의미함. 클수록 정상패턴에서 벗어남을 의미
        CL : float
            trScore Control Limit

        """
        
        km = KMeans(n_clusters=n_clusters,max_iter=max_iter,random_state=0).fit(trdat)
        km_score = km.predict(trdat)
        cluster_df = pd.concat([pd.DataFrame(trdat),pd.DataFrame(km_score, columns=['cluster'])], axis=1)

        # 실제 사용할 클러스터 개수 (클러스터 내 관측치가 한개인 데이터 제거)
        cluster_cnt = np.unique(km_score, return_counts = True)[1]
        clusterId = np.where(cluster_cnt!=1)[0]

        cluster_df = cluster_df[cluster_df['cluster'].isin(clusterId)]
        
        
        if isinstance(trdat,(np.ndarray)):
            trdat = pd.DataFrame(trdat)
            
        km = KMeans(n_clusters=n_clusters,max_iter=max_iter,random_state=0).fit(trdat)
        km_score = km.predict(trdat)
        cluster_df = pd.concat([pd.DataFrame(trdat),pd.DataFrame(km_score, columns=['cluster'])], axis=1)
        
        # kmeans 예외처리
        cluster_cnt = np.unique(km_score, return_counts = True)[1]
        clusterId = np.where(cluster_cnt!=1)[0]
        cluster_df = cluster_df[cluster_df['cluster'].isin(clusterId)].reset_index(drop=True)
        
        self.tr_k_mu = np.zeros((trdat.shape[1], len(clusterId)))
        self.tr_k_cov = np.zeros((len(clusterId), trdat.shape[1], trdat.shape[1]))
        self.n_clusters = len(clusterId)
        
        
        for i in range(len(clusterId)):
            
            self.tr_k_mu[:,i] = cluster_df.drop('cluster', axis=1)[cluster_df['cluster'] == clusterId[i]].mean()
            self.tr_k_cov[i] = cluster_df.drop('cluster', axis=1)[cluster_df['cluster'] == clusterId[i]].cov()

        trCbmMat = np.zeros((len(trdat), self.n_clusters))
        
        # train fit
        for i in range(len(trdat)):
            for j in range(self.n_clusters):
                # cluster 별 마할라노비스 계산
                trCbmMat[i,j] = (trdat.values[i,:] - self.tr_k_mu[:,j]) @ np.linalg.pinv(self.tr_k_cov[j]) @ (trdat.values[i,:] - self.tr_k_mu[:,j]).transpose()
        
        # CBM 이상감지 통계량
        self.trScore = trCbmMat.min(axis=1)
        self.cl = bootstrap_limit(self.trScore, alpha=alpha, bootstrap=100)
        
        return {"trScore" : self.trScore}
    
    def CL_printor(self) :
        """
        
        Returns
        -------
        CL: float
            Control Limit,
            
        """
        
        return {'CL' : self.cl}
    
    def predict(self, tsdat):
        """

        Parameters
        ----------
        tsdat : array
            Test data. 예측 대상이 되는 데이터

        Returns
        -------
        tsScore : array
            Test data의 이상치 값

        """
        if isinstance(tsdat,(np.ndarray)):
            tsdat = pd.DataFrame(tsdat)

        tsCbmMat = np.zeros((len(tsdat), self.n_clusters))     
        
        # test fit
        for i in range(len(tsdat)):
            for j in range(self.n_clusters):
                # cluster 별 마할라노비스 계산
                tsCbmMat[i,j] = (tsdat.values[i,:] - self.tr_k_mu[:,j]) @ np.linalg.pinv(self.tr_k_cov[j]) @ (tsdat.values[i,:] - self.tr_k_mu[:,j]).transpose()
    
        # CBM 이상감지 통계량
        tsScore = tsCbmMat.min(axis=1)
    
        return {"tsScore" : tsScore}   


def cbm(trdat, tsdat, n_clusters=3, alpha=0.05):
    """

    Parameters
    ----------
    trdat : array
        Train data. 학습 대상이 되는 데이터
    tsdat : array
        Test data. 예측 대상이 되는 데이터
    n_clusters : int
        클러스터의 개수
    alpha : float, 0~1
            Bootstrap Limit value. The default is 0.05.

    Returns
    -------
    trScore : array
        Train data의 이상치 값
    tsScore : array
        Test data의 이상치 값
    CL : float 
        Control Limit

    """
    model = CBM()
    fit = model.fit(trdat, n_clusters=n_clusters, max_iter=100, alpha=alpha)
    CL = model.CL_printor()
    pred = model.predict(tsdat)
    
     # CBM model pickle 파일로 저장
    saved_model = joblib.dump(model, 'cbm.pkl')
    
    return {'trScore':fit['trScore'], 'tsScore':pred['tsScore'], 'CL': CL['CL']}     


# Testing Model load
import joblib
def CBM_model_loader(pickleFile, tsdat) :
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
    model = joblib.load(pickleFile)  
    CL = model.CL_printor()
    pred = model.predict(tsdat)

    return {'tsScore' : pred['tsScore'], 'CL' : CL['CL']}


df = pd.read_csv('./data/titanic.csv', encoding='euc-kr')

df = df.dropna()

# +
# df = pd.read_csv('test_data.csv', encoding='euc-kr')

trdat = df.values[0:600,:]
tsdat = df.values[600:891, :]
# -

cb = cbm(trdat, tsdat, n_clusters=5, alpha=0.05)

cb


