"""
🎯 문제: 결측치 처리 전략 구현

다양한 결측치 처리 방법을 구현하고, 데이터 특성에 따라 적절한 방법을 선택하는 클래스를 만드세요.

요구사항:
1. 클래스 기반 구현 (MissingValueHandler)
2. 수치형/범주형 변수 구분 처리
3. 다양한 대체 전략 구현
4. 결측치 패턴 분석 기능
5. 처리 전후 비교 시각화

구현할 대체 전략:
- 평균/중앙값/최빈값 대체
- 전진/후진 채우기 (시계열)
- KNN 기반 대체
- 선형 보간
- 상수값 대체
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union, Dict, List, Optional
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

class MissingValueHandler:
    def __init__(self):
        """
        결측치 처리 핸들러 초기화
        """
        # TODO: 초기화 구현
        pass
    
    def analyze_missing_pattern(self, df: pd.DataFrame) -> Dict:
        """
        결측치 패턴 분석
        
        Args:
            df: 분석할 데이터프레임
            
        Returns:
            analysis: 결측치 분석 결과
        """
        # TODO: 결측치 패턴 분석 구현
        # 1. 각 컬럼별 결측치 비율
        # 2. 결측치 조합 패턴
        # 3. 결측치 상관관계
        pass
    
    def fill_numerical(self, series: pd.Series, strategy: str = 'mean', **kwargs) -> pd.Series:
        """
        수치형 변수 결측치 처리
        
        Args:
            series: 처리할 시리즈
            strategy: 대체 전략 ('mean', 'median', 'mode', 'constant', 'interpolate', 'knn')
            **kwargs: 추가 파라미터
            
        Returns:
            filled_series: 결측치가 처리된 시리즈
        """
        # TODO: 수치형 결측치 처리 구현
        pass
    
    def fill_categorical(self, series: pd.Series, strategy: str = 'mode', **kwargs) -> pd.Series:
        """
        범주형 변수 결측치 처리
        
        Args:
            series: 처리할 시리즈
            strategy: 대체 전략 ('mode', 'constant', 'knn')
            **kwargs: 추가 파라미터
            
        Returns:
            filled_series: 결측치가 처리된 시리즈
        """
        # TODO: 범주형 결측치 처리 구현
        pass
    
    def knn_impute(self, df: pd.DataFrame, target_col: str, n_neighbors: int = 5) -> pd.Series:
        """
        KNN 기반 결측치 대체
        
        Args:
            df: 전체 데이터프레임
            target_col: 대체할 컬럼명
            n_neighbors: 이웃 수
            
        Returns:
            imputed_series: KNN으로 대체된 시리즈
        """
        # TODO: KNN 기반 대체 구현
        pass
    
    def fit_transform(self, df: pd.DataFrame, strategies: Dict[str, str] = None) -> pd.DataFrame:
        """
        전체 데이터프레임에 대한 결측치 처리
        
        Args:
            df: 처리할 데이터프레임
            strategies: 컬럼별 처리 전략 딕셔너리
            
        Returns:
            processed_df: 처리된 데이터프레임
        """
        # TODO: 전체 데이터프레임 처리 구현
        pass
    
    def plot_missing_pattern(self, df: pd.DataFrame):
        """
        결측치 패턴 시각화
        """
        # TODO: 결측치 패턴 시각화 구현
        pass
    
    def compare_before_after(self, original_df: pd.DataFrame, processed_df: pd.DataFrame):
        """
        처리 전후 비교 시각화
        """
        # TODO: 전후 비교 시각화 구현
        pass

# 테스트 코드
if __name__ == "__main__":
    # 샘플 데이터 생성 (결측치 포함)
    np.random.seed(42)
    
    data = {
        'age': np.random.normal(35, 10, 1000),
        'income': np.random.normal(50000, 15000, 1000),
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 1000),
        'city': np.random.choice(['Seoul', 'Busan', 'Incheon', 'Daegu'], 1000)
    }
    
    df = pd.DataFrame(data)
    
    # 인위적으로 결측치 생성
    missing_indices = np.random.choice(df.index, size=int(0.1 * len(df)), replace=False)
    df.loc[missing_indices[:50], 'age'] = np.nan
    df.loc[missing_indices[50:100], 'income'] = np.nan
    df.loc[missing_indices[100:150], 'education'] = np.nan
    
    # 결측치 처리 실행
    handler = MissingValueHandler()
    
    # 패턴 분석
    analysis = handler.analyze_missing_pattern(df)
    print("결측치 분석 결과:")
    print(analysis)
    
    # 처리 전략 정의
    strategies = {
        'age': 'median',
        'income': 'mean',
        'education': 'mode',
        'city': 'mode'
    }
    
    # 결측치 처리
    processed_df = handler.fit_transform(df, strategies)
    
    print(f"\n처리 전 결측치 수: {df.isnull().sum().sum()}")
    print(f"처리 후 결측치 수: {processed_df.isnull().sum().sum()}")
    
    # 시각화
    handler.plot_missing_pattern(df)
    handler.compare_before_after(df, processed_df)