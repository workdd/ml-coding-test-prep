"""
🎯 문제: K-Means 클러스터링 구현

다음 요구사항을 만족하는 K-Means 알고리즘을 NumPy만 사용하여 구현하세요.

요구사항:
1. 클래스 기반 구현 (KMeans)
2. fit() 메서드로 학습
3. predict() 메서드로 예측
4. 중심점 초기화는 랜덤으로
5. 최대 반복 횟수 제한
6. 수렴 조건 체크

입력:
- X: (n_samples, n_features) 형태의 데이터
- k: 클러스터 개수
- max_iters: 최대 반복 횟수 (기본값: 100)
- tol: 수렴 허용 오차 (기본값: 1e-4)

출력:
- labels: 각 데이터 포인트의 클러스터 라벨
- centroids: 최종 중심점들
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

class KMeans:
    def __init__(self, k: int, max_iters: int = 100, tol: float = 1e-4):
        """
        K-Means 클러스터링 초기화
        
        Args:
            k: 클러스터 개수
            max_iters: 최대 반복 횟수
            tol: 수렴 허용 오차
        """
        # TODO: 여기에 초기화 코드를 작성하세요
        pass
    
    def _initialize_centroids(self, X: np.ndarray) -> np.ndarray:
        """
        중심점을 랜덤하게 초기화
        
        Args:
            X: 입력 데이터 (n_samples, n_features)
            
        Returns:
            centroids: 초기 중심점들 (k, n_features)
        """
        # TODO: 랜덤 중심점 초기화 구현
        pass
    
    def _assign_clusters(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """
        각 데이터 포인트를 가장 가까운 중심점에 할당
        
        Args:
            X: 입력 데이터
            centroids: 현재 중심점들
            
        Returns:
            labels: 클러스터 라벨 (n_samples,)
        """
        # TODO: 클러스터 할당 구현 (유클리드 거리 사용)
        pass
    
    def _update_centroids(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        새로운 중심점 계산
        
        Args:
            X: 입력 데이터
            labels: 현재 클러스터 라벨
            
        Returns:
            new_centroids: 업데이트된 중심점들
        """
        # TODO: 중심점 업데이트 구현
        pass
    
    def fit(self, X: np.ndarray) -> 'KMeans':
        """
        K-Means 알고리즘 학습
        
        Args:
            X: 학습 데이터
            
        Returns:
            self: 학습된 모델
        """
        # TODO: 메인 K-Means 알고리즘 구현
        # 1. 중심점 초기화
        # 2. 반복:
        #    - 클러스터 할당
        #    - 중심점 업데이트
        #    - 수렴 체크
        pass
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        새로운 데이터에 대한 클러스터 예측
        
        Args:
            X: 예측할 데이터
            
        Returns:
            labels: 예측된 클러스터 라벨
        """
        # TODO: 예측 구현
        pass
    
    def plot_clusters(self, X: np.ndarray, labels: np.ndarray = None):
        """
        클러스터링 결과 시각화 (2D 데이터만)
        """
        if X.shape[1] != 2:
            print("2D 데이터만 시각화 가능합니다.")
            return
        
        if labels is None:
            labels = self.predict(X)
        
        plt.figure(figsize=(10, 8))
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray']
        
        for i in range(self.k):
            cluster_points = X[labels == i]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                       c=colors[i % len(colors)], label=f'Cluster {i}', alpha=0.6)
        
        # 중심점 표시
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], 
                   c='black', marker='x', s=200, linewidths=3, label='Centroids')
        
        plt.title('K-Means Clustering Results')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

# 테스트 코드
if __name__ == "__main__":
    # 샘플 데이터 생성
    np.random.seed(42)
    
    # 3개의 클러스터를 가진 데이터 생성
    cluster1 = np.random.normal([2, 2], 0.5, (50, 2))
    cluster2 = np.random.normal([6, 6], 0.5, (50, 2))
    cluster3 = np.random.normal([2, 6], 0.5, (50, 2))
    
    X = np.vstack([cluster1, cluster2, cluster3])
    
    # K-Means 실행
    kmeans = KMeans(k=3, max_iters=100)
    kmeans.fit(X)
    
    # 결과 출력
    labels = kmeans.predict(X)
    print(f"클��스터 중심점:\n{kmeans.centroids}")
    print(f"각 클러스터별 데이터 개수: {np.bincount(labels)}")
    
    # 시각화
    kmeans.plot_clusters(X, labels)