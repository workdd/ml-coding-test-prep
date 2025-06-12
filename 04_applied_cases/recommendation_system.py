"""
🎯 문제: 추천 시스템 구현

협업 필터링과 콘텐츠 기반 필터링을 결합한 하이브리드 추천 시스템을 구현하세요.

요구사항:
1. 사용자-아이템 평점 매트릭스 처리
2. 협업 필터링 (User-based, Item-based)
3. 콘텐츠 기반 필터링
4. 하이브리드 추천 (가중 결합)
5. 추천 성능 평가 (RMSE, MAE, Precision@K)
6. Cold Start 문제 해결

수학적 배경:
- 코사인 유사도: sim(u,v) = (u·v) / (||u|| ||v||)
- 피어슨 상관계수: r = Σ(x-x̄)(y-ȳ) / √(Σ(x-x̄)²Σ(y-ȳ)²)
- 매트릭스 분해: R ≈ UV^T
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

class RecommendationSystem:
    def __init__(self, alpha: float = 0.5):
        """
        하이브리드 추천 시스템 초기화
        
        Args:
            alpha: 협업 필터링과 콘텐츠 기반 필터링의 가중치
        """
        # TODO: 초기화 구현
        pass
    
    def fit(self, ratings_df: pd.DataFrame, items_df: pd.DataFrame = None):
        """
        추천 시스템 학습
        
        Args:
            ratings_df: 사용자-아이템-평점 데이터 (user_id, item_id, rating)
            items_df: 아이템 메타데이터 (item_id, features...)
        """
        # TODO: 시스템 학습 구현
        pass
    
    def _compute_user_similarity(self) -> np.ndarray:
        """
        사용자 간 유사도 계산
        
        Returns:
            user_similarity: 사용자 유사도 매트릭스
        """
        # TODO: 사용자 유사도 계산 구현
        pass
    
    def _compute_item_similarity(self) -> np.ndarray:
        """
        아이템 간 유사도 계산
        
        Returns:
            item_similarity: 아이템 유사도 매트릭스
        """
        # TODO: 아이템 유사도 계산 구현
        pass
    
    def _collaborative_filtering_predict(self, user_id: int, item_id: int, method: str = 'user') -> float:
        """
        협업 필터링 예측
        
        Args:
            user_id: 사용자 ID
            item_id: 아이템 ID
            method: 'user' 또는 'item' 기반
            
        Returns:
            predicted_rating: 예측 평점
        """
        # TODO: 협업 필터링 예측 구현
        pass
    
    def _content_based_predict(self, user_id: int, item_id: int) -> float:
        """
        콘텐츠 기반 예측
        
        Args:
            user_id: 사용자 ID
            item_id: 아이템 ID
            
        Returns:
            predicted_rating: 예측 평점
        """
        # TODO: 콘텐츠 기반 예측 구현
        pass
    
    def predict(self, user_id: int, item_id: int) -> float:
        """
        하이브리드 예측
        
        Args:
            user_id: 사용자 ID
            item_id: 아이템 ID
            
        Returns:
            predicted_rating: 하이브리드 예측 평점
        """
        # TODO: 하이브리드 예측 구현
        pass
    
    def recommend(self, user_id: int, n_recommendations: int = 10) -> List[Tuple[int, float]]:
        """
        사용자에게 아이템 추천
        
        Args:
            user_id: 사용자 ID
            n_recommendations: 추천할 아이템 수
            
        Returns:
            recommendations: (item_id, predicted_rating) 리스트
        """
        # TODO: 추천 구현
        pass
    
    def evaluate(self, test_ratings: pd.DataFrame) -> Dict[str, float]:
        """
        추천 시스템 성능 평가
        
        Args:
            test_ratings: 테스트 평점 데이터
            
        Returns:
            metrics: 평가 지표 딕셔너리
        """
        # TODO: 성능 평가 구현
        pass

# 테스트 코드
if __name__ == "__main__":
    # 샘플 데이터 생성
    np.random.seed(42)
    
    # 사용자-아이템-평점 데이터
    n_users, n_items = 100, 50
    n_ratings = 2000
    
    ratings_data = {
        'user_id': np.random.randint(0, n_users, n_ratings),
        'item_id': np.random.randint(0, n_items, n_ratings),
        'rating': np.random.randint(1, 6, n_ratings)
    }
    ratings_df = pd.DataFrame(ratings_data)
    
    # 중복 제거
    ratings_df = ratings_df.drop_duplicates(['user_id', 'item_id'])
    
    # 아이템 메타데이터
    items_data = {
        'item_id': range(n_items),
        'genre': np.random.choice(['Action', 'Comedy', 'Drama', 'Horror'], n_items),
        'year': np.random.randint(1990, 2024, n_items)
    }
    items_df = pd.DataFrame(items_data)
    
    # 추천 시스템 실행
    rec_sys = RecommendationSystem(alpha=0.7)
    rec_sys.fit(ratings_df, items_df)
    
    # 추천 생성
    user_id = 0
    recommendations = rec_sys.recommend(user_id, n_recommendations=5)
    
    print(f"사용자 {user_id}에 대한 추천:")
    for item_id, score in recommendations:
        print(f"아이템 {item_id}: {score:.3f}")
    
    # 성능 평가 (일부 데이터를 테스트용으로 분리)
    test_size = int(0.2 * len(ratings_df))
    test_ratings = ratings_df.sample(n=test_size, random_state=42)
    
    metrics = rec_sys.evaluate(test_ratings)
    print("\n성능 평가 결과:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")