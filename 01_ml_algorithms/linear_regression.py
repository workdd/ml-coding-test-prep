"""
🎯 문제: 선형 회귀 구현

경사하강법을 사용하여 선형 회귀를 구현하세요.

요구사항:
1. 클래스 기반 구현 (LinearRegression)
2. fit() 메서드로 학습
3. predict() 메서드로 예측
4. 경사하강법 사용
5. 비용함수(MSE) 계산
6. 학습 과정 시각화

수학적 배경:
- 가설: h(x) = θ₀ + θ₁x₁ + θ₂x₂ + ... + θₙxₙ
- 비용함수: J(θ) = (1/2m) * Σ(h(x⁽ⁱ⁾) - y⁽ⁱ⁾)²
- 경사하강법: θⱼ := θⱼ - α * ∂J(θ)/∂θⱼ
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List

class LinearRegression:
    def __init__(self, learning_rate: float = 0.01, max_iters: int = 1000, tol: float = 1e-6):
        """
        선형 회귀 초기화
        
        Args:
            learning_rate: 학습률
            max_iters: 최대 반복 횟수
            tol: 수렴 허용 오차
        """
        # TODO: 초기화 구현
        pass
    
    def _add_intercept(self, X: np.ndarray) -> np.ndarray:
        """
        절편(bias) 항을 위한 1 컬럼 추가
        
        Args:
            X: 입력 특성 (n_samples, n_features)
            
        Returns:
            X_with_intercept: 절편 항이 추가된 특성 (n_samples, n_features + 1)
        """
        # TODO: 절편 항 추가 구현
        pass
    
    def _compute_cost(self, X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
        """
        평균 제곱 오차(MSE) 계산
        
        Args:
            X: 입력 특성 (절편 항 포함)
            y: 타겟 값
            theta: 모델 파라미터
            
        Returns:
            cost: MSE 비용
        """
        # TODO: MSE 비용함수 구현
        # J(θ) = (1/2m) * Σ(h(x⁽ⁱ⁾) - y⁽ⁱ⁾)²
        pass
    
    def _compute_gradients(self, X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """
        경사(gradient) 계산
        
        Args:
            X: 입력 특성 (절편 항 포함)
            y: 타겟 값
            theta: 현재 파라미터
            
        Returns:
            gradients: 각 파라미터에 대한 경사
        """
        # TODO: 경사 계산 구현
        # ∂J(θ)/∂θⱼ = (1/m) * Σ(h(x⁽ⁱ⁾) - y⁽ⁱ⁾) * x⁽ⁱ⁾ⱼ
        pass
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegression':
        """
        선형 회귀 모델 학습
        
        Args:
            X: 학습 데이터 특성
            y: 학습 데이터 타겟
            
        Returns:
            self: 학습된 모델
        """
        # TODO: 경사하강법을 사용한 학습 구현
        # 1. 절편 항 추가
        # 2. 파라미터 초기화
        # 3. 경사하강법 반복:
        #    - 비용 계산
        #    - 경사 계산
        #    - 파라미터 업데이트
        #    - 수렴 체크
        pass
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        예측 수행
        
        Args:
            X: 예측할 데이터
            
        Returns:
            predictions: 예측 값들
        """
        # TODO: 예측 구현
        pass
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        R² 점수 계산
        
        Args:
            X: 테스트 데이터 특성
            y: 테스트 데이터 타겟
            
        Returns:
            r2_score: R² 점수
        """
        # TODO: R² 점수 구현
        # R² = 1 - (SS_res / SS_tot)
        pass
    
    def plot_cost_history(self):
        """
        비용 함수 변화 시각화
        """
        if not hasattr(self, 'cost_history'):
            print("모델이 학습되지 않았습니다.")
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.cost_history)
        plt.title('Cost Function Over Iterations')
        plt.xlabel('Iterations')
        plt.ylabel('Cost (MSE)')
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def plot_predictions(self, X: np.ndarray, y: np.ndarray):
        """
        예측 결과 시각화 (1D 특성만)
        """
        if X.shape[1] != 1:
            print("1D 특성 데이터만 시각화 가능합니다.")
            return
        
        predictions = self.predict(X)
        
        plt.figure(figsize=(10, 6))
        plt.scatter(X, y, alpha=0.6, label='Actual')
        plt.plot(X, predictions, color='red', linewidth=2, label='Predicted')
        plt.title('Linear Regression Results')
        plt.xlabel('Feature')
        plt.ylabel('Target')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

# 테스트 코드
if __name__ == "__main__":
    # 샘플 데이터 생성
    np.random.seed(42)
    
    # 1D 선형 데이터
    X = np.random.randn(100, 1)
    y = 4 + 3 * X.ravel() + np.random.randn(100) * 0.5
    
    # 선형 회귀 실행
    lr = LinearRegression(learning_rate=0.01, max_iters=1000)
    lr.fit(X, y)
    
    # 결과 출력
    predictions = lr.predict(X)
    r2_score = lr.score(X, y)
    
    print(f"학습된 파라미터: {lr.theta}")
    print(f"R² 점수: {r2_score:.4f}")
    print(f"최종 비용: {lr.cost_history[-1]:.6f}")
    
    # 시각화
    lr.plot_cost_history()
    lr.plot_predictions(X, y)