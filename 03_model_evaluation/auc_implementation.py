"""
🎯 문제: AUC (Area Under Curve) 계산 구현

ROC 곡선과 AUC를 scratch부터 구현하고, 다양한 임계값에서의 성능을 분석하세요.

요구사항:
1. ROC 곡선 계산 함수
2. AUC 수치 계산 (사다리꼴 공식 사용)
3. 최적 임계값 찾기
4. PR 곡선 및 AP 계산
5. 시각화 기능

수학적 배경:
- TPR (True Positive Rate) = TP / (TP + FN)
- FPR (False Positive Rate) = FP / (FP + TN)
- AUC = ∫₀¹ TPR(FPR⁻¹(x)) dx
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

class AUCCalculator:
    def __init__(self):
        """
        AUC 계산기 초기화
        """
        # TODO: 초기화 구현
        pass
    
    def compute_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
        """
        혼동 행렬 계산
        
        Args:
            y_true: 실제 라벨
            y_pred: 예측 라벨
            
        Returns:
            confusion_dict: TP, TN, FP, FN 딕셔너리
        """
        # TODO: 혼동 행렬 계산 구현
        pass
    
    def compute_roc_curve(self, y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        ROC 곡선 계산
        
        Args:
            y_true: 실제 라벨 (0 또는 1)
            y_scores: 예측 확률 또는 점수
            
        Returns:
            fpr: False Positive Rate 배열
            tpr: True Positive Rate 배열
            thresholds: 임계값 배열
        """
        # TODO: ROC 곡선 계산 구현
        # 1. 임계값들을 정렬된 점수로 설정
        # 2. 각 임계값에서 TPR, FPR 계산
        # 3. (0,0)과 (1,1) 포인트 추가
        pass
    
    def compute_auc(self, fpr: np.ndarray, tpr: np.ndarray) -> float:
        """
        AUC 계산 (사다리꼴 공식 사용)
        
        Args:
            fpr: False Positive Rate 배열
            tpr: True Positive Rate 배열
            
        Returns:
            auc: AUC 값
        """
        # TODO: 사다리꼴 공식으로 AUC 계산
        # AUC = Σ(x[i+1] - x[i]) * (y[i+1] + y[i]) / 2
        pass
    
    def find_optimal_threshold(self, y_true: np.ndarray, y_scores: np.ndarray, 
                             metric: str = 'youden') -> Tuple[float, float]:
        """
        최적 임계값 찾기
        
        Args:
            y_true: 실제 라벨
            y_scores: 예측 점수
            metric: 최적화 기준 ('youden', 'f1', 'precision_recall')
            
        Returns:
            optimal_threshold: 최적 임계값
            optimal_score: 해당 임계값에서의 점수
        """
        # TODO: 최적 임계값 찾기 구현
        # Youden's J statistic: J = TPR - FPR
        pass
    
    def compute_pr_curve(self, y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Precision-Recall 곡선 계산
        
        Args:
            y_true: 실제 라벨
            y_scores: 예측 점수
            
        Returns:
            precision: Precision 배열
            recall: Recall 배열
            thresholds: 임계값 배열
        """
        # TODO: PR 곡선 계산 구현
        pass
    
    def compute_average_precision(self, precision: np.ndarray, recall: np.ndarray) -> float:
        """
        Average Precision (AP) 계산
        
        Args:
            precision: Precision 배열
            recall: Recall 배열
            
        Returns:
            ap: Average Precision 값
        """
        # TODO: AP 계산 구현
        pass
    
    def plot_roc_curve(self, fpr: np.ndarray, tpr: np.ndarray, auc: float):
        """
        ROC 곡선 시각화
        """
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def plot_pr_curve(self, precision: np.ndarray, recall: np.ndarray, ap: float):
        """
        PR 곡선 시각화
        """
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {ap:.3f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def evaluate_model(self, y_true: np.ndarray, y_scores: np.ndarray) -> Dict:
        """
        모델 종합 평가
        
        Args:
            y_true: 실제 라벨
            y_scores: 예측 점수
            
        Returns:
            evaluation: 평가 결과 딕셔너리
        """
        # TODO: 종합 평가 구현
        pass

# 테스트 코드
if __name__ == "__main__":
    # 샘플 데이터 생성
    X, y = make_classification(n_samples=1000, n_features=20, n_redundant=0, 
                             n_informative=20, random_state=42, n_clusters_per_class=1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 로지스틱 회귀 모델 학습
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    
    # 예측 확률 계산
    y_scores = model.predict_proba(X_test)[:, 1]
    
    # AUC 계산기 사용
    auc_calc = AUCCalculator()
    
    # ROC 곡선 및 AUC 계산
    fpr, tpr, roc_thresholds = auc_calc.compute_roc_curve(y_test, y_scores)
    auc_score = auc_calc.compute_auc(fpr, tpr)
    
    # PR 곡선 및 AP 계산
    precision, recall, pr_thresholds = auc_calc.compute_pr_curve(y_test, y_scores)
    ap_score = auc_calc.compute_average_precision(precision, recall)
    
    # 최적 임계값 찾기
    optimal_threshold, optimal_score = auc_calc.find_optimal_threshold(y_test, y_scores)
    
    # 결과 출력
    print(f"AUC Score: {auc_score:.4f}")
    print(f"Average Precision: {ap_score:.4f}")
    print(f"Optimal Threshold: {optimal_threshold:.4f}")
    print(f"Optimal Score: {optimal_score:.4f}")
    
    # 시각화
    auc_calc.plot_roc_curve(fpr, tpr, auc_score)
    auc_calc.plot_pr_curve(precision, recall, ap_score)
    
    # 종합 평가
    evaluation = auc_calc.evaluate_model(y_test, y_scores)
    print("\n종합 평가 결과:")
    for key, value in evaluation.items():
        print(f"{key}: {value}")