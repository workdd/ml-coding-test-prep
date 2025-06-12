"""
🏆 HackerRank Style Problem: 분류 성능 평가

문제 설명:
이진 분류 결과에 대해 다음 성능 지표들을 계산하세요:
1. Precision (정밀도)
2. Recall (재현율)
3. F1-Score
4. Accuracy (정확도)

입력 형식:
첫 번째 줄: 테스트 케이스 개수 n
다음 n줄: 실제값 예측값 (0 또는 1, 공백으로 구분)

출력 형식:
각 줄에 하나씩 (소수점 셋째 자리까지):
- Precision
- Recall  
- F1-Score
- Accuracy

제약 조건:
- 10 ≤ n ≤ 1000
- 실제값, 예측값은 0 또는 1

예제 입력:
10
1 1
1 0
0 1
0 0
1 1
1 1
0 0
0 1
1 0
0 0

예제 출력:
0.600
0.750
0.667
0.600

수학적 정의:
- TP: True Positive, TN: True Negative
- FP: False Positive, FN: False Negative
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)
- F1-Score = 2 * (Precision * Recall) / (Precision + Recall)
- Accuracy = (TP + TN) / (TP + TN + FP + FN)
"""

from typing import List, Tuple

def calculate_confusion_matrix(y_true: List[int], y_pred: List[int]) -> Tuple[int, int, int, int]:
    """
    혼동 행렬 계산
    
    Args:
        y_true: 실제 라벨 리스트
        y_pred: 예측 라벨 리스트
        
    Returns:
        (TP, TN, FP, FN) 튜플
    """
    # TODO: 혼동 행렬 계산 구현
    # TP: y_true=1, y_pred=1
    # TN: y_true=0, y_pred=0  
    # FP: y_true=0, y_pred=1
    # FN: y_true=1, y_pred=0
    pass

def calculate_precision(tp: int, fp: int) -> float:
    """
    정밀도 계산
    
    Args:
        tp: True Positive
        fp: False Positive
        
    Returns:
        Precision = TP / (TP + FP)
    """
    # TODO: 정밀도 계산 구현
    # 분모가 0인 경우 처리 필요
    pass

def calculate_recall(tp: int, fn: int) -> float:
    """
    재현율 계산
    
    Args:
        tp: True Positive
        fn: False Negative
        
    Returns:
        Recall = TP / (TP + FN)
    """
    # TODO: 재현율 계산 구현
    pass

def calculate_f1_score(precision: float, recall: float) -> float:
    """
    F1-Score 계산
    
    Args:
        precision: 정밀도
        recall: 재현율
        
    Returns:
        F1-Score = 2 * (Precision * Recall) / (Precision + Recall)
    """
    # TODO: F1-Score 계산 구현
    pass

def calculate_accuracy(tp: int, tn: int, fp: int, fn: int) -> float:
    """
    정확도 계산
    
    Args:
        tp, tn, fp, fn: 혼동 행렬 요소들
        
    Returns:
        Accuracy = (TP + TN) / (TP + TN + FP + FN)
    """
    # TODO: 정확도 계산 구현
    pass

def solve():
    """
    메인 솔루션 함수
    """
    # 입력 읽기
    n = int(input().strip())
    y_true = []
    y_pred = []
    
    for _ in range(n):
        true_val, pred_val = map(int, input().strip().split())
        y_true.append(true_val)
        y_pred.append(pred_val)
    
    # 혼동 행렬 계산
    tp, tn, fp, fn = calculate_confusion_matrix(y_true, y_pred)
    
    # 성능 지표 계산
    precision = calculate_precision(tp, fp)
    recall = calculate_recall(tp, fn)
    f1_score = calculate_f1_score(precision, recall)
    accuracy = calculate_accuracy(tp, tn, fp, fn)
    
    # 결과 출력
    print(f"{precision:.3f}")
    print(f"{recall:.3f}")
    print(f"{f1_score:.3f}")
    print(f"{accuracy:.3f}")

if __name__ == "__main__":
    solve()

# 테스트용 입력 데이터
"""
테스트 케이스 1:
입력:
10
1 1
1 0
0 1
0 0
1 1
1 1
0 0
0 1
1 0
0 0

예상 출력:
0.600
0.750
0.667
0.600

분석:
TP=3, TN=3, FP=2, FN=1
Precision = 3/(3+2) = 0.6
Recall = 3/(3+1) = 0.75
F1 = 2*0.6*0.75/(0.6+0.75) = 0.667
Accuracy = (3+3)/(3+3+2+1) = 0.6
"""