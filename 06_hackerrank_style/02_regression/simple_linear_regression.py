"""
🏆 HackerRank Style Problem: 단순 선형회귀

문제 설명:
주어진 (x, y) 데이터 포인트들에 대해 단순 선형회귀를 수행하고,
새로운 x 값에 대한 y 값을 예측하세요.

회귀식: y = a + bx
여기서:
- b = Σ((xi - x̄)(yi - ȳ)) / Σ((xi - x̄)²)
- a = ȳ - b * x̄

입력 형식:
첫 번째 줄: 데이터 포인트 개수 n
다음 n줄: xi yi (공백으로 구분)
마지막 줄: 예측할 x 값

출력 형식:
예측된 y 값 (소수점 셋째 자리까지)

제약 조건:
- 5 ≤ n ≤ 100
- 0 ≤ xi, yi ≤ 100

예제 입력:
5
95 85
85 95
80 70
70 65
60 70
80

예제 출력:
78.288
"""

import sys
from typing import List, Tuple

def calculate_regression_coefficients(data: List[Tuple[float, float]]) -> Tuple[float, float]:
    """
    선형회귀 계수 계산
    
    Args:
        data: (x, y) 튜플들의 리스트
        
    Returns:
        (a, b): 절편과 기울기
    """
    n = len(data)
    
    # TODO: 회귀계수 계산 구현
    # 1. x, y의 평균 계산
    # 2. 기울기 b 계산: Σ((xi - x̄)(yi - ȳ)) / Σ((xi - x̄)²)
    # 3. 절편 a 계산: ȳ - b * x̄
    
    pass

def predict(a: float, b: float, x: float) -> float:
    """
    선형회귀 모델로 예측
    
    Args:
        a: 절편
        b: 기울기
        x: 예측할 x 값
        
    Returns:
        예측된 y 값
    """
    # TODO: 예측 구현
    # y = a + bx
    pass

def solve():
    """
    메인 솔루션 함수
    """
    # 입력 읽기
    n = int(input().strip())
    data = []
    
    for _ in range(n):
        x, y = map(float, input().strip().split())
        data.append((x, y))
    
    predict_x = float(input().strip())
    
    # 회귀계수 계산
    a, b = calculate_regression_coefficients(data)
    
    # 예측
    predicted_y = predict(a, b, predict_x)
    
    # 결과 출력
    print(f"{predicted_y:.3f}")

if __name__ == "__main__":
    solve()

# 테스트용 입력 데이터
"""
테스트 케이스 1:
입력:
5
95 85
85 95
80 70
70 65
60 70
80

예상 출력:
78.288

테스트 케이스 2:
입력:
3
1 2
2 4
3 6
4

예상 출력:
8.000
"""