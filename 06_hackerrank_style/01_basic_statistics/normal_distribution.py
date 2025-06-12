"""
🏆 HackerRank Style Problem: 정규분포 확률 계산

문제 설명:
정규분포 N(μ, σ²)를 따르는 확률변수 X에 대해 주어진 구간의 확률을 계산하세요.

입력 형식:
첫 번째 줄: 평균 μ, 표준편차 σ (공백으로 구분)
두 번째 줄: 하한 a, 상한 b (공백으로 구분)

출력 형식:
P(a ≤ X ≤ b)를 소수점 셋째 자리까지 출력

제약 조건:
- -100 ≤ μ ≤ 100
- 0 < σ ≤ 10
- -1000 ≤ a ≤ b ≤ 1000

예제 입력:
20 2
19.5 20.5

예제 출력:
0.197

수학적 배경:
정규분포의 확률밀도함수: f(x) = (1/√(2πσ²)) * exp(-(x-μ)²/(2σ²))
Z-score: z = (x - μ) / σ
표준정규분포 누적분포함수를 이용하여 계산
"""

import math
from typing import Tuple

def erf_approximation(x: float) -> float:
    """
    오차함수(Error Function) 근사 계산
    erf(x) ≈ sign(x) * sqrt(1 - exp(-x² * (4/π + ax²) / (1 + ax²)))
    여기서 a ≈ 0.147
    
    Args:
        x: 입력값
        
    Returns:
        erf(x) 근사값
    """
    # TODO: 오차함수 근사 구현
    # 힌트: Abramowitz and Stegun 근사식 사용
    pass

def standard_normal_cdf(z: float) -> float:
    """
    표준정규분포 누적분포함수 계산
    Φ(z) = 0.5 * (1 + erf(z/√2))
    
    Args:
        z: 표준화된 값 (Z-score)
        
    Returns:
        P(Z ≤ z)
    """
    # TODO: 표준정규분포 CDF 구현
    pass

def normal_probability(mu: float, sigma: float, a: float, b: float) -> float:
    """
    정규분포에서 구간 [a, b]의 확률 계산
    P(a ≤ X ≤ b) = Φ((b-μ)/σ) - Φ((a-μ)/σ)
    
    Args:
        mu: 평균
        sigma: 표준편차
        a: 하한
        b: 상한
        
    Returns:
        P(a ≤ X ≤ b)
    """
    # TODO: 정규분포 확률 계산 구현
    # 1. Z-score 계산
    # 2. 표준정규분포 CDF 이용
    # 3. P(a ≤ X ≤ b) = CDF(z_b) - CDF(z_a)
    pass

def solve():
    """
    메인 솔루션 함수
    """
    # 입력 읽기
    mu, sigma = map(float, input().strip().split())
    a, b = map(float, input().strip().split())
    
    # 확률 계산
    probability = normal_probability(mu, sigma, a, b)
    
    # 결과 출력 (소수점 셋째 자리까지)
    print(f"{probability:.3f}")

if __name__ == "__main__":
    solve()

# 테스트용 입력 데이터
"""
테스트 케이스 1:
입력:
20 2
19.5 20.5

예상 출력:
0.197

테스트 케이스 2:
입력:
0 1
-1 1

예상 출력:
0.683

테스트 케이스 3:
입력:
100 15
70 130

예상 출력:
0.954
"""