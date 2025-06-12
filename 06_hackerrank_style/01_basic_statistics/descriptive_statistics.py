"""
🏆 HackerRank Style Problem: 기술통계량 계산

문제 설명:
주어진 데이터셋에 대해 다음 통계량들을 계산하세요:
1. 평균 (Mean)
2. 중앙값 (Median) 
3. 최빈값 (Mode) - 가장 작은 값 출력
4. 표준편차 (Standard Deviation)

입력 형식:
첫 번째 줄: 데이터 개수 n
두 번째 줄: n개의 정수 (공백으로 구분)

출력 형식:
각 줄에 하나씩:
- 평균 (소수점 첫째 자리까지)
- 중앙값 (소수점 첫째 자리까지)
- 최빈값 (정수)
- 표준편차 (소수점 첫째 자리까지)

제약 조건:
- 10 ≤ n ≤ 2500
- 0 ≤ 각 원소 ≤ 10^5

예제 입력:
10
64630 11735 14216 99233 14470 4978 73429 38120 51135 67060

예제 출력:
43900.6
44627.5
4978
30466.9
"""

import sys
import math
from collections import Counter
from typing import List

def calculate_mean(data: List[int]) -> float:
    """
    평균 계산
    
    Args:
        data: 정수 리스트
        
    Returns:
        평균값
    """
    # TODO: 평균 계산 구현
    pass

def calculate_median(data: List[int]) -> float:
    """
    중앙값 계산
    
    Args:
        data: 정수 리스트
        
    Returns:
        중앙값
    """
    # TODO: 중앙값 계산 구현
    # 힌트: 데이터를 정렬한 후 중간값 찾기
    pass

def calculate_mode(data: List[int]) -> int:
    """
    최빈값 계산 (가장 작은 값 반환)
    
    Args:
        data: 정수 리스트
        
    Returns:
        최빈값 (동일한 빈도가 여러 개면 가장 작은 값)
    """
    # TODO: 최빈값 계산 구현
    # 힌트: Counter 사용하여 빈도 계산
    pass

def calculate_std_dev(data: List[int]) -> float:
    """
    표준편차 계산
    
    Args:
        data: 정수 리스트
        
    Returns:
        표준편차
    """
    # TODO: 표준편차 계산 구현
    # 공식: sqrt(sum((x - mean)^2) / n)
    pass

def solve():
    """
    메인 솔루션 함수
    """
    # 입력 읽기
    n = int(input().strip())
    data = list(map(int, input().strip().split()))
    
    # 통계량 계산
    mean = calculate_mean(data)
    median = calculate_median(data)
    mode = calculate_mode(data)
    std_dev = calculate_std_dev(data)
    
    # 결과 출력 (소수점 첫째 자리까지)
    print(f"{mean:.1f}")
    print(f"{median:.1f}")
    print(mode)
    print(f"{std_dev:.1f}")

if __name__ == "__main__":
    solve()

# 테스트용 입력 데이터
"""
테스트 케이스 1:
입력:
10
64630 11735 14216 99233 14470 4978 73429 38120 51135 67060

예상 출력:
43900.6
44627.5
4978
30466.9

테스트 케이스 2:
입력:
5
1 2 3 4 5

예상 출력:
3.0
3.0
1
1.4
"""