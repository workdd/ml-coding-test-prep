"""
🏆 HackerRank Style Problem: K-Means 클러스터링

문제 설명:
2차원 점들에 대해 K-Means 클러스터링을 수행하고,
최종 중심점들의 좌표를 출력하세요.

알고리즘:
1. K개의 초기 중심점을 첫 K개 점으로 설정
2. 각 점을 가장 가까운 중심점에 할당
3. 각 클러스터의 중심점을 새로 계산
4. 중심점이 변하지 않을 때까지 2-3 반복

입력 형식:
첫 번째 줄: 점의 개수 n, 클러스터 개수 k
다음 n줄: xi yi (공백으로 구분)

출력 형식:
k줄에 걸쳐 최종 중심점 좌표 (소수점 둘째 자리까지)
중심점은 클러스터 번호 순으로 출력

제약 조건:
- 2 ≤ k ≤ n ≤ 100
- 0 ≤ xi, yi ≤ 100
- 최대 100번 반복

예제 입력:
6 2
1 1
2 1
1 2
8 8
9 8
8 9

예제 출력:
1.33 1.33
8.33 8.33
"""

import math
from typing import List, Tuple

def euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """
    두 점 사이의 유클리드 거리 계산
    
    Args:
        p1, p2: (x, y) 좌표 튜플
        
    Returns:
        유클리드 거리
    """
    # TODO: 유클리드 거리 계산 구현
    # distance = sqrt((x1-x2)² + (y1-y2)²)
    pass

def assign_clusters(points: List[Tuple[float, float]], 
                   centroids: List[Tuple[float, float]]) -> List[int]:
    """
    각 점을 가장 가까운 중심점에 할당
    
    Args:
        points: 데이터 점들
        centroids: 현재 중심점들
        
    Returns:
        각 점의 클러스터 번호 리스트
    """
    # TODO: 클러스터 할당 구현
    # 각 점에 대해 가장 가까운 중심점 찾기
    pass

def update_centroids(points: List[Tuple[float, float]], 
                    assignments: List[int], 
                    k: int) -> List[Tuple[float, float]]:
    """
    새로운 중심점 계산
    
    Args:
        points: 데이터 점들
        assignments: 각 점의 클러스터 할당
        k: 클러스터 개수
        
    Returns:
        새로운 중심점들
    """
    # TODO: 중심점 업데이트 구현
    # 각 클러스터에 속한 점들의 평균 계산
    pass

def kmeans_clustering(points: List[Tuple[float, float]], k: int, max_iters: int = 100) -> List[Tuple[float, float]]:
    """
    K-Means 클러스터링 수행
    
    Args:
        points: 데이터 점들
        k: 클러스터 개수
        max_iters: 최대 반복 횟수
        
    Returns:
        최종 중심점들
    """
    # TODO: K-Means 알고리즘 구현
    # 1. 초기 중심점 설정 (첫 k개 점)
    # 2. 수렴할 때까지 반복:
    #    - 클러스터 할당
    #    - 중심점 업데이트
    #    - 수렴 체크
    pass

def solve():
    """
    메인 솔루션 함수
    """
    # 입력 읽기
    n, k = map(int, input().strip().split())
    points = []
    
    for _ in range(n):
        x, y = map(float, input().strip().split())
        points.append((x, y))
    
    # K-Means 클러스터링 수행
    final_centroids = kmeans_clustering(points, k)
    
    # 결과 출력
    for centroid in final_centroids:
        print(f"{centroid[0]:.2f} {centroid[1]:.2f}")

if __name__ == "__main__":
    solve()

# 테스트용 입력 데이터
"""
테스트 케이스 1:
입력:
6 2
1 1
2 1
1 2
8 8
9 8
8 9

예상 출력:
1.33 1.33
8.33 8.33

분석:
클러스터 0: (1,1), (2,1), (1,2) → 중심점 (4/3, 4/3)
클러스터 1: (8,8), (9,8), (8,9) → 중심점 (25/3, 25/3)
"""