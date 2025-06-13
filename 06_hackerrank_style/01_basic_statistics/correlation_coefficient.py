"""
🏆 HackerRank Style Problem: 피어슨 상관계수 계산

문제 설명:
두 데이터셋 X, Y에 대해 피어슨 상관계수를 계산하세요.

입력 형식:
첫 번째 줄: 데이터 개수 n
두 번째 줄: n개의 X 값 (공백 구분)
세 번째 줄: n개의 Y 값 (공백 구분)

출력 형식:
피어슨 상관계수 (소수점 셋째 자리까지)

제약 조건:
- 2 ≤ n ≤ 1000

예제 입력:
5
1 2 3 4 5
2 4 6 8 10

예제 출력:
1.000

수학적 배경:
r = Σ((xi-x̄)(yi-ȳ)) / (sqrt(Σ(xi-x̄)^2) * sqrt(Σ(yi-ȳ)^2))
"""

def pearson_correlation(x, y):
    # TODO: 피어슨 상관계수 계산 구현
    pass

def solve():
    n = int(input())
    x = list(map(float, input().split()))
    y = list(map(float, input().split()))
    r = pearson_correlation(x, y)
    print(f"{r:.3f}")

if __name__ == "__main__":
    solve()

"""
테스트 케이스 1:
입력:
5
1 2 3 4 5
2 4 6 8 10
예상 출력:
1.000
"""