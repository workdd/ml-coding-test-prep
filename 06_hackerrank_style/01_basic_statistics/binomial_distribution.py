"""
🏆 HackerRank Style Problem: 이항분포 확률 계산

문제 설명:
이항분포 B(n, p)를 따르는 확률변수 X에 대해, X = k일 확률을 계산하세요.

입력 형식:
첫 번째 줄: 시행 횟수 n, 성공 확률 p (공백 구분)
두 번째 줄: 성공 횟수 k

출력 형식:
P(X = k)를 소수점 셋째 자리까지 출력

제약 조건:
- 1 ≤ n ≤ 100
- 0 < p < 1
- 0 ≤ k ≤ n

예제 입력:
10 0.5
3

예제 출력:
0.117

수학적 배경:
P(X = k) = C(n, k) * p^k * (1-p)^(n-k)
여기서 C(n, k)는 조합
"""

def factorial(n):
    return 1 if n == 0 else n * factorial(n-1)

def combination(n, k):
    return factorial(n) // (factorial(k) * factorial(n-k))

def binomial_probability(n, p, k):
    # TODO: 이항분포 확률 계산 구현
    pass

def solve():
    n, p = input().split()
    n = int(n)
    p = float(p)
    k = int(input())
    prob = binomial_probability(n, p, k)
    print(f"{prob:.3f}")

if __name__ == "__main__":
    solve()

"""
테스트 케이스 1:
입력:
10 0.5
3
예상 출력:
0.117
"""