from typing import Sequence


def prob_rain_more_than_n(p: Sequence[float], n: int) -> float:
    m = len(p)
    if n < 0:
        return 1.0
    if n >= m:
        return 0.0

    # dp[i][k] = probability of exactly k rainy days after i days
    dp = [[0.0] * (m + 1) for _ in range(m + 1)]
    dp[0][0] = 1.0

    for i, r in enumerate(p, start=1):
        dp[i][0] = dp[i - 1][0] * (1.0 - r)
        for k in range(1, i + 1):
            dp[i][k] = dp[i - 1][k] * (1.0 - r) + dp[i - 1][k - 1] * r

    return sum(dp[m][n + 1 :])


if __name__ == "__main__":
    sample_p = [0.1, 0.5, 0.2, 0.8, 0.3]
    sample_n = 2
    result = prob_rain_more_than_n(sample_p, sample_n)
    print(f"probability of more than {sample_n} rainy days: {result}")



# Explanation:
"""
Let (X_i \in {0,1}) denote whether day (i) is rainy, with (P(X_i = 1) = p_i), and define the total number of rainy days
over (m) days as (S_m = \sum_{i=1}^m X_i). We are interested in computing (P(S_m > n)).
To do so, define (P_i(k) = P(S_i = k)), the probability of observing exactly (k) rainy days after the first (i) days.
By the law of total probability and independence across days, conditioning on whether day (i) is rainy or not yields
the recursive relation (P_i(k) = P_{i-1}(k)(1 - p_i) + P_{i-1}(k - 1)p_i), with the base condition (P_0(0) = 1)
and (P_0(k>0)=0). Iterating this recursion for (i=1,...,m) yields the full probability mass function of (S_m),
and the desired probability is obtained by summing the upper tail of this distribution, (P(S_m > n) = \sum_{k=n+1}^m P_m(k)).
"""
