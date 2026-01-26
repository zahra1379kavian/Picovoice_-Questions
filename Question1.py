def chance_rain_at_least_n_days(p, n):
    if n <= 0:
        return 1.0
    m = len(p)
    if n > m:
        return 0.0

    dp = [0.0] * (m + 1)
    dp[0] = 1.0
    for r in p:
        for k in range(m, 0, -1):
            dp[k] = dp[k] * (1.0 - r) + dp[k - 1] * r
        dp[0] *= 1.0 - r
    return sum(dp[n:])
