from typing import Iterable, Sequence, Tuple, Optional, List, Dict

PRONUNCIATIONS = {"ABACUS": [["AE", "B", "AH", "K", "AH", "S"]], "BOOK": [["B", "UH", "K"]],
                  "THEIR": [["DH", "EH", "R"]], "THERE": [["DH", "EH", "R"]],
                  "TOMATO": [["T", "AH", "M", "AA", "T", "OW"], ["T", "AH", "M", "EY", "T", "OW"]]}


def _build_matches(phonemes: Sequence[str]) -> Dict[int, List[Tuple[str, int]]]:
    n = len(phonemes)
    matches_at: Dict[int, List[Tuple[str, int]]] = {i: [] for i in range(n)}

    for word, variants in PRONUNCIATIONS.items():
        for phones in variants:
            phones_t = tuple(phones)
            m = len(phones_t)
            if m == 0 or m > n:
                # Skip empty pronunciations to avoid infinite recursion.
                continue
            for i in range(n - m + 1):
                if phonemes[i : i + m] == phones_t:
                    matches_at[i].append((word, m))

    return matches_at


def _count_combos(matches_at: Dict[int, List[Tuple[str, int]]], n: int, max_results: Optional[int]) -> List[int]:
    counts = [0] * (n + 1)
    counts[n] = 1
    cap = max_results if max_results is not None else None

    for i in range(n - 1, -1, -1):
        total = 0
        for _word, m in matches_at.get(i, []):
            total += counts[i + m]
            if cap is not None and total > cap:
                total = cap + 1
                break
        counts[i] = total
    return counts


def find_word_combos_with_pronunciation(phonemes: Iterable[str], max_results: Optional[int] = 10000) -> Sequence[Sequence[str]]:
    if max_results is not None and max_results < 0:
        raise ValueError("max_results must be >= 0 or None.")

    n = len(phonemes)
    if n == 0:
        return [[]]

    matches_at = _build_matches(phonemes)

    counts = _count_combos(matches_at, n, max_results)
    if max_results is not None and counts[0] > max_results:
        raise ValueError(f"Too many combinations ({counts[0]}). Refine input or pass a larger max_results (or None).")

    memo: Dict[int, List[List[str]]] = {}

    def dfs(i: int) -> List[List[str]]:
        if i in memo:
            return memo[i]
        if i == n:
            return [[]]
        combos: List[List[str]] = []
        for word, m in matches_at.get(i, []):
            for tail in dfs(i + m):
                combos.append([word] + tail)
        memo[i] = combos
        return combos

    return dfs(0)


# Explanation: normalize input to a tuple (supports generators), pre-scan for matching word
# pronunciations, cap the total combinations to avoid runaway memory/time, then DFS with
# memoization to build the full-coverage sequences.


if __name__ == "__main__":
    sample = ["DH", "EH", "R", "DH", "EH", "R"]
    got = sorted(tuple(x) for x in find_word_combos_with_pronunciation(sample))
    print(got)
