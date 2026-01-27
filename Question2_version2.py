from typing import Sequence

PRONUNCIATIONS = {
    "ABACUS": [["AE", "B", "AH", "K", "AH", "S"]],
    "BOOK": [["B", "UH", "K"]],
    "THEIR": [["DH", "EH", "R"]],
    "THERE": [["DH", "EH", "R"]],
    "TOMATO": [["T", "AH", "M", "AA", "T", "OW"], ["T", "AH", "M", "EY", "T", "OW"]],
}


def _find_related_words_and_matches(phonemes: Sequence[str]):
    #find all words whose pronunciations appear in the input, and collect the positions where they match.
    n = len(phonemes)
    matches_at = {i: [] for i in range(n)}
    related_words = set()

    for word, variants in PRONUNCIATIONS.items():
        for phones in variants:
            m = len(phones)
            if m == 0 or m > n:
                continue
            for i in range(n - m + 1):
                if phonemes[i : i + m] == phones:
                    matches_at[i].append((word, m))
                    related_words.add(word)

    return related_words, matches_at


def find_word_combos_with_pronunciation(phonemes: Sequence[str]) -> Sequence[Sequence[str]]:
    related_words, matches_at = _find_related_words_and_matches(phonemes)
    n = len(phonemes)
    memo = {}

    def dfs(i: int):
        if i in memo:
            return memo[i]
        if i == n:
            return [[]]
        
        combos = []
        for word, m in matches_at.get(i, []):
            for tail in dfs(i + m):
                combos.append([word] + tail)
        memo[i] = combos
        return combos

    return dfs(0)


# Explanation: first scan the input to find words whose pronunciations match
# contiguous slices ("related words"), record all match positions, then Depth‑First Search (DFS) across those positions to build every full-coverage sequence.


if __name__ == "__main__":
    sample = ["DH", "EH", "R", "DH", "EH", "R"]
    got = sorted(tuple(x) for x in find_word_combos_with_pronunciation(sample))
    print(got)
