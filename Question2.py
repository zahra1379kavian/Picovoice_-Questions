from typing import Sequence

PRONUNCIATIONS = {"ABACUS": [["AE", "B", "AH", "K", "AH", "S"]], "BOOK": [["B", "UH", "K"]],
                  "THEIR": [["DH", "EH", "R"]], "THERE": [["DH", "EH", "R"]],
                  "TOMATO": [["T", "AH", "M", "AA", "T", "OW"], ["T", "AH", "M", "EY", "T", "OW"]]}


def find_word_combos_with_pronunciation(phonemes: Sequence[str]) -> Sequence[Sequence[str]]:
    n = len(phonemes)
    memo = {}

    def dfs(i: int):
        if i in memo:
            return memo[i]
        if i == n:
            return [[]]
        combos = []
        
        for word, variants in PRONUNCIATIONS.items():
            for phones in variants:
                if phonemes[i : i + len(phones)] == phones:
                    for tail in dfs(i + len(phones)):
                        combos.append([word] + tail)
        memo[i] = combos
        return combos

    return dfs(0)


# Explanation: use DFS with memoization to build all word sequences whose pronunciations exactly cover the phoneme list; 
# at each index, try every word pronunciation that matches the next slice and append the best tails.


if __name__ == "__main__":
    sample = ["DH", "EH", "R", "DH", "EH", "R"]
    got = sorted(tuple(x) for x in find_word_combos_with_pronunciation(sample))
    print(got)
