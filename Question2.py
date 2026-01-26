from typing import Sequence

PRONUNCIATIONS = {
    "ABACUS": [["AE", "B", "AH", "K", "AH", "S"]],
    "BOOK": [["B", "UH", "K"]],
    "THEIR": [["DH", "EH", "R"]],
    "THERE": [["DH", "EH", "R"]],
    "TOMATO": [["T", "AH", "M", "AA", "T", "OW"], ["T", "AH", "M", "EY", "T", "OW"]],
}


class _TrieNode:
    def __init__(self) -> None:
        self.children = {}
        self.words = []


def _build_trie(pronunciations: dict) -> _TrieNode:
    root = _TrieNode()
    for word, variants in pronunciations.items():
        for phones in variants:
            node = root
            for ph in phones:
                node = node.children.setdefault(ph, _TrieNode())
            if word not in node.words:
                node.words.append(word)
    return root


_TRIE = _build_trie(PRONUNCIATIONS)


def find_word_combos_with_pronunciation(phonemes: Sequence[str]) -> Sequence[Sequence[str]]:
    n = len(phonemes)
    memo = {}

    def dfs(i: int):
        if i in memo:
            return memo[i]
        if i == n:
            return [[]]
        combos = []
        node = _TRIE
        j = i
        while j < n and phonemes[j] in node.children:
            node = node.children[phonemes[j]]
            j += 1
            if node.words:
                for tail in dfs(j):
                    for w in node.words:
                        combos.append([w] + tail)
        memo[i] = combos
        return combos

    return dfs(0)


if __name__ == "__main__":
    sample = ["DH", "EH", "R", "DH", "EH", "R"]
    got = sorted(tuple(x) for x in find_word_combos_with_pronunciation(sample))
    expected = sorted(
        [
            ("THEIR", "THEIR"),
            ("THEIR", "THERE"),
            ("THERE", "THEIR"),
            ("THERE", "THERE"),
        ]
    )
    assert got == expected
