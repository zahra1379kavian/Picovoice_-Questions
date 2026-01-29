from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict
import itertools
import numpy as np


def softmax(u: np.ndarray, axis: int = -1) -> np.ndarray:
    u_max = np.max(u, axis=axis, keepdims=True)
    e = np.exp(u - u_max)
    return e / np.sum(e, axis=axis, keepdims=True)


def B_map(pi: List[int], blank: int) -> List[int]:
    """
      remove repeated labels (collapse consecutive duplicates) & remove blanks
    """
    collapsed: List[int] = []
    prev = None
    for sym in pi:
        if sym != prev:
            collapsed.append(sym)
        prev = sym
    return [sym for sym in collapsed if sym != blank]


def extend_with_blanks(z: List[int], blank: int) -> List[int]:
    """
      l^0 = (b, z1, b, z2, b, ..., z_U, b) so |l^0| = 2|z| + 1.
    """
    l0: List[int] = [blank]
    for lab in z:
        l0.append(lab)
        l0.append(blank)
    return l0


@dataclass
class CTCForwardCache:
    u: np.ndarray              # u^t_k
    y: np.ndarray              # y^t_k
    z: List[int]               # target labelling z
    l0: List[int]              # modified label sequence l^0
    alpha_hat: np.ndarray      # \hat{α}_t(s)
    C: np.ndarray              # C_t
    blank: int
    eps: float


class CTC_class:
    def __init__(self, blank: int = 0, eps: float = 1e-12):
        self.blank = blank
        self.eps = eps

    # -----------------------
    # Forward propagation
    # -----------------------
    def forward(self, u: np.ndarray, z: List[int]) -> Tuple[float, CTCForwardCache]:
        """
        Inputs:
          u : shape (T,K)
          z : list of labels

        Returns:
          loss  : -ln p(z|x)
          cache : quantities needed for backward propagation
        """
        T, K = u.shape
        if len(z) > T:
            raise ValueError("|z| must be <= T")
        if any(lab == self.blank for lab in z):
            raise ValueError("Target z must not contain the blank symbol.")

        # y^t_k = softmax(u^t)_k
        y = softmax(u, axis=1)

        # Modified label sequence l^0
        l0 = extend_with_blanks(z, blank=self.blank)
        S = len(l0)  # S = |l^0| = 2|z| + 1

        alpha_hat = np.zeros((T, S), dtype=np.float64)  # \hat{α}_t(s)
        C = np.zeros(T, dtype=np.float64)               # C_t = sum_s α_t(s)

        # Initialisation for α_1(s)
        alpha = np.zeros(S, dtype=np.float64)
        alpha[0] = y[0, self.blank]  # α_1(1) = y^1_b
        if S > 1:
            alpha[1] = y[0, l0[1]]   # α_1(2) = y^1_{l1}
        C[0] = np.sum(alpha)         # C_1 = sum_s α_1(s)
        alpha_hat[0] = alpha / C[0]  # \hat{α}_1(s) = α_1(s)/C_1

        # Recursion for α_t(s)
        for t in range(1, T):
            t1 = t + 1
            alpha = np.zeros(S, dtype=np.float64)

            # α_t(s)=0 for s < |l^0| - 2(T-t) - 1 and for s<1.
            s_min = max(1, S - 2*(T - t1) - 1) 
            s_max = min(S, 2*t1)                

            for s1 in range(s_min, s_max + 1): 
                s = s1 - 1             
                # \bar{α}_t(s) = α_{t-1}(s) + α_{t-1}(s-1)
                alpha_bar = alpha_hat[t-1, s]
                if s - 1 >= 0:
                    alpha_bar += alpha_hat[t-1, s-1]

                if (l0[s] == self.blank) or (s - 2 >= 0 and l0[s-2] == l0[s]):
                    # α_t(s) = \bar{α}_t(s) * y^t_{l^0_s}
                    alpha[s] = alpha_bar * y[t, l0[s]]
                else:
                    # α_t(s) = (\bar{α}_t(s) + α_{t-1}(s-2)) * y^t_{l^0_s}
                    alpha[s] = (alpha_bar + alpha_hat[t-1, s-2]) * y[t, l0[s]]

            C[t] = np.sum(alpha)   
            alpha_hat[t] = alpha / C[t]   # \hat{α}_t(s) = α_t(s)/C_t

        # ln p(z|x) = Σ_t ln(C_t)  (sec. 4.1)
        log_p = float(np.sum(np.log(C)))
        loss = -log_p

        cache = CTCForwardCache(u=u, y=y, z=z, l0=l0, alpha_hat=alpha_hat, C=C, blank=self.blank, eps=self.eps)
        return loss, cache

    # -----------------------
    # Backward propagation
    # -----------------------
    def backward(self, cache: CTCForwardCache) -> np.ndarray:
        """
        Backward propagation returns the error signal.
        """
        u = cache.u
        y = cache.y
        l0 = cache.l0
        alpha_hat = cache.alpha_hat
        C = cache.C
        blank = cache.blank
        eps = cache.eps

        T, K = u.shape
        S = len(l0)

        beta_hat = np.zeros((T, S), dtype=np.float64)  # \hat{β}_t(s)
        D = np.zeros(T, dtype=np.float64)              # D_t = sum_s β_t(s)

        # Initialisation for β_T(s)
        beta = np.zeros(S, dtype=np.float64)
        beta[S-1] = y[T-1, blank]       # β_T(|l^0|) = y^T_b
        if S > 1:
            beta[S-2] = y[T-1, l0[S-2]] # β_T(|l^0|-1) = y^T_{l_|l|}
        D[T-1] = np.sum(beta)
        beta_hat[T-1] = beta / D[T-1]

        # Recursion for β_t(s)
        for t in range(T-2, -1, -1):
            t1 = t + 1
            beta = np.zeros(S, dtype=np.float64)

            # β_t(s)=0 for s>2t and s>|l^0| 
            s_max = min(S, 2*t1)
            for s in range(s_max):
                # \bar{β}_t(s) = β_{t+1}(s) + β_{t+1}(s+1)
                beta_bar = beta_hat[t+1, s]
                if s + 1 < S:
                    beta_bar += beta_hat[t+1, s+1]

                # β_t(s) transition rule
                if (l0[s] == blank) or (s + 2 < S and l0[s+2] == l0[s]):
                    # β_t(s) = \bar{β}_t(s) * y^t_{l^0_s}
                    beta[s] = beta_bar * y[t, l0[s]]
                else:
                    # β_t(s) = (\bar{β}_t(s) + β_{t+1}(s+2)) * y^t_{l^0_s}
                    beta_s2 = beta_hat[t+1, s+2] if (s + 2 < S) else 0.0
                    beta[s] = (beta_bar + beta_s2) * y[t, l0[s]]

            D[t] = np.sum(beta)
            beta_hat[t] = beta / D[t]

        Q = np.zeros(T, dtype=np.float64)
        Q[T-1] = D[T-1]
        for t in range(T-2, -1, -1):
            Q[t] = D[t] * Q[t+1] / C[t+1]

        l0_arr = np.array(l0, dtype=np.int64)
        lab_pos: Dict[int, np.ndarray] = {k: np.where(l0_arr == k)[0] for k in range(K)}

        grad_u = np.zeros_like(u, dtype=np.float64)
        for t in range(T):
            for k in range(K):
                pos = lab_pos[k]
                sum_ab = float(np.sum(alpha_hat[t, pos] * beta_hat[t, pos])) if pos.size else 0.0
                grad_u[t, k] = y[t, k] - (Q[t] * sum_ab) / (y[t, k] + eps)

        return grad_u

    def forward_backward(self, u: np.ndarray, z: List[int]) -> Tuple[float, np.ndarray]:
        loss, cache = self.forward(u, z)
        grad_u = self.backward(cache)
        return loss, grad_u



def brute_force_p(y: np.ndarray, z: List[int], blank: int) -> float:
    """
    Brute-force p(z|x) by summing over all paths π ∈ L_0^T. Only feasible for very small T and K (used here for tests).
    """
    T, K = y.shape
    p = 0.0
    for pi in itertools.product(range(K), repeat=T):
        if B_map(list(pi), blank=blank) == list(z):
            prob = 1.0
            for t, sym in enumerate(pi):
                prob *= float(y[t, sym])
            p += prob
    return p


def finite_difference_check(ctc: CTC_class, u: np.ndarray, z: List[int], eps: float = 1e-6, num_checks: int = 20, seed: int = 0) -> None:
    """Compare analytic gradient from backward() with finite differences."""
    rng = np.random.default_rng(seed)
    loss, grad = ctc.forward_backward(u, z)

    T, K = u.shape
    for _ in range(num_checks):
        t = int(rng.integers(T))
        k = int(rng.integers(K))

        u_pos = u.copy()
        u_neg = u.copy()
        u_pos[t, k] += eps
        u_neg[t, k] -= eps

        loss_pos, _ = ctc.forward(u_pos, z)
        loss_neg, _ = ctc.forward(u_neg, z)

        grad_fd = (loss_pos - loss_neg) / (2.0 * eps)
        grad_an = float(grad[t, k])

        rel_err = abs(grad_fd - grad_an) / (abs(grad_fd) + abs(grad_an) + 1e-12)
        assert rel_err < 1e-5, f"finite-diff check failed at (t={t},k={k}): rel_err={rel_err:e}"


if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True)

    # -----------------
    # Test 1
    # -----------------
    np.random.seed(0)
    T, K = 3, 3         
    blank = 0
    z = [1, 2]
    u = np.random.randn(T, K)

    ctc = CTC_class(blank=blank)
    loss, cache = ctc.forward(u, z)

    p_forward = float(np.exp(-loss))
    p_brute = brute_force_p(cache.y, z, blank=blank)

    print(f"Test 1: p_forward={p_forward:.12f}, p_brute={p_brute:.12f}")
    assert abs(p_forward - p_brute) < 1e-12

    # -----------------
    # Test 2: with repeated labels 
    # -----------------
    np.random.seed(1)
    z_rep = [1, 1]
    u = np.random.randn(T, K)
    loss_rep, cache_rep = ctc.forward(u, z_rep)
    p_forward_rep = float(np.exp(-loss_rep))
    p_brute_rep = brute_force_p(cache_rep.y, z_rep, blank=blank)

    print(f"Test 2: p_forward(repeated)={p_forward_rep:.12f}, p_brute(repeated)={p_brute_rep:.12f}")
    assert abs(p_forward_rep - p_brute_rep) < 1e-12

    # -----------------
    # Test 3: gradient finite-difference check 
    # -----------------
    np.random.seed(42)
    T, K = 4, 4  # blank + 3 labels
    u = np.random.randn(T, K)
    z = [1, 3]
    finite_difference_check(ctc=CTC_class(blank=blank), u=u, z=z, eps=1e-6, num_checks=30, seed=0)
    print("Test 3: finite-difference gradient check passed.")

    # -----------------
    # Test 4: softmax invariance sanity check
    # -----------------
    loss, grad = ctc.forward_backward(u, z)
    grad_row_sums = np.sum(grad, axis=1)
    print("Test 4: row sums of grad (should be ~0):", grad_row_sums)
    assert np.max(np.abs(grad_row_sums)) < 1e-8

    # -----------------
    # Test 5: empty target z=[] has probability Π_t y^t_blank
    # -----------------
    np.random.seed(7)
    T, K = 5, 3
    u = np.random.randn(T, K)
    z_empty: List[int] = []
    loss_empty, cache_empty = ctc.forward(u, z_empty)
    expected_loss = -float(np.sum(np.log(cache_empty.y[:, blank])))
    print(f"Test 5: loss_empty={loss_empty:.12f}, expected={expected_loss:.12f}")
    assert abs(loss_empty - expected_loss) < 1e-12

    print("All tests passed.")
