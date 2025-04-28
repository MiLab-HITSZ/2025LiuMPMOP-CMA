import numpy as np
from pprint import pprint

def get_groups_indices(N: int, K: int) -> list[np.ndarray]:
    """Evenly partition indices 0...N-1 into K groups"""
    groups = []
    base = N // K
    remainder = N % K

    # first K regular groups
    for i in range(K):
        start, end = i * base, (i + 1) * base
        if end > start:
            groups.append(np.arange(start, end))

    # leftover dimensions -> extra group
    if remainder > 0:
        groups.append(np.arange(K * base, N))

    return groups


def simulate_process(
    n_samples: int, N: int, K1: int, K2: int, seed: int = None
) -> tuple[float, int]:
    """Simulate the process of drawing samples from two distributions
    and checking for violations of the coarse and fine grouping conditions.

    Steps
    -----
    1. Draw x, y in [0,1]^N (i.i.d. uniform);  b_x, b_y in [0, 1].
    2. Keep only samples where b_y > b_x; set d = b_y − b_x.
    3. Coarse grouping (K1):
         All groups must satisfy |mean(x) − mean(y)| <= d.
    4. Fine  grouping (K2):
         Check if at least one group violates |mean(x) − mean(y)| >= d.
    5. Return the probability of a violation given the coarse condition
       and the number of valid coarse samples.

    Returns
    -------
    prob : float
        Conditional probability of at least one fine-group violation.
    n_coarse : int
        Number of samples that satisfied the coarse-group condition.
    """
    if seed is not None:
        np.random.seed(seed)

    # ----- sample generation
    x = np.random.rand(n_samples, N)
    y = np.random.rand(n_samples, N)
    bx = np.random.rand(n_samples)
    by = np.random.rand(n_samples)

    mask = by > bx
    x, y, bx, by = x[mask], y[mask], bx[mask], by[mask]
    d = by - bx  # positive scalar per sample

    # ----- grouping indices
    groups_coarse = get_groups_indices(N, K1)
    groups_fine = get_groups_indices(N, K2)

    # ----- coarse-group test
    n_valid = x.shape[0]
    coarse_diff = np.empty((n_valid, len(groups_coarse)))

    for g, idx in enumerate(groups_coarse):
        coarse_diff[:, g] = np.abs(
            np.mean(x[:, idx], axis=1) - np.mean(y[:, idx], axis=1)
        )

    coarse_ok = np.all(coarse_diff <= d[:, None], axis=1)

    if coarse_ok.sum() == 0:
        return 0.0, 0

    x_c, y_c, d_c = x[coarse_ok], y[coarse_ok], d[coarse_ok]

    # ----- fine-group test
    n_c = x_c.shape[0]
    fine_diff = np.empty((n_c, len(groups_fine)))

    for g, idx in enumerate(groups_fine):
        fine_diff[:, g] = np.abs(
            np.mean(x_c[:, idx], axis=1) - np.mean(y_c[:, idx], axis=1)
        )

    fine_violation = np.any(fine_diff >= d_c[:, None], axis=1)
    prob = fine_violation.mean()

    return prob, n_c


if __name__ == "__main__":
    # single demo run
    prob, valid_samples = simulate_process(
        n_samples=10**6, N=10, K1=1, K2=10, seed=1
    )
    print("Number of coarse-valid samples:", valid_samples)
    print("Conditional violation probability:", prob)

    # table for (K1, K2) pairs at N = 5
    results = []
    for K1 in (1, 2, 3, 5):
        for K2 in (1, 2, 3, 5):
            if K1 == K2:
                continue
            p, n_coarse = simulate_process(
                n_samples=10**6, N=5, K1=K1, K2=K2, seed=1
            )
            results.append([5, K1, K2, p])

    # df = pd.DataFrame(results, columns=["N", "K1", "K2", "prob"])
    # print(df.to_latex(index=False, float_format="%.2f"))
    pprint(results)
