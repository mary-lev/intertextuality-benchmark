"""Smith–Waterman local alignment on lemma sequences, returning the aligned span.
Match reward = idf(lemma); mismatch/gap = small negative (same params as passim_baseline)."""
MISMATCH = -0.5
GAP = -0.5


def sw_span(Q, P, idf):
    """Return (best_score, q_start, q_end, p_start, p_end) for the best local alignment.
    Indices are half-open into Q and P. Returns (0.0,0,0,0,0) if no positive alignment."""
    nQ, nP = len(Q), len(P)
    H = [[0.0] * (nP + 1) for _ in range(nQ + 1)]
    ptr = [[0] * (nP + 1) for _ in range(nQ + 1)]  # 0=stop 1=diag 2=up 3=left
    best, bi, bj = 0.0, 0, 0
    for i in range(1, nQ + 1):
        wa = idf.get(Q[i - 1], 0.0)
        for j in range(1, nP + 1):
            diag = H[i - 1][j - 1] + (wa if Q[i - 1] == P[j - 1] else MISMATCH)
            up = H[i - 1][j] + GAP
            left = H[i][j - 1] + GAP
            v, p = diag, 1
            if up > v:
                v, p = up, 2
            if left > v:
                v, p = left, 3
            if v <= 0.0:
                v, p = 0.0, 0
            H[i][j], ptr[i][j] = v, p
            if v > best:
                best, bi, bj = v, i, j
    if best == 0.0:
        return 0.0, 0, 0, 0, 0
    i, j = bi, bj
    while ptr[i][j] != 0:
        p = ptr[i][j]
        if p == 1:
            i -= 1; j -= 1
        elif p == 2:
            i -= 1
        else:
            j -= 1
    return best, i, bi, j, bj
