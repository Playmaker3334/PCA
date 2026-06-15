import numpy as np


def _prompt_text(p):
    if isinstance(p, (tuple, list)):
        return str(p[0])
    return str(p)


def length_matched_indices(safe_prompts, unsafe_prompts, tol=15, k=3, seed=42):
    rng = np.random.default_rng(seed)

    safe_lens = np.array([len(_prompt_text(p)) for p in safe_prompts], dtype=np.int64)
    unsafe_lens = np.array([len(_prompt_text(p)) for p in unsafe_prompts], dtype=np.int64)

    safe_order = np.argsort(safe_lens, kind="stable")
    safe_lens_sorted = safe_lens[safe_order]

    unsafe_order = np.argsort(unsafe_lens, kind="stable")

    kept_safe = []
    kept_unsafe = []
    used = set()

    for u_pos in unsafe_order:
        uL = unsafe_lens[u_pos]
        lo = int(np.searchsorted(safe_lens_sorted, uL - tol, side="left"))
        hi = int(np.searchsorted(safe_lens_sorted, uL + tol, side="right"))
        candidates = [int(safe_order[j]) for j in range(lo, hi)
                      if int(safe_order[j]) not in used]
        if len(candidates) == 0:
            continue
        n_pick = min(k, len(candidates))
        pick = rng.choice(candidates, size=n_pick, replace=False)
        for idx in pick:
            used.add(int(idx))
            kept_safe.append(int(idx))
        kept_unsafe.append(int(u_pos))

    kept_safe = np.array(sorted(kept_safe), dtype=np.int64)
    kept_unsafe = np.array(sorted(kept_unsafe), dtype=np.int64)
    return kept_safe, kept_unsafe


def length_auroc(safe_lengths, unsafe_lengths):
    pos = list(unsafe_lengths)
    neg = list(safe_lengths)
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    allv = [(v, 1) for v in pos] + [(v, 0) for v in neg]
    allv.sort(key=lambda x: x[0])
    rank_sum_pos = 0.0
    i = 0
    n = len(allv)
    while i < n:
        j = i
        while j < n and allv[j][0] == allv[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for m in range(i, j):
            if allv[m][1] == 1:
                rank_sum_pos += avg_rank
        i = j
    n_pos, n_neg = len(pos), len(neg)
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def apply_length_matching(safe_acts, unsafe_acts, safe_prompts, unsafe_prompts,
                          tol=15, k=3, seed=42, logger=None):
    safe_idx, unsafe_idx = length_matched_indices(
        safe_prompts, unsafe_prompts, tol=tol, k=k, seed=seed
    )

    safe_acts_m = safe_acts[safe_idx] if len(safe_idx) else safe_acts[:0]
    unsafe_acts_m = unsafe_acts[unsafe_idx] if len(unsafe_idx) else unsafe_acts[:0]
    safe_prompts_m = [safe_prompts[i] for i in safe_idx]
    unsafe_prompts_m = [unsafe_prompts[i] for i in unsafe_idx]

    if logger is not None:
        s_lens = [len(_prompt_text(p)) for p in safe_prompts_m]
        u_lens = [len(_prompt_text(p)) for p in unsafe_prompts_m]
        auc = length_auroc(s_lens, u_lens)
        s_mean = float(np.mean(s_lens)) if s_lens else 0.0
        u_mean = float(np.mean(u_lens)) if u_lens else 0.0
        logger.info(
            f"length-matching (tol={tol}, k={k}, seed={seed}): "
            f"safe {len(safe_prompts)}->{len(safe_prompts_m)} "
            f"unsafe {len(unsafe_prompts)}->{len(unsafe_prompts_m)} "
            f"| len_mean safe={s_mean:.0f} unsafe={u_mean:.0f} "
            f"| length-AUROC={auc:.4f}"
        )

    return safe_acts_m, unsafe_acts_m, safe_prompts_m, unsafe_prompts_m