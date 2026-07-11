"""Alignment and repeatability tools for phoneme-scale neural trajectories."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class AlignedEvent:
    """A label occurrence and its fixed-width time-resolved representation."""

    example_index: int
    label_index: int
    label_id: int
    onset_token: int
    offset_token: int
    center_token: int
    trajectory: np.ndarray


def _log_softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    shifted = x - np.max(x, axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def ctc_forced_align(
    logits: np.ndarray,
    labels: np.ndarray | list[int],
    *,
    blank_index: int,
) -> list[tuple[int, int]]:
    """Viterbi-align a known label sequence to CTC frames.

    Returns inclusive-exclusive ``(onset, offset)`` spans, one per reference
    label. This alignment is model-assisted and must not be presented as
    independently measured phoneme timing.
    """
    logits = np.asarray(logits)
    labels = np.asarray(labels, dtype=np.int64)
    if logits.ndim != 2:
        raise ValueError("logits must have shape [time, classes]")
    if labels.ndim != 1 or len(labels) == 0:
        raise ValueError("labels must be a non-empty one-dimensional sequence")
    if np.any(labels == int(blank_index)):
        raise ValueError("reference labels must not include the CTC blank")

    extended = np.full(2 * len(labels) + 1, int(blank_index), dtype=np.int64)
    extended[1::2] = labels
    time_steps, state_count = logits.shape[0], len(extended)
    if time_steps < len(labels):
        raise ValueError("fewer CTC frames than reference labels")
    log_probs = _log_softmax(logits)
    score = np.full((time_steps, state_count), -np.inf, dtype=np.float64)
    back = np.full((time_steps, state_count), -1, dtype=np.int32)
    score[0, 0] = log_probs[0, int(blank_index)]
    if state_count > 1:
        score[0, 1] = log_probs[0, int(extended[1])]

    for t in range(1, time_steps):
        for state in range(state_count):
            candidates = [state]
            if state > 0:
                candidates.append(state - 1)
            if (
                state > 1
                and extended[state] != int(blank_index)
                and extended[state] != extended[state - 2]
            ):
                candidates.append(state - 2)
            previous = np.asarray([score[t - 1, candidate] for candidate in candidates])
            best = int(np.argmax(previous))
            score[t, state] = previous[best] + log_probs[t, int(extended[state])]
            back[t, state] = int(candidates[best])

    final_candidates = [state_count - 1]
    if state_count > 1:
        final_candidates.append(state_count - 2)
    final_scores = np.asarray([score[-1, state] for state in final_candidates])
    if not np.isfinite(final_scores).any():
        raise ValueError("no valid CTC alignment path")
    state = int(final_candidates[int(np.argmax(final_scores))])
    path = np.empty(time_steps, dtype=np.int32)
    for t in range(time_steps - 1, -1, -1):
        path[t] = state
        if t:
            state = int(back[t, state])
            if state < 0:
                raise ValueError("broken CTC backtrace")

    spans: list[tuple[int, int]] = []
    for label_index in range(len(labels)):
        frames = np.flatnonzero(path == 2 * label_index + 1)
        if not len(frames):
            raise ValueError(f"alignment skipped reference label {label_index}")
        spans.append((int(frames[0]), int(frames[-1]) + 1))
    return spans


def extract_aligned_events(
    values: np.ndarray,
    logits: np.ndarray,
    example_indices: np.ndarray,
    references: dict[int, list[int] | np.ndarray],
    *,
    blank_index: int,
    before: int = 3,
    after: int = 3,
) -> list[AlignedEvent]:
    """Extract fixed-width trajectories around forced-aligned label centers."""
    values = np.asarray(values)
    logits = np.asarray(logits)
    example_indices = np.asarray(example_indices)
    if values.ndim != 2 or logits.ndim != 2 or len(values) != len(logits):
        raise ValueError("values and logits must be token-aligned matrices")
    events: list[AlignedEvent] = []
    for example_index, reference in references.items():
        rows = np.flatnonzero(example_indices == int(example_index))
        if not len(rows):
            continue
        spans = ctc_forced_align(logits[rows], reference, blank_index=blank_index)
        for label_index, ((onset, offset), label_id) in enumerate(zip(spans, reference)):
            center = (int(onset) + int(offset) - 1) // 2
            start, stop = center - int(before), center + int(after) + 1
            if start < 0 or stop > len(rows):
                continue
            window_rows = rows[start:stop]
            if np.all(np.diff(window_rows) == 1):
                # Standard exports store every example contiguously. Preserve
                # a view here so collecting all eligible events does not copy
                # large flattened input windows before balanced subsampling.
                trajectory = values[int(window_rows[0]) : int(window_rows[-1]) + 1]
            else:
                trajectory = values[window_rows]
            events.append(
                AlignedEvent(
                    example_index=int(example_index),
                    label_index=int(label_index),
                    label_id=int(label_id),
                    onset_token=int(onset),
                    offset_token=int(offset),
                    center_token=int(center),
                    trajectory=np.asarray(trajectory, dtype=np.float32),
                )
            )
    return events


def fit_shared_pca(
    trajectories: list[np.ndarray],
    *,
    n_components: int = 3,
) -> tuple[list[np.ndarray], StandardScaler, PCA]:
    """Fit one standardized PCA basis to all timepoints and transform each path."""
    if not trajectories:
        raise ValueError("at least one trajectory is required")
    lengths = [len(trajectory) for trajectory in trajectories]
    pooled = np.concatenate(trajectories, axis=0)
    scaler = StandardScaler()
    standardized = scaler.fit_transform(pooled)
    component_count = min(int(n_components), pooled.shape[0], pooled.shape[1])
    pca = PCA(n_components=component_count)
    transformed_pool = pca.fit_transform(standardized)
    boundaries = np.cumsum([0, *lengths])
    transformed = [
        transformed_pool[boundaries[idx] : boundaries[idx + 1]]
        for idx in range(len(lengths))
    ]
    return transformed, scaler, pca


def split_half_reliability(
    trajectories: list[np.ndarray],
    conditions: list[int | str],
    *,
    repetitions: int = 200,
    seed: int = 7,
    temporal_center: bool = True,
) -> dict[int | str, float]:
    """Median correlation between random split-half condition-mean paths.

    By default each mean path is centered over time before correlation, so a
    reliable but stationary condition offset cannot masquerade as repeatable
    trajectory shape.
    """
    if len(trajectories) != len(conditions):
        raise ValueError("trajectories and conditions must have equal length")
    rng = np.random.default_rng(seed)
    results: dict[int | str, float] = {}
    for condition in dict.fromkeys(conditions):
        indices = np.flatnonzero(np.asarray(conditions, dtype=object) == condition)
        if len(indices) < 4:
            continue
        correlations: list[float] = []
        for _ in range(int(repetitions)):
            shuffled = rng.permutation(indices)
            midpoint = len(shuffled) // 2
            first_path = np.mean([trajectories[idx] for idx in shuffled[:midpoint]], axis=0)
            second_path = np.mean([trajectories[idx] for idx in shuffled[midpoint:]], axis=0)
            if temporal_center:
                first_path = first_path - first_path.mean(axis=0, keepdims=True)
                second_path = second_path - second_path.mean(axis=0, keepdims=True)
            first = first_path.ravel()
            second = second_path.ravel()
            if np.std(first) > 0 and np.std(second) > 0:
                correlations.append(float(np.corrcoef(first, second)[0, 1]))
        if correlations:
            results[condition] = float(np.median(correlations))
    return results


def trajectory_separation(
    trajectories: list[np.ndarray],
    conditions: list[int | str],
    *,
    permutations: int = 1000,
    seed: int = 7,
    temporal_center: bool = True,
    max_pairs: int = 50_000,
) -> dict[str, float]:
    """Compare pairwise within-condition and between-condition path distances.

    Temporal centering makes this a path-shape comparison rather than a test of
    static condition centroids.
    """
    if len(trajectories) != len(conditions):
        raise ValueError("trajectories and conditions must have equal length")
    if len(trajectories) < 2:
        raise ValueError("at least two trajectories are required")
    prepared = [np.asarray(trajectory) for trajectory in trajectories]
    if temporal_center:
        prepared = [path - path.mean(axis=0, keepdims=True) for path in prepared]
    flat = np.stack([path.ravel() for path in prepared])
    condition_array = np.asarray(conditions, dtype=object)
    pair_count = len(flat) * (len(flat) - 1) // 2
    if pair_count <= int(max_pairs):
        i, j = np.triu_indices(len(flat), k=1)
    else:
        # Uniformly sample ordered off-diagonal pairs. Distance and same-label
        # status are symmetric, so this estimates the full unordered-pair
        # distribution without allocating an O(n^2) index array.
        rng = np.random.default_rng(seed)
        i = rng.integers(0, len(flat), size=int(max_pairs))
        j = rng.integers(0, len(flat) - 1, size=int(max_pairs))
        j = j + (j >= i)
    distances = np.sqrt(np.mean((flat[i] - flat[j]) ** 2, axis=1))

    def effect(labels: np.ndarray) -> float:
        same = labels[i] == labels[j]
        if not same.any() or same.all():
            return float("nan")
        return float(np.mean(distances[~same]) - np.mean(distances[same]))

    observed = effect(condition_array)
    rng = np.random.default_rng(seed + 1)
    null = np.asarray([effect(rng.permutation(condition_array)) for _ in range(int(permutations))])
    finite = np.isfinite(null)
    p_value = (
        float((1 + np.sum(null[finite] >= observed)) / (1 + np.sum(finite)))
        if np.isfinite(observed)
        else float("nan")
    )
    same = condition_array[i] == condition_array[j]
    return {
        "within_distance": float(np.mean(distances[same])) if same.any() else float("nan"),
        "between_distance": float(np.mean(distances[~same])) if (~same).any() else float("nan"),
        "between_minus_within": observed,
        "permutation_p_value": p_value,
        "sampled_pair_count": int(len(distances)),
        "total_pair_count": int(pair_count),
    }
