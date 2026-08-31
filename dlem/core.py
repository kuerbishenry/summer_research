import jax
import jax.numpy as jnp
import functools
import numpy as np
from jax import core, lax, vmap
from collections.abc import Mapping, Sequence

### GPU Utilities ###

def _block_until_ready(value):
    """
    Ensures GPU is done doing work on input before returning.
    
    """
    if isinstance(value, np.ndarray):
        return
    if hasattr(value, "block_until_ready"):
        value.block_until_ready()
        return
    if isinstance(value, Mapping):
        for v in value.values():
            _block_until_ready(v)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for v in value:
            _block_until_ready(v)

### Data Utilities ###

def _normalized_metrics_impl(
    p_left: jnp.ndarray,
    p_right: jnp.ndarray,
    target_flat: jnp.ndarray,
    mask_flat: jnp.ndarray,
    slowdown: float,
    rows: int,
    band_rows: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    generated = jax_forward_generate(p_left, p_right, slowdown, rows)
    band = generated[(rows - band_rows) :, :]
    normalized_generated = normalize_expected_observed(band).reshape(-1)
    mask_flat = mask_flat.astype(normalized_generated.dtype)
    masked_gen = normalized_generated * mask_flat
    masked_target = target_flat * mask_flat
    sum_mask = jnp.sum(mask_flat)
    mean_gen = jnp.where(sum_mask > 0, jnp.sum(masked_gen) / sum_mask, 0.0)
    mean_target = jnp.where(sum_mask > 0, jnp.sum(masked_target) / sum_mask, 0.0)
    centered_gen = (normalized_generated - mean_gen) * mask_flat
    centered_target = (target_flat - mean_target) * mask_flat
    cov = jnp.where(
        sum_mask > 0, jnp.sum(centered_gen * centered_target) / sum_mask, 0.0
    )
    var_gen = jnp.where(
        sum_mask > 0, jnp.sum(centered_gen * centered_gen) / sum_mask, 0.0
    )
    var_target = jnp.where(
        sum_mask > 0, jnp.sum(centered_target * centered_target) / sum_mask, 0.0
    )
    std_gen = jnp.sqrt(jnp.maximum(var_gen, 1e-12))
    std_target = jnp.sqrt(jnp.maximum(var_target, 1e-12))
    denom = std_gen * std_target
    corr = jnp.where(denom > 0, cov / denom, 0.0)
    diff = (normalized_generated - target_flat) * mask_flat
    mse = jnp.where(sum_mask > 0, jnp.sum(diff * diff) / sum_mask, 0.0)
    return corr, mse, normalized_generated, centered_gen, centered_target, band, generated 

@functools.partial(jax.jit, static_argnames=("rows", "band_rows"))
def normalized_correlation_and_mse(
    p_left: jnp.ndarray,
    p_right: jnp.ndarray,
    target_flat: jnp.ndarray,
    mask_flat: jnp.ndarray,
    slowdown: float,
    rows: int,
    band_rows: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    return _normalized_metrics_impl(
        p_left, p_right, target_flat, mask_flat, slowdown, rows, band_rows
    )


def normalize_expected_observed(band, log=True):
    """
    Compute correlation and MSE between generated and 
    target bands in the expected/observed normalised space.
    Used for tracking training progress.

    """    
    band = jnp.asarray(band, dtype=jnp.float32)
    if band.ndim != 2:
        raise ValueError("normalize_expected_observed expects a 2D array.")

    width, n = band.shape
    if width == 0 or n == 0:
        return jnp.zeros_like(band)

    row_idx = jnp.arange(width, dtype=jnp.int32)[:, None]
    col_idx = jnp.arange(n, dtype=jnp.int32)[None, :]
    mask = col_idx >= row_idx

    masked_band = jnp.where(mask, band, 0.0)
    counts = mask.sum(axis=1).astype(band.dtype)
    sums = masked_band.sum(axis=1)
    means = jnp.where(counts > 0, sums / counts, 0.0)
    means_expanded = means[:, None]
    offset = 1e-6
    valid = mask & (means_expanded > 0.0)
    denom = jnp.where(valid, means_expanded, 1.0)
    ratios = jnp.where(valid, (band + offset) / (denom + offset), 1.0)
    if not log:
        normalized = jnp.where(valid, ratios, 0.0)
        return normalized
    normalized = jnp.where(valid, jnp.log(ratios), 0.0)
    return normalized

def fill_band_nans(
    band: np.ndarray | jnp.ndarray,
    *,
    min_neighbors: int = 4,
) -> jnp.ndarray:
    "Replaces NaNs in the valid upper-triangular portion of a band with the row mean."
    if min_neighbors < 0 or min_neighbors > 8:
        raise ValueError("min_neighbors must be between 0 and 8.")

    arr = np.asarray(band, dtype=np.float32)
    width, n = arr.shape
    fallback = np.asarray(_fill_band_nan_with_row_mean(arr), dtype=np.float32)
    result = np.array(arr, copy=True)

    def _lookup(source: np.ndarray, row_idx: int, col_idx: int) -> float:
        if not (0 <= row_idx < n and 0 <= col_idx < n):
            return np.nan
        if row_idx > col_idx:
            row_idx, col_idx = col_idx, row_idx
        diag = col_idx - row_idx
        if diag >= width:
            return np.nan
        value = source[diag, col_idx]
        return float(value) if np.isfinite(value) else np.nan

    for diag in range(width):
        start_col = diag
        if start_col >= n:
            break
        for col in range(start_col, n):
            if not np.isnan(arr[diag, col]):
                continue
            row_idx = col - diag
            col_idx = col
            neighbors: list[float] = []
            for d_row in (-1, 0, 1):
                for d_col in (-1, 0, 1):
                    if d_row == 0 and d_col == 0:
                        continue
                    n_row = row_idx + d_row
                    n_col = col_idx + d_col
                    val = _lookup(arr, n_row, n_col)
                    if np.isfinite(val):
                        neighbors.append(val)
            if len(neighbors) >= min_neighbors:
                result[diag, col] = float(np.mean(neighbors))
            else:
                result[diag, col] = fallback[diag, col]

    return jnp.asarray(result, dtype=jnp.float32)

def _fill_band_nan_with_row_mean(band: jnp.ndarray) -> jnp.ndarray:
    band = jnp.asarray(band, dtype=jnp.float32)
    width, n = band.shape

    row_idx = jnp.arange(width, dtype=jnp.int32)[:, None]
    col_idx = jnp.arange(n, dtype=jnp.int32)[None, :]
    valid_mask = col_idx >= row_idx

    valid_values = jnp.where(valid_mask, band, jnp.nan)
    row_means = jnp.nanmean(valid_values, axis=1)
 
    row_means = jnp.nan_to_num(row_means, nan=0.0)[:, None]

    replacements = jnp.where(valid_mask, row_means, band)
    return jnp.where(valid_mask & jnp.isnan(band), replacements, band)


### Training utilities ###

def _compute_rights_table(p_right, mat_rows):
    "Function to precompute right probabilities for all rows."
    n = p_right.shape[0]
    if mat_rows is None:
        limit = n
    else:
        limit = max(min(int(mat_rows), n), 0)
    length = max(limit - 1, 0)
    if length == 0:
        return jnp.zeros((0, n), dtype=p_right.dtype)

    def scan_body(this_right, _):
        next_right = jnp.concatenate((this_right[-1:], this_right[:-1]))
        return next_right, this_right

    _, rights = lax.scan(scan_body, p_right, xs=None, length=length)
    return rights

def _apply_vmap_update(prev, p_left, rights_table, slowdown, start_row):
    "Function to apply dLEM update to multiple rows via vmap."
    n_rows = prev.shape[0]
    n = p_left.shape[0]
    max_rows = max(n_rows - 1 - start_row, 0)
    max_cols = max(n - 1 - start_row, 0)
    max_rights = max(rights_table.shape[0] - start_row, 0)
    row_count = min(max_rows, max_cols, max_rights)
    if row_count == 0:
        return prev

    pl_l = p_left[:-1]
    pl_r = p_left[1:]
    rows = prev[start_row : start_row + row_count]
    rights = rights_table[start_row : start_row + row_count]

    def update(row, right):
        nom = row[:-1] * pl_l + row[1:] * right[1:]
        denom = pl_r + right[:-1] + slowdown
        return nom / denom

    new_rows = vmap(update)(rows, rights)
    return prev.at[start_row + 1 : start_row + 1 + row_count, 1:].set(new_rows)

def _forward_scan(
    p_left,
    p_right,
    mat,
    slowdown,
    start_row,
    steps,
    collect=False,
):
    "Function to perform multiple dLEM updates via lax.scan."
    start_row = core.concrete_or_error(
        int, start_row, "start_row must be a static integer."
    )
    steps = core.concrete_or_error(
        int, steps, "steps must be a static integer for _forward_scan."
    )
    slowdown = jnp.asarray(slowdown, dtype=mat.dtype)
    rights_table = _compute_rights_table(p_right, mat.shape[0])

    def body(prev, _):
        next_mat = _apply_vmap_update(prev, p_left, rights_table, slowdown, start_row)
        return next_mat, prev if collect else None

    final, history = lax.scan(body, mat, xs=None, length=steps)
    return final, history

@functools.partial(
    jax.jit, static_argnames=("start_row", "steps")
)
def _forward_impl(p_left, p_right, mat, slowdown, *, start_row, steps):
    "JIT-compiled dLEM forward pass implementation."
    final, _ = _forward_scan(
        p_left,
        p_right,
        mat,
        slowdown,
        start_row=start_row,
        steps=steps,
        collect=False,
    )
    return final

def forward_custom(p_left, p_right, mat, slowdown, *, start_row=0, steps=1):
    "Custom dLEM forward pass with adjustable start row and steps."
    return _forward_impl(
        p_left,
        p_right,
        mat,
        slowdown,
        start_row=start_row,
        steps=steps,
    )

def jax_forward_generate(p_left, p_right, slowdown, rows):
    "Basic dLEM forward pass that produces a full matrix of row updates (first row all ones)."
    p_left = jnp.asarray(p_left, dtype=jnp.float32)
    p_right = jnp.asarray(p_right, dtype=jnp.float32)
    slowdown = jnp.asarray(slowdown, dtype=p_left.dtype)
    rows = int(rows)
    if rows <= 0:
        raise ValueError("rows must be positive")

    n = p_left.shape[0]
    initial_row = jnp.ones((n,), dtype=p_left.dtype)

    def scan_body(carry, _):
        prev_row, this_right = carry
        nom = prev_row[:-1] * p_left[:-1] + prev_row[1:] * this_right[1:]
        denom = p_left[1:] + this_right[:-1] + slowdown
        updated = nom / denom
        next_row = prev_row.at[1:].set(updated)
        next_right = jnp.concatenate((this_right[-1:], this_right[:-1]))
        return (next_row, next_right), next_row

    length = max(rows - 1, 0)
    (_, _), tail = lax.scan(scan_body, (initial_row, p_right), xs=None, length=length)
    if rows == 1:
        return initial_row[None, :]
    return jnp.vstack((initial_row, tail))


def adaptive_coarsegrain(ar, countar, cutoff=5, max_levels=8, min_shape=8):
    """
    Copied over from cooltools to avoid dependency.
    https://cooltools.readthedocs.io/en/latest/cooltools.lib.html#cooltools.lib.numutils.adaptive_coarsegrain
    """

    def _coarsen(ar, operation=np.sum):
        """Coarsegrains an array by a factor of 2"""
        M = ar.shape[0] // 2
        newar = np.reshape(ar, (M, 2, M, 2))
        cg = operation(newar, axis=1)
        cg = operation(cg, axis=2)
        return cg

    def _expand(ar, counts=None):
        """
        Performs an inverse of nancoarsen
        """
        N = ar.shape[0] * 2
        newar = np.zeros((N, N))
        newar[::2, ::2] = ar
        newar[1::2, ::2] = ar
        newar[::2, 1::2] = ar
        newar[1::2, 1::2] = ar
        return newar

    ar = np.asarray(ar, float)
    countar = np.asarray(countar, float)

    Norig = ar.shape[0]
    Nlog = np.log2(Norig)
    if not np.allclose(Nlog, np.rint(Nlog)):
        newN = int(2 ** np.ceil(Nlog))
        newar = np.empty((newN, newN), dtype=float)
        newar[:] = np.nan
        newcountar = np.zeros((newN, newN), dtype=float)
        newar[:Norig, :Norig] = ar
        newcountar[:Norig, :Norig] = countar
        ar = newar
        countar = newcountar

    armask = np.isfinite(ar)
    countar[~armask] = 0
    ar[~armask] = 0

    assert np.isfinite(countar).all()
    assert countar.shape == ar.shape

    ar_cg = [ar]
    countar_cg = [countar]
    armask_cg = [armask]

    for i in range(max_levels):
        if countar_cg[-1].shape[0] > min_shape:
            countar_cg.append(_coarsen(countar_cg[-1]))
            armask_cg.append(_coarsen(armask_cg[-1]))
            ar_cg.append(_coarsen(ar_cg[-1]))

    ar_cur = ar_cg.pop()
    countar_cur = countar_cg.pop()
    armask_cur = armask_cg.pop()

    for i in range(len(countar_cg)):
        ar_next = ar_cg.pop()
        countar_next = countar_cg.pop()
        armask_next = armask_cg.pop()

        val_cur = ar_cur / armask_cur
        val_exp = _expand(val_cur)
        addar_exp = val_exp * armask_next

        countar_next_mask = np.array(countar_next)
        countar_next_mask[armask_next == 0] = np.nan 
        countar_exp = _expand(_coarsen(countar_next, operation=np.nanmin))

        curmask = countar_exp < cutoff
        ar_next[curmask] = addar_exp[curmask]
        ar_next[armask_next == 0] = 0

        ar_cur = ar_next
        countar_cur = countar_next
        armask_cur = armask_next

    ar_next[armask_next == 0] = np.nan
    ar_next = ar_next[:Norig, :Norig]

    return ar_next