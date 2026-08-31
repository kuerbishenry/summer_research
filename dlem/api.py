import jax
import jax.numpy as jnp
import optax
import functools
import numpy as np
import os
import time
import hictkpy as htk
import pandas
from scipy.optimize import curve_fit   
from rich import print
import re
import pyranges as pr 
import pyarrow
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from rich.progress import track

from dlem.core import (
    _block_until_ready,
    forward_custom,
    normalize_expected_observed,
    normalized_correlation_and_mse,
    jax_forward_generate,
    fill_band_nans,
    adaptive_coarsegrain
)

### GPU Utilities ###

def set_device(device) -> str:
    """
        Set the JAX device configuration for computation.

    Args:
        device (str): 'cpu', 'gpu', or 'tpu'.

    Returns:
        The current JAX platform configuration string.
    """
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.3")
    os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

    if device == 'gpu':
        try:
            device = jax.devices("gpu")
            jax.config.update("jax_platforms", "cuda,cpu")
        except RuntimeError:
            print("GPU backend not available, falling back to CPU")
            jax.config.update("jax_platforms", "cpu")
    elif device == 'tpu':
        jax.config.update("jax_platforms", "tpu,cpu")
    else:
        jax.config.update("jax_platforms", "cpu")
        
    return jax.config.jax_platforms

### Data Manipulation Utilities ###

def _geometric_row_mean(arr: np.ndarray, 
                       valid_mask: np.ndarray | None = None
                       ) -> np.ndarray:
    """
    Compute the geometric mean of each row in a 2D array, optionally using a validity mask.

    Args:
        arr (np.ndarray): Input 2D array.
        valid_mask (np.ndarray, optional): Boolean mask for valid entries.

    Returns:
        np.ndarray: Array of geometric means for each row.
    """
    if valid_mask is None:
        valid_mask = np.isfinite(arr) & (arr > 0)
    else:
        valid_mask = valid_mask & np.isfinite(arr) & (arr > 0)
    row_means = np.full(arr.shape[0], np.nan, dtype=np.float32)
    for i in range(arr.shape[0]):
        row_vals = arr[i][valid_mask[i]]
        if row_vals.size == 0:
            continue
        row_means[i] = float(np.exp(np.mean(np.log(row_vals))))
    return row_means

def norm_band(cur_pred: np.ndarray, 
                       cur_goal: np.ndarray
                       ) -> np.ndarray:
    """
   Normalize prediction band to match the geometric mean of the goal band per row.

    Args:
        cur_pred (np.ndarray): Predicted band matrix.
        cur_goal (np.ndarray): Target band matrix.

    Returns:
        np.ndarray: Normalized prediction band.
    """

    rows, span = cur_goal.shape
    row_idx = np.arange(rows)[:, None]
    col_idx = np.arange(span)[None, :]
    valid_mask = col_idx >= row_idx
    rng = np.random.default_rng(seed=42)
    sample_size_geom = min(500, span)
    rows_for_norm = min(cur_goal.shape[0], cur_pred.shape[0], valid_mask.shape[0])
    if rows_for_norm <= 0:
        raise ValueError(
            f"(band rows={cur_goal.shape[0]}, pred rows={cur_pred.shape[0]})."
        )
    if rows_for_norm < cur_pred.shape[0]:
        print("(available input rows); remaining rows left as-is.")

    def sample_rows(arr, base_cols):
        sampled = np.full((rows_for_norm, len(base_cols)), np.nan, dtype=np.float32)
        for r in range(rows_for_norm):
            cols_r = base_cols + r
            mask = cols_r < span
            if not np.any(mask):
                continue
            sampled[r, : np.sum(mask)] = arr[r, cols_r[mask]]
        return sampled

    base_cols = np.sort(rng.choice(span, size=sample_size_geom, replace=False))
    band_interp_sample = sample_rows(cur_goal, base_cols)
    pred_band_sample = sample_rows(cur_pred, base_cols)
    valid_mask_sample = sample_rows(valid_mask.astype(np.float32), base_cols)
    valid_mask_sample = np.isfinite(valid_mask_sample) & (valid_mask_sample > 0)
    input_gm = _geometric_row_mean(
        band_interp_sample[:rows_for_norm], valid_mask_sample[:rows_for_norm]
    )
    pred_gm = _geometric_row_mean(
        pred_band_sample[:rows_for_norm], valid_mask_sample[:rows_for_norm]
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        scales = input_gm / pred_gm
    scales = np.where(np.isfinite(scales) & (pred_gm > 0), scales, 1.0)
    if scales.shape[0] < span:
        scales_full = np.ones(span, dtype=np.float32)
        scales_full[:rows] = scales
        scales = scales_full
    cur_pred = cur_pred * np.squeeze(scales)
    
    return cur_pred

def fetch_band(cool_filename: str,
               resolution: int,
                cur_region: str,
                width: int,
                passthrough: bool = False,
                fill_nans: bool = True) -> np.ndarray | pyarrow.Table:
    """
    Extract a band matrix from a .cool file for a given region and width.

    Args:
        cool_filename (str): Path to .cool file.
        cur_region (str): Genomic region string (e.g., 'chr1:0-10000').
        width (int): Band width.
        passthrough (bool): If True, return raw data as DataFrame.
        fill_nans (bool): If True, fill NaNs in the band.

    Returns:
        np.ndarray or pd.DataFrame: Band matrix or DataFrame if passthrough.
    """
    if width <= 0:
        raise ValueError("width must be positive.")

    with htk.File(cool_filename, resolution) as cool_handle:

        split_region = re.split(r"[-:]", cur_region)
        bins = cool_handle.bins().to_pandas(range=cur_region)
        step = width // 2
        block = width + step
        n = len(bins)

        if not passthrough:
            band = np.zeros((width, n), dtype=np.float32)
            for i in range(0, n, step):
                i0, i1 = i, min(n, i + block)
                region = f'{split_region[0]}:{bins.start.iloc[i0]}-{bins.end.iloc[i1-1]}'
                raw = cool_handle.fetch(region).to_numpy()
                bal = cool_handle.fetch(region, normalization="weight").to_numpy()
                with np.errstate(divide="ignore", invalid="ignore"):
                    sub = adaptive_coarsegrain(bal, raw, cutoff=3, max_levels=8)

                for k in range(width):
                    vals = jnp.diag(sub, k)
                    start = i0 + k
                    end = start + len(vals)
                    if start < n:
                        band[k, start:end] = vals[: max(0, n - start)]
            if fill_nans:
                band = fill_band_nans(band)
        else:
            region = f'{split_region[0]}:{split_region[1]}-{split_region[2]}'
            band = cool_handle.fetch(region).to_pandas()
            
        return band
    

def band_generate_pixels(
    band: np.ndarray | None,
    ref_cool_filename: str,
    resolution: int,
    chrom_name: str,
    offset: int = 0,
) -> pandas.DataFrame:
    """
    Convert a band matrix to pixel format for a given chromosome using a reference .cool file.

    Args:
        band (np.ndarray or None): Band matrix or None for passthrough.
        ref_cool_filename (str): Path to reference .cool file.
        chrom_name (str): Chromosome name.
        offset (int): Bin offset for pixel coordinates.

    Returns:
        pd.DataFrame: Pixel table with bin1_id, bin2_id, and count columns.
    """
    with htk.File(ref_cool_filename, resolution) as cool_ref_htk_handle:
        chrom_sizes = cool_ref_htk_handle.chromosomes()
        resolution = cool_ref_htk_handle.resolution()
        bin_offset = cool_ref_htk_handle.bins().get(chrom_name,0).id + offset
    bin1_arr = []
    bin2_arr = []
    cnt_arr = []
    if band is not None:
        print(f"Massaging band data for {chrom_name} into pixels")
        width, n = band.shape
        for bin1 in range(n):
            max_bin2 = min(n, bin1 + width)
            if max_bin2 > bin1:
                bin2_range = np.arange(bin1, max_bin2, dtype=np.int64)
                offsets = bin2_range - bin1
                values = np.array(band[offsets, bin2_range], dtype=np.float32)
                mask = np.isfinite(values) & (values != 0)
                if mask.any():
                    bin1_arr.extend(np.full(mask.sum(), bin_offset + bin1, dtype=np.int64))
                    bin2_arr.extend(bin_offset + bin2_range[mask])
                    cnt_arr.extend(values[mask])
        pixels = pyarrow.table({
                "bin1_id": bin1_arr,
                "bin2_id": bin2_arr,
                "count": cnt_arr
        }).to_pandas()
    else:
        print(f"Passing through data for {chrom_name} into pixels")
        chr_start = 0
        chr_end = chrom_sizes[chrom_name]
        region_name = f"{chrom_name}:{chr_start}-{chr_end}"
        pixels = fetch_band(ref_cool_filename, 
                            resolution=resolution,
                            cur_region=region_name, 
                            width=chr_end // resolution,
                            passthrough=True)
    return(pixels)

### Plotting ###

def plot_map(contact_map: np.array, 
             p_left: np.array, 
             p_right: np.array, 
             region: str, 
             resolution: int) -> go.Figure:
    """
    Plot a contact map and associated left/right parameter tracks using Plotly.

    Args:
        contact_map (np.array): 2D contact map.
        p_left (np.array): Left parameter vector.
        p_right (np.array): Right parameter vector.
        region (str): Genomic region string.
        resolution (int): Bin resolution.

    Returns:
        plotly.graph_objs.Figure: Plotly figure object.
    """
    split_region = re.split(r"[-:]", region)
    in_region_gr = pr.PyRanges(
        chromosomes=[split_region[0]],
        starts=[int(split_region[1])],
        ends=[int(split_region[2])]
    )
    labels_gr = in_region_gr.tile(resolution)
    cur_regions_labels = labels_gr.as_df().agg('{0[Chromosome]}:{0[Start]}-{0[End]}'.format, axis=1)
    tick_labels = cur_regions_labels[range(0,len(cur_regions_labels),int(len(cur_regions_labels)/4))]
    
    fig = make_subplots(rows=3, cols=1, 
                        row_heights=[0.1, 0.1, .8],
                        vertical_spacing = 0.01,
                        shared_xaxes=True)
    
    trace_pred = go.Heatmap(
        x=cur_regions_labels,
        y=cur_regions_labels,
        z=contact_map,
        colorscale='RdPu',
        showscale=True,
        hovertemplate="Bin1: %{x}<br>Bin2: %{y}<br>Count: %{z:.2f}<extra></extra>"
    
    )
    
    trace_pred_l = go.Scatter(
        mode='lines', 
        x=cur_regions_labels, 
        y=p_left,
        line=dict(color='#337FC2',width=2)
    )
    
    trace_pred_r = go.Scatter(
        mode='lines', 
        x=cur_regions_labels, 
        y=p_right,
        line=dict(color='#F2B340',width=2)
    )
    
    fig.add_trace(trace_pred_l, row=1, col=1)
    fig.add_trace(trace_pred_r, row=2, col=1)
    fig.add_trace(trace_pred, row=3, col=1)
    
    fig.update_xaxes(showticklabels=False, row=1, col=1, showgrid=False)
    fig.update_yaxes(showticklabels=True, row=1, col=1, range=[0, 1], showgrid=False)
    
    fig.update_xaxes(showticklabels=False, row=2, col=1, showgrid=False)
    fig.update_yaxes(showticklabels=True, row=2, col=1, range=[0, 1], showgrid=False)
    
    fig.update_yaxes(showticklabels=False, row=3, col=1, autorange="reversed")
    fig.update_xaxes(showticklabels=True, row=3, col=1)

    fig.update_layout(
        autosize=True,
        title_text=f"Learned parameters and prediction for {split_region[0]}:{split_region[1]}-{split_region[2]} ",
        title_x=0.5,
        showlegend=False,
    )
    
    fig.update_xaxes(
        tickmode='array',
        ticktext= tick_labels,
        tickvals= tick_labels.index,
        row=3, col=2
    )
    return fig

def flip_diag_row(mat, zero_lower=False) -> np.ndarray:
    """
    Flip a matrix from diagonal rows to full matrix, optionally zeroing the lower triangle.

    Args:
        mat (np.ndarray): Input matrix.
        zero_lower (bool): If True, zero out lower triangle.

    Returns:
        np.ndarray: Flipped matrix.
    """
    n = mat.shape[0]
    ii = np.arange(n)
    iy = ii.reshape(1, -1) * np.ones(n).reshape(-1, 1)
    ix = (ii[::-1].reshape(-1, 1) - ii[::-1].reshape(1, -1)) % n
    out = mat[ix.astype(int), iy.astype(int)]
    if zero_lower:
        out = np.triu(out)
    return out

def symmetrize_upper(mat):
    """
    Symmetrize a matrix by copying the upper triangle to the lower triangle.

    Args:
        mat (np.ndarray): Input matrix.

    Returns:
        np.ndarray: Symmetrized matrix.
    """
    mat = np.triu(mat)
    return mat + mat.T - np.diag(np.diag(mat))

def compute_contact_heatmap(
    vals,
    *,
    normalize: bool = True,
    log: bool = True,
    lower_vals=None,
    match_lower_range: bool = True,
) -> np.ndarray:
    """
    Compute a symmetrized contact heatmap from upper and optional lower band matrices.

    Args:
        vals (np.ndarray): Upper band matrix.
        normalize (bool): Whether to normalize expected/observed.
        log (bool): Whether to log-transform values.
        lower_vals (np.ndarray, optional): Lower band matrix.
        match_lower_range (bool): Match lower triangle range to upper.

    Returns:
        np.ndarray: Symmetrized contact heatmap.
    """
    def _prepare(band):
        data = band
        if normalize:
            data = normalize_expected_observed(data, log=log)
        pad_width = max(0,data.shape[1]-data.shape[0])
        data_padded = jnp.pad(data, 
                    pad_width=((0, pad_width)), 
                    mode='constant', 
                    constant_values=0)
        prepared = flip_diag_row(data_padded)
        if (not normalize) and log:
            prepared = np.log(prepared + 1e-6)
        return np.asarray(prepared)

    def _tri_mask(shape, upper=True):
        if upper:
            return np.triu(np.ones(shape, dtype=bool), k=0)
        return np.tril(np.ones(shape, dtype=bool), k=-1)

    def _min_max(data, mask):
        masked = data[mask]
        if masked.size == 0:
            return None, None
        return float(masked.min()), float(masked.max())

    upper = _prepare(vals)
    if lower_vals is None:
        return symmetrize_upper(upper)

    lower = np.array(_prepare(lower_vals).T, copy=True)
    if match_lower_range:
        upper_mask = _tri_mask(upper.shape, upper=True)
        lower_mask = _tri_mask(lower.shape, upper=False)
        upper_min, upper_max = _min_max(upper, upper_mask)
        lower_min, lower_max = _min_max(lower, lower_mask)
        upper_range = (
            None
            if (upper_min is None or upper_max is None)
            else (upper_max - upper_min)
        )
        lower_range = (
            None
            if (lower_min is None or lower_max is None)
            else (lower_max - lower_min)
        )
        if (
            lower_min is not None
            and lower_max is not None
            and upper_min is not None
            and upper_max is not None
        ):
            if lower_range and lower_range > 0 and upper_range and upper_range > 0:
                scaled = (lower[lower_mask] - lower_min) / lower_range
                lower[lower_mask] = scaled * upper_range + upper_min
            elif upper_range == 0:
                lower[lower_mask] = upper_min
            else:
                lower[lower_mask] = upper_min
    return np.triu(upper) + np.tril(lower, k=-1)


### dLEM Train/Prediction Utilities ###

def train_dlem(
    noisy_target,
    *,
    steps: int,
    start_row: int,
    slowdown: float,
    learning_rate: float = 1e-2,
    train_steps: int = 600,
    verbose: bool = False,
    loss_type: str = "multinomial",
    weight_power: int = 0,
    auto_stop_metric: str = "mse",
) -> dict[str, float]:
    """
    Unified trainer for the dLEM model.

    Args:
        noisy_target (array-like): Target band matrix with noise.
        steps (int): Number of steps for the model.
        start_row (int): Starting row for training.
        slowdown (float): Slowdown parameter for the model.
        learning_rate (float): Learning rate for optimizer.
        train_steps (int): Number of training steps.
        verbose (bool): If True, print progress.
        loss_type (str): 'multinomial' or 'mse'.
        weight_power (int): Weighting power for MSE loss.
        auto_stop_metric (str): Early stopping metric ('mse', 'corr', or 'none').

    Returns:
        dict: Training results and learned parameters.
    """

    loss_type = loss_type.lower()
    if loss_type not in {"multinomial", "mse"}:
        raise ValueError("loss_type must be either 'multinomial' or 'mse'.")
    use_mse_loss = loss_type == "mse"
    if use_mse_loss and weight_power not in (0, 1):
        raise ValueError("weight_power must be 0 or 1 when using mse loss.")
    auto_stop_metric = auto_stop_metric.lower()
    if auto_stop_metric not in {"mse", "corr", "none"}:
        raise ValueError("auto_stop_metric must be one of {'mse', 'corr', 'none'}.")

    noisy_target = jnp.asarray(noisy_target, dtype=jnp.float32)
    rows, n = noisy_target.shape
    if steps <= 0:
        raise ValueError("steps must be positive.")
    if rows < steps:
        raise ValueError("noisy_target must have at least `steps` rows.")

    mat = jnp.zeros((rows, n), dtype=jnp.float32)
    mat = mat.at[0].set(1.0)

    target_band = noisy_target[(steps - 1) :, :]
    band_rows = int(target_band.shape[0])

    row_idx = jnp.arange(band_rows, dtype=jnp.int32)[:, None]
    col_idx = jnp.arange(n, dtype=jnp.int32)[None, :]
    valid_mask = col_idx >= (row_idx + start_row)
    mask_float = valid_mask.astype(jnp.float32)
    mask_flat = mask_float.reshape(-1)
    target_band = (target_band * mask_float).astype(jnp.float32)

    normalized_target = normalize_expected_observed(target_band)
    normalized_target_flat = normalized_target.reshape(-1)

    forward_fn = functools.partial(
        forward_custom,
        slowdown=slowdown,
        start_row=start_row,
        steps=steps,
    )

    def multinomial_loss(pred_counts, target_counts, mask):
        eps = 1e-8
        masked_pred = pred_counts * mask
        masked_target = target_counts * mask
        total_pred = jnp.sum(masked_pred, axis=-1, keepdims=True)
        probs = jnp.where(
            total_pred > 0,
            masked_pred / jnp.maximum(total_pred, eps),
            0.0,
        )
        probs = jnp.clip(probs, min=eps, max=None)
        log_probs = jnp.log(probs)
        total_target = jnp.sum(masked_target, axis=-1, keepdims=True)

        nll = -jnp.sum(masked_target * log_probs, axis=-1)

        total_target_scalar = total_target.squeeze(-1)
        log_factorial_sum = jax.scipy.special.gammaln(total_target_scalar + 1.0)
        log_factorial_terms = jnp.sum(
            jax.scipy.special.gammaln(masked_target + 1.0), axis=-1
        )
        loss = nll - log_factorial_sum + log_factorial_terms

        row_mask = jnp.sum(mask, axis=-1) > 0
        row_mask_f = row_mask.astype(pred_counts.dtype)
        loss = loss * row_mask_f
        denom = jnp.sum(row_mask_f)
        return jnp.where(denom > 0, jnp.sum(loss) / denom, 0.0)

    def mse_loss_from_band(pred_band):
        pred_norm = normalize_expected_observed(pred_band)
        diff = (pred_norm - normalized_target)
        if weight_power == 0:
            weights = mask_float
        else:
            weights = jnp.where(mask_float > 0, target_band, 0.0)
        weight_sum = jnp.sum(weights)
        diff_sq = (diff * diff) * weights
        return jnp.where(weight_sum > 0, jnp.sum(diff_sq) / weight_sum, 0.0)

    def loss_fn(current_params):
        p_left_param = jnp.clip(current_params["p_left"], 1e-4, None)
        p_right_param = jnp.clip(current_params["p_right"], 1e-4, None)
        pred = forward_fn(p_left_param, p_right_param, noisy_target)
        pred_band = pred[(steps - 1) :, :]
        if use_mse_loss:
            return mse_loss_from_band(pred_band)
        return multinomial_loss(pred_band, target_band, mask_float)

    value_and_grad = jax.jit(
        jax.value_and_grad(loss_fn)
    )

    params = {
        "p_left": jnp.ones(n, dtype=jnp.float32),
        "p_right": jnp.ones(n, dtype=jnp.float32),
    }
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    history: list[float] = []
    corr_history: list[tuple[int, float]] = []
    mse_history: list[tuple[int, float]] = []
    start_time = time.time()
    best_corr = -np.inf
    best_corr_params = None
    best_corr_mse = np.inf
    best_mse = np.inf
    best_mse_params = None
    best_corr_step = None
    best_mse_step = None

    best_loss = np.inf

    for step in track(range(1, train_steps + 1), description="Training..."):
        loss_value, grads = value_and_grad(params)
        grads = {
            "p_left": grads["p_left"],
            "p_right": grads["p_right"],
        }
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        params = {
            "p_left": jnp.clip(params["p_left"], 0.0, 1.0),
            "p_right": jnp.clip(params["p_right"], 0.0, 1.0),
        }

        loss_float = float(jax.device_get(loss_value))
        history.append(loss_float)

        if loss_float < best_loss:
            best_loss = loss_float

        corr_float = None
        mse_float = None
        if step % 20 == 0 or step == train_steps:
            corr_value, mse_value, normalized_generated, centered_gen, centered_target, band, generated = normalized_correlation_and_mse(
                params["p_left"],
                params["p_right"],
                normalized_target_flat,
                mask_flat,
                slowdown,
                rows,
                band_rows,
            )
            corr_float = float(jax.device_get(corr_value))
            mse_float = float(jax.device_get(mse_value))
            corr_history.append((step, corr_float))
            mse_history.append((step, mse_float))
            if corr_float > best_corr:
                best_corr = corr_float
                best_corr_params = {
                    "p_left": params["p_left"],
                    "p_right": params["p_right"],
                }
                best_corr_mse = mse_float
                best_corr_step = step
            if mse_float < best_mse:
                best_mse = mse_float
                best_mse_params = {
                    "p_left": params["p_left"],
                    "p_right": params["p_right"],
                }
                best_normalized_generated = normalized_generated
                best_centered_gen = centered_gen
                best_centered_target = centered_target
                best_generated = generated
                best_band = band
                best_mse_step = step

        if verbose and (step == 1 or step % 20 == 0 or step == train_steps):
            msg = f"[train] step={step:04d} loss={loss_float:.6f}"
            if corr_float is not None:
                msg += f" corr={corr_float:.4f}"
            if mse_float is not None:
                msg += f" mse={mse_float:.6f}"
            print(msg)

        if auto_stop_metric != "none" and mse_float is not None and corr_float is not None:
            if auto_stop_metric == "mse" and best_mse_step is not None and step > best_mse_step and step < train_steps:
                if verbose:
                    print(
                        f"[train] early stopping at step {step} (no MSE improvement after step {best_mse_step}).",
                        flush=True,
                    )
                break
            if auto_stop_metric == "corr" and best_corr_step is not None and step > best_corr_step and step < train_steps:
                if verbose:
                    print(
                        f"[train] early stopping at step {step} (no correlation improvement after step {best_corr_step}).",
                        flush=True,
                    )
                break

    elapsed = time.time() - start_time

    learned_left = jnp.clip(params["p_left"], 1e-4, None)
    learned_right = jnp.clip(params["p_right"], 1e-4, None)
    final_pred = forward_fn(learned_left, learned_right, mat)
    final_pred_band = final_pred[(steps - 1) :, :]


    if use_mse_loss:
        final_loss = float(jax.device_get(mse_loss_from_band(final_pred_band)))
    else:
        final_loss = float(
            jax.device_get(
                multinomial_loss(final_pred_band, target_band, mask_float)
            )
        )

    final_corr_value, final_mse_value, final_normalized_generated, final_centered_gen, final_centered_target, final_band,final_generated = normalized_correlation_and_mse(
        learned_left,
        learned_right,
        normalized_target_flat,
        mask_flat,
        slowdown,
        rows,
        band_rows,
    )
    final_corr = float(jax.device_get(final_corr_value))
    final_mse = float(jax.device_get(final_mse_value))

    if best_corr_params is None:
        best_corr = final_corr
        best_corr_params = {
            "p_left": learned_left,
            "p_right": learned_right,
        }
        best_corr_mse = final_mse

    if best_mse_params is None:
        best_mse = final_mse
        best_mse_params = {
            "p_left": learned_left,
            "p_right": learned_right,
        }

    best_corr_pred = forward_fn(
        best_corr_params["p_left"], best_corr_params["p_right"], mat
    )
    best_corr_band = best_corr_pred[(steps - 1) :, :]

    forward_fn_mse_params = forward_fn(best_mse_params["p_left"], best_mse_params["p_right"], noisy_target)
    forward_fn_corr_params = forward_fn(best_corr_params["p_left"], best_corr_params["p_right"], noisy_target)

    _block_until_ready(
        {
            "final_pred": final_pred,
            "best_corr_pred": best_corr_pred,
            "learned_left": learned_left,
            "learned_right": learned_right,
            
        }
    )

    results = {
        "loss_history": np.asarray(history, dtype=np.float32),
        "final_loss": final_loss,
        "p_left": np.asarray(jax.device_get(learned_left)),
        "p_right": np.asarray(jax.device_get(learned_right)),
        "corr_history": np.asarray(corr_history, dtype=np.float32)
        if corr_history
        else np.empty((0, 2), dtype=np.float32),
        "mse_history": np.asarray(mse_history, dtype=np.float32)
        if mse_history
        else np.empty((0, 2), dtype=np.float32),
        "final_corr": final_corr,
        "final_mse": final_mse,
        "best_corr": best_corr,
        "best_corr_mse": best_corr_mse,
        "best_mse": best_mse,
        "best_corr_params": {
            "p_left": np.asarray(jax.device_get(best_corr_params["p_left"])),
            "p_right": np.asarray(jax.device_get(best_corr_params["p_right"])),
        },
        "best_mse_params": {
            "p_left": np.asarray(jax.device_get(best_mse_params["p_left"])),
            "p_right": np.asarray(jax.device_get(best_mse_params["p_right"])),
        },
        "p_left_cor": np.asarray(jax.device_get(best_corr_params["p_left"])),
        "p_right_cor": np.asarray(jax.device_get(best_corr_params["p_right"])),
        "p_left_mse": np.asarray(jax.device_get(best_mse_params["p_left"])),
        "p_right_mse": np.asarray(jax.device_get(best_mse_params["p_right"])),
        "best_corr_band": np.asarray(jax.device_get(best_corr_band)),
        "target_band": np.asarray(jax.device_get(target_band)),
        "mask": np.asarray(jax.device_get(mask_float)),
        "noisy_target": np.asarray(jax.device_get(noisy_target)),
        "normalized_target": np.asarray(jax.device_get(normalized_target)),
        "elapsed": elapsed,
        "loss_type": loss_type,
        "weight_power": weight_power if use_mse_loss else 0,

        "forward_fn_mse_params": np.asarray(jax.device_get(forward_fn_mse_params)),
        "forward_fn_corr_params": np.asarray(jax.device_get(forward_fn_corr_params)),
        "final_pred_band": np.asarray(jax.device_get(final_pred_band)),

        "final_normalized_generated": np.asarray(jax.device_get(final_normalized_generated)),
        "best_normalized_generated": np.asarray(jax.device_get(best_normalized_generated)),

        "final_centered_gen": np.asarray(jax.device_get(final_centered_gen)),
        "best_centered_gen": np.asarray(jax.device_get(best_centered_gen)),
        "final_centered_target": np.asarray(jax.device_get(final_centered_target)),
        "best_centered_target": np.asarray(jax.device_get(best_centered_target)),
        "best_band": np.asarray(jax.device_get(best_band)),
        "best_generated": np.asarray(jax.device_get(best_generated)),
    }
    return results

def fit_band_row_profile_sliding(
    band,
    *,
    start_row: int,
    extent: int,
    window_size: int,
    window_step: int | None = None,
    log_y=True,
) -> dict[str, float]:
    """
    Fit a row profile model to a band matrix using sliding windows and aggregate fits.

    Args:
        band (np.ndarray): Input band matrix.
        start_row (int): Starting row for fitting.
        extent (int): Number of rows to fit.
        window_size (int): Size of sliding window.
        window_step (int, optional): Step size for sliding window.
        log_y (bool): If True, fit in log space.

    Returns:
        dict: Aggregated fit parameters and statistics.
    """
    band = np.asarray(band, dtype=float)
    width, n = band.shape
    if not (0 <= start_row < width):
        raise ValueError("start_row must fall within the band height")
    if extent <= 0:
        raise ValueError("extent must be positive")
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if window_size > n:
        raise ValueError("window_size cannot exceed the number of columns")

    if window_step is None:
        window_step = max(1, window_size // 2)
    if window_step <= 0:
        raise ValueError("window_step must be positive")

    end_row = min(start_row + extent, width)
    x_vals = np.arange(width, dtype=float)[start_row:end_row]

    final_start = n - window_size
    window_starts = list(range(0, n - window_size + 1, window_step))
    if window_starts and window_starts[-1] != final_start:
        window_starts.append(final_start)
    elif not window_starts:
        window_starts = [0]


    def model(x, a, b, c, d):
        safe_x = np.maximum(x, 1e-6)
        
        if log_y:
            t1 = np.log(a + 1e-15) + b * np.log(safe_x)
            with np.errstate(invalid='ignore', divide='ignore'):
                t2 = np.log(c + 1e-15) + d * x
            m = np.maximum(t1, t2)
            return m + np.log(np.exp(t1 - m) + np.exp(t2 - m))
        else:
            return a * np.power(safe_x, b) + c * np.exp(d * x)
     
    window_fits: list[np.ndarray] = []
    window_params: list[dict[str, float]] = []
    window_y_vals: list[np.ndarray] = []
    window_bounds: list[tuple[int, int]] = []

    for start in window_starts:
        stop = start + window_size
        row_means, _ = upper_triangular_row_means(
            band[:, start:stop], col_offset=start, empty_value=np.nan
        )
        y_vals = row_means[start_row:end_row]
        finite_mask = np.isfinite(y_vals)
        if finite_mask.sum() < 2:
            continue

        x_fit = x_vals[finite_mask]
        y_fit_target = y_vals[finite_mask]
        if log_y:
            y_fit_target=np.log(y_fit_target+1e-10)
        if not log_y:
            guess = (
                float(np.max(y_fit_target)) if y_fit_target.size else 1.0,
                -1.0,
                float(np.min(y_fit_target)) if y_fit_target.size else 0.1,
                -0.01,
            )
        else:
            guess = (
                float(np.max(np.exp(y_fit_target))) if y_fit_target.size else 1.0,
                -1.0,
                float(np.min(np.exp(y_fit_target))) if y_fit_target.size else 0.1,
                -0.01,
            )
        try:
            popt, _ = curve_fit(model, x_fit, y_fit_target, p0=guess, maxfev=10000)
        except (RuntimeError, ValueError):
            continue

        y_fit_full = model(x_vals, *popt)
        window_fits.append(y_fit_full)
        window_params.append({"a": float(popt[0]), "b": float(popt[1]), "c": float(popt[2]), "d": float(popt[3])})
        window_y_vals.append(y_vals)
        window_bounds.append((start, stop))

    if not window_fits:
        raise RuntimeError("No successful fits were produced; adjust window parameters.")

    stacked_fits = np.vstack(window_fits)
    median_fit = np.median(stacked_fits, axis=0)
    stacked_y = np.vstack(window_y_vals)
    median_y = np.nanmedian(stacked_y, axis=0)
    params_median = {
        "a": float(np.median([p["a"] for p in window_params])),
        "b": float(np.median([p["b"] for p in window_params])),
        "c": float(np.median([p["c"] for p in window_params])),
        "d": float(np.median([p["d"] for p in window_params])),
    }

    return {
        "params": params_median,
        "x": x_vals,
        "y": median_y,
        "y_fit": median_fit,
        "window_row_means": stacked_y,
        "window_params": window_params,
        "window_fits": stacked_fits,
        "window_indices": window_bounds,

    }

def upper_triangular_row_means(band_slice, *, col_offset: int = 0, empty_value: float = 0.0):
    """
    Compute the mean of each row in the upper triangle of a band matrix.

    Args:
        band_slice (np.ndarray): Input band matrix.
        col_offset (int): Column offset for mask.
        empty_value (float): Value to use for empty rows.

    Returns:
        tuple: (row_means, counts) for each row.
    """
    band_slice = np.asarray(band_slice, dtype=float)
    width, n = band_slice.shape
    row_idx = np.arange(width)[:, None]
    col_idx = (np.arange(n) + col_offset)[None, :]
    mask = col_idx >= row_idx
    masked = np.where(mask, band_slice, 0.0)
    sums = masked.sum(axis=1)
    counts = mask.sum(axis=1)
    row_means = np.divide(
        sums,
        counts,
        out=np.full(width, empty_value, dtype=float),
        where=counts > 0,
    )
    return row_means, counts

def generate_dlem_prediction(
    fit_result: dict,
    *,
    start: int,
    span: int,
    slowdown: float,
    rows: int,
    mode: str = "mse",
    offset: int = 0,
) -> tuple[np.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Generate a dLEM model prediction using learned parameters and jax_forward_generate.

    Args:
        fit_result (dict): Output from train_dlem with learned parameters.
        start (int): Start index for slicing parameters.
        span (int): Number of bins to predict.
        slowdown (float): Slowdown parameter for the model.
        rows (int): Number of rows for prediction.
        mode (str): Which parameter set to use ('mse', 'corr', 'final').
        offset (int): Offset for right parameter vector.

    Returns:
        tuple: (prediction array, left parameter slice, right parameter slice)
    """
    mode = mode.lower()
    key_map = {
        "mse": ("p_left_mse", "p_right_mse"),
        "corr": ("p_left_cor", "p_right_cor"),
        "final": ("p_left", "p_right"),
    }
    if mode not in key_map:
        raise ValueError(f"mode must be one of {tuple(key_map)}")
    left_key, right_key = key_map[mode]
    if left_key not in fit_result or right_key not in fit_result:
        missing = [k for k in (left_key, right_key) if k not in fit_result]
        raise KeyError(f"fit_result missing keys for mode '{mode}': {missing}")

    p_left = np.asarray(fit_result[left_key])
    p_right = np.asarray(fit_result[right_key])
    if start < 0 or span <= 0:
        raise ValueError("start must be non-negative and span must be positive.")
    end = start + span
    right_start = start + offset
    right_end = right_start + span
    if end > len(p_left) or right_end > len(p_right):
        raise ValueError("Requested slice exceeds parameter vector lengths.")

    left_slice = jnp.asarray(p_left[start:end], dtype=jnp.float32)
    right_slice = jnp.asarray(p_right[right_start:right_end], dtype=jnp.float32)
    pred = jax_forward_generate(left_slice, right_slice, slowdown=slowdown, rows=rows)

    return pred, left_slice, right_slice

