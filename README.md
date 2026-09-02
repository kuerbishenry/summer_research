# Summer Research Write-Up

**Henry Kuerbis**
9/1/2026

## summer_research Repo Overview

This document contains descriptions and documentation of the codebase and dataset I have been working with and the pipelines, results, and figures I created during my exploration of the effect of noise in Hi-C data on dLEM performance.

## Folders

### Data

- `GSE263229_GlambdaLmerged.mcool` – Hi-C chromatin conformation capture dataset from a *Drosophila* embryo from https://www.ncbi.nlm.nih.gov/geo/
- `generated_band_denormalized` – processed dLEM prediction used to measure baseline performance
- `normalizaed_generated_band` – dLEM-generated contact map built from raw data, chromosome 3R at 1600 bp resolution

### Dlem

Dlem codebase from: https://github.com/chikinalab/dLEM

Added a few extra lines to results dictionary in `api.py` for additional variable tracking during training

### Pipeline

**`Interation_pipeline_reverse_normalization`**

Inputs: `.mcool` file, `noise_levels`, iterations per noise level, rng seed, plus all other dLEM inputs

- Takes raw hi-c data, fits the slowdown parameter
- Takes a synthetic dataset, introduces multiplicative noise
  - `noise = rng.uniform(-1, 1, size=band.shape) * noise_level * band`
- Trains dLEM
- Denormalizes predicted band
- Plots MSE between input data and dLEM predicted contact matrix at different noise levels
- Plots MSE between cohesin movement parameters found from input data (noise-free) and parameters found from noisy data
- Plots change in correlation between input and predicted contact matrix at different noise levels

### Results_figures

- Pkl files contain record dictionary from each (noise_level, iteration) pair

```python
  record = {
      "left_param":  np.asarray(model_result["p_left"]),
      "right_param": np.asarray(model_result["p_right"]),
      "band_mse":    mse,
      "band_corr":   corr,
      "noise_level": noise_level,
      "iteration":   iteration,
      "seed":        seed,
  }
```

  - `best_corr` and `best_mse` are training metrics and calculated on normalized scale
  - `band_corr` and `band_mse` are calculated in the pipeline and on denormalized scale
- PNG naming: `MSE measurement_noise range_iterations_other run adjustments`
- PDF files contain flowcharts outlining dLEM steps and draft pipeline

## Findings

The addition of noise gradually decreased model performance; at no level tested did the model performance significantly drop off. Noise was added proportionally to the value of the pixels and was applied to every pixel, which on a large scale doesn't seem to affect the structure of the data much. Correlation fell from 0.892 to 0.752 across the full range, and MSE rose from 0.525 to 1.639.

## Future directions

- Measure the robustness of cohesin parameters by introducing noise and comparing predicted contact matrices
- Test on datasets from other organisms
- Other methods of noise introduction, more significant perturbations
