import os
from pathlib import Path
import pickle
import numpy as np
import typer
from typing import Optional, List
from rich import print
import re
import pyranges as pr 
import pandas as pd
from enum import Enum
import hictkpy as htk
import cooler

from dlem.api import (band_generate_pixels, 
                      fetch_band, 
                      fit_band_row_profile_sliding, 
                      train_dlem, 
                      generate_dlem_prediction,
                      plot_map,
                      compute_contact_heatmap,
                      norm_band,
                      set_device,
                      )


# CLI utils
def comma_list_parser(value: str) -> List[str]:
    return [item.strip() for item in value.split(",")]

class PredMode(str, Enum):
    mse = "mse"
    corr = "corr"
    none = "final"

class EarlyStop(str, Enum):
    mse = "mse"
    corr = "corr"
    none = "none"

class CalcDevice(str, Enum):
    gpu = "gpu"
    cpu = "cpu"
    tpu = "tpu"

def region_tile(region, resolution):
    in_region_gr = pr.PyRanges(pd.DataFrame([region]))
    return(in_region_gr.tile(resolution))

dlem_cli = typer.Typer(
    help="""Differentiable Loop Extrusion Model (dLEM) is a tool to learn the parameters
    of loop extrusion from Hi-C or Micro-C data.
    dLEM is implemented in Jax and can be run on CPU or GPU.
    dLEM learns the speed of Left and Right leg of Cohesin extrusion at each genomic bin or pixel 
    dLEM can be used to predict contact maps from extrusion parameters, 
    harmonize chromatin contact, and generate tracks of chromatin extrusion speed""",
    context_settings={"help_option_names": ["-h", "--help"]}
)


@dlem_cli.command()
def main(
    device: CalcDevice = typer.Option(
        CalcDevice.cpu, 
        "--device", 
        help="Select computation device.",
    ),
    region: Optional[str] = typer.Option(
        None,
        "--region",
        "-r",
        help="""
            Print details about a specific target chromosomal region 
            (ex: chr16:51906425-55560999), this includes plotting and 
            track output. By default dLEM learns the whole chromosome 
            (ex: chr16) the target region lies in.
        """,
    ),
    bed_region_file: Optional[str] = typer.Option(
        None,
        "--bed",
        "-b",
        help="""
            Define target regions in BED file to plot and output tracks for,
            Plot and output tsv for all regions in BED file,
            3 columns [chromosome start end], ex: [chr21 43789210 43789690].
            """,
    ),
    chromosomes: Optional[str] = typer.Option(
        None,
        "--chromosomes",
        parser=comma_list_parser,
        help="Comma-separated list of chromosomes to train dLEM on (e.g., 'chr1,chr2,chr3').",
    ),
    all: bool = typer.Option(
        False, 
        "--all", 
        help="Run dLEM across all chromosomes in input chromatin loop file.",
    ),
    resolution: Optional[int] = typer.Option(
        None,
        "--resolution",
        "-l",
        help="Resolution to pull .cool file from .mcool file, must exist.",
    ),
    start_diag: Optional[int] = typer.Option(
        5,
        "--start-diag",
        help="Diagonal offset where updates start during training.",
    ),
    train_rows: Optional[int] = typer.Option(
        170,
        "--train-rows",
        help="Number of diagonals to use during training dLEM.",
    ),
    steps: Optional[int] = typer.Option(
        10,
        "--steps",
        help="Number of steps for training.",
    ),
    window_size: Optional[int] = typer.Option(
        3000,
        "--window-size",
        "-w",
        help="Column window size for slowdown fitting",
    ),
    window_step: Optional[int] = typer.Option(
        200,
        "--window-step",
        "-s",
        help="Step between slowdown-fitting windows.",
    ), 
    decay_extent: Optional[int] = typer.Option(
        500,
        "--decay-extent",
        "-d",
        help="Number of band rows to use when fitting slowdown.",
    ),   
    iterations: Optional[int] = typer.Option(
        300,
        "--iterations",
        "-i",
        help="Number of iterations to fit the model over.",
    ),
    learning_rate: Optional[float] = typer.Option(
        1e-2,
        "--lr",
        "-lr",
        help="Change speed of parameter tuning, useful if loss is too fast/slow or model not converging.",
    ),
    prediction_mode: PredMode = typer.Option(
        PredMode.mse,
        "--prediction-mode",
        "-pm",
        help="Which parameter set to use for predictions.",
    ),
    prediction_rows: Optional[int] = typer.Option(
        700,
        "--prediction-rows",
        "-pr",
        help="Number of diagonals to predict when outputting dLEM predictions",
    ),
    full_prediction: bool = typer.Option(
        False,
        "--full-output",
        help="Predict all diagonals in output instead of a set number",
    ),
    early_stop_metric: EarlyStop = typer.Option(
        EarlyStop.mse,
        "--early-stop-metric",
        "-esm",
        help="Metric for early stopping",
    ),
    debug: bool = typer.Option(
        False, 
        "--debug", 
        help="Verbose output during training",
    ),
    output_cool: bool = typer.Option(
        False,
        "--output-cool",
        "-c",
        help="Output the prediction as a .cool file",
    ),
    tracks: bool = typer.Option(
        False,
        "--output-tracks",
        "-t",
        help="Output the parameters as genomic tracks for target regions in BED and whole genome in BigWig format",
    ),
    row_normalization: bool = typer.Option(
        False,
        "--norm",
        help="Normalize the predicted band using geometric mean",
    ),
    plot: bool = typer.Option(
        False, 
        "--plot", 
        "-p",
        help="Plot fits to compare to input",
    ),
    dna_ints: Optional[Path] = typer.Argument(
        ...,
        help="Chromatin input dataset in the form of a .cool file",
    ),
    output_location: Optional[Path] = typer.Argument(
        ...,
        help="Location of output directory, will be created if not existing",
    ),
):
    """
   dLEM: differentiable Loop Extrusion Model

    """
    print(f"Running dLEM on {dna_ints} and outputting to {output_location}")

    device = set_device(device.value)
    print(f"Using device: {device}")
    
    os.makedirs(output_location, exist_ok=True)
    dna_ints = os.path.abspath(dna_ints)

    if htk.is_mcool_file(dna_ints) and resolution:
        cool_filename = f"{dna_ints}::/resolutions/{resolution}"
        with htk.File(dna_ints, resolution) as cool_handle:
            chrom_sizes = cool_handle.chromosomes()
            cool_norms = cool_handle.avail_normalizations()
            bins = cool_handle.bins()
            resolution = cool_handle.resolution()
        if output_cool:
            pred_cool_filename = os.path.basename(dna_ints).replace('.mcool', '_dlem_pred.cool')
            pred_cool_filename = os.path.join(os.path.abspath(output_location), pred_cool_filename)
            if os.path.exists(pred_cool_filename):
                os.remove(pred_cool_filename)
    elif htk.is_cooler(dna_ints):
        cool_filename = dna_ints
        with htk.File(dna_ints) as cool_handle:
            chrom_sizes = cool_handle.chromosomes()
            cool_norms = cool_handle.avail_normalizations()
            bins = cool_handle.bins()
            resolution = cool_handle.resolution()
        if output_cool:
            pred_cool_filename = os.path.basename(dna_ints).replace('.cool', '_dlem_pred.cool')
            pred_cool_filename = os.path.join(os.path.abspath(output_location), pred_cool_filename)
            if os.path.exists(pred_cool_filename):
                os.remove(pred_cool_filename)
    elif not os.path.isfile(dna_ints):
        print(f"Input chromatin looping file {dna_ints} not found!")
        raise typer.Exit(code=1)
    else:
        print("""
              Check chromatin looping file.
              Chromatin looping input must be in .cool or .mcool format. 
              If mcool you must provide a resolution using --resolution that is present in the mcool
              """)
        raise typer.Exit(code=1)  
    

    if "weight" not in cool_norms:
        print("Please balance the input data using cooler, ex: hictk balance ice input_data.cool")
        raise typer.Exit(code=1)  
    
    target_regions_gr = None
    in_region_gr = None
    if region:
        split_region = re.split(r"[-:]", region)
        chroms_to_use = [chrom for chrom in chrom_sizes.keys() if chrom == split_region[0]]
        target_regions_gr = pr.PyRanges(
            chromosomes=[split_region[0]],
            starts=[int(split_region[1])],
            ends=[int(split_region[2])]
        )
        in_region_gr = pr.PyRanges(
            chromosomes=chroms_to_use,
            starts=[0] * len(chroms_to_use),
            ends=[chrom_sizes[chrom] for chrom in chroms_to_use]
        )
    elif bed_region_file:
        target_regions_gr = pr.read_bed(bed_region_file)
        chroms_to_use = target_regions_gr.chromosomes
        in_region_gr = pr.PyRanges(
            chromosomes=chroms_to_use,
            starts=[0] * len(chroms_to_use),
            ends=[chrom_sizes[chrom] for chrom in chroms_to_use]
        )
    if chromosomes:
        chroms_to_use = [chrom for chrom in chromosomes if chrom in chrom_sizes]
        if len(chroms_to_use) == 0:
            print("None of the specified chromosomes were found in the cool file.")
            raise typer.Exit(code=1)
        in_region_gr = pr.PyRanges(
            chromosomes=chroms_to_use,
            starts=[0] * len(chroms_to_use),
            ends=[chrom_sizes[chrom] for chrom in chroms_to_use]
        )
    elif all:
        chroms_to_use = [key for key, value in chrom_sizes.items() if value > 1000000]
        in_region_gr = pr.PyRanges(
            chromosomes=chroms_to_use,
            starts=[0] * len(chroms_to_use),
            ends=[chrom_sizes[chrom] for chrom in chroms_to_use]
        )
    if in_region_gr is None:
        print("Please specify chromosomes to train dLEM on using --region, --bed, --chromosomes, or --all")
        raise typer.Exit(code=1)
    if target_regions_gr is not None:
        target_regions_gr = pr.genomicfeatures.genome_bounds(target_regions_gr, chrom_sizes, clip=True)
    print(f"Running dLEM on chromosomes: {', '.join(in_region_gr.chromosomes)}")      

    output_location = os.path.abspath(output_location)
    if not os.path.exists(output_location):
        os.makedirs(output_location)
    if tracks:
        out_left_bw = os.path.join(output_location,'pred_left.bw')
        out_right_bw = os.path.join(output_location,'pred_right.bw')

    out_data = []

    print("Gathering input data")
    band_width = train_rows + steps + 1
    for cur_chr in in_region_gr.chromosomes:
        cur_region = in_region_gr[cur_chr].as_df().iloc[0]
        region_name = f"{cur_region.Chromosome}:{cur_region.Start}-{cur_region.End}"
        cur_data = fetch_band(cool_filename,
                              resolution=resolution,
                              cur_region=region_name,
                              width=band_width)
        out_data.append(cur_data)

# Fit slowdown parameters
    print("Fitting slowdown parameters")
    batched_detach_result = []
    for cur_data in out_data:
        available_rows = cur_data.shape[0] - start_diag
        extent = min(decay_extent, max(available_rows, 1))
        window_size = min(window_size, cur_data.shape[1])

        batched_detach_result.append(
            fit_band_row_profile_sliding(cur_data,
                                         start_row=start_diag,
                                         extent=extent,
                                         window_size=window_size,
                                         window_step=window_step,
                                         log_y=True)
                                         )
    
    d_values = np.asarray([p["params"]["d"] for p in batched_detach_result], dtype=float)
    finite = d_values[np.isfinite(d_values)]
    if finite.size == 0:
        raise RuntimeError("Unable to infer slowdown: no valid decay parameters found.")
    slowdown = -float(np.median(finite))

    print("Training dLEM across selected chromosomes")
    batched_result = []
    print(cur_data.shape)
    for cur_data in out_data: 
        batched_result.append(
            train_dlem(cur_data[0:train_rows, :], 
                                            steps=steps,
                                            start_row=start_diag,
                                            slowdown=slowdown,
                                            learning_rate=learning_rate,
                                            train_steps=iterations,
                                            verbose=debug,
                                            auto_stop_metric=early_stop_metric
                                            )
        )
    
    out_pickle = os.path.join(output_location,
                            'dLEM_results.pkl')
    with open(out_pickle, 'wb') as file:
        pickle.dump(batched_result, file)
    print("Saved dLEM results to pickle")

    print("Generating dLEM predictions for selected chromosomes")
    out_gr_l = []
    out_gr_r = []
    out_figs = []
    out_names = []
    pred_chroms = {}
    pred_ls = {}
    pred_rs = {}

    for out_chrom in chrom_sizes.keys():
        chr_start = 0
        chr_end = chrom_sizes[out_chrom]
        region_name = f"{out_chrom}:{chr_start}-{chr_end}"

        if out_chrom in in_region_gr.chromosomes:
            print(f"Generating {out_chrom} predictions")
            ix_chr = in_region_gr.chromosomes.index(out_chrom)
            cur_model_res = batched_result[ix_chr]
            cur_goal = out_data[ix_chr]
            rows, span = cur_goal.shape
            if full_prediction:
                prediction_rows = span
            cur_pred, cur_l, cur_r = generate_dlem_prediction(cur_model_res, 
                                                                start=0,
                                                                span=span, 
                                                                mode=prediction_mode,
                                                                slowdown=slowdown, 
                                                                rows=prediction_rows,)
            
            if row_normalization:
                cur_pred = norm_band(cur_pred, cur_goal)
            pred_chroms[out_chrom] = cur_pred
            pred_ls[out_chrom] = cur_l
            pred_rs[out_chrom] = cur_r
        else:
            pred_chroms[out_chrom] = None

    print("Plotting and generating tracks for dLEM selected target regions")

    for out_chrom in in_region_gr.chromosomes:
            chr_start = 0
            chr_end = chrom_sizes[out_chrom]
            region_name = f"{out_chrom}:{chr_start}-{chr_end}"
            ix_chr = in_region_gr.chromosomes.index(out_chrom)
            cur_model_res = batched_result[ix_chr]
            cur_goal = out_data[ix_chr]
            rows, span = cur_goal.shape
            cur_pred = pred_chroms[out_chrom]
            cur_l = pred_ls[out_chrom]
            cur_r = pred_rs[out_chrom]

            if tracks:
                chr_gr = pr.PyRanges(
                    chromosomes=[out_chrom],
                    starts=[chr_start],
                    ends=[chr_end]
                )

                bw_res = 100
                cur_l_gr = chr_gr.tile(resolution)
                cur_l_gr.Score = cur_l
                cur_l_gr = cur_l_gr.tile(bw_res)
                cur_l_gr.End = cur_l_gr.End-1
                out_gr_l.append(cur_l_gr)

                cur_r_gr = chr_gr.tile(resolution)
                cur_r_gr.Score = cur_r
                cur_r_gr = cur_r_gr.tile(bw_res)
                cur_r_gr.End = cur_r_gr.End-1
                out_gr_r.append(cur_r_gr)
            
            if target_regions_gr:
                plot_regions_df = target_regions_gr[out_chrom].as_df()
                for plot_ix in range(len(plot_regions_df)):
                    plot_region = plot_regions_df.iloc[plot_ix]
                    plot_gr = pr.PyRanges(
                        chromosomes=[plot_region.Chromosome],
                        starts=[plot_region.Start],
                        ends=[plot_region.End]
                    )
                    plot_start_offset = int(np.floor(plot_region.Start / resolution))
                    plot_end_offset = int(np.ceil(plot_region.End / resolution))
                    plot_region_name = f"{plot_region.Chromosome}_{plot_region.Start}_{plot_region.End}"
                    plot_region_chr_name = f"{plot_region.Chromosome}:{plot_region.Start}-{plot_region.End}"
                    cur_l_plot = cur_l[plot_start_offset:plot_end_offset]
                    cur_r_plot = cur_r[plot_start_offset:plot_end_offset]
                    if tracks:
                        out_left_csv = os.path.join(output_location,f'{plot_region_name}_pred_left.bed')
                        plot_l_gr = plot_gr.tile(resolution)
                        plot_l_gr.Score = cur_l_plot
                        plot_l_gr.to_csv(out_left_csv, sep="\t", header=False)

                        out_right_csv = os.path.join(output_location,f'{plot_region_name}_pred_right.bed')
                        plot_r_gr = plot_gr.tile(resolution)
                        plot_r_gr.Score = cur_r_plot
                        plot_r_gr.to_csv(out_right_csv, sep="\t", header=False)

                    if plot:
                        print(f"Plotting predictions as HTML and PNG for {plot_region_name}")
                        band_height = min(plot_end_offset - plot_start_offset, 
                                          cur_pred.shape[0],
                                          cur_goal.shape[0])
                        cur_pred_plot = cur_pred[0:band_height,
                                                 plot_start_offset:plot_end_offset]
                        cur_goal_plot = cur_goal[0:band_height,
                                                 plot_start_offset:plot_end_offset]
                        heatmap = compute_contact_heatmap(
                            cur_pred_plot,
                            lower_vals=cur_goal_plot,
                            normalize=True,
                            log=True,
                            match_lower_range=True
                        )
                        
                        out_name = os.path.join(output_location,
                                                f'in_vs_pred_{plot_region_name}.png')
                        out_html = os.path.join(output_location,
                                                f'in_vs_pred_{plot_region_name}.html')
                        fig = plot_map(heatmap,
                                        cur_l_plot, 
                                        cur_r_plot, 
                                        plot_region_chr_name, 
                                        resolution)
                        fig.write_html(out_html)
                        fig.update_layout(
                            width=1400,
                            height=700,
                        )
                        out_names.append(out_name)
                        out_figs.append(fig)

    if output_cool:
        print(f"Outputting dLEM predicted contact map to {pred_cool_filename}")
        # Old Cooler write version
        bins = cooler.binnify(pd.Series(chrom_sizes), resolution)
        bins["weight"] = 1.0
        custom_dtypes = {'count': np.float32 }
        cooler.create_cooler(os.path.abspath(pred_cool_filename),
                             bins=bins,
                             pixels=[
                                 band_generate_pixels(band=pred_chroms[out_chrom], 
                                                      ref_cool_filename=cool_filename, 
                                                      resolution=resolution,
                                                      chrom_name=out_chrom
                                                      ) for out_chrom in chrom_sizes.keys()],
                             dtypes=custom_dtypes,
                             metadata={},
                             ordered=True,
                             symmetric_upper=True,
                             triucheck=False,
                             dupcheck=False)
        
        # with htk.cooler.FileWriter(pred_cool_filename,
        #                            bins=bins
        #                            ) as pred_cool_filehandle:
        #     for out_chrom in chrom_sizes.keys():
        #         pred_cool_filehandle.add_pixels(sorted = False,
        #                                         pixels = band_generate_pixels(
        #                                             band=pred_chroms[out_chrom],
        #                                             ref_cool_filename=cool_filename,
        #                                             chrom_name=out_chrom,
        #                                             ))
                
    if tracks:
        all_gr_l = pr.concat(out_gr_l)
        all_gr_r = pr.concat(out_gr_r)
        all_gr_l.to_bigwig(out_left_bw, 
                           chrom_sizes,
                           rpm=False,
                           value_col="Score")
        all_gr_r.to_bigwig(out_right_bw, 
                           chrom_sizes,
                           rpm=False,
                           value_col="Score")

    print(f"dLEM analysis complete! Outputs written to {output_location}")