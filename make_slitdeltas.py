#!/usr/bin/env python3
"""
Analyze FITS files to find and track peak positions across rows.

This script identifies spectral peaks in each row of a FITS image, fits Gaussians
to determine precise peak positions, and calculates how peaks shift between rows.
The median offsets are saved for use in spectral extraction algorithms.
"""

import os
from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scipy.optimize import curve_fit
from scipy.signal import find_peaks


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class PeakFindingConfig:
    """Configuration parameters for peak finding and fitting."""

    # Peak detection parameters
    height_multiplier: float = 1.5  # Minimum peak height as multiple of mean
    min_peak_distance: int = 10  # Minimum distance between peaks (pixels)

    # Gaussian fitting parameters
    fit_window_size: int = 15  # Window size around peak for fitting (pixels)
    initial_sigma: float = 5.0  # Initial guess for Gaussian sigma (pixels)

    # Directory configuration
    data_dir: str = "data"
    plots_dir: str = "plots"


DEFAULT_CONFIG = PeakFindingConfig()


# =============================================================================
# Core Functions
# =============================================================================


def gaussian(
    x: np.ndarray, amplitude: float, mean: float, sigma: float, offset: float
) -> np.ndarray:
    """Gaussian function with vertical offset."""
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * sigma**2)) + offset


def fit_gaussian_to_peak(
    x_vals: np.ndarray,
    row_data: np.ndarray,
    peak_loc: int,
    window_size: int,
    initial_sigma: float,
) -> tuple[float, dict]:
    """
    Fit a Gaussian to a peak and return the fitted position.

    Args:
        x_vals: X-axis values (pixel indices)
        row_data: Intensity values for the row
        peak_loc: Initial peak location (pixel index)
        window_size: Window size around peak for fitting
        initial_sigma: Initial guess for Gaussian sigma

    Returns:
        Tuple of (fitted_position, fit_parameters_dict)
    """
    num_cols = len(row_data)

    # Define fitting window
    left_bound = max(0, peak_loc - window_size)
    right_bound = min(num_cols, peak_loc + window_size + 1)

    x_window = x_vals[left_bound:right_bound]
    y_window = row_data[left_bound:right_bound]

    try:
        # Initial parameter guess [amplitude, mean, sigma, offset]
        p0 = [
            row_data[peak_loc] - np.min(y_window),  # amplitude
            peak_loc,  # mean
            initial_sigma,  # sigma
            np.min(y_window),  # offset
        ]

        # Fit the Gaussian
        popt, _ = curve_fit(gaussian, x_window, y_window, p0=p0)

        fit_params = {
            "amplitude": popt[0],
            "position": popt[1],
            "sigma": popt[2],
            "offset": popt[3],
            "fit_failed": False,
        }

        return popt[1], fit_params

    except (RuntimeError, ValueError):
        # If fitting fails, use the original peak location
        fit_params = {
            "amplitude": row_data[peak_loc] - np.min(y_window),
            "position": float(peak_loc),
            "sigma": initial_sigma,
            "offset": np.min(y_window),
            "fit_failed": True,
        }

        return float(peak_loc), fit_params


def find_and_fit_peaks(
    data: np.ndarray, config: PeakFindingConfig = DEFAULT_CONFIG
) -> tuple:
    """
    Process FITS data row by row to find and fit peaks.

    1. Identify peaks in each row
    2. Fit a Gaussian to each peak to get the precise position
    3. Calculate absolute offsets from the median position of each peak

    Args:
        data: 2D numpy array (rows x columns)
        config: Configuration for peak finding

    Returns:
        Tuple of (peak_positions, median_offsets, all_peak_fits)
        - peak_positions: List of arrays containing fitted peak positions for each row
        - median_offsets: Array of absolute offsets from median for each row
        - all_peak_fits: List of lists containing fit parameters for each peak
    """
    num_rows, num_cols = data.shape
    x_vals = np.arange(num_cols)

    # Store peak positions for each row
    peak_positions = []
    all_peak_fits = []

    # Find peaks in each row
    for row_idx in range(num_rows):
        row_data = data[row_idx]

        # Find peaks
        peaks, _ = find_peaks(
            row_data,
            height=np.mean(row_data) * config.height_multiplier,
            distance=config.min_peak_distance,
        )

        if len(peaks) == 0:
            peak_positions.append(np.array([]))
            all_peak_fits.append([])
            continue

        # Fit Gaussian to each peak
        fitted_positions = []
        row_fits = []

        for peak_idx, peak_loc in enumerate(peaks):
            fitted_pos, fit_params = fit_gaussian_to_peak(
                x_vals, row_data, peak_loc, config.fit_window_size, config.initial_sigma
            )

            fitted_positions.append(fitted_pos)
            fit_params.update({"peak_idx": peak_idx, "row": row_idx})
            row_fits.append(fit_params)

        # Sort peaks by position to maintain consistent order across rows
        sort_idx = np.argsort(fitted_positions)
        fitted_positions = np.array(fitted_positions)[sort_idx]
        row_fits = [row_fits[i] for i in sort_idx]

        peak_positions.append(fitted_positions)
        all_peak_fits.append(row_fits)

    # Calculate median offsets
    median_offsets = calculate_median_offsets(peak_positions, num_rows)

    return peak_positions, median_offsets, all_peak_fits


def calculate_median_offsets(peak_positions: list, num_rows: int) -> np.ndarray:
    """
    Calculate absolute offsets from median for each peak across all rows.

    Args:
        peak_positions: List of arrays containing peak positions for each row
        num_rows: Total number of rows

    Returns:
        Array of median offsets for each row
    """
    median_offsets = np.zeros(num_rows)

    # Find the most common number of peaks across rows
    peak_counts = [len(pos) for pos in peak_positions if len(pos) > 0]
    if not peak_counts:
        return median_offsets

    most_common_peak_count = max(set(peak_counts), key=peak_counts.count)

    # Use only rows with the most common number of peaks
    valid_rows = [
        i for i, pos in enumerate(peak_positions) if len(pos) == most_common_peak_count
    ]

    if len(valid_rows) < 2:
        print("Warning: Not enough valid rows with consistent peak counts")
        return median_offsets

    # Organize peak positions by peak index across all rows
    peak_positions_by_index = [[] for _ in range(most_common_peak_count)]
    row_indices_by_peak = [[] for _ in range(most_common_peak_count)]

    for row_idx in valid_rows:
        for peak_idx in range(most_common_peak_count):
            if peak_idx < len(peak_positions[row_idx]):
                peak_positions_by_index[peak_idx].append(
                    peak_positions[row_idx][peak_idx]
                )
                row_indices_by_peak[peak_idx].append(row_idx)

    # Calculate median position for each peak
    peak_medians = [
        np.median(positions) if positions else None
        for positions in peak_positions_by_index
    ]

    # Calculate offsets from median for each peak in each row
    for peak_idx in range(most_common_peak_count):
        if peak_medians[peak_idx] is not None:
            for i, row_idx in enumerate(row_indices_by_peak[peak_idx]):
                offset = peak_positions_by_index[peak_idx][i] - peak_medians[peak_idx]
                if median_offsets[row_idx] == 0:  # Only set if not already set
                    median_offsets[row_idx] = offset
                else:  # Average with existing offset
                    median_offsets[row_idx] = (median_offsets[row_idx] + offset) / 2

    # Interpolate missing offsets
    median_offsets = interpolate_missing_offsets(median_offsets)

    return median_offsets


def interpolate_missing_offsets(median_offsets: np.ndarray) -> np.ndarray:
    """
    Fill in missing (zero) offsets using linear interpolation.

    Args:
        median_offsets: Array of offsets (zeros indicate missing values)

    Returns:
        Array with interpolated values filled in
    """
    num_rows = len(median_offsets)
    valid_indices = np.where(median_offsets != 0)[0]

    if len(valid_indices) == 0:
        return median_offsets

    for i in range(num_rows):
        if median_offsets[i] == 0:
            # Find nearest valid indices
            left_indices = valid_indices[valid_indices < i]
            right_indices = valid_indices[valid_indices > i]

            if len(left_indices) > 0 and len(right_indices) > 0:
                # Linear interpolation
                left_idx = left_indices[-1]
                right_idx = right_indices[0]
                left_val = median_offsets[left_idx]
                right_val = median_offsets[right_idx]
                weight = (i - left_idx) / (right_idx - left_idx)
                median_offsets[i] = left_val + weight * (right_val - left_val)
            elif len(left_indices) > 0:
                # Use left value
                median_offsets[i] = median_offsets[left_indices[-1]]
            elif len(right_indices) > 0:
                # Use right value
                median_offsets[i] = median_offsets[right_indices[0]]

    return median_offsets


# =============================================================================
# File Processing
# =============================================================================


def process_fits_file(
    filename: str, config: PeakFindingConfig = DEFAULT_CONFIG
) -> Optional[dict]:
    """
    Process a FITS file and return peak finding results.

    Args:
        filename: Path to FITS file
        config: Configuration for peak finding

    Returns:
        Dictionary containing analysis results, or None if processing fails
    """
    print(f"Processing {filename}...")
    try:
        with fits.open(filename) as hdul:
            data = hdul[0].data

        # Find and fit peaks
        peak_positions, median_offsets, all_peak_fits = find_and_fit_peaks(data, config)

        # Calculate statistics
        avg_offset = np.mean(median_offsets)
        std_offset = np.std(median_offsets)

        print(
            f"  Average offset from median: {avg_offset:.4f} ± {std_offset:.4f} pixels"
        )

        return {
            "filename": filename,
            "peak_positions": peak_positions,
            "median_offsets": median_offsets,
            "all_peak_fits": all_peak_fits,
            "avg_offset": avg_offset,
            "std_offset": std_offset,
        }
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return None


# =============================================================================
# Plotting
# =============================================================================


def plot_results(
    results: list[dict], config: PeakFindingConfig = DEFAULT_CONFIG
) -> None:
    """
    Generate plots for the analysis results.

    Args:
        results: List of result dictionaries from process_fits_file
        config: Configuration (for output directory)
    """
    os.makedirs(config.plots_dir, exist_ok=True)

    for result in results:
        if not result:
            continue

        filename = result["filename"]
        basename = os.path.basename(filename).replace(".fits", "")
        median_offsets = result["median_offsets"]
        peak_positions = result["peak_positions"]

        # Plot 1: Absolute offsets from median
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(len(median_offsets)), median_offsets, "o-")
        plt.xlabel("Row Index")
        plt.ylabel("Offset from Median (pixels)")
        plt.title(f"Absolute Offsets from Median - {basename}")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(config.plots_dir, f"{basename}_offsets.png"))

        # Plot 2: Peak positions across rows
        plt.figure(figsize=(10, 6))

        # Find rows with peaks
        rows_with_peaks = [i for i, pos in enumerate(peak_positions) if len(pos) > 0]
        max_peaks = max([len(pos) for pos in peak_positions], default=0)

        if max_peaks > 0:
            for peak_idx in range(max_peaks):
                positions = []
                rows = []

                for row_idx in rows_with_peaks:
                    if peak_idx < len(peak_positions[row_idx]):
                        positions.append(peak_positions[row_idx][peak_idx])
                        rows.append(row_idx)

                if positions:
                    plt.plot(rows, positions, "o-", label=f"Peak {peak_idx+1}")

            plt.xlabel("Row Index")
            plt.ylabel("Peak Position (pixel)")
            plt.title(f"Peak Positions Across Rows - {basename}")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(
                os.path.join(config.plots_dir, f"{basename}_peak_positions.png")
            )

        plt.close("all")


def save_results(
    results: list[dict], config: PeakFindingConfig = DEFAULT_CONFIG
) -> list[str]:
    """
    Save numerical results to NPZ files.

    Args:
        results: List of result dictionaries
        config: Configuration (for output directory)

    Returns:
        List of saved file paths
    """
    saved_files = []

    for result in results:
        if not result:
            continue

        # Create filename based on original FITS file
        basename = os.path.splitext(os.path.basename(result["filename"]))[0]
        output_file = os.path.join(config.data_dir, f"slitdeltas_{basename}.npz")

        # Save individual dataset
        np.savez(
            output_file,
            filename=result["filename"],
            avg_offset=result["avg_offset"],
            std_offset=result["std_offset"],
            median_offsets=result["median_offsets"],
        )
        saved_files.append(output_file)

    return saved_files


# =============================================================================
# Main
# =============================================================================


def main():
    """Main function to process all test data files."""
    config = DEFAULT_CONFIG

    # Check for data directory
    if not os.path.exists(config.data_dir):
        print(f"Error: {config.data_dir}/ directory not found!")
        return

    # Find all FITS files in data directory
    fits_files = [
        os.path.join(config.data_dir, f)
        for f in os.listdir(config.data_dir)
        if f.endswith(".fits") and f.startswith("test_data")
    ]

    if not fits_files:
        print(f"No test data FITS files found in {config.data_dir}/!")
        return

    print(f"Found {len(fits_files)} FITS files to process\n")

    # Process each file
    results = []
    for fits_file in fits_files:
        result = process_fits_file(fits_file, config)
        if result:
            results.append(result)

    if not results:
        print("No files were successfully processed!")
        return

    # Generate plots
    print("\nGenerating plots...")
    plot_results(results, config)

    # Save numerical results
    print("\nSaving results...")
    saved_files = save_results(results, config)

    print(f"\nResults saved to {len(saved_files)} NPZ files:")
    for file in saved_files:
        print(f"  - {file}")

    print(f"\nPlots saved to {config.plots_dir}/")


if __name__ == "__main__":
    main()
