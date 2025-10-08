#!/usr/bin/env python3
"""
Generate synthetic test data for CharSlit extraction testing.

This script creates various FITS files with Gaussian peaks that have different
characteristics (shifted, unshifted, discontinuous shifts, multislope, etc.)
to test the spectral extraction algorithm under different conditions.
"""

import os
from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits


# =============================================================================
# Configuration and Constants
# =============================================================================


@dataclass
class TestDataConfig:
    """Configuration parameters for test data generation."""

    # Image dimensions
    rows: int = 100
    cols: int = 150

    # Peak parameters
    fwhm: float = 10.0  # Full Width at Half Maximum
    peak1_center: float = 50.0
    peak2_center: float = 120.0
    peak_amplitude: float = 9.99e3

    # Background and noise
    background: float = 5.0
    random_seed: int = 42

    # Shift parameters (for linear shift variant)
    shift_start: float = -5.0
    shift_end: float = 5.0

    # Cosmic ray parameters
    cosmic_ray_count: int = 0

    @property
    def sigma(self) -> float:
        """Convert FWHM to Gaussian sigma."""
        return self.fwhm / 2.355


# Default configuration
DEFAULT_CONFIG = TestDataConfig()


# =============================================================================
# Shift Calculation Functions
# =============================================================================


def calculate_linear_shift(
    row: int, total_rows: int, shift_start: float, shift_end: float
) -> float:
    """Calculate linear shift from start to end across all rows."""
    return shift_start + (row / (total_rows - 1)) * (shift_end - shift_start)


def calculate_discontinuous_shift(row: int, base_shift: float) -> float:
    """
    Apply discontinuous shifts at specific row boundaries.

    Rows 0-29: base_shift
    Rows 30-59: base_shift - 4.0
    Rows 60-99: base_shift - 8.0
    """
    if row >= 60:
        return base_shift - 8.0
    elif row >= 30:
        return base_shift - 4.0
    else:
        return base_shift


def create_continuous_multislope_shifts(
    rows: int,
    slopes: list[float] = [0.05, -0.15, 0.11],
    row_ranges: list[tuple[int, int]] = [(0, 29), (30, 59), (60, 99)],
) -> np.ndarray:
    """
    Generate continuous multi-slope shifts across different row segments.

    Each segment has its own slope, but transitions are continuous (no jumps).

    Args:
        rows: Total number of rows
        slopes: List of slopes for each segment
        row_ranges: List of (start_row, end_row) tuples for each segment

    Returns:
        Array of delta values, one per row
    """
    all_deltas = np.zeros(rows)
    current_pos = 0.0

    for (start_row, end_row), slope in zip(row_ranges, slopes):
        segment_rows = end_row - start_row + 1
        segment_deltas = np.zeros(segment_rows)

        for j in range(segment_rows):
            segment_deltas[j] = current_pos + j * slope

        # Update position for next segment to maintain continuity
        current_pos = segment_deltas[-1] + slope

        all_deltas[start_row : end_row + 1] = segment_deltas

    return all_deltas


# =============================================================================
# Data Generation
# =============================================================================


def create_test_data(
    config: TestDataConfig = DEFAULT_CONFIG,
    with_shift: bool = True,
    with_row_scaling: bool = True,
    with_discontinuous_shifts: bool = False,
    custom_deltas: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Create synthetic 2D spectral image with Gaussian peaks.

    Args:
        config: Configuration parameters
        with_shift: Apply row-dependent peak shifts
        with_row_scaling: Apply triangular amplitude scaling (peak at middle row)
        with_discontinuous_shifts: Apply discontinuous shifts at rows 30 and 60
        custom_deltas: Custom shift values for each row (overrides other shift options)

    Returns:
        2D numpy array with synthetic spectral data
    """
    data = np.zeros((config.rows, config.cols))
    x_vals = np.arange(config.cols)
    middle_row = config.rows // 2

    # Pre-calculate cosmic ray positions
    cosmic_ray_positions = []
    if config.cosmic_ray_count > 0:
        cr_rows = np.random.randint(0, config.rows, config.cosmic_ray_count)
        cr_cols = np.random.randint(0, config.cols, config.cosmic_ray_count)
        cosmic_ray_positions = list(zip(cr_rows, cr_cols))

    # Generate each row
    for row in range(config.rows):
        # Calculate peak positions for this row
        if custom_deltas is not None:
            current_shift = custom_deltas[row]
        elif with_shift:
            current_shift = calculate_linear_shift(
                row, config.rows, config.shift_start, config.shift_end
            )
            if with_discontinuous_shifts:
                current_shift = calculate_discontinuous_shift(row, current_shift)
        else:
            current_shift = 0.0

        peak1_center = config.peak1_center + current_shift
        peak2_center = config.peak2_center + current_shift

        # Calculate row-dependent amplitude scaling
        if with_row_scaling:
            row_scale = 1.0 - abs(row - middle_row) / middle_row
        else:
            row_scale = 1.0

        # Build row: background + two Gaussian peaks
        row_data = np.ones(config.cols) * config.background
        row_data += (
            row_scale
            * config.peak_amplitude
            * np.exp(-0.5 * ((x_vals - peak1_center) / config.sigma) ** 2)
        )
        row_data += (
            row_scale
            * config.peak_amplitude
            * np.exp(-0.5 * ((x_vals - peak2_center) / config.sigma) ** 2)
        )

        # Add Poisson noise
        row_data = np.random.poisson(row_data)

        data[row] = row_data

    # Add cosmic rays
    for row, col in cosmic_ray_positions:
        data[row, col] += 1.0e5

    return data


# =============================================================================
# File I/O Helpers
# =============================================================================


def save_fits_file(data: np.ndarray, filename: str) -> None:
    """Save 2D array to FITS file."""
    hdu = fits.PrimaryHDU(data)
    hdul = fits.HDUList([hdu])
    hdul.writeto(filename, overwrite=True)
    print(f"Created: {filename}")


def save_npz_file(data: np.ndarray, filename: str, key: str = "deltas") -> None:
    """Save array to NPZ file."""
    np.savez(filename, **{key: data})
    print(f"Created: {filename}")


def create_symlink(target: str, link_name: str) -> None:
    """Create symbolic link, removing existing link if present."""
    if os.path.exists(link_name):
        os.remove(link_name)
    os.symlink(target, link_name)
    print(f"Created symbolic link: {link_name} -> {target}")


# =============================================================================
# Plotting Functions
# =============================================================================


def plot_row_samples(
    ax: plt.Axes,
    data: np.ndarray,
    config: TestDataConfig,
    sample_rows: list[int],
    shift_deltas: Optional[np.ndarray] = None,
    title: str = "Row Profiles",
) -> None:
    """
    Plot sample rows from a 2D spectral image.

    Args:
        ax: Matplotlib axis to plot on
        data: 2D data array
        config: Configuration (for peak positions)
        sample_rows: List of row indices to plot
        shift_deltas: Optional array of shift values per row
        title: Plot title
    """
    x_vals = np.arange(config.cols)

    for i, row_idx in enumerate(sample_rows):
        ax.plot(x_vals, data[row_idx], label=f"Row {row_idx}")

        # Mark expected peak positions
        if shift_deltas is not None:
            current_shift = shift_deltas[row_idx]
        else:
            current_shift = 0.0

        p1 = config.peak1_center + current_shift
        p2 = config.peak2_center + current_shift
        ax.axvline(x=p1, color=f"C{i}", linestyle="--", alpha=0.5)
        ax.axvline(x=p2, color=f"C{i}", linestyle="--", alpha=0.5)

    ax.set_title(title)
    ax.set_xlabel("X Position (pixels)")
    ax.set_ylabel("Intensity")
    ax.legend()


def plot_2d_image(
    ax: plt.Axes, data: np.ndarray, title: str = "2D Image", cmap: str = "viridis"
) -> None:
    """Plot 2D spectral image."""
    im = ax.imshow(data, origin="lower", aspect="auto", cmap=cmap)
    plt.colorbar(im, ax=ax, label="Intensity")
    ax.set_title(title)
    ax.set_xlabel("X Position (pixels)")
    ax.set_ylabel("Y Position (rows)")


def create_comparison_plot(
    datasets: dict[str, np.ndarray],
    config: TestDataConfig,
    output_file: str = "test_data_all_versions.png",
) -> None:
    """
    Create comprehensive comparison plot showing all data variants.

    Args:
        datasets: Dictionary mapping variant names to 2D data arrays
        config: Configuration used to generate data
        output_file: Output filename for plot
    """
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()

    # Plot 1: Row scaling profile
    all_rows = np.arange(config.rows)
    middle_row = config.rows // 2
    row_scaling = 1.0 - np.abs(all_rows - middle_row) / middle_row
    axes[0].plot(all_rows, row_scaling)
    axes[0].axhline(y=1.0, color="r", linestyle="--", alpha=0.5)
    axes[0].axhline(y=0.0, color="r", linestyle="--", alpha=0.5)
    axes[0].set_title("Row Amplitude Scaling Profile")
    axes[0].set_xlabel("Row Index")
    axes[0].set_ylabel("Scaling Factor")
    axes[0].grid(True, alpha=0.3)

    # Sample rows for profile plots
    sample_rows = [0, middle_row, config.rows - 1]

    # Create linear shift array for reference
    linear_shifts = np.array(
        [
            calculate_linear_shift(
                row, config.rows, config.shift_start, config.shift_end
            )
            for row in range(config.rows)
        ]
    )

    # Plot remaining variants
    plot_configs = [
        ("shifted_scaled", "Shifted Peaks + Scaling", linear_shifts),
        ("unshifted_scaled", "Fixed Peaks + Scaling", None),
        ("shifted_flat", "Shifted Peaks, No Scaling", linear_shifts),
        ("unshifted_flat", "Fixed Peaks, No Scaling", None),
    ]

    for idx, (key, title, shifts) in enumerate(plot_configs, start=1):
        if key in datasets:
            plot_row_samples(
                axes[idx], datasets[key], config, sample_rows, shifts, title
            )

    # 2D images
    for idx, (key, title, _) in enumerate(plot_configs, start=5):
        if key in datasets:
            plot_2d_image(axes[idx], datasets[key], f"2D: {title}")

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    print(f"Created comparison plot: {output_file}")


# =============================================================================
# Main Test Data Generation
# =============================================================================


def main():
    """Generate all test data variants."""
    config = DEFAULT_CONFIG

    # Ensure data and plots directories exist
    data_dir = "data"
    plots_dir = "plots"
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    print("Generating test data with configuration:")
    print(f"  Image size: {config.rows} x {config.cols}")
    print(f"  Peak FWHM: {config.fwhm} pixels (sigma={config.sigma:.2f})")
    print(f"  Peak amplitude: {config.peak_amplitude}")
    print(f"  Background: {config.background}")
    print(f"  Output directory: {data_dir}/")
    print()

    # Set random seed once at the start (not in create_test_data)
    np.random.seed(config.random_seed)

    # Define all variants to generate (order matches original script for RNG compatibility)
    variants = {
        "shifted_scaled": {
            "data": create_test_data(config, with_shift=True, with_row_scaling=True),
            "file": os.path.join(data_dir, "test_data_shifted.fits"),
            "description": "Shifted peaks with row scaling",
        },
        "unshifted_scaled": {
            "data": create_test_data(config, with_shift=False, with_row_scaling=True),
            "file": os.path.join(data_dir, "test_data_unshifted.fits"),
            "description": "Fixed peaks with row scaling",
        },
        "discontinuous": {
            "data": create_test_data(
                config,
                with_shift=True,
                with_row_scaling=True,
                with_discontinuous_shifts=True,
            ),
            "file": os.path.join(data_dir, "test_data_discontinuous.fits"),
            "description": "Discontinuous shifts at rows 30 and 60",
        },
    }

    # Generate multislope deltas (must happen before flat variants in original order)
    multislope_deltas = create_continuous_multislope_shifts(
        config.rows,
        slopes=[0.05, -0.15, 0.11],
        row_ranges=[(0, 29), (30, 59), (60, 99)],
    )

    # Add flat variants (these came after multislope_deltas in original)
    variants["shifted_flat"] = {
        "data": create_test_data(config, with_shift=True, with_row_scaling=False),
        "file": os.path.join(data_dir, "test_data_shifted_flat.fits"),
        "description": "Shifted peaks without row scaling",
    }
    variants["unshifted_flat"] = {
        "data": create_test_data(config, with_shift=False, with_row_scaling=False),
        "file": os.path.join(data_dir, "test_data_unshifted_flat.fits"),
        "description": "Fixed peaks without row scaling",
    }

    # Add multislope variant (reuse deltas calculated above)
    variants["multislope"] = {
        "data": create_test_data(
            config,
            with_shift=True,
            with_row_scaling=True,
            custom_deltas=multislope_deltas,
        ),
        "file": os.path.join(data_dir, "test_data_multislope.fits"),
        "description": "Continuous multislope shifts",
    }

    # Save all FITS files
    print("Saving FITS files:")
    for variant_info in variants.values():
        save_fits_file(variant_info["data"], variant_info["file"])

    # Save multislope deltas
    save_npz_file(
        multislope_deltas, os.path.join(data_dir, "multislope_deltas.npz"), key="deltas"
    )

    # Create symbolic link for backward compatibility (relative path within data/)
    # We need to create the symlink relative to the data directory
    symlink_target = "test_data_shifted.fits"
    symlink_path = os.path.join(data_dir, "test_data.fits")
    if os.path.exists(symlink_path):
        os.remove(symlink_path)
    os.symlink(symlink_target, symlink_path)
    print(f"Created symbolic link: {symlink_path} -> {symlink_target}")

    # Create comparison plot
    print("\nCreating visualization:")
    plot_datasets = {key: variant["data"] for key, variant in variants.items()}
    create_comparison_plot(
        plot_datasets,
        config,
        output_file=os.path.join(plots_dir, "test_data_all_versions.png"),
    )

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Data shape: {config.rows} x {config.cols}")
    print(f"\nGenerated {len(variants)} test data variants:")
    for i, (key, info) in enumerate(variants.items(), 1):
        print(f"  {i}. {info['file']}")
        print(f"     {info['description']}")
    print("\nAll variants have identical noise characteristics and peak amplitudes")


if __name__ == "__main__":
    main()
