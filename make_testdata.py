#!/usr/bin/env python3

import os

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

# Define parameters
rows = 100
cols = 150
x_vals = np.arange(cols)

# Define FWHM (Full Width at Half Maximum)
fwhm = 10.0

# Convert FWHM to Gaussian sigma
# sigma = FWHM / (2 * sqrt(2 * ln(2)))
sigma = fwhm / 2.355


# Prepare indices for cosmic rays (will be added later)
cosmic_ray_count = 0
# Generate random row and column indices for cosmic rays
cr_rows = np.random.randint(0, rows, cosmic_ray_count)
cr_cols = np.random.randint(0, cols, cosmic_ray_count)
# Store indices as (row, col) pairs for later use
cosmic_ray_positions = list(zip(cr_rows, cr_cols))

# Initial peak positions
peak1_center_init = 50.0
peak2_center_init = 120.0
peak_amplitude = 9.99e3

# Background level to add
background = 5.0

# Define shift parameters
shift_per_row = 0.1  # 1/10 of a pixel per row
shift_start = -5.0  # Start shift at -5 pixels
shift_end = 5.0  # End shift at +5 pixels

# Set random seed for reproducibility
np.random.seed(42)


# Function to create test data
def create_test_data(
    with_shift=True,
    with_row_scaling=True,
    with_discontinuous_shifts=False,
    custom_deltas=None,
):
    # Initialize data array
    data = np.zeros((rows, cols))

    # Middle row index (for scaling calculations)
    middle_row = rows // 2  # Integer division to get middle row index
    # Create each row with or without peak shifts
    for row in range(rows):
        # Calculate peak centers for this row
        if with_shift:
            if custom_deltas is not None:
                # Use provided custom delta values
                current_shift = custom_deltas[row]
            else:
                # Apply shift from -5 to +5 pixels across all rows
                current_shift = shift_start + (row / (rows - 1)) * (
                    shift_end - shift_start
                )

                # Apply additional discontinuous shifts if requested
                if with_discontinuous_shifts:
                    if row >= 60:
                        # Rows 60-100: shift by additional 8 pixels left (negative)
                        current_shift -= 8.0
                    elif row >= 30:
                        # Rows 30-59: shift by additional 4 pixels left (negative)
                        current_shift -= 4.0

            peak1_center = peak1_center_init + current_shift
            peak2_center = peak2_center_init + current_shift
        else:
            # No shift between rows
            peak1_center = peak1_center_init
            peak2_center = peak2_center_init

        # Calculate row-dependent scaling factor (triangular profile)
        if with_row_scaling:
            # Scale linearly from 0 at top/bottom to 1 at middle
            row_scale = 1.0 - abs(row - middle_row) / middle_row
        else:
            # No row-dependent scaling
            row_scale = 1.0

        # Create this row with the appropriate peaks
        # Start with the background level (background is constant for all rows)
        row_data = np.ones(cols) * background

        # Add the Gaussian peaks with row-dependent scaling
        row_data += (
            row_scale
            * peak_amplitude
            * np.exp(-0.5 * ((x_vals - peak1_center) / sigma) ** 2)
        )
        row_data += (
            row_scale
            * peak_amplitude
            * np.exp(-0.5 * ((x_vals - peak2_center) / sigma) ** 2)
        )

        # Add Poisson noise to simulate photon counts
        # For each pixel, draw from a Poisson distribution with mean = expected count
        row_data = np.random.poisson(row_data)

        # Store this row in the data array
        data[row] = row_data

    # Add cosmic rays (random pixels with values 1E5 higher)
    for row, col in cosmic_ray_positions:
        data[row, col] += 1.0e5

    return data


# Function to generate continuous multi-slope shifts
def create_continuous_multislope_shifts(
    slopes=[0.05, -0.15, 0.11], row_ranges=[(0, 29), (30, 59), (60, 99)]
):
    # Initialize array for all delta values
    all_deltas = np.zeros(rows)

    # Starting position - we'll start at 0
    current_pos = 0.0

    # For each segment
    for i, ((start_row, end_row), slope) in enumerate(zip(row_ranges, slopes)):
        # Number of rows in this segment
        segment_rows = end_row - start_row + 1

        # Generate deltas for this segment
        segment_deltas = np.zeros(segment_rows)

        # Fill the segment with cumulative slope values
        for j in range(segment_rows):
            segment_deltas[j] = current_pos + j * slope

        # Update the starting position for the next segment to maintain continuity
        current_pos = segment_deltas[-1] + slope

        # Store in the main array
        all_deltas[start_row : end_row + 1] = segment_deltas

    return all_deltas


# Create all data versions
shifted_data = create_test_data(with_shift=True, with_row_scaling=True)
unshifted_data = create_test_data(with_shift=False, with_row_scaling=True)

# Create variant with discontinuous shifts
discontinuous_data = create_test_data(
    with_shift=True, with_row_scaling=True, with_discontinuous_shifts=True
)

# Create multislope variant with continuous transitions
# Generate the deltas with the specified slopes
multislope_deltas = create_continuous_multislope_shifts(
    slopes=[0.05, -0.15, 0.11], row_ranges=[(0, 29), (30, 59), (60, 99)]
)

# Save the delta values to an npz file
np.savez("multislope_deltas.npz", deltas=multislope_deltas)

# Save shifted data to FITS file
hdu_shifted = fits.PrimaryHDU(shifted_data)
hdul_shifted = fits.HDUList([hdu_shifted])
output_file_shifted = "test_data_shifted.fits"
hdul_shifted.writeto(output_file_shifted, overwrite=True)
print(f"Created FITS file with shifted peaks and row scaling: {output_file_shifted}")

# Save unshifted data to FITS file
hdu_unshifted = fits.PrimaryHDU(unshifted_data)
hdul_unshifted = fits.HDUList([hdu_unshifted])
output_file_unshifted = "test_data_unshifted.fits"
hdul_unshifted.writeto(output_file_unshifted, overwrite=True)
print(f"Created FITS file without shifts, with row scaling: {output_file_unshifted}")

# For compatibility with previous tests, also create versions without row scaling
shifted_data_flat = create_test_data(with_shift=True, with_row_scaling=False)
unshifted_data_flat = create_test_data(with_shift=False, with_row_scaling=False)

# Save flat profile versions
hdu_shifted_flat = fits.PrimaryHDU(shifted_data_flat)
hdul_shifted_flat = fits.HDUList([hdu_shifted_flat])
output_file_shifted_flat = "test_data_shifted_flat.fits"
hdul_shifted_flat.writeto(output_file_shifted_flat, overwrite=True)
print(
    f"Created FITS file with shifted peaks, no row scaling: {output_file_shifted_flat}"
)

hdu_unshifted_flat = fits.PrimaryHDU(unshifted_data_flat)
hdul_unshifted_flat = fits.HDUList([hdu_unshifted_flat])
output_file_unshifted_flat = "test_data_unshifted_flat.fits"
hdul_unshifted_flat.writeto(output_file_unshifted_flat, overwrite=True)
print(f"Created FITS file without shifts, no row scaling: {output_file_unshifted_flat}")

# Save the discontinuous shifts version
hdu_discontinuous = fits.PrimaryHDU(discontinuous_data)
hdul_discontinuous = fits.HDUList([hdu_discontinuous])
output_file_discontinuous = "test_data_discontinuous.fits"
hdul_discontinuous.writeto(output_file_discontinuous, overwrite=True)
print(f"Created FITS file with discontinuous shifts: {output_file_discontinuous}")

# Create and save the multislope variant
multislope_data = create_test_data(
    with_shift=True, with_row_scaling=True, custom_deltas=multislope_deltas
)
hdu_multislope = fits.PrimaryHDU(multislope_data)
hdul_multislope = fits.HDUList([hdu_multislope])
output_file_multislope = "test_data_multislope.fits"
hdul_multislope.writeto(output_file_multislope, overwrite=True)
print(f"Created FITS file with continuous multislope shifts: {output_file_multislope}")

# Create a comparison plot showing row scaling and peak shifts
plt.figure(figsize=(15, 15))

# Define consistent sample rows for all plots
middle_row = rows // 2
sample_rows = [0, middle_row, rows - 1]  # First, middle, and last row

# Plot row amplitude scaling factors
plt.subplot(331)
all_rows = np.arange(rows)
row_scaling = 1.0 - np.abs(all_rows - middle_row) / middle_row
plt.plot(all_rows, row_scaling)
plt.axhline(y=1.0, color="r", linestyle="--", alpha=0.5)
plt.axhline(y=0.0, color="r", linestyle="--", alpha=0.5)
plt.title("Row Amplitude Scaling Profile")
plt.xlabel("Row Index")
plt.ylabel("Scaling Factor")
plt.grid(True, alpha=0.3)

# Plot sample rows from shifted data with scaling
plt.subplot(332)
for i, row_idx in enumerate(sample_rows):
    plt.plot(x_vals, shifted_data[row_idx], label=f"Row {row_idx}")
    # Calculate the peak positions for this row
    current_shift = shift_start + (row_idx / (rows - 1)) * (shift_end - shift_start)
    p1 = peak1_center_init + current_shift
    p2 = peak2_center_init + current_shift
    plt.axvline(x=p1, color=f"C{i}", linestyle="--", alpha=0.5)
    plt.axvline(x=p2, color=f"C{i}", linestyle="--", alpha=0.5)

plt.title("Row Profiles: Shifted Peaks + Scaling")
plt.xlabel("X Position (pixels)")
plt.ylabel("Intensity")
plt.legend()

# Plot sample rows from unshifted data with scaling
plt.subplot(333)
for i, row_idx in enumerate(sample_rows):
    plt.plot(x_vals, unshifted_data[row_idx], label=f"Row {row_idx}")
    # For unshifted data, all rows have same peak position
    plt.axvline(x=peak1_center_init, color=f"C{i}", linestyle="--", alpha=0.5)
    plt.axvline(x=peak2_center_init, color=f"C{i}", linestyle="--", alpha=0.5)

plt.title("Row Profiles: Fixed Peaks + Scaling")
plt.xlabel("X Position (pixels)")
plt.ylabel("Intensity")
plt.legend()

# Plot sample rows from shifted data without scaling
plt.subplot(334)
for i, row_idx in enumerate(sample_rows):
    plt.plot(x_vals, shifted_data_flat[row_idx], label=f"Row {row_idx}")
    # Calculate the peak positions for this row
    current_shift = shift_start + (row_idx / (rows - 1)) * (shift_end - shift_start)
    p1 = peak1_center_init + current_shift
    p2 = peak2_center_init + current_shift
    plt.axvline(x=p1, color=f"C{i}", linestyle="--", alpha=0.5)
    plt.axvline(x=p2, color=f"C{i}", linestyle="--", alpha=0.5)

plt.title("Row Profiles: Shifted Peaks, No Scaling")
plt.xlabel("X Position (pixels)")
plt.ylabel("Intensity")
plt.legend()

# Plot sample rows from unshifted data without scaling
plt.subplot(335)
for i, row_idx in enumerate(sample_rows):
    plt.plot(x_vals, unshifted_data_flat[row_idx], label=f"Row {row_idx}")
    plt.axvline(x=peak1_center_init, color=f"C{i}", linestyle="--", alpha=0.5)
    plt.axvline(x=peak2_center_init, color=f"C{i}", linestyle="--", alpha=0.5)

plt.title("Row Profiles: Fixed Peaks, No Scaling")
plt.xlabel("X Position (pixels)")
plt.ylabel("Intensity")
plt.legend()

# Plot the 2D images
# Shifted data with row scaling
plt.subplot(336)
im1 = plt.imshow(shifted_data, origin="lower", aspect="auto", cmap="viridis")
plt.colorbar(im1, label="Intensity")
plt.title("2D: Shifted Peaks + Scaling")
plt.xlabel("X Position (pixels)")
plt.ylabel("Y Position (rows)")

# Unshifted data with row scaling
plt.subplot(337)
im2 = plt.imshow(unshifted_data, origin="lower", aspect="auto", cmap="viridis")
plt.colorbar(im2, label="Intensity")
plt.title("2D: Fixed Peaks + Scaling")
plt.xlabel("X Position (pixels)")
plt.ylabel("Y Position (rows)")

# Shifted data without row scaling
plt.subplot(338)
im3 = plt.imshow(shifted_data_flat, origin="lower", aspect="auto", cmap="viridis")
plt.colorbar(im3, label="Intensity")
plt.title("2D: Shifted Peaks, No Scaling")
plt.xlabel("X Position (pixels)")
plt.ylabel("Y Position (rows)")

# Unshifted data without row scaling
plt.subplot(339)
im4 = plt.imshow(unshifted_data_flat, origin="lower", aspect="auto", cmap="viridis")
plt.colorbar(im4, label="Intensity")
plt.title("2D: Fixed Peaks, No Scaling")
plt.xlabel("X Position (pixels)")
plt.ylabel("Y Position (rows)")

plt.tight_layout()
plt.savefig("test_data_all_versions.png")
plt.close()

print("Created comparison preview image: test_data_all_versions.png")
print(f"Data shape: {shifted_data.shape}")
print("Generated six versions of test data:")
print("  1. Shifted peaks with row scaling: test_data_shifted.fits")
print("  2. Fixed peaks with row scaling: test_data_unshifted.fits")
print("  3. Shifted peaks without row scaling: test_data_shifted_flat.fits")
print("  4. Fixed peaks without row scaling: test_data_unshifted_flat.fits")
print(
    "  5. Peaks with discontinuous shifts at rows 30 and 60: test_data_discontinuous.fits"
)
print("  6. Peaks with continuous multislope shifts: test_data_multislope.fits")
print("All versions have same noise characteristics and peak amplitudes")
print("Delta values for the multislope variant saved to: multislope_deltas.npz")

# Also create a symbolic link to match the name used in previous tests
if os.path.exists("test_data.fits"):
    os.remove("test_data.fits")
os.symlink("test_data_shifted.fits", "test_data.fits")
print("Created symbolic link: test_data.fits -> test_data_shifted.fits")
