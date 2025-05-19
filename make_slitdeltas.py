#!/usr/bin/env python3

import numpy as np
from astropy.io import fits
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os

def gaussian(x, amplitude, mean, sigma, offset):
    """
    Gaussian function with vertical offset
    """
    return amplitude * np.exp(-(x - mean)**2 / (2 * sigma**2)) + offset

def find_and_fit_peaks(data):
    """
    Process FITS data row by row:
    1. Identify peaks in each row
    2. Fit a Gaussian to each peak to get the precise position
    3. Calculate absolute offsets from the median position of each peak
    
    Returns:
    - peak_positions: List of arrays, each containing fitted peak positions for each row
    - median_offsets: Array of absolute offsets from the median for each row
    - all_peak_fits: List of dictionaries containing fit parameters for each peak
    """
    num_rows, num_cols = data.shape
    x_vals = np.arange(num_cols)
    
    # Store peak positions for each row
    peak_positions = []
    all_peak_fits = []
    
    for row_idx in range(num_rows):
        row_data = data[row_idx]
        
        # Find peaks in the row
        peaks, properties = find_peaks(row_data, height=np.mean(row_data)*1.5, distance=10)
        
        if len(peaks) == 0:
            # No peaks found in this row
            peak_positions.append(np.array([]))
            continue
        
        # Fitted peak positions for this row
        fitted_positions = []
        row_fits = []
        
        for peak_idx, peak_loc in enumerate(peaks):
            # Define a window around the peak for fitting
            window_size = 15  # Adjust based on peak width
            left_bound = max(0, peak_loc - window_size)
            right_bound = min(num_cols, peak_loc + window_size + 1)
            
            x_window = x_vals[left_bound:right_bound]
            y_window = row_data[left_bound:right_bound]
            
            try:
                # Initial guess for parameters [amplitude, mean, sigma, offset]
                p0 = [
                    row_data[peak_loc] - np.min(y_window),  # amplitude
                    peak_loc,  # mean
                    5.0,  # sigma (based on FWHM of 10.0 / 2.355)
                    np.min(y_window)  # offset
                ]
                
                # Fit the Gaussian
                popt, pcov = curve_fit(gaussian, x_window, y_window, p0=p0)
                
                # Extract the fitted mean (peak position)
                fitted_pos = popt[1]
                fitted_positions.append(fitted_pos)
                
                # Store all fit parameters
                row_fits.append({
                    'peak_idx': peak_idx,
                    'row': row_idx,
                    'amplitude': popt[0],
                    'position': fitted_pos,
                    'sigma': popt[2],
                    'offset': popt[3]
                })
                
            except RuntimeError:
                # If fitting fails, use the original peak location
                fitted_positions.append(peak_loc)
                row_fits.append({
                    'peak_idx': peak_idx,
                    'row': row_idx,
                    'amplitude': row_data[peak_loc] - np.min(y_window),
                    'position': peak_loc,
                    'sigma': 5.0,
                    'offset': np.min(y_window),
                    'fit_failed': True
                })
        
        # Sort peaks by position to maintain consistent order across rows
        sort_idx = np.argsort(fitted_positions)
        fitted_positions = np.array(fitted_positions)[sort_idx]
        peak_positions.append(fitted_positions)
        
        # Sort the fit parameters in the same way
        all_peak_fits.append([row_fits[i] for i in sort_idx])
    
    # Calculate absolute offsets from median for each peak
    median_offsets = np.zeros(num_rows)
    
    # Find the most common number of peaks across rows
    peak_counts = [len(pos) for pos in peak_positions if len(pos) > 0]
    if not peak_counts:
        return peak_positions, np.zeros(num_rows), all_peak_fits
    
    most_common_peak_count = max(set(peak_counts), key=peak_counts.count)
    
    # Use only rows with the most common number of peaks
    valid_rows = [i for i, pos in enumerate(peak_positions) if len(pos) == most_common_peak_count]
    
    if len(valid_rows) < 2:
        print("Not enough valid rows with consistent peak counts")
        return peak_positions, np.zeros(num_rows), all_peak_fits
    
    # Organize peak positions by peak index across all rows
    peak_positions_by_index = [[] for _ in range(most_common_peak_count)]
    row_indices_by_peak = [[] for _ in range(most_common_peak_count)]
    
    for row_idx in valid_rows:
        for peak_idx in range(most_common_peak_count):
            if peak_idx < len(peak_positions[row_idx]):
                peak_positions_by_index[peak_idx].append(peak_positions[row_idx][peak_idx])
                row_indices_by_peak[peak_idx].append(row_idx)
    
    # Calculate median position for each peak
    peak_medians = [np.median(positions) if positions else None 
                   for positions in peak_positions_by_index]
    
    # Calculate offsets from median for each peak in each row
    for peak_idx in range(most_common_peak_count):
        if peak_medians[peak_idx] is not None:
            for i, row_idx in enumerate(row_indices_by_peak[peak_idx]):
                offset = peak_positions_by_index[peak_idx][i] - peak_medians[peak_idx]
                if median_offsets[row_idx] == 0:  # Only set if not already set
                    median_offsets[row_idx] = offset
                else:  # Average with existing offset
                    median_offsets[row_idx] = (median_offsets[row_idx] + offset) / 2
    
    # Fill in missing offsets using interpolation
    valid_indices = np.where(median_offsets != 0)[0]
    if len(valid_indices) > 0:
        for i in range(num_rows):
            if median_offsets[i] == 0:
                # Find nearest valid indices
                left_indices = valid_indices[valid_indices < i]
                right_indices = valid_indices[valid_indices > i]
                
                if len(left_indices) > 0 and len(right_indices) > 0:
                    # Interpolate
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
    
    return peak_positions, median_offsets, all_peak_fits

def process_fits_file(filename):
    """
    Process a FITS file and return the results of peak finding and analysis
    """
    print(f"Processing {filename}...")
    try:
        with fits.open(filename) as hdul:
            data = hdul[0].data
        
        # Find and fit peaks
        peak_positions, median_offsets, all_peak_fits = find_and_fit_peaks(data)
        
        # Calculate the average and std deviation of absolute offsets
        avg_offset = np.mean(median_offsets)
        std_offset = np.std(median_offsets)
        
        print(f"  Average offset from median: {avg_offset:.4f} ± {std_offset:.4f} pixels")
        
        return {
            'filename': filename,
            'peak_positions': peak_positions,
            'median_offsets': median_offsets,
            'all_peak_fits': all_peak_fits,
            'avg_offset': avg_offset,
            'std_offset': std_offset
        }
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return None

def plot_results(results):
    """
    Generate plots for the analysis results
    """
    # Create output directory for plots
    os.makedirs('slitdelta_plots', exist_ok=True)
    
    for result in results:
        if not result:
            continue
            
        filename = result['filename']
        basename = os.path.basename(filename).replace('.fits', '')
        median_offsets = result['median_offsets']
        peak_positions = result['peak_positions']
        
        # Plot 1: Absolute offsets from median
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(len(median_offsets)), median_offsets, 'o-')
        plt.xlabel('Row Index')
        plt.ylabel('Offset from Median (pixels)')
        plt.title(f'Absolute Offsets from Median - {basename}')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'slitdelta_plots/{basename}_offsets.png')
        
        # Plot 2: Peak positions across rows
        plt.figure(figsize=(10, 6))
        
        # Find rows with peaks
        rows_with_peaks = [i for i, pos in enumerate(peak_positions) if len(pos) > 0]
        
        # Plot each peak's position across rows
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
                    plt.plot(rows, positions, 'o-', label=f'Peak {peak_idx+1}')
            
            plt.xlabel('Row Index')
            plt.ylabel('Peak Position (pixel)')
            plt.title(f'Peak Positions Across Rows - {basename}')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f'slitdelta_plots/{basename}_peak_positions.png')
        
        plt.close('all')

def main():
    """
    Main function to process all test data files
    """
    # Find all FITS files
    fits_files = [f for f in os.listdir('.') if f.endswith('.fits') and f.startswith('test_data')]
    
    if not fits_files:
        print("No test data FITS files found!")
        return
    
    # Process each file
    results = []
    for fits_file in fits_files:
        result = process_fits_file(fits_file)
        if result:
            results.append(result)
    
    # Plot the results
    plot_results(results)
    
    # Save the numerical results for each dataset separately
    saved_files = []
    for result in results:
        if result:
            # Create filename based on original FITS file
            basename = os.path.splitext(os.path.basename(result['filename']))[0]
            output_file = f'slitdeltas_{basename}.npz'
            
            # Save individual dataset
            np.savez(
                output_file,
                filename=result['filename'],
                avg_offset=result['avg_offset'],
                std_offset=result['std_offset'],
                median_offsets=result['median_offsets']
            )
            saved_files.append(output_file)
    
    print(f"Results saved to {len(saved_files)} separate NPZ files:")
    for file in saved_files:
        print(f"  - {file}")

if __name__ == "__main__":
    main()
