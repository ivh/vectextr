import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.stats import sigma_clip

# Try to import vectextr module
try:
    import vectextr
except ImportError:
    raise ImportError("vectextr module not installed")

def test_extract_basic(fname, slope=None, slitdeltas=None):
    """
    Basic test for the extract function with a simple Gaussian-like pattern
    """
    # Define test parameters
    osample = 10
    
    # Add regularization for spectrum to improve numerical stability
    lambda_sP = 0.0
    lambda_sL = 2.0
    maxiter = 20
    
    # Load the FITS file data
    with fits.open(fname) as hdul:
        # Get the data from the 0th extension
        im = hdul[0].data.astype(np.float64)
    
    # Update dimensions based on the actual image
    nrows, ncols = im.shape
    print(f"Loaded image with shape: {im.shape}")
    
    # Calculate derived values
    ny = osample * (nrows + 1) + 1
    
    if slitdeltas is None and slope is None:
        print("Error: either slitdeltas or slope must be provided")
        return
    elif slitdeltas is None:
        slitdeltas_os = np.linspace(-slope*nrows/2, slope*nrows/2, ny, dtype=np.float64)
    else: # oversample incoming slitdeltas
        slitdeltas_os = np.interp(np.linspace(0, nrows-1, ny), np.arange(nrows), slitdeltas)

    # Create other required arrays
    pix_unc = np.ones_like(im) * 0.1  # Constant uncertainty
    mask = np.ones_like(im, dtype=np.uint8)  # All pixels valid
    
    # For the central line, use the middle of the image
    ycen = np.ones(ncols, dtype=np.float64) * (nrows / 2.0)  # Central line
    
    # Initial slit function as horizontal mean of im, with outliers rejected first
    slit_func_in = sigma_clip(im, sigma=6).mean(axis=1)
    
    # oversample slit_func_in
    slit_func_in = np.interp(np.linspace(0, nrows-1, ny), np.arange(nrows), slit_func_in)
    # normalize 
    slit_func_in = slit_func_in / np.sum(slit_func_in)
    np.savez('slit_func_in.npz', slit_func_in=slit_func_in)

    # Call the extract function with error handling
    try:
        result, sL, sP, model, unc, info, img_mad, img_mad_mask = vectextr.extract(
            im,
            pix_unc,
            mask,
            ycen,
            slitdeltas_os,
            osample,
            lambda_sP,
            lambda_sL,
            maxiter,
            slit_func_in
        )
    except Exception as e:
        print(f"\nError during extraction: {str(e)}")
        # Return some minimal data to avoid downstream errors
        return -1, np.ones(ny), np.zeros(ncols), np.zeros_like(im), np.zeros_like(im)
    
    print(f"Extract function returned: {result}")
    print(f"Info array: {info}")
    print(f"Finite values in spectrum: {np.sum(np.isfinite(sP))}/{len(sP)}")

    # Visualize the input image, model, residuals, extracted spectrum, and slit function
    plt.figure(figsize=(15, 12))
    
    # Input image
    plt.subplot(3, 2, 1)
    peak_ampl = np.percentile(im, 95)
    plt.imshow(im, origin='lower', aspect='auto', cmap='viridis', vmin=0, vmax=peak_ampl)
    plt.colorbar(label='Flux')
    plt.title('Input Image')
    plt.xlabel('Column')
    plt.ylabel('Row')
    
    # Plot slitdeltas on the input image
    ref_col = int(ncols/2)
    if 0 <= ref_col < ncols:
        # Downsample the slitdeltas to match image rows
        y_trace = np.linspace(0, nrows-1, nrows)
        if slitdeltas is not None:
            x_trace = ref_col + slitdeltas
        else:       
            # Calculate x-positions based on slitdeltas, downsampled to match image rows
            # Convert from oversampled slitdeltas to image coordinates
            x_trace = ref_col + np.interp(y_trace, 
                                        np.linspace(0, nrows-1, ny), 
                                        slitdeltas_os)
        plt.plot(x_trace, y_trace, 'w-', lw=1.5, alpha=0.7)
    
    # Model
    plt.subplot(3, 2, 3)
    plt.imshow(model, origin='lower', aspect='auto', cmap='viridis', vmin=0, vmax=peak_ampl)
    plt.colorbar(label='Flux')
    plt.title('Model')
    plt.xlabel('Column')
    plt.ylabel('Row')
    
    # Plot slitdeltas on the model image
    if 0 <= ref_col < ncols:
        plt.plot(x_trace, y_trace, 'w-', lw=1.5, alpha=0.7)
    
    # Calculate residuals where model has values
    residuals = im - model
    rms = np.sqrt(np.mean(residuals[np.isfinite(residuals)]**2))
    
    # Residuals
    plt.subplot(3, 2, 5)
    vmax = np.maximum(np.abs(np.nanpercentile(residuals, 3)), np.nanpercentile(residuals, 97))
    plt.imshow(residuals, origin='lower', aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    plt.colorbar(label='Residual Flux')
    plt.title(f'Residuals (Data - Model), RMS: {rms:.4f}')
    plt.xlabel('Column')
    plt.ylabel('Row')
        
    # Plot slitdeltas on the residuals image
    if 0 <= ref_col < ncols:
        plt.plot(x_trace, y_trace, 'k-', lw=1.5, alpha=0.7)
    
    # Extracted spectrum
    plt.subplot(3, 2, 2)
    plt.plot(np.arange(ncols), sP)
    plt.title(f'Extracted Spectrum')
    plt.xlabel('Column')
    plt.ylabel('Flux')
    plt.grid(True, alpha=0.3)
        
    # Slit function (added to the main figure as a 5th panel)
    plt.subplot(3, 2, 4)
    y_coords = np.linspace(0, nrows-1, ny)
    plt.plot(y_coords, sL)
    plt.title('Slit Function')
    plt.xlabel('Row (interpolated)')
    plt.ylabel('Value')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the figure to a file instead of showing it
    output_file = fname.replace('.fits', '_visualization.png')
    plt.savefig(output_file, dpi=150)
    print(f"Visualization saved to {output_file}")
    
    # Close the figures to free up memory
    plt.close('all')
    
    return result, sL, sP, model, unc

def run_single_test(fname, slope=None, slitdeltas=None):
    """Wrapper function to run a single test case via subprocess"""
    import sys
    import json
    
    try:
        # Check if we're using a custom slitdeltas array
        if slitdeltas is not None:
            print(f"Using custom slitdeltas for {fname}")
            
        result, sL, sP, model, unc = test_extract_basic(fname=fname, slope=slope, slitdeltas=slitdeltas)
        if result == -1:
            print(f"Test failed but gracefully handled for {fname}")
            return False
        else:
            print(f"Test completed successfully for {fname}")
            return True
    except Exception as e:
        print(f"Test execution error for {fname}: {str(e)}")
        return False

if __name__ == "__main__":
    import sys
    import subprocess
    import os
    
    # Check if we're running a single test (called from subprocess)
    if len(sys.argv) > 1 and sys.argv[1] == "--single-test":
        fname = sys.argv[2]
        slope = float(sys.argv[3]) if sys.argv[3] != "None" else None
        
        # Handle the discontinuous slitdeltas case specially
        if fname.endswith('discontinuous.fits'):
            slitdeltas = np.concatenate([
                np.linspace(-5, 5, 100)[:30],                   # Rows 0-29: Regular gradual shift
                np.linspace(-5, 5, 100)[30:60] - 4.0,           # Rows 30-59: Extra 4 pixel left shift
                np.linspace(-5, 5, 100)[60:] - 8.0             # Rows 60-99: Extra 8 pixel left shift
            ])
            run_single_test(fname=fname, slitdeltas=slitdeltas)
        # Handle the multislope case with continuous transitions
        elif fname.endswith('multislope.fits'):
            # Load the pre-computed deltas from the NPZ file
            multislope_data = np.load('multislope_deltas.npz')
            deltas = multislope_data['deltas']
            run_single_test(fname=fname, slitdeltas=deltas)
        # Handle any data file with computed slitdeltas from make_slitdeltas.py
        elif os.path.exists(f'slitdeltas_{os.path.basename(fname).replace(".fits", "")}.npz'):
            # Load slitdeltas computed by make_slitdeltas.py
            npz_file = f'slitdeltas_{os.path.basename(fname).replace(".fits", "")}.npz'
            print(f"Loading slitdeltas from {npz_file}")
            slitdeltas_data = np.load(npz_file)
            deltas = slitdeltas_data['median_offsets']
            run_single_test(fname=fname, slitdeltas=deltas)
        else:
            run_single_test(fname=fname, slope=slope)
        sys.exit(0)
    
    # Main process - run each test in a separate subprocess
    datasets = [
        {
            'fname': '/Users/tom/vectextr/test_data_unshifted.fits',
            'slope': 0.0
        },
        {
            'fname': '/Users/tom/vectextr/test_data_shifted.fits',
            'slope': 0.1
        },
        {
            'fname': '/Users/tom/vectextr/test_data_discontinuous.fits',
            'slope': None  # Will be handled specially in subprocess
        },
        {
            'fname': '/Users/tom/vectextr/test_data_multislope.fits',
            'slope': None  # Will be handled specially in subprocess
        },
        {
            'fname': '/Users/tom/vectextr/test_data_Rsim.fits',
            'slope': None  # Will use slitdeltas from NPZ file
        },
        {
            'fname': '/Users/tom/vectextr/test_data_Hsim.fits',
            'slope': None  # Will use slitdeltas from NPZ file
        }
    ]
    
    for dataset in datasets[-1:]:
        print(f"\n===== Testing {dataset['fname']} =====")
        cmd = [sys.executable, __file__, "--single-test", dataset['fname'], str(dataset['slope'])]
        try:
            # Run the test in a separate process with a timeout
            process = subprocess.run(cmd, timeout=60, capture_output=True, text=True)
            print(process.stdout)
            if process.returncode != 0:
                print(f"Test process exited with code {process.returncode}")
                if process.stderr:
                    print(f"Error output: {process.stderr}")
        except subprocess.TimeoutExpired:
            print(f"Test timed out for {dataset['fname']}")
        except Exception as e:
            print(f"Error running test for {dataset['fname']}: {str(e)}")
        
        print(f"===== End of test for {dataset['fname']} =====\n")
