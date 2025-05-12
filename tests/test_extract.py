import pytest
import numpy as np

# Try to import vectextr module
try:
    import vectextr
except ImportError:
    pytest.skip("vectextr module not installed", allow_module_level=True)

def test_extract_basic():
    """
    Basic test for the extract function with a simple Gaussian-like pattern
    """
    # Define test parameters
    ncols = 10
    nrows = 5
    osample = 2
    delta_x = 1
    y_lower_lim = 2
    error_factor = 1.0
    lambda_sP = 0.1
    lambda_sL = 0.1
    sP_stop = 1e-3
    maxiter = 10
    kappa = 5.0
    
    # Calculate derived values
    ny = osample * (nrows + 1) + 1
    
    # Create a simple test image with a Gaussian-like pattern along y-axis
    im = np.zeros((nrows, ncols), dtype=np.float64)
    for j in range(nrows):
        for i in range(ncols):
            y_dist = abs(j - nrows/2.0)
            im[j, i] = np.exp(-y_dist * y_dist / 2.0)
    
    # Create other required arrays
    pix_unc = np.ones_like(im) * 0.1  # Constant uncertainty
    mask = np.ones_like(im, dtype=np.int32)  # All pixels valid
    ycen = np.ones(ncols, dtype=np.float64) * (nrows / 2.0)  # Central line
    ycen_offset = np.zeros(ncols, dtype=np.int32)  # No offsets
    
    # Slit curve coefficients - all zeros for a straight slit
    slitdeltas = np.zeros(ncols * osample * 3, dtype=np.float64)
    
    # Initial slit function (optional)
    slit_func_in = np.zeros(ny, dtype=np.float64)
    for i in range(ny):
        y_norm = (i - ny/2.0) / (ny/4.0)
        slit_func_in[i] = np.exp(-y_norm * y_norm / 2.0)
    
    # Call the extract function
    result, sL, sP, model, unc, img_mad, img_mad_mask = vectextr.extract(
        error_factor,
        im,
        pix_unc,
        mask,
        ycen,
        ycen_offset,
        y_lower_lim,
        slitdeltas,
        delta_x,
        osample,
        lambda_sP,
        lambda_sL,
        sP_stop,
        maxiter,
        kappa,
        slit_func_in
    )
    
    # Basic sanity checks
    assert result == 0, "Extract function should return 0 for success"
    assert sP.shape == (ncols,), "Spectrum should have shape (ncols,)"
    assert sL.shape == (ny,), "Slit function should have shape (ny,)"
    assert model.shape == (nrows, ncols), "Model should have shape (nrows, ncols)"
    assert unc.shape == (ncols,), "Uncertainty should have shape (ncols,)"
    
    # Check that spectrum has reasonable values
    assert np.all(np.isfinite(sP)), "Spectrum should have finite values"
    assert np.all(sP > 0), "Spectrum should be positive for this test case"
    
    # Check that slit function is normalized
    slit_integral = np.sum(sL) / osample
    assert abs(slit_integral - 1.0) < 0.1, "Slit function should be normalized"
    
    # Check that model reproduces data reasonably well
    residuals = im - model
    rms = np.sqrt(np.mean(residuals**2))
    assert rms < 0.5, "Model should fit the data reasonably well"
    
    # Print some results
    print(f"Extract function returned: {result}")
    print(f"Spectrum mean: {np.mean(sP)}")
    print(f"Slit function sum/osample: {slit_integral}")
    print(f"RMS of residuals: {rms}")
    
    return result, sL, sP, model, unc

if __name__ == "__main__":
    # Run the test directly
    test_extract_basic()
    print("All tests passed!")
