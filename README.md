# vectextr

Python bindings for the vectextr C library that implements spectral extraction algorithms.

## Installation

Using `uv`:

```bash
uv pip install -e .
```

## Testing

Run tests with:

```bash
uv run -m pytest
```

## Usage Example

```python
import numpy as np
import vectextr

# Create test data
ncols = 10
nrows = 5
osample = 2

# Create a simple image
im = np.ones((nrows, ncols), dtype=np.float64)
pix_unc = np.ones_like(im) * 0.1
mask = np.ones_like(im, dtype=np.int32)
ycen = np.ones(ncols, dtype=np.float64) * (nrows / 2.0)
ycen_offset = np.zeros(ncols, dtype=np.int32)
slitdeltas = np.zeros(ncols * osample * 3, dtype=np.float64)

# Extract the spectrum
result, sL, sP, model, unc, img_mad, img_mad_mask = vectextr.extract(

    im=im,
    pix_unc=pix_unc,
    mask=mask,
    ycen=ycen,
    ycen_offset=ycen_offset,
    y_lower_lim=2,
    slitdeltas=slitdeltas,
    delta_x=1,
    osample=osample,
    lambda_sP=0.1,
    lambda_sL=0.1,
    sP_stop=1e-3,
    maxiter=10,
    kappa=5.0
)

# Check the results
print(f"Extraction status: {result}")
print(f"Spectrum: {sP}")
```
