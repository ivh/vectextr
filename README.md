# CharSlit

**Flexible Slit-Decomposition for Echelle Spectroscopy**

CharSlit extends the slit-decomposition algorithm from [PyReduce](https://github.com/ivh/PyReduce) to support arbitrary slit shapes beyond polynomial approximations.

## Background

The original slit-decomposition method (Piskunov, Wehrhahn & Marquart 2021) models curved slits using polynomial functions. CharSlit generalizes this approach by representing the slit shape as an array of **per-row pixel offsets** (Δx values), where each element describes the horizontal shift from vertical for a given row in the spectral order.

This allows:
- **Characterization of complex slit geometries** that cannot be approximated by low-order polynomials
- **Empirical slit shape measurement** directly from calibration data
- **Improved extraction accuracy** for instruments with irregular optical distortions

## Workflow

1. **Measure slit shape**: Extract Δx offsets for each row from calibration frames
2. **Apply decomposition**: Use the measured offsets in the extraction algorithm

## Installation

Fork the repository, clone it, and build with [uv](https://github.com/astral-sh/uv):

```bash
git clone https://github.com/yourusername/CharSlit.git
cd CharSlit
uv sync
uv build
```

**Also see CLAUDE.md** for some more technical info.

## References

- Piskunov, N. E., & Valenti, J. A. (2002). *Spectroscopy Made Easy: A New Tool for Fitting Observations with Synthetic Spectra.* A&A, 385, 1095-1106
- Piskunov, N. E., Wehrhahn, A., & Marquart, T. (2021). *Optimizing Echelle Spectroscopy: Curved Slit Extraction.* A&A, 646, A32

## Status

This project is under active development as an experimental extension of the PyReduce framework.
