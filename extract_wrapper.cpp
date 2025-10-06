#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <cstring>
#include <cmath>

extern "C" {
#include "extract.h"
}

namespace nb = nanobind;
using namespace nb::literals;

// Use simplified ndarray types
using ndarray_double_2d = nb::ndarray<double, nb::ndim<2>, nb::device::cpu>;
using ndarray_double_1d = nb::ndarray<double, nb::ndim<1>, nb::device::cpu>;
using ndarray_uint8_2d = nb::ndarray<uint8_t, nb::ndim<2>, nb::device::cpu>;

nb::tuple extract_wrapper(
    ndarray_double_2d im,
    ndarray_double_2d pix_unc,
    ndarray_uint8_2d mask,
    ndarray_double_1d ycen,
    ndarray_double_1d slitdeltas,
    int osample,
    double lambda_sP,
    double lambda_sL,
    int maxiter,
    nb::object slit_func_in = nb::none()
) {
    // Extract dimensions from image
    if (im.ndim() != 2) {
        throw std::runtime_error("Input image must be 2-dimensional");
    }

    int nrows = static_cast<int>(im.shape(0));
    int ncols = static_cast<int>(im.shape(1));

    // Calculate ny (needed for output sL array)
    int ny = osample * (nrows + 1) + 1;

    // Create output arrays using numpy arrays
    auto sP = nb::ndarray<nb::numpy, double>(nullptr, {static_cast<size_t>(ncols)});
    auto sL = nb::ndarray<nb::numpy, double>(nullptr, {static_cast<size_t>(ny)});
    auto model = nb::ndarray<nb::numpy, double>(nullptr, {static_cast<size_t>(nrows), static_cast<size_t>(ncols)});
    auto unc = nb::ndarray<nb::numpy, double>(nullptr, {static_cast<size_t>(ncols)});
    auto info = nb::ndarray<nb::numpy, double>(nullptr, {5});
    auto img_mad = nb::ndarray<nb::numpy, double>(nullptr, {static_cast<size_t>(nrows), static_cast<size_t>(ncols)});
    auto img_mad_mask = nb::ndarray<nb::numpy, uint8_t>(nullptr, {static_cast<size_t>(nrows), static_cast<size_t>(ncols)});

    // Get buffer pointers for all arrays
    auto im_ptr = im.data();
    auto pix_unc_ptr = pix_unc.data();
    auto mask_ptr = mask.data();
    auto ycen_ptr = ycen.data();
    auto slitdeltas_ptr = slitdeltas.data();

    // Get buffer pointers for output arrays
    auto sP_ptr = sP.data();
    auto sL_ptr = sL.data();
    auto model_ptr = model.data();
    auto unc_ptr = unc.data();
    auto info_ptr = info.data();
    auto img_mad_ptr = img_mad.data();
    auto img_mad_mask_ptr = img_mad_mask.data();

    // If slit_func_in is provided, initialize sL with it
    if (!slit_func_in.is_none()) {
        auto slit_func_in_arr = nb::cast<ndarray_double_1d>(slit_func_in);
        auto slit_func_in_ptr = slit_func_in_arr.data();
        memcpy(sL_ptr, slit_func_in_ptr, ny * sizeof(double));
    }

    // Call the C extract function
    int result = extract(
        ncols, nrows,
        im_ptr, pix_unc_ptr, mask_ptr,
        ycen_ptr, slitdeltas_ptr,
        osample, lambda_sP, lambda_sL, maxiter,
        sP_ptr, sL_ptr, model_ptr, unc_ptr, info_ptr
    );

    // Calculate img_mad as absolute difference between image and model
    for (int i = 0; i < nrows * ncols; i++) {
        img_mad_ptr[i] = std::abs(im_ptr[i] - model_ptr[i]);
        // Simple example mask based on a threshold
        img_mad_mask_ptr[i] = (img_mad_ptr[i] > 0.1) ? 1 : 0;
    }

    // Return the result code and all output arrays
    return nb::make_tuple(
        result,
        sL,
        sP,
        model,
        unc,
        info,
        img_mad,
        img_mad_mask
    );
}

NB_MODULE(vectextr, m) {
    m.doc() = "Python wrapper for the vectextr C library";

    m.def("extract", &extract_wrapper,
        "im"_a,
        "pix_unc"_a,
        "mask"_a,
        "ycen"_a,
        "slitdeltas"_a,
        "osample"_a,
        "lambda_sP"_a,
        "lambda_sL"_a,
        "maxiter"_a,
        "slit_func_in"_a = nb::none(),
        "Extract spectrum from a 2D image using optimal extraction algorithm");
}
