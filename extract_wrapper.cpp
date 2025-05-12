#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

// Properly include C headers in C++
extern "C" {
#include "extract.h"
}

namespace py = pybind11;

// Function to wrap the C extract function to use NumPy arrays
py::tuple py_extract(
    py::array_t<double> im_py,
    py::array_t<double> pix_unc_py,
    py::array_t<int> mask_py,
    py::array_t<double> ycen_py,
    py::array_t<int> ycen_offset_py,
    int y_lower_lim,
    py::array_t<double> slitdeltas_py,
    int delta_x,
    int osample,
    double lambda_sP,
    double lambda_sL,
    double sP_stop,
    int maxiter,
    double kappa,
    py::array_t<double, py::array::c_style | py::array::forcecast> slit_func_in_py = py::none()
) {
    // Get array dimensions
    py::buffer_info im_info = im_py.request();
    if (im_info.ndim != 2) {
        throw std::runtime_error("im array must be 2-dimensional");
    }
    
    int nrows = im_info.shape[0];
    int ncols = im_info.shape[1];
    
    // Calculate derived dimensions
    int ny = osample * (nrows + 1) + 1;
    int nx = 4 * delta_x + 1;
    
    // Check input array shapes
    if (pix_unc_py.request().ndim != 2 || 
        pix_unc_py.request().shape[0] != nrows || 
        pix_unc_py.request().shape[1] != ncols) {
        throw std::runtime_error("pix_unc array must have same shape as im");
    }
    
    if (mask_py.request().ndim != 2 || 
        mask_py.request().shape[0] != nrows || 
        mask_py.request().shape[1] != ncols) {
        throw std::runtime_error("mask array must have same shape as im");
    }
    
    if (ycen_py.request().ndim != 1 || ycen_py.request().shape[0] != ncols) {
        throw std::runtime_error("ycen array must be 1D with length ncols");
    }
    
    if (ycen_offset_py.request().ndim != 1 || ycen_offset_py.request().shape[0] != ncols) {
        throw std::runtime_error("ycen_offset array must be 1D with length ncols");
    }
    
    // Access data pointers
    double* im = static_cast<double*>(im_info.ptr);
    double* pix_unc = static_cast<double*>(pix_unc_py.request().ptr);
    int* mask = static_cast<int*>(mask_py.request().ptr);
    double* ycen = static_cast<double*>(ycen_py.request().ptr);
    int* ycen_offset = static_cast<int*>(ycen_offset_py.request().ptr);
    double* slitdeltas = static_cast<double*>(slitdeltas_py.request().ptr);
    
    // Prepare output arrays
    py::array_t<double> sL_py = py::array_t<double>(ny);
    py::array_t<double> sP_py = py::array_t<double>(ncols);
    py::array_t<double> model_py = py::array_t<double>(nrows * ncols);
    py::array_t<double> unc_py = py::array_t<double>(ncols);
    py::array_t<double> sP_old_py = py::array_t<double>(ncols);
    py::array_t<double> l_Aij_py = py::array_t<double>(ny * (4 * osample + 1));
    py::array_t<double> p_Aij_py = py::array_t<double>(ncols * nx);
    py::array_t<double> l_bj_py = py::array_t<double>(ny);
    py::array_t<double> p_bj_py = py::array_t<double>(ncols);
    py::array_t<double> img_mad_py = py::array_t<double>(nrows * ncols);
    py::array_t<int> img_mad_mask_py = py::array_t<int>(nrows * ncols);
    
    // Get pointers to data in output arrays
    double* sL = static_cast<double*>(sL_py.request().ptr);
    double* sP = static_cast<double*>(sP_py.request().ptr);
    double* model = static_cast<double*>(model_py.request().ptr);
    double* unc = static_cast<double*>(unc_py.request().ptr);
    double* sP_old = static_cast<double*>(sP_old_py.request().ptr);
    double* l_Aij = static_cast<double*>(l_Aij_py.request().ptr);
    double* p_Aij = static_cast<double*>(p_Aij_py.request().ptr);
    double* l_bj = static_cast<double*>(l_bj_py.request().ptr);
    double* p_bj = static_cast<double*>(p_bj_py.request().ptr);
    double* img_mad = static_cast<double*>(img_mad_py.request().ptr);
    int* img_mad_mask = static_cast<int*>(img_mad_mask_py.request().ptr);
    
    // Handle the slit_func_in parameter (optional)
    double* slit_func_in = nullptr;
    if (!py::isinstance<py::none>(slit_func_in_py)) {
        py::buffer_info slit_info = slit_func_in_py.request();
        if (slit_info.ndim != 1 || slit_info.shape[0] != ny) {
            throw std::runtime_error("slit_func_in must be 1D with length ny");
        }
        slit_func_in = static_cast<double*>(slit_info.ptr);
    }
    
    // Allocate memory for the tensors
    std::vector<xi_ref> xi(ncols * ny * 4);
    std::vector<zeta_ref> zeta(ncols * nrows * 3 * (osample + 1));
    std::vector<int> m_zeta(ncols * nrows, 0);
    
    // Call the extract function
    int result = extract(
        ncols,
        nrows,
        osample,
        im,
        pix_unc,
        mask,
        ycen,
        ycen_offset,
        y_lower_lim,
        slitdeltas,
        delta_x,
        sL,
        sP,
        model,
        unc,
        lambda_sP,
        lambda_sL,
        sP_stop,
        maxiter,
        kappa,
        slit_func_in,
        sP_old,
        l_Aij,
        p_Aij,
        l_bj,
        p_bj,
        img_mad,
        img_mad_mask,
        xi.data(),
        zeta.data(),
        m_zeta.data()
    );
    
    // Reshape model and img_mad arrays to 2D
    py::array_t<double> model_2d = py::array_t<double>({nrows, ncols});
    py::array_t<double> img_mad_2d = py::array_t<double>({nrows, ncols});
    py::array_t<int> img_mad_mask_2d = py::array_t<int>({nrows, ncols});
    
    auto model_2d_buf = model_2d.request();
    auto img_mad_2d_buf = img_mad_2d.request();
    auto img_mad_mask_2d_buf = img_mad_mask_2d.request();
    
    double* model_2d_ptr = static_cast<double*>(model_2d_buf.ptr);
    double* img_mad_2d_ptr = static_cast<double*>(img_mad_2d_buf.ptr);
    int* img_mad_mask_2d_ptr = static_cast<int*>(img_mad_mask_2d_buf.ptr);
    
    // Copy data to correctly shaped arrays
    for (int y = 0; y < nrows; y++) {
        for (int x = 0; x < ncols; x++) {
            model_2d_ptr[y * ncols + x] = model[y * ncols + x];
            img_mad_2d_ptr[y * ncols + x] = img_mad[y * ncols + x];
            img_mad_mask_2d_ptr[y * ncols + x] = img_mad_mask[y * ncols + x];
        }
    }
    
    // Return the output arrays in a tuple
    return py::make_tuple(
        result,        // Return code
        sL_py,         // Slit function
        sP_py,         // Spectrum
        model_2d,      // Model (2D)
        unc_py,        // Uncertainty
        img_mad_2d,    // MAD image (2D)
        img_mad_mask_2d // MAD mask (2D)
    );
}

PYBIND11_MODULE(vectextr, m) {
    m.doc() = "Python bindings for vectextr's extract function";
    
    m.def("extract", &py_extract, 

          py::arg("im"),
          py::arg("pix_unc"),
          py::arg("mask"),
          py::arg("ycen"),
          py::arg("ycen_offset"),
          py::arg("y_lower_lim"),
          py::arg("slitdeltas"),
          py::arg("delta_x"),
          py::arg("osample"),
          py::arg("lambda_sP"),
          py::arg("lambda_sL"),
          py::arg("sP_stop"),
          py::arg("maxiter"),
          py::arg("kappa"),
          py::arg("slit_func_in") = py::none(),
          "Extract a spectrum from an image using the C implementation");
}
