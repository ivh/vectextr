

int cr2res_extract_slitdec_curved(
    const hdrl_image    *   img_hdrl,
    const cpl_table     *   trace_tab,
    const cpl_vector    *   slit_func_vec_in,
    int                     order,
    int                     trace_id,
    int                     height,
    int                     swath,
    int                     oversample,
    double                  smooth_slit,
    double                  smooth_spec,
    int                     niter,
    double                  kappa,
    double                  error_factor,
    cpl_vector          **  slit_func,
    cpl_bivector        **  spec,
    hdrl_image          **  model)

{
    double          *   ycen_rest;
    double          *   ycen_sw;
    int             *   ycen_offset_sw;
    double          *   slitfu_sw_data;
    
    double          *   model_sw;
    const double    *   slit_func_in;
    int             *   mask_sw;
    const cpl_image *   img_in;
    const cpl_image *   err_in;
    cpl_image       *   img_sw;
    cpl_image       *   err_sw;
    cpl_image       *   img_rect;
    cpl_image       *   err_rect;
    cpl_image       *   model_rect;
    cpl_vector      *   ycen ;
    cpl_image       *   img_out;
    cpl_vector      *   slitfu_sw;
    cpl_vector      *   unc_sw;
    cpl_vector      *   spc;
    cpl_vector      *   slitfu;
    cpl_vector      *   weights_sw;
    cpl_vector      *   tmp_vec;
    cpl_vector      *   bins_begin;
    cpl_vector      *   bins_end;
    cpl_vector      *   unc_decomposition;
    cpl_size            lenx, leny, pow;
    cpl_type            imtyp;
    cpl_polynomial      *slitcurve_A, *slitcurve_B, *slitcurve_C;
    cpl_polynomial  **  slitcurves_sw;
    hdrl_image      *   model_out;
    cpl_bivector    *   spectrum_loc;
    double          *   sP_old;
    double          *   l_Aij;
    double          *   p_Aij;
    double          *   l_bj;
    double          *   p_bj;
    cpl_image       *   img_mad;
    xi_ref          *   xi;
    zeta_ref        *   zeta;
    int             *   m_zeta;
    char            *   path;
    double              pixval, errval;
    double              trace_cen, trace_height;
    int                 i, j, k, nswaths, col, x, y, ny_os,
                        badpix, delta_x;
    int                 ny, nx;
  

    /* Check Entries */
    if (img_hdrl == NULL || trace_tab == NULL) return -1 ;

    if (smooth_slit == 0.0) {
        cpl_msg_error(__func__, "Slit-smoothing cannot be 0.0");
        return -1;
    } else if (smooth_slit < 0.1) {
        cpl_msg_warning(__func__, "Slit-smoothing unreasonably small");
    } else if (smooth_slit > 100.0) {
        cpl_msg_warning(__func__, "Slit-smoothing unreasonably big");
    }
    if (oversample < 3){
        cpl_msg_error(__func__, "Oversampling too small");
        return -1;
    } else if (oversample > 15) {
        cpl_msg_warning(__func__, "Large oversampling, runtime will be long");
    }
    if (niter < 5){
        cpl_msg_warning(__func__,
                "Allowing at least 5 iterations is recommended.");
    }
    if (kappa < 4){
        cpl_msg_warning(__func__,
                "Rejecting outliers < 4 sigma risks making good data.");
    }
    img_in = hdrl_image_get_image_const(img_hdrl);
    err_in = hdrl_image_get_error_const(img_hdrl);

    /* Initialise */
    imtyp = cpl_image_get_type(img_in);
    lenx = cpl_image_get_size_x(img_in);
    leny = cpl_image_get_size_y(img_in);
   
    /* Compute height if not given */
    if (height <= 0) {
        height = cr2res_trace_get_height(trace_tab, order, trace_id);
        if (height <= 0) {
            cpl_msg_error(__func__, "Cannot compute height");
            return -1;
        }
    }
    if (height > leny){
        height = leny;
        cpl_msg_warning(__func__,
                "Given height larger than image, clipping height");
    }

    /* Get ycen */
    if ((ycen = cr2res_trace_get_ycen(trace_tab, order,
                    trace_id, lenx)) == NULL) {
        cpl_msg_error(__func__, "Cannot get ycen");
        return -1 ;
    }
    trace_cen = cpl_vector_get(ycen, cpl_vector_get_size(ycen)/2) ;
    trace_height = (double)cr2res_trace_get_height(trace_tab, order, trace_id) ;
    cpl_msg_info(__func__, "Y position of the trace: %g -> %g", 
            trace_cen-(trace_height/2), trace_cen+(trace_height/2)) ;
    if (trace_cen-(height/2) < 0.0 || 
                trace_cen+(height/2) > CR2RES_DETECTOR_SIZE) {
        cpl_msg_error(__func__, "Extraction outside detector edges impossible");
        cpl_vector_delete(ycen);
        return -1;
    }

    // Get cut-out rectified order
    img_rect = cr2res_image_cut_rectify(img_in, ycen, height);
    if (img_rect == NULL){
        cpl_msg_error(__func__, "Cannot rectify order");
        cpl_vector_delete(ycen);
        return -1;
    }
    if (cpl_msg_get_level() == CPL_MSG_DEBUG) {
        cpl_image_save(img_rect, "debug_rectorder_curved.fits", imtyp,
                NULL, CPL_IO_CREATE);
    }
    err_rect = cr2res_image_cut_rectify(err_in, ycen, height);
    ycen_rest = cr2res_vector_get_rest(ycen);

    /* Retrieve the polynomials that describe the slit tilt and curvature*/
    slitcurve_A = cr2res_get_trace_wave_poly(trace_tab, CR2RES_COL_SLIT_CURV_A,
                    order, trace_id);
    slitcurve_B = cr2res_get_trace_wave_poly(trace_tab, CR2RES_COL_SLIT_CURV_B,
                    order, trace_id);
    slitcurve_C = cr2res_get_trace_wave_poly(trace_tab, CR2RES_COL_SLIT_CURV_C,
                    order, trace_id);
    if ((slitcurve_A == NULL) || (slitcurve_B == NULL) || (slitcurve_C == NULL))
    {
        cpl_msg_error(__func__, 
                "No (or incomplete) slitcurve data found in trace table");
        cpl_vector_delete(ycen);
        cpl_free(ycen_rest) ;
        cpl_image_delete(err_rect) ;
        cpl_image_delete(img_rect) ;
        cpl_polynomial_delete(slitcurve_A);
        cpl_polynomial_delete(slitcurve_B);
        cpl_polynomial_delete(slitcurve_C);
        return -1;
    }

    /* Maximum horizontal shift in detector pixels due to slit image curv. */
    delta_x=0;
    for (i=1; i<=lenx; i+=swath/2){
        double delta_tmp, a, b, c, yc;
        /* Do a coarse sweep through the order and evaluate the slitcurve */
        /* polynomials at  +- height/2, update the value. */
        /* Note: The index i is subtracted from a because the polys have */
        /* their origin at the edge of the full frame */
        //a = cpl_polynomial_eval_1d(slitcurve_A, i, NULL); this is ignored apparently?
        b = cpl_polynomial_eval_1d(slitcurve_B, i, NULL);
        c = cpl_polynomial_eval_1d(slitcurve_C, i, NULL);
        yc = cpl_vector_get(ycen, i-1);

        // Shift polynomial to local frame
        // We fix a to 0, see comment later on, when we create the
        // polynomials for the extraction
        a = 0; 
        b += 2 * yc * c;

        delta_tmp = max( fabs(a + (c*height/2. + b)*height/2.),
                fabs(a + (c*height/-2. + b)*height/-2.));
        if (delta_tmp > delta_x) delta_x = (int)ceil(delta_tmp);
    }
    delta_x += 1;
    cpl_msg_debug(__func__, "Max delta_x from slit curv: %d pix.", delta_x);

    if (delta_x >= swath / 4){
        cpl_msg_error(__func__, 
            "Curvature is larger than the swath, try again with a larger swath size");
        cpl_vector_delete(ycen);
        cpl_free(ycen_rest) ;
        cpl_image_delete(err_rect) ;
        cpl_image_delete(img_rect) ;
        cpl_polynomial_delete(slitcurve_A);
        cpl_polynomial_delete(slitcurve_B);
        cpl_polynomial_delete(slitcurve_C);
        return -1;
    }

    /* Number of rows after oversampling */
    ny_os = oversample*(height+1) +1;
    if ((swath = cr2res_extract_slitdec_adjust_swath(ycen, height, leny, swath, 
                    lenx, delta_x, &bins_begin, &bins_end)) == -1){
        cpl_msg_error(__func__, "Cannot calculate swath size");
        cpl_vector_delete(ycen);
        cpl_free(ycen_rest) ;
        cpl_image_delete(err_rect) ;
        cpl_image_delete(img_rect) ;
        cpl_polynomial_delete(slitcurve_A);
        cpl_polynomial_delete(slitcurve_B);
        cpl_polynomial_delete(slitcurve_C);
        return -1;
    }
    nswaths = cpl_vector_get_size(bins_begin);

    /* Use existing slitfunction if given */
    slit_func_in = NULL;
    if (slit_func_vec_in != NULL) {
        cpl_size size;
        size = cpl_vector_get_size(slit_func_vec_in);
        if (size == ny_os){
            slit_func_in = cpl_vector_get_data_const(slit_func_vec_in);
        } else {
            cpl_msg_warning(__func__, "Ignoring the given slit_func since it is"
                " of the wrong size, expected %i but got %lli points.",
                ny_os, size);
        }
    }
   
    /* Allocate */
    mask_sw = cpl_malloc(height * swath*sizeof(int));
    model_sw = cpl_malloc(height * swath*sizeof(double));
    unc_sw = cpl_vector_new(swath);
    img_sw = cpl_image_new(swath, height, CPL_TYPE_DOUBLE);
    err_sw = cpl_image_new(swath, height, CPL_TYPE_DOUBLE);
    ycen_sw = cpl_malloc(swath*sizeof(double));
    ycen_offset_sw = cpl_malloc(swath * sizeof(int));

    slitcurves_sw = cpl_malloc(swath * sizeof(cpl_polynomial*));
    for (i=0; i<swath; i++) slitcurves_sw[i]= cpl_polynomial_new(1);

    // Local versions of return data
    slitfu = cpl_vector_new(ny_os);
    spectrum_loc = cpl_bivector_new(lenx);
    spc = cpl_bivector_get_x(spectrum_loc);
    unc_decomposition = cpl_bivector_get_y(spectrum_loc);
    for (j=0; j<lenx ; j++){
        cpl_vector_set(spc, j, 0.);
        cpl_vector_set(unc_decomposition, j, 0.);
    }
    model_out = hdrl_image_new(lenx, leny);
    img_out = hdrl_image_get_image(model_out);
    model_rect = cpl_image_new(lenx, height, CPL_TYPE_DOUBLE);

    // Work vectors
    slitfu_sw = cpl_vector_new(ny_os);
    for (j=0; j < ny_os; j++) cpl_vector_set(slitfu_sw, j, 0);
    slitfu_sw_data = cpl_vector_get_data(slitfu_sw);
    weights_sw = cpl_vector_new(swath);
    for (i = 0; i < swath; i++) cpl_vector_set(weights_sw, i, 0);

    /* Pre-calculate the weights for overlapping swaths*/
    for (i=delta_x; i < swath/2; i++) {
        j = i - delta_x + 1;
        cpl_vector_set(weights_sw, i, j);
        cpl_vector_set(weights_sw, swath - i - 1, j);
    }
    // normalize such that max(w)=1
    cpl_vector_divide_scalar(weights_sw, swath/2 - delta_x + 1);

    // assert cpl_vector_get_sum(weights_sw) == swath / 2 - delta_x
    // Assign memory for extract_curved algorithm
    // Since the arrays always have the same size, we can reuse allocated memory
    ny = oversample * (height + 1) + 1;
    nx = 4 * delta_x + 1;
    if(nx < 3) nx = 3;

    sP_old = cpl_malloc(swath * sizeof(double));
    l_Aij  = cpl_malloc(ny * (4*oversample+1) * sizeof(double));
    p_Aij  = cpl_malloc(swath * nx * sizeof(double));
    l_bj   = cpl_malloc(ny * sizeof(double));
    p_bj   = cpl_malloc(swath * sizeof(double));
    img_mad = cpl_image_new(swath, height, CPL_TYPE_DOUBLE);

    /*
       Convolution tensor telling the coordinates of detector pixels on which
       {x, iy} element falls and the corresponding projections. [ncols][ny][4]
    */
    xi = cpl_malloc(swath * ny * 4 * sizeof(xi_ref));

    /* Convolution tensor telling the coordinates of subpixels {x, iy}
       contributing to detector pixel {x, y}. [ncols][nrows][3*(osample+1)]
    */
    zeta = cpl_malloc(swath * height * 3 * (oversample + 1)
                                    * sizeof(zeta_ref));

    /* The actual number of contributing elements in zeta  [ncols][nrows]  */
    m_zeta = cpl_malloc(swath * height * sizeof(int));

    for (i = 0; i < nswaths; i++) {
        double *img_sw_data;
        double *err_sw_data;
        double *spec_sw_data;
        double *unc_sw_data;

        cpl_image *img_tmp;
        cpl_vector *spec_sw;
        cpl_vector *spec_tmp;

        int sw_start, sw_end, y_lower_limit;

        double img_sum;

        sw_start = cpl_vector_get(bins_begin, i);
        sw_end = cpl_vector_get(bins_end, i);

        /* Prepare swath cut-outs and auxiliary data */
        for(col=1; col<=swath; col++){   // col is x-index in swath
            x = sw_start + col;          // coords in large image

            /* prepare signal, error and mask */
            for(y=1;y<=height;y++){
                errval = cpl_image_get(err_rect, x, y, &badpix);
                if (isnan(errval) | badpix){
                    // default to errval of 1 instead of 0
                    // this avoids division by 0
                    errval = 1;
                }
                pixval = cpl_image_get(img_rect, x, y, &badpix);
                if (isnan(pixval) | badpix){
                    // We set bad pixels to neg. infinity, to make sure they are
                    // rejected in the extraction
                    // The algorithm does not like NANs!
                    badpix = 1;
                    pixval = -DBL_MAX;
                    errval = 1;
                } 
                cpl_image_set(img_sw, col, y, pixval);
                cpl_image_set(err_sw, col, y, errval);
                if (badpix){
                    // Reject the pixel here, so it is not used for the initial
                    // guess of the spectrum
                    cpl_image_reject(img_sw, col, y);
                }
                
                // raw index for mask, start with 0!
                j = (y-1)*swath + (col-1) ;
                // The mask value is inverted for the extraction
                // 1 for good pixel and 0 for bad pixel
                mask_sw[j] = !badpix;
            }

            /* set slit curvature polynomials */
            /* subtract col because we want origin relative to here */
            pow = 2;
            cpl_polynomial_set_coeff(slitcurves_sw[col-1], &pow,
                cpl_polynomial_eval_1d(slitcurve_C, x, NULL));
            pow = 1;
            cpl_polynomial_set_coeff(slitcurves_sw[col-1], &pow,
                cpl_polynomial_eval_1d(slitcurve_B, x, NULL));
            pow = 0;
            cpl_polynomial_set_coeff(slitcurves_sw[col-1], &pow,
                cpl_polynomial_eval_1d(slitcurve_A, x, NULL) - x);

            // Shift polynomial to local frame
            // -------------------------------
            // The slit curvature has been determined in the global reference
            // frame, with the a coefficient set to 0 in the local frame.
            // The following transformation will shift it into the local frame
            // again and should result in a = 0.
            //      a - x + yc * b + yc * yc * c
            // However this only works, as long as ycen
            // is the same ycen that was used for the slitcurvature. If e.g. we
            // switch traces, then ycen will change and a will be unequal 0.
            // in fact a will be the offset due to the curvature between the
            // old ycen and the new. This will then cause an offset in the
            // pixels used for the extraction, so that all traces will have the
            // same spectrum, with no relative offsets.
            // Which would be great, if we didn't have an offset in the
            // wavelength calibration of the different traces.
            // Therefore we force a to be 0 in the local frame regardless of
            // ycen. For the extraction we only need the b and c coefficient
            // anyways.
            // Note that this means, we use the curvature a few pixels offset.
            // Usually this is no problem, since it only varies slowly over the
            // order.
            cpl_polynomial_shift_1d(slitcurves_sw[col-1], 0,
                                            cpl_vector_get(ycen, x-1));
            cpl_polynomial_set_coeff(slitcurves_sw[col-1], &pow, 0);
        }

        for (j=0; j< height * swath; j++) model_sw[j] = 0;
        img_sw_data = cpl_image_get_data_double(img_sw);
        err_sw_data = cpl_image_get_data_double(err_sw);
        unc_sw_data = cpl_vector_get_data(unc_sw);      
        // First guess for the spectrum
        // img_tmp = cpl_image_collapse_median_create(img_sw, 0, 0, 0);
        img_tmp = cpl_image_collapse_median_create(img_sw, 0, 0, 0);
        spec_tmp = cpl_vector_new_from_image_row(img_tmp, 1);
        cpl_vector_multiply_scalar(spec_tmp, 
                            (double)cpl_image_get_size_y(img_sw));
        spec_sw = cpl_vector_filter_median_create(spec_tmp, 1);
        cpl_vector_delete(spec_tmp);
        cpl_image_delete(img_tmp);
        spec_sw_data = cpl_vector_get_data(spec_sw);

        for (j=sw_start;j<sw_end;j++){
            ycen_sw[j-sw_start] = ycen_rest[j];
            ycen_offset_sw[j-sw_start] = (int) cpl_vector_get(ycen, j);
        }
        y_lower_limit = height / 2;

        img_tmp = cpl_image_wrap_int(swath, height, mask_sw);
        img_sum = cpl_image_get_flux(img_tmp);
        cpl_msg_debug(__func__, "img_sum = %.0f, swath = %d, height = %d", img_sum, swath, height);
        if (img_sum < 0.5 * swath*height){
            cpl_msg_error(__func__,
                    "Only %.0f %% of pixels not masked, cannot extract",
                    100*img_sum/(swath*height));
            cpl_image_unwrap(img_tmp);
            cpl_vector_delete(spec_sw);
            break;
        }
        if (cpl_msg_get_level() == CPL_MSG_DEBUG)
        {
            cpl_image_save(img_tmp, "debug_mask_before_sw.fits", CPL_TYPE_INT, NULL, CPL_IO_CREATE);
            cpl_vector_save(spec_sw, "debug_spc_initial_guess.fits",
                    CPL_TYPE_DOUBLE, NULL, CPL_IO_CREATE);
        }
        cpl_image_unwrap(img_tmp);
        
        /* Finally ready to call the slit-decomp */
        cr2res_extract_slit_func_curved(error_factor, swath, height, oversample, 
                img_sw_data, err_sw_data, mask_sw, ycen_sw, ycen_offset_sw, 
                y_lower_limit, slitcurves_sw, delta_x, slitfu_sw_data, 
                spec_sw_data, model_sw, unc_sw_data, smooth_spec, smooth_slit, 
                5.e-5, niter, kappa, slit_func_in, sP_old, l_Aij, p_Aij, l_bj, 
                p_bj, img_mad, xi, zeta, m_zeta);

        // add up slit-functions, divide by nswaths below to get average
        if (i==0) cpl_vector_copy(slitfu,slitfu_sw);
        else cpl_vector_add(slitfu,slitfu_sw);

        if (cpl_msg_get_level() == CPL_MSG_DEBUG) {
            path = cpl_sprintf("debug_spc_%i.fits", i);
            cpl_vector_save(spec_sw, path , CPL_TYPE_DOUBLE, NULL,
                    CPL_IO_CREATE);
            cpl_free(path);

            path = cpl_sprintf("debug_mask_%i.fits", i);
            img_tmp = cpl_image_wrap_int(swath, height, mask_sw);
            cpl_image_save(img_tmp, path, CPL_TYPE_INT, NULL, CPL_IO_CREATE);
            cpl_free(path);
            cpl_image_unwrap(img_tmp);

            tmp_vec = cpl_vector_wrap(swath, ycen_sw);
            path = cpl_sprintf("debug_ycen_%i.fits", i);
            cpl_vector_save(tmp_vec, path, CPL_TYPE_DOUBLE, NULL,
                    CPL_IO_CREATE);
            cpl_vector_unwrap(tmp_vec);
            cpl_free(path);

            cpl_vector_save(weights_sw, "debug_weights.fits", CPL_TYPE_DOUBLE,
                    NULL, CPL_IO_CREATE);
            path = cpl_sprintf("debug_slitfu_%i.fits", i);
            cpl_vector_save(slitfu_sw, path, CPL_TYPE_DOUBLE,
                    NULL, CPL_IO_CREATE);
            cpl_free(path);

            path = cpl_sprintf("debug_model_%i.fits", i);
            img_tmp = cpl_image_wrap_double(swath, height, model_sw);
            cpl_image_save(img_tmp, path, CPL_TYPE_DOUBLE,
                    NULL, CPL_IO_CREATE);
            cpl_image_unwrap(img_tmp);
            cpl_free(path);

            path = cpl_sprintf("debug_img_sw_%i.fits", i);
            cpl_image_save(img_sw, path, CPL_TYPE_DOUBLE, NULL,
                    CPL_IO_CREATE);
            cpl_free(path);

            path = cpl_sprintf("debug_img_mad_%i.fits", i);
            cpl_image_save(img_mad, path,  CPL_TYPE_DOUBLE, NULL,
                    CPL_IO_CREATE);
            cpl_free(path);
        }

        // The last bins are shifted, overwriting the first k values
        // this is the same amount the bin was shifted to the front before
        // (when creating the bins)
        // The duplicate values in the vector will not matter as they are
        // not used below
        if ((i == nswaths - 1) && (i != 0)){
            k = cpl_vector_get(bins_end, i-1) -
                cpl_vector_get(bins_begin, i) - swath / 2 - delta_x;

            for (j = 0; j < swath - k; j++){
                cpl_vector_set(spec_sw, j, cpl_vector_get(spec_sw, j + k));
                cpl_vector_set(unc_sw, j, cpl_vector_get(unc_sw, j + k));
                for (y = 0; y < height; y++)
                    model_sw[y * swath + j] = model_sw[y * swath + j + k];
            }
            sw_start = cpl_vector_get(bins_begin, i-1) + swath / 2 - delta_x;
            cpl_vector_set(bins_begin, i, sw_start);
            // for the following k's
            cpl_vector_set(bins_end, i, lenx);
        }

        if (nswaths==1) ; // no weighting if only one swath 
        else if (i==0){ // first and last half swath are not weighted
            for (j = 0; j < delta_x; j++)
            {
                cpl_vector_set(spec_sw, j, 0);
                cpl_vector_set(unc_sw, j, 0.);
                for (y = 0; y < height; y++) model_sw[y * swath + j] = 0;
            }
            for (j = swath/2; j < swath; j++) {
                cpl_vector_set(spec_sw, j,
                    cpl_vector_get(spec_sw,j) * cpl_vector_get(weights_sw,j));
                cpl_vector_set(unc_sw, j,
                    cpl_vector_get(unc_sw, j) * cpl_vector_get(weights_sw, j));
                for (y = 0; y < height; y++) {
                    model_sw[y * swath + j] *= cpl_vector_get(weights_sw, j);
                }
            }
        } else if (i == nswaths - 1) {
            for (j = sw_end-sw_start-1; j >= sw_end-sw_start-delta_x-1; j--)
            {
                cpl_vector_set(spec_sw, j, 0);
                cpl_vector_set(unc_sw, j, 0);
                for (y = 0; y < height; y++) model_sw[y * swath + j] = 0;
            }
            for (j = 0; j < swath / 2; j++) {
                cpl_vector_set(spec_sw, j,
                    cpl_vector_get(spec_sw,j) * cpl_vector_get(weights_sw,j));
                cpl_vector_set(unc_sw, j,
                    cpl_vector_get(unc_sw,j) * cpl_vector_get(weights_sw,j));
                for (y = 0; y < height; y++) {
                    model_sw[y * swath + j] *= cpl_vector_get(weights_sw,j);
                }
            }
        } else {
            /* Multiply by weights and add to output array */
            cpl_vector_multiply(spec_sw, weights_sw);
            cpl_vector_multiply(unc_sw, weights_sw);
            for (y = 0; y < height; y++) {
                for (j = 0; j < swath; j++){
                    model_sw[y * swath + j] *= cpl_vector_get(weights_sw,j);
                }
            }
        }

        if (cpl_msg_get_level() == CPL_MSG_DEBUG) {
            img_tmp = cpl_image_wrap_double(swath, height, model_sw);
            cpl_image_save(img_tmp, "debug_model_after_sw.fits", CPL_TYPE_DOUBLE, 
                    NULL, CPL_IO_CREATE);
            cpl_image_unwrap(img_tmp);
        }

        // Save swath to output vector
        for (j=sw_start;j<sw_end;j++) {
            cpl_vector_set(spc, j,
                cpl_vector_get(spec_sw, j-sw_start) + cpl_vector_get(spc, j));
            // just add weighted errors (instead of squared sum)
            // as they are not independent
            cpl_vector_set(unc_decomposition, j, 
                cpl_vector_get(unc_sw, j - sw_start)
                + cpl_vector_get(unc_decomposition, j));

            for(y = 0; y < height; y++){
                cpl_image_set(model_rect, j+1, y+1, 
                    cpl_image_get(model_rect, j+1, y+1, &badpix)
                    + model_sw[y * swath + j - sw_start]);
                if (badpix) cpl_image_reject(model_rect, j+1, y+1);
            }
        }

        if (cpl_msg_get_level() == CPL_MSG_DEBUG) {
            cpl_image_save(model_rect, "debug_model_after_merge.fits",
                CPL_TYPE_DOUBLE, NULL, CPL_IO_CREATE);
        }

        cpl_vector_delete(spec_sw);
    }  // End loop over swaths

    // divide by nswaths to make the slitfu into the average over all swaths.
    cpl_vector_divide_scalar(slitfu, nswaths);

    // Deallocate loop memory
    cpl_image_delete(img_mad);
    cpl_free(sP_old);
    cpl_free(l_Aij);
    cpl_free(p_Aij);
    cpl_free(l_bj);
    cpl_free(p_bj);

    cpl_free(xi);
    cpl_free(zeta);
    cpl_free(m_zeta);

    cpl_image_delete(img_rect);
    cpl_image_delete(err_rect);
    cpl_image_delete(img_sw);
    cpl_image_delete(err_sw);

    cpl_free(mask_sw);
    cpl_free(model_sw);
    cpl_vector_delete(unc_sw);
    cpl_free(ycen_rest);
    cpl_free(ycen_sw);
    cpl_free(ycen_offset_sw);

    cpl_vector_delete(bins_begin);
    cpl_vector_delete(bins_end);
    cpl_vector_delete(slitfu_sw);
    cpl_vector_delete(weights_sw);

    cpl_polynomial_delete(slitcurve_A);
    cpl_polynomial_delete(slitcurve_B);
    cpl_polynomial_delete(slitcurve_C);
    for (i=0; i<swath; i++) cpl_polynomial_delete(slitcurves_sw[i]);
    cpl_free(slitcurves_sw);

    // insert model_rect into large frame
    if (cr2res_image_insert_rect(model_rect, ycen, img_out) == -1) {
        // Cancel
        cpl_msg_error(__func__, "failed to reinsert model swath into model image");
        cpl_image_delete(model_rect);
        hdrl_image_delete(model_out);
        cpl_vector_delete(ycen);
        cpl_bivector_delete(spectrum_loc);
        cpl_vector_delete(slitfu);
        return -1; 
    }

    if (cpl_msg_get_level() == CPL_MSG_DEBUG) {
        cpl_image_save(model_rect, "debug_model_rect.fits", CPL_TYPE_DOUBLE,
                NULL, CPL_IO_CREATE);
        cpl_image_save(img_out, "debug_model_all.fits", CPL_TYPE_DOUBLE,
                NULL, CPL_IO_CREATE);
        cpl_vector_save(spc, "debug_spc_all.fits", CPL_TYPE_DOUBLE,
                NULL, CPL_IO_CREATE);
    }

    cpl_image_delete(model_rect);
    cpl_vector_delete(ycen);

    if (cpl_error_get_code() != CPL_ERROR_NONE){
        cpl_msg_error(__func__, 
            "Something went wrong in the extraction. Error Code: %i, loc: %s", 
            cpl_error_get_code(), cpl_error_get_where());
        cpl_error_reset();
        cpl_vector_delete(slitfu);
        cpl_bivector_delete(spectrum_loc);
        hdrl_image_delete(model_out);
        return -1;
    }

    *slit_func = slitfu;
    *spec = spectrum_loc;
    *model = model_out;
    return 0;
}