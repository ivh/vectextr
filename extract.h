#ifndef EXTRACT_H
#define EXTRACT_H

/* Type definitions needed for function declarations */
typedef struct {
    int     x ;
    int     y ;     /* Coordinates of target pixel x,y  */
    double  w ;     /* Contribution weight <= 1/osample */
} xi_ref;

typedef struct {
    int     x ;
    int     iy ;    /* Contributing subpixel  x,iy      */
    double  w;      /* Contribution weight <= 1/osample */
} zeta_ref;

/* Helper macro for min function if not already defined elsewhere */
#ifndef min
#define min(a,b) (((a)<(b))?(a):(b))
#endif

/* Function declarations */
int xi_zeta_tensors(
        int         ncols,
        int         nrows,
        int         ny,
        double  *   ycen,
        const int   *   ycen_offset,
        int         y_lower_lim,
        int         osample,
        double  *   slitdeltas,
        xi_ref   *  xi,
        zeta_ref *  zeta,
        int      *  m_zeta);

int bandsol(
    double  *   a,
    double  *   r,
    int         n,
    int         nd,
    double      lambda);

int extract(
        double      error_factor,
        int         ncols,
        int         nrows,
        int         osample,
        double  *   im,
        double  *   pix_unc,
        int     *   mask,
        double  *   ycen,
        int     *   ycen_offset,
        int         y_lower_lim,
        double  *   slitdeltas,
        int         delta_x,
        double  *   sL,
        double  *   sP,
        double  *   model,
        double  *   unc,
        double      lambda_sP,
        double      lambda_sL,
        double      sP_stop,
        int         maxiter,
        double      kappa,
        const double  *   slit_func_in,
        double    *  sP_old,
        double    *  l_Aij,
        double    *  p_Aij,
        double    *  l_bj,
        double    *  p_bj,
        double    *  img_mad,
        int       *  img_mad_mask,
        xi_ref    *  xi,
        zeta_ref  *  zeta,
        int       *  m_zeta);

#endif /* EXTRACT_H */
