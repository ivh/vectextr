
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>

/* Logging level constants (replacing CPL_MSG_*) */
#define MSG_DEBUG 0
#define MSG_INFO 1
#define MSG_WARNING 2
#define MSG_ERROR 3

/* Current message level */
static int current_msg_level = MSG_WARNING;

typedef unsigned char byte;
#define min(a,b) (((a)<(b))?(a):(b))
#define max(a,b) (((a)>(b))?(a):(b))
#define signum(a) (((a)>0)?1:((a)<0)?-1:0)
#define zeta_index(x, y, z) (z * ncols * nrows) + (y * ncols) + x
#define mzeta_index(x, y) (y * ncols) + x
#define xi_index(x, y, z) (z * ncols * ny) + (y * ncols) + x

// Alternative indexing, probably faster
//#define zeta_index(x, y, z) (x * nrows + y) * 3*(osample+1) + z
//#define mzeta_index(x, y) (x * nrows) + y
//#define xi_index(x, y, z) (x * ny + y) * 4 + z


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


static int debug_output( 
    int         ncols,
    int         nrows,
    int         osample,
    double  *   im,
    double  *   pix_unc,
    int     *   mask,
    double  *   ycen,
    int     *   ycen_offset,
    int         y_lower_lim,
    double  *   slitdeltas)
{
    FILE *fp;
    int i, j;
    
    /* Write image as text file */
    fp = fopen("debug_image_at_error.txt", "w");
    if (fp) {
        fprintf(fp, "# Debug image: %d x %d\n", ncols, nrows);
        fprintf(fp, "# Properties:\n");
        fprintf(fp, "# osample = %d\n", osample);
        fprintf(fp, "# y_lower_lim = %d\n", y_lower_lim);
        fprintf(fp, "# Data:\n");
        for (j = 0; j < nrows; j++) {
            for (i = 0; i < ncols; i++) {
                fprintf(fp, "%g ", im[j * ncols + i]);
            }
            fprintf(fp, "\n");
        }
        fclose(fp);
    }
    
    /* Write mask as text file */
    fp = fopen("debug_mask_after_error.txt", "w");
    if (fp) {
        fprintf(fp, "# Debug mask: %d x %d\n", ncols, nrows);
        fprintf(fp, "# Data:\n");
        for (j = 0; j < nrows; j++) {
            for (i = 0; i < ncols; i++) {
                fprintf(fp, "%d ", mask[j * ncols + i]);
            }
            fprintf(fp, "\n");
        }
        fclose(fp);
    }
    
    /* Write uncertainty as text file */
    fp = fopen("debug_unc_at_error.txt", "w");
    if (fp) {
        fprintf(fp, "# Debug uncertainty: %d x %d\n", ncols, nrows);
        fprintf(fp, "# Data:\n");
        for (j = 0; j < nrows; j++) {
            for (i = 0; i < ncols; i++) {
                fprintf(fp, "%g ", pix_unc[j * ncols + i]);
            }
            fprintf(fp, "\n");
        }
        fclose(fp);
    }
    
    /* Write ycen as text file */
    fp = fopen("debug_ycen_after_error.txt", "w");
    if (fp) {
        fprintf(fp, "# Debug ycen: %d\n", ncols);
        fprintf(fp, "# Data:\n");
        for (i = 0; i < ncols; i++) {
            fprintf(fp, "%g\n", ycen[i]);
        }
        fclose(fp);
    }
    
    /* Write offset as text file */
    fp = fopen("debug_offset_after_error.txt", "w");
    if (fp) {
        fprintf(fp, "# Debug offset: %d\n", ncols);
        fprintf(fp, "# Data:\n");
        for (i = 0; i < ncols; i++) {
            fprintf(fp, "%d\n", ycen_offset[i]);
        }
        fclose(fp);
    }
    
    /* Uncomment if needed
    fp = fopen("debug_slitcurves_at_error.txt", "w");
    if (fp) {
        fprintf(fp, "# Debug slitcurves: %d x %d\n", ncols, 3);
        fprintf(fp, "# Data:\n");
        for (j = 0; j < 3; j++) {
            for (i = 0; i < ncols; i++) {
                fprintf(fp, "%g ", slitdeltas[i*osample+j]);
            }
            fprintf(fp, "\n");
        }
        fclose(fp);
    }
    */
    
    return 0;
}


/*----------------------------------------------------------------------------*/
/**
  @brief    Solve a sparse system of linear equations
  @param    a   2D array [n,nd]i
  @param    r   array of RHS of size n
  @param    n   number of equations
  @param    nd  width of the band (3 for tri-diagonal system)
  @return   0 on success, -1 on incorrect size of "a" and -4 on
            degenerate matrix

  Solve a sparse system of linear equations with band-diagonal matrix.
  Band is assumed to be symmetric relative to the main diagonal.

  nd must be an odd number. The main diagonal should be in a(*,nd/2)
  The first lower subdiagonal should be in a(1:n-1,nd/2-1), the first
  upper subdiagonal is in a(0:n-2,nd/2+1) etc. For example:
                    / 0 0 X X X \
                    | 0 X X X X |
                    | X X X X X |
                    | X X X X X |
              A =   | X X X X X |
                    | X X X X X |
                    | X X X X X |
                    | X X X X 0 |
                    \ X X X 0 0 /
 */
/*----------------------------------------------------------------------------*/
int bandsol(
    double  *   a,
    double  *   r,
    int         n,
    int         nd,
    double      lambda)
{
double aa;
int i, j, k;

//if(fmod(nd,2)==0) return -1;

/* Forward sweep */
for(i=0; i<n-1; i++)
{
    aa=a[i+n*(nd/2)];
    if(aa==0.e0) aa = lambda; //return -3;
    r[i]/=aa;
    for(j=0; j<nd; j++) a[i+j*n]/=aa;
    for(j=1; j<min(nd/2+1,n-i); j++)
    {
        aa=a[i+j+n*(nd/2-j)];
        r[i+j]-=r[i]*aa;
        for(k=0; k<n*(nd-j); k+=n) a[i+j+k]-=a[i+k+n*j]*aa;
    }
}

/* Backward sweep */
aa = a[n-1+n*(nd/2)];
if (aa == 0) aa = lambda; //return -4;
r[n-1]/=aa;
for(i=n-1; i>0; i--)
{
    for(j=1; j<=min(nd/2,i); j++){
        r[i-j]-=r[i]*a[i-j+n*(nd/2+j)];
    }
    aa = a[i-1+n*(nd/2)];
    if(aa==0.e0) aa = lambda; //return -5;
    
    r[i-1]/=aa;
}

aa = a[n*(nd/2)];
if(aa==0.e0) aa = lambda; //return -6;
r[0]/=aa;
return 0;
}

/*----------------------------------------------------------------------------*/
/**
  @brief    Helper function for extract
  @param ncols          Swath width in pixels
  @param nrows          Extraction slit height in pixels
  @param ny             Size of the slit function array: ny=osample(nrows+1)+1
  @param ycen           Order centre line offset from pixel row boundary
  @param ycen_offset    Order image column shift
  @param y_lower_lim    Number of detector pixels below the pixel
                        containing the central line yc
  @param osample        Subpixel oversampling factor
  @param PSF_curve      Parabolic fit to the slit image curvature
                        For column d_x = PSF_curve[ncols][0] +
                                        PSF_curve[ncols][1] *d_y +
                                        PSF_curve[ncols][2] *d_y^2
                        where d_y is the offset from the central line ycen.
                        Thus central subpixel of omega[x][y'][delta_x][iy']
                        does not stick out of column x
  @param xi[ncols][ny][4]   Convolution tensor telling the coordinates
                            of detector pixels on which {x, iy} element
                            falls and the corresponding projections
  @param zeta[ncols][nrows][3 * (osample + 1)]
                        Convolution tensor telling the coordinates
                        of subpixels {x, iy} contributing to detector pixel
                        {x, y}
  @param m_zeta[ncols][nrows]
                        The actual number of contributing elements in zeta

  @return
 */
/*----------------------------------------------------------------------------*/
int xi_zeta_tensors(
        int         ncols,
        int         nrows,
        int         ny,
        double  *   ycen,
        const int     *   ycen_offset,
        int         y_lower_lim,
        int         osample,
        double  *   slitdeltas,
        xi_ref   *  xi,
        zeta_ref *  zeta,
        int      *  m_zeta)
{
    int x, xx, y, yy, ix, ix1, ix2, iy, m;
    double step, delta, w;
    step = 1.e0 / osample;

    /* Clean xi */
    for (x = 0; x < ncols; x++)
    {
        for (iy = 0; iy < ny; iy++)
        {
            for (m = 0; m < 4; m++)
            {
                xi[xi_index(x, iy, m)].x = -1;
                xi[xi_index(x, iy, m)].y = -1;
                xi[xi_index(x, iy, m)].w = 0.;
            }
        }
    }

    /* Clean zeta */
    for (x = 0; x < ncols; x++)
    {
        for (y = 0; y < nrows; y++)
        {
            m_zeta[mzeta_index(x, y)] = 0;
            for (ix = 0; ix < 3 * (osample + 1); ix++)
            {
                zeta[zeta_index(x, y, ix)].x = -1;
                zeta[zeta_index(x, y, ix)].iy = -1;
                zeta[zeta_index(x, y, ix)].w = 0.;
            }
        }
    }

    /*
    Construct the xi and zeta tensors. They contain pixel references and contribution. 
    values going from a given subpixel to other pixels (xi) and coming from other subpixels
    to a given detector pixel (zeta).
    Note, that xi and zeta are used in the equations for sL, sP and for the model but they
    do not involve the data, only the geometry. Thus it can be pre-computed once.
    */
    for (x = 0; x < ncols; x++)
    {
        int iy1, iy2;
        double d1, d2, dy;
        /*
        I promised to reconsider the initial offset. Here it is. For the original layout
        (no column shifts and discontinuities in ycen) there is pixel y that contains the
        central line yc. There are two options here (by construction of ycen that can be 0
        but cannot be 1): (1) yc is inside pixel y and (2) yc falls at the boundary between
        pixels y and y-1. yc cannot be at the boundary of pixels y+1 and y because we would
        select y+1 to be pixel y in that case.

        Next we need to define starting and ending indices iy for sL subpixels that contribute
        to pixel y. I call them iy1 and iy2. For both cases we assume osample+1 subpixels covering
        pixel y (weird). So for case 1 iy1 will be (y-1)*osample and iy2 == y*osample. Special
        treatment of the boundary subpixels will compensate for introducing extra subpixel in
        case 1. In case 2 things are more logical: iy1=(yc-y)*osample+(y-1)*osample;
        iy2=(y+1-yc)*osample)+(y-1)*osample. ycen is yc-y making things simpler. Note also that
        the same pattern repeats for all rows: we only need to initialize iy1 and iy2 and keep
        incrementing them by osample. 
        */

        iy2 = osample - floor(ycen[x] * osample);
        iy1 = iy2 - osample;

        /*
        Handling partial subpixels cut by detector pixel rows is again tricky. Here we have three
        cases (mostly because of the decision to assume that we always have osample+1 subpixels
        per one detector pixel). Here d1 is the fraction of the subpixel iy1 inside detector pixel y.
        d2 is then the fraction of subpixel iy2 inside detector pixel y. By definition d1+d2==step.
        Case 1: ycen falls on the top boundary of each detector pixel (ycen == 1). Here we conclude
                that the first subpixel is fully contained inside pixel y and d1 is set to step.
        Case 2: ycen falls on the bottom boundary of each detector pixel (ycen == 0). Here we conclude
                that the first subpixel is totally outside of pixel y and d1 is set to 0.
        Case 3: ycen falls inside of each pixel (0>ycen>1). In this case d1 is set to the fraction of
                the first step contained inside of each pixel.
        And BTW, this also means that central line coincides with the upper boundary of subpixel iy2
        when the y loop reaches pixel y_lower_lim. In other words:

        dy=(iy-(y_lower_lim+ycen[x])*osample)*step-0.5*step
        */

        d1 = fmod(ycen[x], step);
        if (d1 == 0)
            d1 = step;
        d2 = step - d1;

        /*
        The final hurdle for 2D slit decomposition is to construct two 3D reference tensors. We proceed
        similar to 1D case except that now each iy subpixel can be shifted left or right following
        the curvature of the slit image on the detector. We assume for now that each subpixel is
        exactly 1 detector pixel wide. This may not be exactly true if the curvature changes across
        the focal plane but will deal with it when the necessity will become apparent. For now we
        just assume that a shift delta the weight w assigned to subpixel iy is divided between
        ix1=int(delta) and ix2=int(delta)+signum(delta) as (1-|delta-ix1|)*w and |delta-ix1|*w.

        The curvature is given by a quadratic polynomial evaluated from an approximation for column
        x: delta = PSF_curve[x][0] + PSF_curve[x][1] * (y-yc[x]) + PSF_curve[x][2] * (y-yc[x])^2.
        It looks easy except that y and yc are set in the global detector coordinate system rather than
        in the shifted and cropped swath passed to slit_func_2d. One possible solution I will try here
        is to modify PSF_curve before the call such as:
        delta = PSF_curve'[x][0] + PSF_curve'[x][1] * (y'-ycen[x]) + PSF_curve'[x][2] * (y'-ycen[x])^2
        where y' = y - floor(yc).
        */

        /* Define initial distance from ycen       */
        /* It is given by the center of the first  */
        /* subpixel falling into pixel y_lower_lim */
        dy = ycen[x] - floor((y_lower_lim + ycen[x]) / step) * step - step;

        /*
        Now we go detector pixels x and y incrementing subpixels looking for their contributions
        to the current and adjacent pixels. Note that the curvature/tilt of the projected slit
        image could be so large that subpixel iy may no contribute to column x at all. On the
        other hand, subpixels around ycen by definition must contribute to pixel x,y. 
        3rd index in xi refers corners of pixel xx,y: 0:LL, 1:LR, 2:UL, 3:UR.
        */
        for (y = 0; y < nrows; y++) {
            iy1 += osample; // Bottom subpixel falling in row y
            iy2 += osample; // Top subpixel falling in row y
            dy -= step;
            for (iy = iy1; iy <= iy2; iy++) {
                if (iy == iy1)      w = d1;
                else if (iy == iy2) w = d2;
                else                w = step;
                dy += step;
                delta = slitdeltas[x];
                ix1 = delta;
                ix2 = ix1 + signum(delta);

                /* Three cases: subpixel on the bottom boundary of row y, intermediate subpixels and top boundary */

                if (iy == iy1) /* Case A: Subpixel iy is entering detector row y */
                {
                    if (ix1 < ix2) /* Subpixel iy shifts to the right from column x  */
                    {
                        if (x + ix1 >= 0 && x + ix2 < ncols)
                        {
                            xx = x + ix1; /* Upper right corner of subpixel iy */
                            yy = y + ycen_offset[x] - ycen_offset[xx];
                            xi[xi_index(x, iy, 3)].x = xx;
                            xi[xi_index(x, iy, 3)].y = yy;
                            xi[xi_index(x, iy, 3)].w = w - fabs(delta - ix1) * w;
                            // xx>=0 && xx<ncols is already given by the loop condition
                            if (xx < ncols && yy >= 0 && yy < nrows && xi[xi_index(x, iy, 3)].w > 0)
                            {
                                m = m_zeta[mzeta_index(xx, yy)];
                                zeta[zeta_index(xx, yy, m)].x = x;
                                zeta[zeta_index(xx, yy, m)].iy = iy;
                                zeta[zeta_index(xx, yy, m)].w = xi[xi_index(x, iy, 3)].w;
                                m_zeta[mzeta_index(xx, yy)]++;
                            }
                            xx = x + ix2; /* Upper left corner of subpixel iy */
                            // This offset is required because the iy subpixel
                            // is going to contribute to the yy row in xx column
                            // of detector pixels where yy and y are in the same
                            // row. In the packed array this is not necessarily true.
                            // Instead, what we know is that:
                            // y+ycen_offset[x] == yy+ycen_offset[xx]
                            yy = y + ycen_offset[x] - ycen_offset[xx];

                            xi[xi_index(x, iy, 2)].x = xx;
                            xi[xi_index(x, iy, 2)].y = yy;
                            xi[xi_index(x, iy, 2)].w = fabs(delta - ix1) * w;
                            if (xx >= 0 && yy >= 0 && yy < nrows && xi[xi_index(x, iy, 2)].w > 0)
                            {
                                m = m_zeta[mzeta_index(xx, yy)];
                                zeta[zeta_index(xx, yy, m)].x = x;
                                zeta[zeta_index(xx, yy, m)].iy = iy;
                                zeta[zeta_index(xx, yy, m)].w = xi[xi_index(x, iy, 2)].w;
                                m_zeta[mzeta_index(xx, yy)]++;
                            }
                        }
                    }
                    else if (ix1 > ix2) /* Subpixel iy shifts to the left from column x */
                    {
                        if (x + ix2 >= 0 && x + ix1 < ncols)
                        {
                            xx = x + ix2; /* Upper left corner of subpixel iy */
                            yy = y + ycen_offset[x] - ycen_offset[xx];
                            xi[xi_index(x, iy, 2)].x = xx;
                            xi[xi_index(x, iy, 2)].y = yy;
                            xi[xi_index(x, iy, 2)].w = fabs(delta - ix1) * w;
                            if (xx < ncols && yy >= 0 && yy < nrows && xi[xi_index(x, iy, 2)].w > 0)
                            {
                                m = m_zeta[mzeta_index(xx, yy)];
                                zeta[zeta_index(xx, yy, m)].x = x;
                                zeta[zeta_index(xx, yy, m)].iy = iy;
                                zeta[zeta_index(xx, yy, m)].w = xi[xi_index(x, iy, 2)].w;
                                m_zeta[mzeta_index(xx, yy)]++;
                            }
                            xx = x + ix1; /* Upper right corner of subpixel iy */
                            yy = y + ycen_offset[x] - ycen_offset[xx];
                            xi[xi_index(x, iy, 3)].x = xx;
                            xi[xi_index(x, iy, 3)].y = yy;
                            xi[xi_index(x, iy, 3)].w = w - fabs(delta - ix1) * w;
                            if (xx >= 0 && yy >= 0 && yy < nrows && xi[xi_index(x, iy, 3)].w > 0)
                            {
                                m = m_zeta[mzeta_index(xx, yy)];
                                zeta[zeta_index(xx, yy, m)].x = x;
                                zeta[zeta_index(xx, yy, m)].iy = iy;
                                zeta[zeta_index(xx, yy, m)].w = xi[xi_index(x, iy, 3)].w;
                                m_zeta[mzeta_index(xx, yy)]++;
                            }
                        }
                    }
                    else
                    {
                        if (x + ix1 >= 0 && x + ix1 < ncols)
                        {
                            xx = x + ix1; /* Subpixel iy stays inside column x */
                            yy = y + ycen_offset[x] - ycen_offset[xx];
                            xi[xi_index(x, iy, 2)].x = xx;
                            xi[xi_index(x, iy, 2)].y = yy;
                            xi[xi_index(x, iy, 2)].w = w;
                            if (yy >= 0 && yy < nrows && w > 0)
                            {
                                m = m_zeta[mzeta_index(xx, yy)];
                                zeta[zeta_index(xx, yy, m)].x = x;
                                zeta[zeta_index(xx, yy, m)].iy = iy;
                                zeta[zeta_index(xx, yy, m)].w = w;
                                m_zeta[mzeta_index(xx, yy)]++;
                            }
                        }
                    }
                }
                else if (iy == iy2) /* Case C: Subpixel iy is leaving detector row y */
                {
                    if (ix1 < ix2) /* Subpixel iy shifts to the right from column x */
                    {
                        if (x + ix1 >= 0 && x + ix2 < ncols)
                        {
                            xx = x + ix1; /* Bottom right corner of subpixel iy */
                            yy = y + ycen_offset[x] - ycen_offset[xx];
                            xi[xi_index(x, iy, 1)].x = xx;
                            xi[xi_index(x, iy, 1)].y = yy;
                            xi[xi_index(x, iy, 1)].w = w - fabs(delta - ix1) * w;
                            if (xx < ncols && yy >= 0 && yy < nrows && xi[xi_index(x, iy, 1)].w > 0)
                            {
                                m = m_zeta[mzeta_index(xx, yy)];
                                zeta[zeta_index(xx, yy, m)].x = x;
                                zeta[zeta_index(xx, yy, m)].iy = iy;
                                zeta[zeta_index(xx, yy, m)].w = xi[xi_index(x, iy, 1)].w;
                                m_zeta[mzeta_index(xx, yy)]++;
                            }
                            xx = x + ix2; /* Bottom left corner of subpixel iy */
                            yy = y + ycen_offset[x] - ycen_offset[xx];
                            xi[xi_index(x, iy, 0)].x = xx;
                            xi[xi_index(x, iy, 0)].y = yy;
                            xi[xi_index(x, iy, 0)].w = fabs(delta - ix1) * w;
                            if (xx >= 0 && yy >= 0 && yy < nrows && xi[xi_index(x, iy, 0)].w > 0)
                            {
                                m = m_zeta[mzeta_index(xx, yy)];
                                zeta[zeta_index(xx, yy, m)].x = x;
                                zeta[zeta_index(xx, yy, m)].iy = iy;
                                zeta[zeta_index(xx, yy, m)].w = xi[xi_index(x, iy, 0)].w;
                                m_zeta[mzeta_index(xx, yy)]++;
                            }
                        }
                    }
                    else if (ix1 > ix2) /* Subpixel iy shifts to the left from column x */
                    {
                        if (x + ix2 >= 0 && x + ix1 < ncols)
                        {
                            xx = x + ix2; /* Bottom left corner of subpixel iy */
                            yy = y + ycen_offset[x] - ycen_offset[xx];
                            xi[xi_index(x, iy, 0)].x = xx;
                            xi[xi_index(x, iy, 0)].y = yy;
                            xi[xi_index(x, iy, 0)].w = fabs(delta - ix1) * w;
                            if (xx < ncols && yy >= 0 && yy < nrows && xi[xi_index(x, iy, 0)].w > 0)
                            {
                                m = m_zeta[mzeta_index(xx, yy)];
                                zeta[zeta_index(xx, yy, m)].x = x;
                                zeta[zeta_index(xx, yy, m)].iy = iy;
                                zeta[zeta_index(xx, yy, m)].w = xi[xi_index(x, iy, 0)].w;
                                m_zeta[mzeta_index(xx, yy)]++;
                            }
                            xx = x + ix1; /* Bottom right corner of subpixel iy */
                            yy = y + ycen_offset[x] - ycen_offset[xx];
                            xi[xi_index(x, iy, 1)].x = xx;
                            xi[xi_index(x, iy, 1)].y = yy;
                            xi[xi_index(x, iy, 1)].w = w - fabs(delta - ix1) * w;
                            if (xx >= 0 && yy >= 0 && yy < nrows && xi[xi_index(x, iy, 1)].w > 0)
                            {
                                m = m_zeta[mzeta_index(xx, yy)];
                                zeta[zeta_index(xx, yy, m)].x = x;
                                zeta[zeta_index(xx, yy, m)].iy = iy;
                                zeta[zeta_index(xx, yy, m)].w = xi[xi_index(x, iy, 1)].w;
                                m_zeta[mzeta_index(xx, yy)]++;
                            }
                        }
                    }
                    else /* Subpixel iy stays inside column x        */
                    {
                        if (x + ix1 >= 0 && x + ix1 < ncols)
                        {
                            xx = x + ix1;
                            yy = y + ycen_offset[x] - ycen_offset[xx];
                            xi[xi_index(x, iy, 0)].x = xx;
                            xi[xi_index(x, iy, 0)].y = yy;
                            xi[xi_index(x, iy, 0)].w = w;
                            if (yy >= 0 && yy < nrows && w > 0)
                            {
                                m = m_zeta[mzeta_index(xx, yy)];
                                zeta[zeta_index(xx, yy, m)].x = x;
                                zeta[zeta_index(xx, yy, m)].iy = iy;
                                zeta[zeta_index(xx, yy, m)].w = w;
                                m_zeta[mzeta_index(xx, yy)]++;
                            }
                        }
                    }
                }
                else /* CASE B: Subpixel iy is fully inside detector row y */
                {
                    if (ix1 < ix2) /* Subpixel iy shifts to the right from column x      */
                    {
                        if (x + ix1 >= 0 && x + ix2 < ncols)
                        {
                            xx = x + ix1; /* Bottom right corner of subpixel iy */
                            yy = y + ycen_offset[x] - ycen_offset[xx];
                            xi[xi_index(x, iy, 1)].x = xx;
                            xi[xi_index(x, iy, 1)].y = yy;
                            xi[xi_index(x, iy, 1)].w = w - fabs(delta - ix1) * w;
                            if (xx < ncols && yy >= 0 && yy < nrows && xi[xi_index(x, iy, 1)].w > 0)
                            {
                                m = m_zeta[mzeta_index(xx, yy)];
                                zeta[zeta_index(xx, yy, m)].x = x;
                                zeta[zeta_index(xx, yy, m)].iy = iy;
                                zeta[zeta_index(xx, yy, m)].w = xi[xi_index(x, iy, 1)].w;
                                m_zeta[mzeta_index(xx, yy)]++;
                            }
                            xx = x + ix2; /* Bottom left corner of subpixel iy */
                            yy = y + ycen_offset[x] - ycen_offset[xx];
                            xi[xi_index(x, iy, 0)].x = xx;
                            xi[xi_index(x, iy, 0)].y = yy;
                            xi[xi_index(x, iy, 0)].w = fabs(delta - ix1) * w;
                            if (xx >= 0 && yy >= 0 && yy < nrows && xi[xi_index(x, iy, 0)].w > 0)
                            {
                                m = m_zeta[mzeta_index(xx, yy)];
                                zeta[zeta_index(xx, yy, m)].x = x;
                                zeta[zeta_index(xx, yy, m)].iy = iy;
                                zeta[zeta_index(xx, yy, m)].w = xi[xi_index(x, iy, 0)].w;
                                m_zeta[mzeta_index(xx, yy)]++;
                            }
                        }
                    }
                    else if (ix1 > ix2) /* Subpixel iy shifts to the left from column x */
                    {
                        if (x + ix2 >= 0 && x + ix1 < ncols)
                        {
                            xx = x + ix2; /* Bottom right corner of subpixel iy */
                            yy = y + ycen_offset[x] - ycen_offset[xx];
                            xi[xi_index(x, iy, 1)].x = xx;
                            xi[xi_index(x, iy, 1)].y = yy;
                            xi[xi_index(x, iy, 1)].w = fabs(delta - ix1) * w;
                            if (xx < ncols && yy >= 0 && yy < nrows && xi[xi_index(x, iy, 1)].w > 0)
                            {
                                m = m_zeta[mzeta_index(xx, yy)];
                                zeta[zeta_index(xx, yy, m)].x = x;
                                zeta[zeta_index(xx, yy, m)].iy = iy;
                                zeta[zeta_index(xx, yy, m)].w = xi[xi_index(x, iy, 1)].w;
                                m_zeta[mzeta_index(xx, yy)]++;
                            }
                            xx = x + ix1; /* Bottom left corner of subpixel iy */
                            yy = y + ycen_offset[x] - ycen_offset[xx];
                            xi[xi_index(x, iy, 0)].x = xx;
                            xi[xi_index(x, iy, 0)].y = yy;
                            xi[xi_index(x, iy, 0)].w = w - fabs(delta - ix1) * w;
                            if (xx >= 0 && yy >= 0 && yy < nrows && xi[xi_index(x, iy, 0)].w > 0)
                            {
                                m = m_zeta[mzeta_index(xx, yy)];
                                zeta[zeta_index(xx, yy, m)].x = x;
                                zeta[zeta_index(xx, yy, m)].iy = iy;
                                zeta[zeta_index(xx, yy, m)].w = xi[xi_index(x, iy, 0)].w;
                                m_zeta[mzeta_index(xx, yy)]++;
                            }
                        }
                    }
                    else /* Subpixel iy stays inside column x */
                    {
                        if (x + ix2 >= 0 && x + ix2 < ncols)
                        {
                            xx = x + ix2;
                            yy = y + ycen_offset[x] - ycen_offset[xx];
                            xi[xi_index(x, iy, 0)].x = xx;
                            xi[xi_index(x, iy, 0)].y = yy;
                            xi[xi_index(x, iy, 0)].w = w;
                            if (yy >= 0 && yy < nrows && w > 0)
                            {
                                m = m_zeta[mzeta_index(xx, yy)];
                                zeta[zeta_index(xx, yy, m)].x = x;
                                zeta[zeta_index(xx, yy, m)].iy = iy;
                                zeta[zeta_index(xx, yy, m)].w = w;
                                m_zeta[mzeta_index(xx, yy)]++;
                            }
                        }
                    }
                }
            }
        }
    }
    return 0;
}

/*----------------------------------------------------------------------------*/
/**
  @brief    Slit decomposition of single swath with slit tilt & curvature
  @param error_factor Factor for error scaling
  @param ncols      Swath width in pixels
  @param nrows      Extraction slit height in pixels
  @param osample    Subpixel oversampling factor
  @param im         Image to be decomposed [nrows][ncols]
  @param pix_unc
  @param mask       Initial and final mask for the swath [nrows][ncols]
  @param ycen       Order centre line offset from pixel row boundary [ncols]
  @param ycen_offset    Order image column shift     [ncols]
  @param y_lower_lim    Number of detector pixels below the pixel containing
                        the central line yc
  @param PSF_curve  Slit curvature
  @param delta_x    Maximum horizontal shift in detector pixels due to slit
                    image curvature
  @param sL         Slit function resulting from decomposition    [ny]
  @param sP         Spectrum resulting from decomposition      [ncols]
  @param model      Model constructed from sp and sf
  @param unc        Spectrum uncertainties based on data - model [ncols]
  @param lambda_sP  Smoothing parameter for the spectrum, could be zero
  @param lambda_sL  Smoothing parameter for the slit function, usually>0
  @param sP_stop
  @param maxiter
  @return
 */
/*----------------------------------------------------------------------------*/
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
        int       *  m_zeta)
{
    int x, xx, xxx, y, yy, iy, jy, n, m, ny, nx;
    double norm, lambda, diag_tot, ww, www, sP_change, sP_med;
    double tmp, sLmax, sum;
    int info, iter;

    /* The size of the sL array. */
    /* Extra osample is because ycen can be between 0 and 1. */
    ny = osample * (nrows + 1) + 1;
    nx = 4 * delta_x + 1;
    if (nx < 3)
        nx = 3;

    xi_zeta_tensors(ncols, nrows, ny, ycen, ycen_offset,
                                   y_lower_lim, osample, slitdeltas, xi, zeta,
                                   m_zeta);

    // If a slit func is given, use that instead of recalculating it
    if (slit_func_in != NULL) {
        // Normalize the input just in case
        norm = 0.e0;
        for (iy = 0; iy < ny; iy++) {
            sL[iy] = slit_func_in[iy];
            norm += sL[iy];
        }
        norm /= osample;
        for (iy = 0; iy < ny; iy++)
            sL[iy] /= norm;
    }

    /* Resetting the mask and img values for outliers and NaN */
    /*    for (y = 0; y < nrows; y++) {
     for (x = 0; x < ncols; x++) {
       mask[y * ncols + x] = 1;
       if(im[y * ncols + x] < -1.e3) {
         mask[y * ncols + x] = 0;
         //im[y * ncols + x] = 0.e0;
       }
     }
    }
*/

    /* Loop through sL , sP reconstruction until convergence is reached */
    iter = 0;
    // cost = 0; Not used without cost_old?
    do {
        //cost_old = cost; this is not used apparently?
        double cost, sigma;
        int isum;
        if (slit_func_in == NULL) {
            /* Compute slit function sL */
            /* Prepare the RHS and the matrix */
            for (iy = 0; iy < ny; iy++) {
                l_bj[iy] = 0.e0;
                /* Clean RHS                */
                for (jy = 0; jy <= 4 * osample; jy++)
                    l_Aij[iy + ny * jy] = 0.e0;
            }
            /* Fill in SLE arrays for slit function */
            diag_tot = 0.e0;
            for (iy = 0; iy < ny; iy++) {
                for (x = 0; x < ncols; x++) {
                    for (n = 0; n < 4; n++) {
                        ww = xi[xi_index(x, iy, n)].w;
                        if (ww > 0) {
                            xx = xi[xi_index(x, iy, n)].x;
                            yy = xi[xi_index(x, iy, n)].y;
                            if (xx >= 0 && xx < ncols && yy >= 0 &&
                                yy < nrows) {
                                if (m_zeta[mzeta_index(xx, yy)] > 0) {
                                    for (m = 0; m < m_zeta[mzeta_index(xx, yy)];
                                         m++) {
                                        xxx = zeta[zeta_index(xx, yy, m)].x;
                                        jy = zeta[zeta_index(xx, yy, m)].iy;
                                        www = zeta[zeta_index(xx, yy, m)].w;
                                        if (jy - iy + 2 * osample >= 0)
                                            l_Aij[iy + ny * (jy - iy +
                                                             2 * osample)] +=
                                                sP[xxx] * sP[x] * www * ww *
                                                mask[yy * ncols + xx];
                                    }
                                    l_bj[iy] += im[yy * ncols + xx] *
                                                mask[yy * ncols + xx] * sP[x] *
                                                ww;
                                }
                            }
                        }
                    }
                }
                diag_tot += fabs(l_Aij[iy + ny * 2 * osample]);
            }
            /* Scale regularization parameters */
            lambda = lambda_sL * diag_tot / ny;
            /* Add regularization parts for the SLE matrix */
            /* Main diagonal  */
            l_Aij[ny * 2 * osample] += lambda;
            /* Upper diagonal */
            l_Aij[ny * (2 * osample + 1)] -= lambda;
            for (iy = 1; iy < ny - 1; iy++) {
                /* Lower diagonal */
                l_Aij[iy + ny * (2 * osample - 1)] -= lambda;
                /* Main diagonal  */
                l_Aij[iy + ny * 2 * osample] += lambda * 2.e0;
                /* Upper diagonal */
                l_Aij[iy + ny * (2 * osample + 1)] -= lambda;
            }
            /* Lower diagonal */
            l_Aij[ny - 1 + ny * (2 * osample - 1)] -= lambda;
            /* Main diagonal  */
            l_Aij[ny - 1 + ny * 2 * osample] += lambda;

            /* Solve the system of equations */
            info = bandsol(l_Aij, l_bj, ny,
                                                  4 * osample + 1, lambda);
            if (info && current_msg_level <= MSG_ERROR) {
                fprintf(stderr, "ERROR (%s): info(sL)=%d\n", __func__, info);
            }

            /* Normalize the slit function */
            norm = 0.e0;
            for (iy = 0; iy < ny; iy++) {
                sL[iy] = l_bj[iy];
                norm += fabs(sL[iy]);
            }
            norm /= osample;
            for (iy = 0; iy < ny; iy++)
                sL[iy] /= norm;
        }

        /*  Compute spectrum sP */
        for (x = 0; x < ncols; x++) {
            for (xx = 0; xx < nx; xx++)
                p_Aij[xx * ncols + x] = 0.;
            p_bj[x] = 0;
        }
        for (x = 0; x < ncols; x++) {
            for (iy = 0; iy < ny; iy++) {
                for (n = 0; n < 4; n++) {
                    ww = xi[xi_index(x, iy, n)].w;
                    if (ww > 0) {
                        xx = xi[xi_index(x, iy, n)].x;
                        yy = xi[xi_index(x, iy, n)].y;
                        if (xx >= 0 && xx < ncols && yy >= 0 && yy < nrows) {
                            if (m_zeta[mzeta_index(xx, yy)] > 0) {
                                for (m = 0; m < m_zeta[mzeta_index(xx, yy)];
                                     m++) {
                                    xxx = zeta[zeta_index(xx, yy, m)].x;
                                    jy = zeta[zeta_index(xx, yy, m)].iy;
                                    www = zeta[zeta_index(xx, yy, m)].w;
                                    p_Aij[x +
                                          ncols * (xxx - x + 2 * delta_x)] +=
                                        sL[jy] * sL[iy] * www * ww *
                                        mask[yy * ncols + xx];
                                }
                                p_bj[x] += im[yy * ncols + xx] *
                                           mask[yy * ncols + xx] * sL[iy] * ww;
                            }
                        }
                    }
                }
            }
        }

        /* Save the previous iteration spectrum */
        for (x = 0; x < ncols; x++)
            sP_old[x] = sP[x];

        lambda = 1;
        if (lambda_sP > 0.e0) {
            lambda = lambda_sP; /* Scale regularization parameter */
            p_Aij[ncols * (2 * delta_x)] += lambda;     /* Main diagonal  */
            p_Aij[ncols * (2 * delta_x + 1)] -= lambda; /* Upper diagonal */
            for (x = 1; x < ncols - 1; x++) {
                /* Lower diagonal */
                p_Aij[x + ncols * (2 * delta_x - 1)] -= lambda;
                /* Main diagonal  */
                p_Aij[x + ncols * (2 * delta_x)] += lambda * 2.e0;
                /* Upper diagonal */
                p_Aij[x + ncols * (2 * delta_x + 1)] -= lambda;
            }
            /* Lower diagonal */
            p_Aij[ncols - 1 + ncols * (2 * delta_x - 1)] -= lambda;
            /* Main diagonal  */
            p_Aij[ncols - 1 + ncols * (2 * delta_x)] += lambda;
        }

        /* Solve the system of equations */
        info = bandsol(p_Aij, p_bj, ncols, nx, lambda);
        if (info && current_msg_level <= MSG_ERROR) {
            fprintf(stderr, "ERROR (%s): info(sP)=%d\n", __func__, info);
        }

        for (x = 0; x < ncols; x++)
            sP[x] = p_bj[x]; /* New Spectrum vector */


        /* Compute median value of the spectrum for normalisation purpose */
        /* First copy and take absolute values */
        double *sP_copy = malloc(ncols * sizeof(double));
        if (sP_copy == NULL) {
            fprintf(stderr, "ERROR (%s): Memory allocation failed\n", __func__);
            return -1;
        }
        
        for (x = 0; x < ncols; x++) {
            sP_copy[x] = fabs(sP[x]);
        }
        
        /* Sort the array */
        for (int i = 0; i < ncols-1; i++) {
            for (int j = 0; j < ncols-i-1; j++) {
                if (sP_copy[j] > sP_copy[j+1]) {
                    double temp = sP_copy[j];
                    sP_copy[j] = sP_copy[j+1];
                    sP_copy[j+1] = temp;
                }
            }
        }
        
        /* Get the median */
        if (ncols % 2 == 0) {
            sP_med = (sP_copy[ncols/2-1] + sP_copy[ncols/2]) / 2.0;
        } else {
            sP_med = sP_copy[ncols/2];
        }
        
        free(sP_copy);

        /* Compute the change in the spectrum */
        sP_change = 0.e0;
        for (x = 0; x < ncols; x++) {
            if (fabs(sP[x] - sP_old[x]) > sP_change)
                sP_change = fabs(sP[x] - sP_old[x]);
        }

        if (isnan(sP[0]) || (sP[ncols / 2] == 0)) {
            if (current_msg_level <= MSG_DEBUG) {
                debug_output(ncols, nrows, osample, im, pix_unc, mask, ycen,
                             ycen_offset, y_lower_lim, slitdeltas);
                fprintf(stderr, "ERROR (%s): Swath failed\n", __func__);
            }
        }

        /* Compute the model */
        for (y = 0; y < nrows * ncols; y++) {
            model[y] = 0.;
        }
        for (y = 0; y < nrows; y++) {
            for (x = 0; x < ncols; x++) {
                for (m = 0; m < m_zeta[mzeta_index(x, y)]; m++) {
                    xx = zeta[zeta_index(x, y, m)].x;
                    iy = zeta[zeta_index(x, y, m)].iy;
                    ww = zeta[zeta_index(x, y, m)].w;
                    model[y * ncols + x] += sP[xx] * sL[iy] * ww;
                }
            }
        }
        /* Compare model and data */
        // We use a simple standard deviation here (which is NOT robust to
        // outliers), since it is less strict than a more robust measurement
        // (e.g. MAD) would be. Initial problems in the guess will be more
        // easily be fixed this way. We would mask them away otherwise.
        // On the other hand the std may get to large and might fail to remove
        // outliers sufficiently in some circumstances.

        cost = 0.e0;
        sum = 0.e0;
        isum = 0;
        for (y = 0; y < nrows; y++) {
            for (x = delta_x; x < ncols - delta_x; x++) {
                if (mask[y * ncols + x]) {
                    tmp = model[y * ncols + x] - im[y * ncols + x];
                    sum += tmp * tmp;
                    tmp /= max(pix_unc[y * ncols + x], 1);
                    cost += tmp * tmp;
                    isum++;
                }
            }
        }
        cost /= (isum - (ncols + ny));
        sigma = sqrt(sum / isum);

        /* Adjust the mask marking outliers */
        for (y = 0; y < nrows; y++) {
            for (x = delta_x; x < ncols - delta_x; x++) {
                if (fabs(model[y * ncols + x] - im[y * ncols + x]) >
                    kappa * sigma)
                    mask[y * ncols + x] = 0;
                else
                    mask[y * ncols + x] = 1;
            }
        }

        for (y = 0; y < nrows; y++) {
            for (x = delta_x; x < ncols - delta_x; x++) {
                img_mad[y * ncols + x] = (model[y * ncols + x] - im[y * ncols + x]);
                img_mad_mask[y * ncols + x] = (mask[y * ncols + x] != 0) && (im[y * ncols + x] != 0);
            }
        }

        if (current_msg_level <= MSG_DEBUG) {
            fprintf(stderr, "DEBUG (%s): Iter: %i, Sigma: %f, Cost: %f, sP_change: %f, sP_lim: %f\n", 
                __func__, iter, sigma, cost, sP_change, sP_stop * sP_med);
        }

        iter++;
    } while (iter == 1 ||
             (iter <= maxiter
              //                      && fabs(cost - cost_old) > sP_stop));
              && sP_change > sP_stop * sP_med));

    if (iter == maxiter && sP_change > sP_stop * sP_med) {
        if (current_msg_level <= MSG_WARNING) {
            fprintf(stderr, "WARNING (%s): Maximum number of %d iterations reached without converging.\n",
                __func__, maxiter);
        }
    }

    /* Flip sign if converged in negative direction */
    sum = 0.0;
    for (y = 0; y < ny; y++)
        sum += sL[y];
    if (sum < 0.0) {
        for (y = 0; y < ny; y++)
            sL[y] *= -1.0;
        for (x = 0; x < ncols; x++)
            sP[x] *= -1.0;
        sum *= -1.0;
    }
    /* Find max value in sL */
    sLmax = sL[0];
    for (y = 1; y < ny; y++) {
        if (sL[y] > sLmax) sLmax = sL[y];
    }
    
    if (current_msg_level <= MSG_DEBUG) {
        fprintf(stderr, "DEBUG (%s): sL-sum, sLmax, osample, nrows, ny: %g, %g, %d, %d, %d\n",
            __func__, sum, sLmax, osample, nrows, ny);
    }


    /*
        for (x = 0; x < ncols; x++) {
            double msum;
            
            unc[x] = 0.0;
            msum = 0.0;
            sum = 0.0;
            for (y = 0; y < nrows; y++) {
                if (mask[y * ncols + x]) {
                    msum += (im[y * ncols + x] * model[y * ncols + x]) *
                            mask[y * ncols + x];
                    sum += (model[y * ncols + x] * model[y * ncols + x]) *
                           mask[y * ncols + x];
                }
            }
            if (msum != 0){
                // This can give NaNs if m/sum is less than zero, i.e. low/no flux
                // due to ignoring background flux.
                unc[x] = sqrt(fabs(sP[x]) * fabs(sum) / fabs(msum) / error_factor);
            } else {
                // Fix bad value to NaN as Phase3 doesn't allow Inf.
                unc[x] = NAN;
            }
        }
    */
    if (error_factor == -1)
    {
        // Uncertainty calculation, following Horne 1986.
        for (x = 0; x < ncols; x++)
        {
            double num_sum;
            double den_sum;
            double model_sum = 0.0;

            unc[x] = 0.0;
            num_sum = 0.0;
            den_sum = 0.0;
            for (y = 0; y < nrows; y++)
            {
                model_sum += model[y * ncols + x];
            }
            for (y = 0; y < nrows; y++)
            {
                double model_norm = model[y * ncols + x] / model_sum;
                num_sum += model_norm * mask[y * ncols + x];
                den_sum += (model_norm * model_norm) * mask[y * ncols + x] / (pix_unc[y * ncols + x] * pix_unc[y * ncols + x]);
            }
            if (den_sum != 0)
            {
                unc[x] = sqrt(fabs(num_sum / den_sum));
            }
            else
            {
                unc[x] = NAN;
            }
        }
    }
    else
    {
        // Uncertainty calculation only using total object flux, needs later correction.
        for (x = 0; x < ncols; x++)
        {
            double msum;

            unc[x] = 0.0;
            msum = 0.0;
            sum = 0.0;
            for (y = 0; y < nrows; y++)
            {
                if (mask[y * ncols + x])
                {
                    msum += (im[y * ncols + x] * model[y * ncols + x]) *
                            mask[y * ncols + x];
                    sum += (model[y * ncols + x] * model[y * ncols + x]) *
                           mask[y * ncols + x];
                }
            }
            if (msum != 0)
            {
                // This can give NaNs if m/sum is less than zero, i.e. low/no flux
                // due to ignoring background flux.
                unc[x] = sqrt(fabs(sP[x]) * fabs(sum) / fabs(msum) / error_factor);
            }
            else
            {
                // Fix bad value to NaN as Phase3 doesn't allow Inf.
                unc[x] = NAN;
            }
        }
    }
    return 0;
}
