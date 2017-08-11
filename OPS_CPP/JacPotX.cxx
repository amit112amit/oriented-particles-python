
def Dphi_pDxi(vi, xi, xj):
    u0,u1,u2 = vi
    x0,x1,x2 = xi
    y0,y1,y2 = xj
    
    vi_mag_sqr = u0*u0 + u1*u1 + u2*u2
    vi_mag = sqrt(vi_mag_sqr)
    sin_alpha_i = sin( 0.5*vi_mag )    
    cos_alpha_i = cos( 0.5*vi_mag )
        
    Dphi_pDxi0 = (-
        4*cos_alpha_i*sin_alpha_i*u1/vi_mag -
         4*sin_alpha_i_2*u0*u2/vi_mag_sqr)*((-x0 +
         y0)*(2*cos_alpha_i*sin_alpha_i*u1/vi_mag +
         2*sin_alpha_i_2*u0*u2/vi_mag_sqr) + (-x1 + y1)*(-
        2*cos_alpha_i*sin_alpha_i*u0/vi_mag +
         2*sin_alpha_i_2*u1*u2/vi_mag_sqr) + (-x2 + y2)*(cos_alpha_i_2 -
         sin_alpha_i_2*u0*u0/vi_mag_sqr - sin_alpha_i_2*u1*u1/vi_mag_sqr +
         sin_alpha_i_2*u2*u2/vi_mag_sqr))

    Dphi_pDxi1 = (4*cos_alpha_i
        *sin_alpha_i*u0/vi_mag - 4*sin_alpha_i_2*u1*u2/vi_mag_sqr)*((-x0 +
         y0)*(2*cos_alpha_i*sin_alpha_i*u1/vi_mag +
         2*sin_alpha_i_2*u0*u2/vi_mag_sqr) + (-x1 + y1)*(-
        2*cos_alpha_i*sin_alpha_i*u0/vi_mag +
         2*sin_alpha_i_2*u1*u2/vi_mag_sqr) + (-x2 + y2)*(cos_alpha_i_2 -
         sin_alpha_i_2*u0*u0/vi_mag_sqr - sin_alpha_i_2*u1*u1/vi_mag_sqr +
         sin_alpha_i_2*u2*u2/vi_mag_sqr))

    Dphi_pDxi2 = ((-x0 +
         y0)*(2*cos_alpha_i*sin_alpha_i*u1/vi_mag +
         2*sin_alpha_i_2*u0*u2/vi_mag_sqr) + (-x1 + y1)*(-
        2*cos_alpha_i*sin_alpha_i*u0/vi_mag +
         2*sin_alpha_i_2*u1*u2/vi_mag_sqr) + (-x2 + y2)*(cos_alpha_i_2 -
         sin_alpha_i_2*u0*u0/vi_mag_sqr - sin_alpha_i_2*u1*u1/vi_mag_sqr +
         sin_alpha_i_2*u2*u2/vi_mag_sqr))*(-2*cos_alpha_i_2 +
         2*sin_alpha_i_2*u0*u0/vi_mag_sqr + 2*sin_alpha_i_2*u1*u1/vi_mag_sqr -
         2*sin_alpha_i_2*u2*u2/vi_mag_sqr)
        
    return array([Dphi_pDxi0,Dphi_pDxi1,Dphi_pDxi2])


def Dphi_pDxj(vi, xi, xj):
    u0,u1,u2 = vi
    x0,x1,x2 = xi
    y0,y1,y2 = xj
    
    vi_mag_sqr = u0*u0 + u1*u1 + u2*u2
    vi_mag = sqrt(vi_mag_sqr)
    sin_alpha_i = sin( 0.5*vi_mag )    
    cos_alpha_i = cos( 0.5*vi_mag )
    
    Dphi_pDxj0 = (4*cos_alpha_i*sin_alpha_i*u1/vi_mag +
         4*sin_alpha_i_2*u0*u2/vi_mag_sqr)*((-x0 +
         y0)*(2*cos_alpha_i*sin_alpha_i*u1/vi_mag +
         2*sin_alpha_i_2*u0*u2/vi_mag_sqr) + (-x1 + y1)*(-
        2*cos_alpha_i*sin_alpha_i*u0/vi_mag +
         2*sin_alpha_i_2*u1*u2/vi_mag_sqr) + (-x2 + y2)*(cos_alpha_i_2 -
         sin_alpha_i_2*u0*u0/vi_mag_sqr - sin_alpha_i_2*u1*u1/vi_mag_sqr +
         sin_alpha_i_2*u2*u2/vi_mag_sqr))

    Dphi_pDxj1 = (-
        4*cos_alpha_i*sin_alpha_i*u0/vi_mag +
         4*sin_alpha_i_2*u1*u2/vi_mag_sqr)*((-x0 +
         y0)*(2*cos_alpha_i*sin_alpha_i*u1/vi_mag +
         2*sin_alpha_i_2*u0*u2/vi_mag_sqr) + (-x1 + y1)*(-
        2*cos_alpha_i*sin_alpha_i*u0/vi_mag +
         2*sin_alpha_i_2*u1*u2/vi_mag_sqr) + (-x2 + y2)*(cos_alpha_i_2 -
         sin_alpha_i_2*u0*u0/vi_mag_sqr - sin_alpha_i_2*u1*u1/vi_mag_sqr +
         sin_alpha_i_2*u2*u2/vi_mag_sqr))

    Dphi_pDxj2 = ((-x0 +
         y0)*(2*cos_alpha_i*sin_alpha_i*u1/vi_mag +
         2*sin_alpha_i_2*u0*u2/vi_mag_sqr) + (-x1 + y1)*(-
        2*cos_alpha_i*sin_alpha_i*u0/vi_mag +
         2*sin_alpha_i_2*u1*u2/vi_mag_sqr) + (-x2 + y2)*(cos_alpha_i_2 -
         sin_alpha_i_2*u0*u0/vi_mag_sqr - sin_alpha_i_2*u1*u1/vi_mag_sqr +
         sin_alpha_i_2*u2*u2/vi_mag_sqr))*(2*cos_alpha_i_2 -
         2*sin_alpha_i_2*u0*u0/vi_mag_sqr - 2*sin_alpha_i_2*u1*u1/vi_mag_sqr +
         2*sin_alpha_i_2*u2*u2/vi_mag_sqr)

    return array([Dphi_pDxj0,Dphi_pDxj1,Dphi_pDxj2])


def Dphi_cDxi(vi, vj, xi, xj):
    u0,u1,u2 = vi
    v0,v1,v2 = vj
    x0,x1,x2 = xi
    y0,y1,y2 = xj
    
    vi_mag_sqr = u0*u0 + u1*u1 + u2*u2
    vj_mag_sqr = v0*v0 + v1*v1 + v2*v2
    vi_mag = sqrt(vi_mag_sqr)
    vj_mag = sqrt(vj_mag_sqr)
    sin_alpha_i = sin( 0.5*vi_mag )
    sin_alpha_j = sin( 0.5*vj_mag )
    cos_alpha_i = cos( 0.5*vi_mag )
    cos_alpha_j = cos( 0.5*vj_mag )
    
    Dphi_cDxi0 = ((-x0 +
         y0)*(2*cos_alpha_i*sin_alpha_i*u1/vi_mag +
         2*cos_alpha_j*sin_alpha_j*v1/vj_mag +
         2*sin_alpha_i_2*u0*u2/vi_mag_sqr +
         2*sin_alpha_j_2*v0*v2/vj_mag_sqr) + (-x1 + y1)*(-
        2*cos_alpha_i*sin_alpha_i*u0/vi_mag -
         2*cos_alpha_j*sin_alpha_j*v0/vj_mag +
         2*sin_alpha_i_2*u1*u2/vi_mag_sqr +
         2*sin_alpha_j_2*v1*v2/vj_mag_sqr) + (-x2 + y2)*(cos_alpha_i_2 +
         cos_alpha_j_2 - sin_alpha_i_2*u0*u0/vi_mag_sqr -
         sin_alpha_i_2*u1*u1/vi_mag_sqr + sin_alpha_i_2*u2*u2/vi_mag_sqr -
         sin_alpha_j_2*v0*v0/vj_mag_sqr - sin_alpha_j_2*v1*v1/vj_mag_sqr +
         sin_alpha_j_2*v2*v2/vj_mag_sqr))*(-
        4*cos_alpha_i*sin_alpha_i*u1/vi_mag -
         4*cos_alpha_j*sin_alpha_j*v1/vj_mag -
         4*sin_alpha_i_2*u0*u2/vi_mag_sqr -
         4*sin_alpha_j_2*v0*v2/vj_mag_sqr)

    Dphi_cDxi1 = ((-x0 +
         y0)*(2*cos_alpha_i*sin_alpha_i*u1/vi_mag +
         2*cos_alpha_j*sin_alpha_j*v1/vj_mag +
         2*sin_alpha_i_2*u0*u2/vi_mag_sqr +
         2*sin_alpha_j_2*v0*v2/vj_mag_sqr) + (-x1 + y1)*(-
        2*cos_alpha_i*sin_alpha_i*u0/vi_mag -
         2*cos_alpha_j*sin_alpha_j*v0/vj_mag +
         2*sin_alpha_i_2*u1*u2/vi_mag_sqr +
         2*sin_alpha_j_2*v1*v2/vj_mag_sqr) + (-x2 + y2)*(cos_alpha_i_2 +
         cos_alpha_j_2 - sin_alpha_i_2*u0*u0/vi_mag_sqr -
         sin_alpha_i_2*u1*u1/vi_mag_sqr + sin_alpha_i_2*u2*u2/vi_mag_sqr -
         sin_alpha_j_2*v0*v0/vj_mag_sqr - sin_alpha_j_2*v1*v1/vj_mag_sqr +
         sin_alpha_j_2*v2*v2/vj_mag_sqr))*(4*cos_alpha_i*sin_alpha_i*
        u0/vi_mag + 4*cos_alpha_j*sin_alpha_j*v0/vj_mag -
         4*sin_alpha_i_2*u1*u2/vi_mag_sqr -
         4*sin_alpha_j_2*v1*v2/vj_mag_sqr)

    Dphi_cDxi2 = ((-x0 +
         y0)*(2*cos_alpha_i*sin_alpha_i*u1/vi_mag +
         2*cos_alpha_j*sin_alpha_j*v1/vj_mag +
         2*sin_alpha_i_2*u0*u2/vi_mag_sqr +
         2*sin_alpha_j_2*v0*v2/vj_mag_sqr) + (-x1 + y1)*(-
        2*cos_alpha_i*sin_alpha_i*u0/vi_mag -
         2*cos_alpha_j*sin_alpha_j*v0/vj_mag +
         2*sin_alpha_i_2*u1*u2/vi_mag_sqr +
         2*sin_alpha_j_2*v1*v2/vj_mag_sqr) + (-x2 + y2)*(cos_alpha_i_2 +
         cos_alpha_j_2 - sin_alpha_i_2*u0*u0/vi_mag_sqr -
         sin_alpha_i_2*u1*u1/vi_mag_sqr + sin_alpha_i_2*u2*u2/vi_mag_sqr -
         sin_alpha_j_2*v0*v0/vj_mag_sqr - sin_alpha_j_2*v1*v1/vj_mag_sqr +
         sin_alpha_j_2*v2*v2/vj_mag_sqr))*(-2*cos_alpha_i_2 -
         2*cos_alpha_j_2 + 2*sin_alpha_i_2*u0*u0/vi_mag_sqr +
         2*sin_alpha_i_2*u1*u1/vi_mag_sqr - 2*sin_alpha_i_2*u2*u2/vi_mag_sqr +
         2*sin_alpha_j_2*v0*v0/vj_mag_sqr + 2*sin_alpha_j_2*v1*v1/vj_mag_sqr -
         2*sin_alpha_j_2*v2*v2/vj_mag_sqr)
 
    return array([Dphi_cDxi0,Dphi_cDxi1,Dphi_cDxi2])


def Dphi_cDxj(vi, vj, xi, xj):
    u0,u1,u2 = vi
    v0,v1,v2 = vj
    x0,x1,x2 = xi
    y0,y1,y2 = xj
    
    vi_mag_sqr = u0*u0 + u1*u1 + u2*u2
    vj_mag_sqr = v0*v0 + v1*v1 + v2*v2
    vi_mag = sqrt(vi_mag_sqr)
    vj_mag = sqrt(vj_mag_sqr)
    sin_alpha_i = sin( 0.5*vi_mag )
    sin_alpha_j = sin( 0.5*vj_mag )
    cos_alpha_i = cos( 0.5*vi_mag )
    cos_alpha_j = cos( 0.5*vj_mag )
    
    Dphi_cDxj0 = ((-x0 +
         y0)*(2*cos_alpha_i*sin_alpha_i*u1/vi_mag +
         2*cos_alpha_j*sin_alpha_j*v1/vj_mag +
         2*sin_alpha_i_2*u0*u2/vi_mag_sqr +
         2*sin_alpha_j_2*v0*v2/vj_mag_sqr) + (-x1 + y1)*(-
        2*cos_alpha_i*sin_alpha_i*u0/vi_mag -
         2*cos_alpha_j*sin_alpha_j*v0/vj_mag +
         2*sin_alpha_i_2*u1*u2/vi_mag_sqr +
         2*sin_alpha_j_2*v1*v2/vj_mag_sqr) + (-x2 + y2)*(cos_alpha_i_2 +
         cos_alpha_j_2 - sin_alpha_i_2*u0*u0/vi_mag_sqr -
         sin_alpha_i_2*u1*u1/vi_mag_sqr + sin_alpha_i_2*u2*u2/vi_mag_sqr -
         sin_alpha_j_2*v0*v0/vj_mag_sqr - sin_alpha_j_2*v1*v1/vj_mag_sqr +
         sin_alpha_j_2*v2*v2/vj_mag_sqr))*(4*cos_alpha_i*sin_alpha_i*
        u1/vi_mag + 4*cos_alpha_j*sin_alpha_j*v1/vj_mag +
         4*sin_alpha_i_2*u0*u2/vi_mag_sqr +
         4*sin_alpha_j_2*v0*v2/vj_mag_sqr)

    Dphi_cDxj1 = ((-x0 +
         y0)*(2*cos_alpha_i*sin_alpha_i*u1/vi_mag +
         2*cos_alpha_j*sin_alpha_j*v1/vj_mag +
         2*sin_alpha_i_2*u0*u2/vi_mag_sqr +
         2*sin_alpha_j_2*v0*v2/vj_mag_sqr) + (-x1 + y1)*(-
        2*cos_alpha_i*sin_alpha_i*u0/vi_mag -
         2*cos_alpha_j*sin_alpha_j*v0/vj_mag +
         2*sin_alpha_i_2*u1*u2/vi_mag_sqr +
         2*sin_alpha_j_2*v1*v2/vj_mag_sqr) + (-x2 + y2)*(cos_alpha_i_2 +
         cos_alpha_j_2 - sin_alpha_i_2*u0*u0/vi_mag_sqr -
         sin_alpha_i_2*u1*u1/vi_mag_sqr + sin_alpha_i_2*u2*u2/vi_mag_sqr -
         sin_alpha_j_2*v0*v0/vj_mag_sqr - sin_alpha_j_2*v1*v1/vj_mag_sqr +
         sin_alpha_j_2*v2*v2/vj_mag_sqr))*(-
        4*cos_alpha_i*sin_alpha_i*u0/vi_mag -
         4*cos_alpha_j*sin_alpha_j*v0/vj_mag +
         4*sin_alpha_i_2*u1*u2/vi_mag_sqr +
         4*sin_alpha_j_2*v1*v2/vj_mag_sqr)

    Dphi_cDxj2 = ((-x0 +
         y0)*(2*cos_alpha_i*sin_alpha_i*u1/vi_mag +
         2*cos_alpha_j*sin_alpha_j*v1/vj_mag +
         2*sin_alpha_i_2*u0*u2/vi_mag_sqr +
         2*sin_alpha_j_2*v0*v2/vj_mag_sqr) + (-x1 + y1)*(-
        2*cos_alpha_i*sin_alpha_i*u0/vi_mag -
         2*cos_alpha_j*sin_alpha_j*v0/vj_mag +
         2*sin_alpha_i_2*u1*u2/vi_mag_sqr +
         2*sin_alpha_j_2*v1*v2/vj_mag_sqr) + (-x2 + y2)*(cos_alpha_i_2 +
         cos_alpha_j_2 - sin_alpha_i_2*u0*u0/vi_mag_sqr -
         sin_alpha_i_2*u1*u1/vi_mag_sqr + sin_alpha_i_2*u2*u2/vi_mag_sqr -
         sin_alpha_j_2*v0*v0/vj_mag_sqr - sin_alpha_j_2*v1*v1/vj_mag_sqr +
         sin_alpha_j_2*v2*v2/vj_mag_sqr))*(2*cos_alpha_i_2 +
         2*cos_alpha_j_2 - 2*sin_alpha_i_2*u0*u0/vi_mag_sqr -
         2*sin_alpha_i_2*u1*u1/vi_mag_sqr + 2*sin_alpha_i_2*u2*u2/vi_mag_sqr -
         2*sin_alpha_j_2*v0*v0/vj_mag_sqr - 2*sin_alpha_j_2*v1*v1/vj_mag_sqr +
         2*sin_alpha_j_2*v2*v2/vj_mag_sqr)

    return array([Dphi_cDxj0,Dphi_cDxj1,Dphi_cDxj2])
