def Dpsi_Dxi(vi, xi, xj, K, a, b):
    u0,u1,u2 = vi
    x0,x1,x2 = xi
    y0,y1,y2 = xj
    
    vi_mag_sqr = u0*u0 + u1*u1 + u2*u2
    vi_mag = sqrt(vi_mag_sqr)    
    sin_alpha_i = sin( 0.5*vi_mag )    
    cos_alpha_i = cos( 0.5*vi_mag )
    
    DpsiDxi0 = K*(-
        (4*cos_alpha_i*sin_alpha_i*u1/vi_mag -
         4*sin_alpha_i_2*u0*u2/vi_mag_sqr)*(cos_alpha_i_2*(-x2 + y2) +
         cos_alpha_i*sin_alpha_i*u0*(-2*x1 + 2*y1)/vi_mag +
         cos_alpha_i*sin_alpha_i*u1*(2*x0 - 2*y0)/vi_mag -
         sin_alpha_i_2*u0*u0*(-x2 + y2)/vi_mag_sqr + sin_alpha_i_2*u0*u2*(-
        2*x0 + 2*y0)/vi_mag_sqr - sin_alpha_i_2*u1*u1*(-x2 + y2)/vi_mag_sqr +
         sin_alpha_i_2*u1*u2*(-2*x1 + 2*y1)/vi_mag_sqr +
         sin_alpha_i_2*u2*u2*(-x2 + y2)/vi_mag_sqr)/(2*b*b) + (-(-
        4*cos_alpha_i*sin_alpha_i*u2/vi_mag -
         4*sin_alpha_i_2*u0*u1/vi_mag_sqr)*(cos_alpha_i_2*(-x1 + y1) -
         cos_alpha_i*sin_alpha_i*u0*(-2*x2 + 2*y2)/vi_mag +
         cos_alpha_i*sin_alpha_i*u2*(-2*x0 + 2*y0)/vi_mag -
         sin_alpha_i_2*u0*u0*(-x1 + y1)/vi_mag_sqr + sin_alpha_i_2*u0*u1*(-
        2*x0 + 2*y0)/vi_mag_sqr + sin_alpha_i_2*u1*u1*(-x1 + y1)/vi_mag_sqr +
         sin_alpha_i_2*u1*u2*(-2*x2 + 2*y2)/vi_mag_sqr -
         sin_alpha_i_2*u2*u2*(-x1 + y1)/vi_mag_sqr) - (-2*cos_alpha_i_2 -
         2*sin_alpha_i_2*u0*u0/vi_mag_sqr + 2*sin_alpha_i_2*u1*u1/vi_mag_sqr +
         2*sin_alpha_i_2*u2*u2/vi_mag_sqr)*(cos_alpha_i_2*(-x0 + y0) +
         cos_alpha_i*sin_alpha_i*u1*(-2*x2 + 2*y2)/vi_mag -
         cos_alpha_i*sin_alpha_i*u2*(-2*x1 + 2*y1)/vi_mag +
         sin_alpha_i_2*u0*u0*(-x0 + y0)/vi_mag_sqr + sin_alpha_i_2*u0*u1*(-
        2*x1 + 2*y1)/vi_mag_sqr + sin_alpha_i_2*u0*u2*(-2*x2 +
         2*y2)/vi_mag_sqr - sin_alpha_i_2*u1*u1*(-x0 + y0)/vi_mag_sqr -
         sin_alpha_i_2*u2*u2*(-x0 + y0)/vi_mag_sqr))/(2*a*a))*exp(-
        pow(cos_alpha_i_2*(-x2 + y2) + cos_alpha_i*sin_alpha_i*u0*(-2*x1 +
         2*y1)/vi_mag + cos_alpha_i*sin_alpha_i*u1*(2*x0 - 2*y0)/vi_mag -
         sin_alpha_i_2*u0*u0*(-x2 + y2)/vi_mag_sqr + sin_alpha_i_2*u0*u2*(-
        2*x0 + 2*y0)/vi_mag_sqr - sin_alpha_i_2*u1*u1*(-x2 + y2)/vi_mag_sqr +
         sin_alpha_i_2*u1*u2*(-2*x1 + 2*y1)/vi_mag_sqr +
         sin_alpha_i_2*u2*u2*(-x2 + y2)/vi_mag_sqr,2)/(2*b*b) + (-
        pow(cos_alpha_i_2*(-x0 + y0) + cos_alpha_i*sin_alpha_i*u1*(-2*x2 +
         2*y2)/vi_mag - cos_alpha_i*sin_alpha_i*u2*(-2*x1 + 2*y1)/vi_mag +
         sin_alpha_i_2*u0*u0*(-x0 + y0)/vi_mag_sqr + sin_alpha_i_2*u0*u1*(-
        2*x1 + 2*y1)/vi_mag_sqr + sin_alpha_i_2*u0*u2*(-2*x2 +
         2*y2)/vi_mag_sqr - sin_alpha_i_2*u1*u1*(-x0 + y0)/vi_mag_sqr -
         sin_alpha_i_2*u2*u2*(-x0 + y0)/vi_mag_sqr,2) - pow(cos_alpha_i_2*(-
        x1 + y1) - cos_alpha_i*sin_alpha_i*u0*(-2*x2 + 2*y2)/vi_mag +
         cos_alpha_i*sin_alpha_i*u2*(-2*x0 + 2*y0)/vi_mag -
         sin_alpha_i_2*u0*u0*(-x1 + y1)/vi_mag_sqr + sin_alpha_i_2*u0*u1*(-
        2*x0 + 2*y0)/vi_mag_sqr + sin_alpha_i_2*u1*u1*(-x1 + y1)/vi_mag_sqr +
         sin_alpha_i_2*u1*u2*(-2*x2 + 2*y2)/vi_mag_sqr -
         sin_alpha_i_2*u2*u2*(-x1 +
         y1)/vi_mag_sqr,2))/(2*a*a));

    Dpsi_Dxi1 = K*(-(-
        4*cos_alpha_i*sin_alpha_i*u0/vi_mag -
         4*sin_alpha_i_2*u1*u2/vi_mag_sqr)*(cos_alpha_i_2*(-x2 + y2) +
         cos_alpha_i*sin_alpha_i*u0*(-2*x1 + 2*y1)/vi_mag +
         cos_alpha_i*sin_alpha_i*u1*(2*x0 - 2*y0)/vi_mag -
         sin_alpha_i_2*u0*u0*(-x2 + y2)/vi_mag_sqr + sin_alpha_i_2*u0*u2*(-
        2*x0 + 2*y0)/vi_mag_sqr - sin_alpha_i_2*u1*u1*(-x2 + y2)/vi_mag_sqr +
         sin_alpha_i_2*u1*u2*(-2*x1 + 2*y1)/vi_mag_sqr +
         sin_alpha_i_2*u2*u2*(-x2 + y2)/vi_mag_sqr)/(2*b*b) + (-
        (4*cos_alpha_i*sin_alpha_i*u2/vi_mag -
         4*sin_alpha_i_2*u0*u1/vi_mag_sqr)*(cos_alpha_i_2*(-x0 + y0) +
         cos_alpha_i*sin_alpha_i*u1*(-2*x2 + 2*y2)/vi_mag -
         cos_alpha_i*sin_alpha_i*u2*(-2*x1 + 2*y1)/vi_mag +
         sin_alpha_i_2*u0*u0*(-x0 + y0)/vi_mag_sqr + sin_alpha_i_2*u0*u1*(-
        2*x1 + 2*y1)/vi_mag_sqr + sin_alpha_i_2*u0*u2*(-2*x2 +
         2*y2)/vi_mag_sqr - sin_alpha_i_2*u1*u1*(-x0 + y0)/vi_mag_sqr -
         sin_alpha_i_2*u2*u2*(-x0 + y0)/vi_mag_sqr) - (-2*cos_alpha_i_2 +
         2*sin_alpha_i_2*u0*u0/vi_mag_sqr - 2*sin_alpha_i_2*u1*u1/vi_mag_sqr +
         2*sin_alpha_i_2*u2*u2/vi_mag_sqr)*(cos_alpha_i_2*(-x1 + y1) -
         cos_alpha_i*sin_alpha_i*u0*(-2*x2 + 2*y2)/vi_mag +
         cos_alpha_i*sin_alpha_i*u2*(-2*x0 + 2*y0)/vi_mag -
         sin_alpha_i_2*u0*u0*(-x1 + y1)/vi_mag_sqr + sin_alpha_i_2*u0*u1*(-
        2*x0 + 2*y0)/vi_mag_sqr + sin_alpha_i_2*u1*u1*(-x1 + y1)/vi_mag_sqr +
         sin_alpha_i_2*u1*u2*(-2*x2 + 2*y2)/vi_mag_sqr -
         sin_alpha_i_2*u2*u2*(-x1 + y1)/vi_mag_sqr))/(2*a*a))*exp(-
        pow(cos_alpha_i_2*(-x2 + y2) + cos_alpha_i*sin_alpha_i*u0*(-2*x1 +
         2*y1)/vi_mag + cos_alpha_i*sin_alpha_i*u1*(2*x0 - 2*y0)/vi_mag -
         sin_alpha_i_2*u0*u0*(-x2 + y2)/vi_mag_sqr + sin_alpha_i_2*u0*u2*(-
        2*x0 + 2*y0)/vi_mag_sqr - sin_alpha_i_2*u1*u1*(-x2 + y2)/vi_mag_sqr +
         sin_alpha_i_2*u1*u2*(-2*x1 + 2*y1)/vi_mag_sqr +
         sin_alpha_i_2*u2*u2*(-x2 + y2)/vi_mag_sqr,2)/(2*b*b) + (-
        pow(cos_alpha_i_2*(-x0 + y0) + cos_alpha_i*sin_alpha_i*u1*(-2*x2 +
         2*y2)/vi_mag - cos_alpha_i*sin_alpha_i*u2*(-2*x1 + 2*y1)/vi_mag +
         sin_alpha_i_2*u0*u0*(-x0 + y0)/vi_mag_sqr + sin_alpha_i_2*u0*u1*(-
        2*x1 + 2*y1)/vi_mag_sqr + sin_alpha_i_2*u0*u2*(-2*x2 +
         2*y2)/vi_mag_sqr - sin_alpha_i_2*u1*u1*(-x0 + y0)/vi_mag_sqr -
         sin_alpha_i_2*u2*u2*(-x0 + y0)/vi_mag_sqr,2) - pow(cos_alpha_i_2*(-
        x1 + y1) - cos_alpha_i*sin_alpha_i*u0*(-2*x2 + 2*y2)/vi_mag +
         cos_alpha_i*sin_alpha_i*u2*(-2*x0 + 2*y0)/vi_mag -
         sin_alpha_i_2*u0*u0*(-x1 + y1)/vi_mag_sqr + sin_alpha_i_2*u0*u1*(-
        2*x0 + 2*y0)/vi_mag_sqr + sin_alpha_i_2*u1*u1*(-x1 + y1)/vi_mag_sqr +
         sin_alpha_i_2*u1*u2*(-2*x2 + 2*y2)/vi_mag_sqr -
         sin_alpha_i_2*u2*u2*(-x1 +
         y1)/vi_mag_sqr,2))/(2*a*a));

    Dpsi_Dxi2 = K*(-(-2*cos_alpha_i_2 +
         2*sin_alpha_i_2*u0*u0/vi_mag_sqr + 2*sin_alpha_i_2*u1*u1/vi_mag_sqr -
         2*sin_alpha_i_2*u2*u2/vi_mag_sqr)*(cos_alpha_i_2*(-x2 + y2) +
         cos_alpha_i*sin_alpha_i*u0*(-2*x1 + 2*y1)/vi_mag +
         cos_alpha_i*sin_alpha_i*u1*(2*x0 - 2*y0)/vi_mag -
         sin_alpha_i_2*u0*u0*(-x2 + y2)/vi_mag_sqr + sin_alpha_i_2*u0*u2*(-
        2*x0 + 2*y0)/vi_mag_sqr - sin_alpha_i_2*u1*u1*(-x2 + y2)/vi_mag_sqr +
         sin_alpha_i_2*u1*u2*(-2*x1 + 2*y1)/vi_mag_sqr +
         sin_alpha_i_2*u2*u2*(-x2 + y2)/vi_mag_sqr)/(2*b*b) + (-
        (4*cos_alpha_i*sin_alpha_i*u0/vi_mag -
         4*sin_alpha_i_2*u1*u2/vi_mag_sqr)*(cos_alpha_i_2*(-x1 + y1) -
         cos_alpha_i*sin_alpha_i*u0*(-2*x2 + 2*y2)/vi_mag +
         cos_alpha_i*sin_alpha_i*u2*(-2*x0 + 2*y0)/vi_mag -
         sin_alpha_i_2*u0*u0*(-x1 + y1)/vi_mag_sqr + sin_alpha_i_2*u0*u1*(-
        2*x0 + 2*y0)/vi_mag_sqr + sin_alpha_i_2*u1*u1*(-x1 + y1)/vi_mag_sqr +
         sin_alpha_i_2*u1*u2*(-2*x2 + 2*y2)/vi_mag_sqr -
         sin_alpha_i_2*u2*u2*(-x1 + y1)/vi_mag_sqr) - (-
        4*cos_alpha_i*sin_alpha_i*u1/vi_mag -
         4*sin_alpha_i_2*u0*u2/vi_mag_sqr)*(cos_alpha_i_2*(-x0 + y0) +
         cos_alpha_i*sin_alpha_i*u1*(-2*x2 + 2*y2)/vi_mag -
         cos_alpha_i*sin_alpha_i*u2*(-2*x1 + 2*y1)/vi_mag +
         sin_alpha_i_2*u0*u0*(-x0 + y0)/vi_mag_sqr + sin_alpha_i_2*u0*u1*(-
        2*x1 + 2*y1)/vi_mag_sqr + sin_alpha_i_2*u0*u2*(-2*x2 +
         2*y2)/vi_mag_sqr - sin_alpha_i_2*u1*u1*(-x0 + y0)/vi_mag_sqr -
         sin_alpha_i_2*u2*u2*(-x0 + y0)/vi_mag_sqr))/(2*a*a))*exp(-
        pow(cos_alpha_i_2*(-x2 + y2) + cos_alpha_i*sin_alpha_i*u0*(-2*x1 +
         2*y1)/vi_mag + cos_alpha_i*sin_alpha_i*u1*(2*x0 - 2*y0)/vi_mag -
         sin_alpha_i_2*u0*u0*(-x2 + y2)/vi_mag_sqr + sin_alpha_i_2*u0*u2*(-
        2*x0 + 2*y0)/vi_mag_sqr - sin_alpha_i_2*u1*u1*(-x2 + y2)/vi_mag_sqr +
         sin_alpha_i_2*u1*u2*(-2*x1 + 2*y1)/vi_mag_sqr +
         sin_alpha_i_2*u2*u2*(-x2 + y2)/vi_mag_sqr,2)/(2*b*b) + (-
        pow(cos_alpha_i_2*(-x0 + y0) + cos_alpha_i*sin_alpha_i*u1*(-2*x2 +
         2*y2)/vi_mag - cos_alpha_i*sin_alpha_i*u2*(-2*x1 + 2*y1)/vi_mag +
         sin_alpha_i_2*u0*u0*(-x0 + y0)/vi_mag_sqr + sin_alpha_i_2*u0*u1*(-
        2*x1 + 2*y1)/vi_mag_sqr + sin_alpha_i_2*u0*u2*(-2*x2 +
         2*y2)/vi_mag_sqr - sin_alpha_i_2*u1*u1*(-x0 + y0)/vi_mag_sqr -
         sin_alpha_i_2*u2*u2*(-x0 + y0)/vi_mag_sqr,2) - pow(cos_alpha_i_2*(-
        x1 + y1) - cos_alpha_i*sin_alpha_i*u0*(-2*x2 + 2*y2)/vi_mag +
         cos_alpha_i*sin_alpha_i*u2*(-2*x0 + 2*y0)/vi_mag -
         sin_alpha_i_2*u0*u0*(-x1 + y1)/vi_mag_sqr + sin_alpha_i_2*u0*u1*(-
        2*x0 + 2*y0)/vi_mag_sqr + sin_alpha_i_2*u1*u1*(-x1 + y1)/vi_mag_sqr +
         sin_alpha_i_2*u1*u2*(-2*x2 + 2*y2)/vi_mag_sqr -
         sin_alpha_i_2*u2*u2*(-x1 + y1)/vi_mag_sqr,2))/(2*a*a));
             
    return array([Dpsi_Dxi0, Dpsi_Dxi1, Dpsi_Dxi2])

"""
 Derivative of kernel with respect to xj
"""
@jit( f8[:](f8[:], f8[:], f8[:], f8, f8, f8), cache=True, nopython=True )
def Dpsi_Dxj(vi, xi, xj, K, a, b):
    u0,u1,u2 = vi
    x0,x1,x2 = xi
    y0,y1,y2 = xj
    
    vi_mag_sqr = u0*u0 + u1*u1 + u2*u2
    vi_mag = sqrt(vi_mag_sqr)    
    sin_alpha_i = sin( 0.5*vi_mag )    
    cos_alpha_i = cos( 0.5*vi_mag )
    
    Dpsi_Dxj0 = K*(-(-
        4*cos_alpha_i*sin_alpha_i*u1/vi_mag +
         4*sin_alpha_i_2*u0*u2/vi_mag_sqr)*(cos_alpha_i_2*(-x2 + y2) +
         cos_alpha_i*sin_alpha_i*u0*(-2*x1 + 2*y1)/vi_mag +
         cos_alpha_i*sin_alpha_i*u1*(2*x0 - 2*y0)/vi_mag -
         sin_alpha_i_2*u0*u0*(-x2 + y2)/vi_mag_sqr + sin_alpha_i_2*u0*u2*(-
        2*x0 + 2*y0)/vi_mag_sqr - sin_alpha_i_2*u1*u1*(-x2 + y2)/vi_mag_sqr +
         sin_alpha_i_2*u1*u2*(-2*x1 + 2*y1)/vi_mag_sqr +
         sin_alpha_i_2*u2*u2*(-x2 + y2)/vi_mag_sqr)/(2*b*b) + (-
        (4*cos_alpha_i*sin_alpha_i*u2/vi_mag +
         4*sin_alpha_i_2*u0*u1/vi_mag_sqr)*(cos_alpha_i_2*(-x1 + y1) -
         cos_alpha_i*sin_alpha_i*u0*(-2*x2 + 2*y2)/vi_mag +
         cos_alpha_i*sin_alpha_i*u2*(-2*x0 + 2*y0)/vi_mag -
         sin_alpha_i_2*u0*u0*(-x1 + y1)/vi_mag_sqr + sin_alpha_i_2*u0*u1*(-
        2*x0 + 2*y0)/vi_mag_sqr + sin_alpha_i_2*u1*u1*(-x1 + y1)/vi_mag_sqr +
         sin_alpha_i_2*u1*u2*(-2*x2 + 2*y2)/vi_mag_sqr -
         sin_alpha_i_2*u2*u2*(-x1 + y1)/vi_mag_sqr) - (2*cos_alpha_i_2 +
         2*sin_alpha_i_2*u0*u0/vi_mag_sqr - 2*sin_alpha_i_2*u1*u1/vi_mag_sqr -
         2*sin_alpha_i_2*u2*u2/vi_mag_sqr)*(cos_alpha_i_2*(-x0 + y0) +
         cos_alpha_i*sin_alpha_i*u1*(-2*x2 + 2*y2)/vi_mag -
         cos_alpha_i*sin_alpha_i*u2*(-2*x1 + 2*y1)/vi_mag +
         sin_alpha_i_2*u0*u0*(-x0 + y0)/vi_mag_sqr + sin_alpha_i_2*u0*u1*(-
        2*x1 + 2*y1)/vi_mag_sqr + sin_alpha_i_2*u0*u2*(-2*x2 +
         2*y2)/vi_mag_sqr - sin_alpha_i_2*u1*u1*(-x0 + y0)/vi_mag_sqr -
         sin_alpha_i_2*u2*u2*(-x0 + y0)/vi_mag_sqr))/(2*a*a))*exp(-
        pow(cos_alpha_i_2*(-x2 + y2) + cos_alpha_i*sin_alpha_i*u0*(-2*x1 +
         2*y1)/vi_mag + cos_alpha_i*sin_alpha_i*u1*(2*x0 - 2*y0)/vi_mag -
         sin_alpha_i_2*u0*u0*(-x2 + y2)/vi_mag_sqr + sin_alpha_i_2*u0*u2*(-
        2*x0 + 2*y0)/vi_mag_sqr - sin_alpha_i_2*u1*u1*(-x2 + y2)/vi_mag_sqr +
         sin_alpha_i_2*u1*u2*(-2*x1 + 2*y1)/vi_mag_sqr +
         sin_alpha_i_2*u2*u2*(-x2 + y2)/vi_mag_sqr,2)/(2*b*b) + (-
        pow(cos_alpha_i_2*(-x0 + y0) + cos_alpha_i*sin_alpha_i*u1*(-2*x2 +
         2*y2)/vi_mag - cos_alpha_i*sin_alpha_i*u2*(-2*x1 + 2*y1)/vi_mag +
         sin_alpha_i_2*u0*u0*(-x0 + y0)/vi_mag_sqr + sin_alpha_i_2*u0*u1*(-
        2*x1 + 2*y1)/vi_mag_sqr + sin_alpha_i_2*u0*u2*(-2*x2 +
         2*y2)/vi_mag_sqr - sin_alpha_i_2*u1*u1*(-x0 + y0)/vi_mag_sqr -
         sin_alpha_i_2*u2*u2*(-x0 + y0)/vi_mag_sqr,2) - pow(cos_alpha_i_2*(-
        x1 + y1) - cos_alpha_i*sin_alpha_i*u0*(-2*x2 + 2*y2)/vi_mag +
         cos_alpha_i*sin_alpha_i*u2*(-2*x0 + 2*y0)/vi_mag -
         sin_alpha_i_2*u0*u0*(-x1 + y1)/vi_mag_sqr + sin_alpha_i_2*u0*u1*(-
        2*x0 + 2*y0)/vi_mag_sqr + sin_alpha_i_2*u1*u1*(-x1 + y1)/vi_mag_sqr +
         sin_alpha_i_2*u1*u2*(-2*x2 + 2*y2)/vi_mag_sqr -
         sin_alpha_i_2*u2*u2*(-x1 +
         y1)/vi_mag_sqr,2))/(2*a*a))

    Dpsi_Dxj1 = K*(-
        (4*cos_alpha_i*sin_alpha_i*u0/vi_mag +
         4*sin_alpha_i_2*u1*u2/vi_mag_sqr)*(cos_alpha_i_2*(-x2 + y2) +
         cos_alpha_i*sin_alpha_i*u0*(-2*x1 + 2*y1)/vi_mag +
         cos_alpha_i*sin_alpha_i*u1*(2*x0 - 2*y0)/vi_mag -
         sin_alpha_i_2*u0*u0*(-x2 + y2)/vi_mag_sqr + sin_alpha_i_2*u0*u2*(-
        2*x0 + 2*y0)/vi_mag_sqr - sin_alpha_i_2*u1*u1*(-x2 + y2)/vi_mag_sqr +
         sin_alpha_i_2*u1*u2*(-2*x1 + 2*y1)/vi_mag_sqr +
         sin_alpha_i_2*u2*u2*(-x2 + y2)/vi_mag_sqr)/(2*b*b) + (-(-
        4*cos_alpha_i*sin_alpha_i*u2/vi_mag +
         4*sin_alpha_i_2*u0*u1/vi_mag_sqr)*(cos_alpha_i_2*(-x0 + y0) +
         cos_alpha_i*sin_alpha_i*u1*(-2*x2 + 2*y2)/vi_mag -
         cos_alpha_i*sin_alpha_i*u2*(-2*x1 + 2*y1)/vi_mag +
         sin_alpha_i_2*u0*u0*(-x0 + y0)/vi_mag_sqr + sin_alpha_i_2*u0*u1*(-
        2*x1 + 2*y1)/vi_mag_sqr + sin_alpha_i_2*u0*u2*(-2*x2 +
         2*y2)/vi_mag_sqr - sin_alpha_i_2*u1*u1*(-x0 + y0)/vi_mag_sqr -
         sin_alpha_i_2*u2*u2*(-x0 + y0)/vi_mag_sqr) - (2*cos_alpha_i_2 -
         2*sin_alpha_i_2*u0*u0/vi_mag_sqr + 2*sin_alpha_i_2*u1*u1/vi_mag_sqr -
         2*sin_alpha_i_2*u2*u2/vi_mag_sqr)*(cos_alpha_i_2*(-x1 + y1) -
         cos_alpha_i*sin_alpha_i*u0*(-2*x2 + 2*y2)/vi_mag +
         cos_alpha_i*sin_alpha_i*u2*(-2*x0 + 2*y0)/vi_mag -
         sin_alpha_i_2*u0*u0*(-x1 + y1)/vi_mag_sqr + sin_alpha_i_2*u0*u1*(-
        2*x0 + 2*y0)/vi_mag_sqr + sin_alpha_i_2*u1*u1*(-x1 + y1)/vi_mag_sqr +
         sin_alpha_i_2*u1*u2*(-2*x2 + 2*y2)/vi_mag_sqr -
         sin_alpha_i_2*u2*u2*(-x1 + y1)/vi_mag_sqr))/(2*a*a))*exp(-
        pow(cos_alpha_i_2*(-x2 + y2) + cos_alpha_i*sin_alpha_i*u0*(-2*x1 +
         2*y1)/vi_mag + cos_alpha_i*sin_alpha_i*u1*(2*x0 - 2*y0)/vi_mag -
         sin_alpha_i_2*u0*u0*(-x2 + y2)/vi_mag_sqr + sin_alpha_i_2*u0*u2*(-
        2*x0 + 2*y0)/vi_mag_sqr - sin_alpha_i_2*u1*u1*(-x2 + y2)/vi_mag_sqr +
         sin_alpha_i_2*u1*u2*(-2*x1 + 2*y1)/vi_mag_sqr +
         sin_alpha_i_2*u2*u2*(-x2 + y2)/vi_mag_sqr,2)/(2*b*b) + (-
        pow(cos_alpha_i_2*(-x0 + y0) + cos_alpha_i*sin_alpha_i*u1*(-2*x2 +
         2*y2)/vi_mag - cos_alpha_i*sin_alpha_i*u2*(-2*x1 + 2*y1)/vi_mag +
         sin_alpha_i_2*u0*u0*(-x0 + y0)/vi_mag_sqr + sin_alpha_i_2*u0*u1*(-
        2*x1 + 2*y1)/vi_mag_sqr + sin_alpha_i_2*u0*u2*(-2*x2 +
         2*y2)/vi_mag_sqr - sin_alpha_i_2*u1*u1*(-x0 + y0)/vi_mag_sqr -
         sin_alpha_i_2*u2*u2*(-x0 + y0)/vi_mag_sqr,2) - pow(cos_alpha_i_2*(-
        x1 + y1) - cos_alpha_i*sin_alpha_i*u0*(-2*x2 + 2*y2)/vi_mag +
         cos_alpha_i*sin_alpha_i*u2*(-2*x0 + 2*y0)/vi_mag -
         sin_alpha_i_2*u0*u0*(-x1 + y1)/vi_mag_sqr + sin_alpha_i_2*u0*u1*(-
        2*x0 + 2*y0)/vi_mag_sqr + sin_alpha_i_2*u1*u1*(-x1 + y1)/vi_mag_sqr +
         sin_alpha_i_2*u1*u2*(-2*x2 + 2*y2)/vi_mag_sqr -
         sin_alpha_i_2*u2*u2*(-x1 +
         y1)/vi_mag_sqr,2))/(2*a*a))

    Dpsi_Dxj2 = K*(-(2*cos_alpha_i_2 -
         2*sin_alpha_i_2*u0*u0/vi_mag_sqr - 2*sin_alpha_i_2*u1*u1/vi_mag_sqr +
         2*sin_alpha_i_2*u2*u2/vi_mag_sqr)*(cos_alpha_i_2*(-x2 + y2) +
         cos_alpha_i*sin_alpha_i*u0*(-2*x1 + 2*y1)/vi_mag +
         cos_alpha_i*sin_alpha_i*u1*(2*x0 - 2*y0)/vi_mag -
         sin_alpha_i_2*u0*u0*(-x2 + y2)/vi_mag_sqr + sin_alpha_i_2*u0*u2*(-
        2*x0 + 2*y0)/vi_mag_sqr - sin_alpha_i_2*u1*u1*(-x2 + y2)/vi_mag_sqr +
         sin_alpha_i_2*u1*u2*(-2*x1 + 2*y1)/vi_mag_sqr +
         sin_alpha_i_2*u2*u2*(-x2 + y2)/vi_mag_sqr)/(2*b*b) + (-(-
        4*cos_alpha_i*sin_alpha_i*u0/vi_mag +
         4*sin_alpha_i_2*u1*u2/vi_mag_sqr)*(cos_alpha_i_2*(-x1 + y1) -
         cos_alpha_i*sin_alpha_i*u0*(-2*x2 + 2*y2)/vi_mag +
         cos_alpha_i*sin_alpha_i*u2*(-2*x0 + 2*y0)/vi_mag -
         sin_alpha_i_2*u0*u0*(-x1 + y1)/vi_mag_sqr + sin_alpha_i_2*u0*u1*(-
        2*x0 + 2*y0)/vi_mag_sqr + sin_alpha_i_2*u1*u1*(-x1 + y1)/vi_mag_sqr +
         sin_alpha_i_2*u1*u2*(-2*x2 + 2*y2)/vi_mag_sqr -
         sin_alpha_i_2*u2*u2*(-x1 + y1)/vi_mag_sqr) -
         (4*cos_alpha_i*sin_alpha_i*u1/vi_mag +
         4*sin_alpha_i_2*u0*u2/vi_mag_sqr)*(cos_alpha_i_2*(-x0 + y0) +
         cos_alpha_i*sin_alpha_i*u1*(-2*x2 + 2*y2)/vi_mag -
         cos_alpha_i*sin_alpha_i*u2*(-2*x1 + 2*y1)/vi_mag +
         sin_alpha_i_2*u0*u0*(-x0 + y0)/vi_mag_sqr + sin_alpha_i_2*u0*u1*(-
        2*x1 + 2*y1)/vi_mag_sqr + sin_alpha_i_2*u0*u2*(-2*x2 +
         2*y2)/vi_mag_sqr - sin_alpha_i_2*u1*u1*(-x0 + y0)/vi_mag_sqr -
         sin_alpha_i_2*u2*u2*(-x0 + y0)/vi_mag_sqr))/(2*a*a))*exp(-
        pow(cos_alpha_i_2*(-x2 + y2) + cos_alpha_i*sin_alpha_i*u0*(-2*x1 +
         2*y1)/vi_mag + cos_alpha_i*sin_alpha_i*u1*(2*x0 - 2*y0)/vi_mag -
         sin_alpha_i_2*u0*u0*(-x2 + y2)/vi_mag_sqr + sin_alpha_i_2*u0*u2*(-
        2*x0 + 2*y0)/vi_mag_sqr - sin_alpha_i_2*u1*u1*(-x2 + y2)/vi_mag_sqr +
         sin_alpha_i_2*u1*u2*(-2*x1 + 2*y1)/vi_mag_sqr +
         sin_alpha_i_2*u2*u2*(-x2 + y2)/vi_mag_sqr,2)/(2*b*b) + (-
        pow(cos_alpha_i_2*(-x0 + y0) + cos_alpha_i*sin_alpha_i*u1*(-2*x2 +
         2*y2)/vi_mag - cos_alpha_i*sin_alpha_i*u2*(-2*x1 + 2*y1)/vi_mag +
         sin_alpha_i_2*u0*u0*(-x0 + y0)/vi_mag_sqr + sin_alpha_i_2*u0*u1*(-
        2*x1 + 2*y1)/vi_mag_sqr + sin_alpha_i_2*u0*u2*(-2*x2 +
         2*y2)/vi_mag_sqr - sin_alpha_i_2*u1*u1*(-x0 + y0)/vi_mag_sqr -
         sin_alpha_i_2*u2*u2*(-x0 + y0)/vi_mag_sqr,2) - pow(cos_alpha_i_2*(-
        x1 + y1) - cos_alpha_i*sin_alpha_i*u0*(-2*x2 + 2*y2)/vi_mag +
         cos_alpha_i*sin_alpha_i*u2*(-2*x0 + 2*y0)/vi_mag -
         sin_alpha_i_2*u0*u0*(-x1 + y1)/vi_mag_sqr + sin_alpha_i_2*u0*u1*(-
        2*x0 + 2*y0)/vi_mag_sqr + sin_alpha_i_2*u1*u1*(-x1 + y1)/vi_mag_sqr +
         sin_alpha_i_2*u1*u2*(-2*x2 + 2*y2)/vi_mag_sqr -
         sin_alpha_i_2*u2*u2*(-x1 + y1)/vi_mag_sqr,2))/(2*a*a))
    
    return array([Dpsi_Dxj0, Dpsi_Dxj1, Dpsi_Dxj2])

"""
 Derivative of kernel with respect to qi
"""
@jit( f8[:](f8[:], f8[:], f8[:], f8, f8, f8), cache=True, nopython=True )
def Dpsi_Dvi(vi, xi, xj, K, a, b):
    u0,u1,u2 = vi
    x0,x1,x2 = xi
    y0,y1,y2 = xj
    
    vi_mag_sqr = u0*u0 + u1*u1 + u2*u2
    vi_mag = sqrt(vi_mag_sqr)    
    sin_alpha_i = sin( 0.5*vi_mag )    
    cos_alpha_i = cos( 0.5*vi_mag )

    Dpsi_Du0 = K*(-(cos_alpha_i_2*(-x2 + y2) +
         cos_alpha_i*sin_alpha_i*u0*(-2*x1 + 2*y1)/vi_mag +
         cos_alpha_i*sin_alpha_i*u1*(2*x0 - 2*y0)/vi_mag -
         sin_alpha_i_2*u0*u0*(-x2 + y2)/vi_mag_sqr + sin_alpha_i_2*u0*u2*(-
        2*x0 + 2*y0)/vi_mag_sqr - sin_alpha_i_2*u1*u1*(-x2 + y2)/vi_mag_sqr +
         sin_alpha_i_2*u1*u2*(-2*x1 + 2*y1)/vi_mag_sqr +
         sin_alpha_i_2*u2*u2*(-x2 + y2)/vi_mag_sqr)*(cos_alpha_i_2*u0*u0*(-
        2*x1 + 2*y1)/vi_mag_sqr + cos_alpha_i_2*u0*u1*(2*x0 -
         2*y0)/vi_mag_sqr - 2*cos_alpha_i*sin_alpha_i*pow(u0,3)*(-x2 +
         y2)/(pow(vi_mag,3)) + 2*cos_alpha_i*sin_alpha_i*u0*u0*u2*(-2*x0 +
         2*y0)/(pow(vi_mag,3)) - 2*cos_alpha_i*sin_alpha_i*u0*u0*(-2*x1 +
         2*y1)/(pow(vi_mag,3)) - 2*cos_alpha_i*sin_alpha_i*u0*u1*u1*(-x2 +
         y2)/(pow(vi_mag,3)) + 2*cos_alpha_i*sin_alpha_i*u0*u1*u2*(-2*x1 +
         2*y1)/(pow(vi_mag,3)) - 2*cos_alpha_i*sin_alpha_i*u0*u1*(2*x0 -
         2*y0)/(pow(vi_mag,3)) + 2*cos_alpha_i*sin_alpha_i*u0*u2*u2*(-x2 +
         y2)/(pow(vi_mag,3)) - 2*cos_alpha_i*sin_alpha_i*u0*(-x2 + y2)/vi_mag +
         2*cos_alpha_i*sin_alpha_i*(-2*x1 + 2*y1)/vi_mag +
         4*sin_alpha_i_2*pow(u0,3)*(-x2 + y2)/(pow(vi_mag,4)) -
         4*sin_alpha_i_2*u0*u0*u2*(-2*x0 + 2*y0)/(pow(vi_mag,4)) -
         sin_alpha_i_2*u0*u0*(-2*x1 + 2*y1)/vi_mag_sqr +
         4*sin_alpha_i_2*u0*u1*u1*(-x2 + y2)/(pow(vi_mag,4)) -
         4*sin_alpha_i_2*u0*u1*u2*(-2*x1 + 2*y1)/(pow(vi_mag,4)) -
         sin_alpha_i_2*u0*u1*(2*x0 - 2*y0)/vi_mag_sqr -
         4*sin_alpha_i_2*u0*u2*u2*(-x2 + y2)/(pow(vi_mag,4)) -
         4*sin_alpha_i_2*u0*(-x2 + y2)/vi_mag_sqr + 2*sin_alpha_i_2*u2*(-
        2*x0 + 2*y0)/vi_mag_sqr)/(2*b*b) + (-(cos_alpha_i_2*(-x0 + y0) +
         cos_alpha_i*sin_alpha_i*u1*(-2*x2 + 2*y2)/vi_mag -
         cos_alpha_i*sin_alpha_i*u2*(-2*x1 + 2*y1)/vi_mag +
         sin_alpha_i_2*u0*u0*(-x0 + y0)/vi_mag_sqr + sin_alpha_i_2*u0*u1*(-
        2*x1 + 2*y1)/vi_mag_sqr + sin_alpha_i_2*u0*u2*(-2*x2 +
         2*y2)/vi_mag_sqr - sin_alpha_i_2*u1*u1*(-x0 + y0)/vi_mag_sqr -
         sin_alpha_i_2*u2*u2*(-x0 + y0)/vi_mag_sqr)*(cos_alpha_i_2*u0*u1*(-
        2*x2 + 2*y2)/vi_mag_sqr - cos_alpha_i_2*u0*u2*(-2*x1 +
         2*y1)/vi_mag_sqr + 2*cos_alpha_i*sin_alpha_i*pow(u0,3)*(-x0 +
         y0)/(pow(vi_mag,3)) + 2*cos_alpha_i*sin_alpha_i*u0*u0*u1*(-2*x1 +
         2*y1)/(pow(vi_mag,3)) + 2*cos_alpha_i*sin_alpha_i*u0*u0*u2*(-2*x2 +
         2*y2)/(pow(vi_mag,3)) - 2*cos_alpha_i*sin_alpha_i*u0*u1*u1*(-x0 +
         y0)/(pow(vi_mag,3)) - 2*cos_alpha_i*sin_alpha_i*u0*u1*(-2*x2 +
         2*y2)/(pow(vi_mag,3)) - 2*cos_alpha_i*sin_alpha_i*u0*u2*u2*(-x0 +
         y0)/(pow(vi_mag,3)) + 2*cos_alpha_i*sin_alpha_i*u0*u2*(-2*x1 +
         2*y1)/(pow(vi_mag,3)) - 2*cos_alpha_i*sin_alpha_i*u0*(-x0 + y0)/vi_mag -
         4*sin_alpha_i_2*pow(u0,3)*(-x0 + y0)/(pow(vi_mag,4)) -
         4*sin_alpha_i_2*u0*u0*u1*(-2*x1 + 2*y1)/(pow(vi_mag,4)) -
         4*sin_alpha_i_2*u0*u0*u2*(-2*x2 + 2*y2)/(pow(vi_mag,4)) +
         4*sin_alpha_i_2*u0*u1*u1*(-x0 + y0)/(pow(vi_mag,4)) -
         sin_alpha_i_2*u0*u1*(-2*x2 + 2*y2)/vi_mag_sqr +
         4*sin_alpha_i_2*u0*u2*u2*(-x0 + y0)/(pow(vi_mag,4)) +
         sin_alpha_i_2*u0*u2*(-2*x1 + 2*y1)/vi_mag_sqr +
         4*sin_alpha_i_2*u0*(-x0 + y0)/vi_mag_sqr + 2*sin_alpha_i_2*u1*(-
        2*x1 + 2*y1)/vi_mag_sqr + 2*sin_alpha_i_2*u2*(-2*x2 +
         2*y2)/vi_mag_sqr) - (cos_alpha_i_2*(-x1 + y1) -
         cos_alpha_i*sin_alpha_i*u0*(-2*x2 + 2*y2)/vi_mag +
         cos_alpha_i*sin_alpha_i*u2*(-2*x0 + 2*y0)/vi_mag -
         sin_alpha_i_2*u0*u0*(-x1 + y1)/vi_mag_sqr + sin_alpha_i_2*u0*u1*(-
        2*x0 + 2*y0)/vi_mag_sqr + sin_alpha_i_2*u1*u1*(-x1 + y1)/vi_mag_sqr +
         sin_alpha_i_2*u1*u2*(-2*x2 + 2*y2)/vi_mag_sqr -
         sin_alpha_i_2*u2*u2*(-x1 + y1)/vi_mag_sqr)*(-cos_alpha_i_2*u0*u0*(-
        2*x2 + 2*y2)/vi_mag_sqr + cos_alpha_i_2*u0*u2*(-2*x0 +
         2*y0)/vi_mag_sqr - 2*cos_alpha_i*sin_alpha_i*pow(u0,3)*(-x1 +
         y1)/(pow(vi_mag,3)) + 2*cos_alpha_i*sin_alpha_i*u0*u0*u1*(-2*x0 +
         2*y0)/(pow(vi_mag,3)) + 2*cos_alpha_i*sin_alpha_i*u0*u0*(-2*x2 +
         2*y2)/(pow(vi_mag,3)) + 2*cos_alpha_i*sin_alpha_i*u0*u1*u1*(-x1 +
         y1)/(pow(vi_mag,3)) + 2*cos_alpha_i*sin_alpha_i*u0*u1*u2*(-2*x2 +
         2*y2)/(pow(vi_mag,3)) - 2*cos_alpha_i*sin_alpha_i*u0*u2*u2*(-x1 +
         y1)/(pow(vi_mag,3)) - 2*cos_alpha_i*sin_alpha_i*u0*u2*(-2*x0 +
         2*y0)/(pow(vi_mag,3)) - 2*cos_alpha_i*sin_alpha_i*u0*(-x1 + y1)/vi_mag -
         2*cos_alpha_i*sin_alpha_i*(-2*x2 + 2*y2)/vi_mag +
         4*sin_alpha_i_2*pow(u0,3)*(-x1 + y1)/(pow(vi_mag,4)) -
         4*sin_alpha_i_2*u0*u0*u1*(-2*x0 + 2*y0)/(pow(vi_mag,4)) +
         sin_alpha_i_2*u0*u0*(-2*x2 + 2*y2)/vi_mag_sqr -
         4*sin_alpha_i_2*u0*u1*u1*(-x1 + y1)/(pow(vi_mag,4)) -
         4*sin_alpha_i_2*u0*u1*u2*(-2*x2 + 2*y2)/(pow(vi_mag,4)) +
         4*sin_alpha_i_2*u0*u2*u2*(-x1 + y1)/(pow(vi_mag,4)) -
         sin_alpha_i_2*u0*u2*(-2*x0 + 2*y0)/vi_mag_sqr -
         4*sin_alpha_i_2*u0*(-x1 + y1)/vi_mag_sqr + 2*sin_alpha_i_2*u1*(-
        2*x0 + 2*y0)/vi_mag_sqr))/(2*a*a))*exp(-pow(cos_alpha_i_2*(-x2 + y2) +
         cos_alpha_i*sin_alpha_i*u0*(-2*x1 + 2*y1)/vi_mag +
         cos_alpha_i*sin_alpha_i*u1*(2*x0 - 2*y0)/vi_mag -
         sin_alpha_i_2*u0*u0*(-x2 + y2)/vi_mag_sqr + sin_alpha_i_2*u0*u2*(-
        2*x0 + 2*y0)/vi_mag_sqr - sin_alpha_i_2*u1*u1*(-x2 + y2)/vi_mag_sqr +
         sin_alpha_i_2*u1*u2*(-2*x1 + 2*y1)/vi_mag_sqr +
         sin_alpha_i_2*u2*u2*(-x2 + y2)/vi_mag_sqr,2)/(2*b*b) + (-
        pow(cos_alpha_i_2*(-x0 + y0) + cos_alpha_i*sin_alpha_i*u1*(-2*x2 +
         2*y2)/vi_mag - cos_alpha_i*sin_alpha_i*u2*(-2*x1 + 2*y1)/vi_mag +
         sin_alpha_i_2*u0*u0*(-x0 + y0)/vi_mag_sqr + sin_alpha_i_2*u0*u1*(-
        2*x1 + 2*y1)/vi_mag_sqr + sin_alpha_i_2*u0*u2*(-2*x2 +
         2*y2)/vi_mag_sqr - sin_alpha_i_2*u1*u1*(-x0 + y0)/vi_mag_sqr -
         sin_alpha_i_2*u2*u2*(-x0 + y0)/vi_mag_sqr,2) - pow(cos_alpha_i_2*(-
        x1 + y1) - cos_alpha_i*sin_alpha_i*u0*(-2*x2 + 2*y2)/vi_mag +
         cos_alpha_i*sin_alpha_i*u2*(-2*x0 + 2*y0)/vi_mag -
         sin_alpha_i_2*u0*u0*(-x1 + y1)/vi_mag_sqr + sin_alpha_i_2*u0*u1*(-
        2*x0 + 2*y0)/vi_mag_sqr + sin_alpha_i_2*u1*u1*(-x1 + y1)/vi_mag_sqr +
         sin_alpha_i_2*u1*u2*(-2*x2 + 2*y2)/vi_mag_sqr -
         sin_alpha_i_2*u2*u2*(-x1 +
         y1)/vi_mag_sqr,2))/(2*a*a))

    Dpsi_Du1 = K*(-(cos_alpha_i_2*(-
        x2 + y2) + cos_alpha_i*sin_alpha_i*u0*(-2*x1 + 2*y1)/vi_mag +
         cos_alpha_i*sin_alpha_i*u1*(2*x0 - 2*y0)/vi_mag -
         sin_alpha_i_2*u0*u0*(-x2 + y2)/vi_mag_sqr + sin_alpha_i_2*u0*u2*(-
        2*x0 + 2*y0)/vi_mag_sqr - sin_alpha_i_2*u1*u1*(-x2 + y2)/vi_mag_sqr +
         sin_alpha_i_2*u1*u2*(-2*x1 + 2*y1)/vi_mag_sqr +
         sin_alpha_i_2*u2*u2*(-x2 + y2)/vi_mag_sqr)*(cos_alpha_i_2*u0*u1*(-
        2*x1 + 2*y1)/vi_mag_sqr + cos_alpha_i_2*u1*u1*(2*x0 -
         2*y0)/vi_mag_sqr - 2*cos_alpha_i*sin_alpha_i*u0*u0*u1*(-x2 +
         y2)/(pow(vi_mag,3)) + 2*cos_alpha_i*sin_alpha_i*u0*u1*u2*(-2*x0 +
         2*y0)/(pow(vi_mag,3)) - 2*cos_alpha_i*sin_alpha_i*u0*u1*(-2*x1 +
         2*y1)/(pow(vi_mag,3)) - 2*cos_alpha_i*sin_alpha_i*pow(u1,3)*(-x2 +
         y2)/(pow(vi_mag,3)) + 2*cos_alpha_i*sin_alpha_i*u1*u1*u2*(-2*x1 +
         2*y1)/(pow(vi_mag,3)) - 2*cos_alpha_i*sin_alpha_i*u1*u1*(2*x0 -
         2*y0)/(pow(vi_mag,3)) + 2*cos_alpha_i*sin_alpha_i*u1*u2*u2*(-x2 +
         y2)/(pow(vi_mag,3)) - 2*cos_alpha_i*sin_alpha_i*u1*(-x2 + y2)/vi_mag +
         2*cos_alpha_i*sin_alpha_i*(2*x0 - 2*y0)/vi_mag +
         4*sin_alpha_i_2*u0*u0*u1*(-x2 + y2)/(pow(vi_mag,4)) -
         4*sin_alpha_i_2*u0*u1*u2*(-2*x0 + 2*y0)/(pow(vi_mag,4)) -
         sin_alpha_i_2*u0*u1*(-2*x1 + 2*y1)/vi_mag_sqr +
         4*sin_alpha_i_2*pow(u1,3)*(-x2 + y2)/(pow(vi_mag,4)) -
         4*sin_alpha_i_2*u1*u1*u2*(-2*x1 + 2*y1)/(pow(vi_mag,4)) -
         sin_alpha_i_2*u1*u1*(2*x0 - 2*y0)/vi_mag_sqr -
         4*sin_alpha_i_2*u1*u2*u2*(-x2 + y2)/(pow(vi_mag,4)) -
         4*sin_alpha_i_2*u1*(-x2 + y2)/vi_mag_sqr + 2*sin_alpha_i_2*u2*(-
        2*x1 + 2*y1)/vi_mag_sqr)/(2*b*b) + (-(cos_alpha_i_2*(-x0 + y0) +
         cos_alpha_i*sin_alpha_i*u1*(-2*x2 + 2*y2)/vi_mag -
         cos_alpha_i*sin_alpha_i*u2*(-2*x1 + 2*y1)/vi_mag +
         sin_alpha_i_2*u0*u0*(-x0 + y0)/vi_mag_sqr + sin_alpha_i_2*u0*u1*(-
        2*x1 + 2*y1)/vi_mag_sqr + sin_alpha_i_2*u0*u2*(-2*x2 +
         2*y2)/vi_mag_sqr - sin_alpha_i_2*u1*u1*(-x0 + y0)/vi_mag_sqr -
         sin_alpha_i_2*u2*u2*(-x0 + y0)/vi_mag_sqr)*(cos_alpha_i_2*u1*u1*(-
        2*x2 + 2*y2)/vi_mag_sqr - cos_alpha_i_2*u1*u2*(-2*x1 +
         2*y1)/vi_mag_sqr + 2*cos_alpha_i*sin_alpha_i*u0*u0*u1*(-x0 +
         y0)/(pow(vi_mag,3)) + 2*cos_alpha_i*sin_alpha_i*u0*u1*u1*(-2*x1 +
         2*y1)/(pow(vi_mag,3)) + 2*cos_alpha_i*sin_alpha_i*u0*u1*u2*(-2*x2 +
         2*y2)/(pow(vi_mag,3)) - 2*cos_alpha_i*sin_alpha_i*pow(u1,3)*(-x0 +
         y0)/(pow(vi_mag,3)) - 2*cos_alpha_i*sin_alpha_i*u1*u1*(-2*x2 +
         2*y2)/(pow(vi_mag,3)) - 2*cos_alpha_i*sin_alpha_i*u1*u2*u2*(-x0 +
         y0)/(pow(vi_mag,3)) + 2*cos_alpha_i*sin_alpha_i*u1*u2*(-2*x1 +
         2*y1)/(pow(vi_mag,3)) - 2*cos_alpha_i*sin_alpha_i*u1*(-x0 + y0)/vi_mag +
         2*cos_alpha_i*sin_alpha_i*(-2*x2 + 2*y2)/vi_mag -
         4*sin_alpha_i_2*u0*u0*u1*(-x0 + y0)/(pow(vi_mag,4)) -
         4*sin_alpha_i_2*u0*u1*u1*(-2*x1 + 2*y1)/(pow(vi_mag,4)) -
         4*sin_alpha_i_2*u0*u1*u2*(-2*x2 + 2*y2)/(pow(vi_mag,4)) +
         2*sin_alpha_i_2*u0*(-2*x1 + 2*y1)/vi_mag_sqr +
         4*sin_alpha_i_2*pow(u1,3)*(-x0 + y0)/(pow(vi_mag,4)) - sin_alpha_i_2*u1*u1*(-
        2*x2 + 2*y2)/vi_mag_sqr + 4*sin_alpha_i_2*u1*u2*u2*(-x0 +
         y0)/(pow(vi_mag,4)) + sin_alpha_i_2*u1*u2*(-2*x1 + 2*y1)/vi_mag_sqr -
         4*sin_alpha_i_2*u1*(-x0 + y0)/vi_mag_sqr) - (cos_alpha_i_2*(-x1 +
         y1) - cos_alpha_i*sin_alpha_i*u0*(-2*x2 + 2*y2)/vi_mag +
         cos_alpha_i*sin_alpha_i*u2*(-2*x0 + 2*y0)/vi_mag -
         sin_alpha_i_2*u0*u0*(-x1 + y1)/vi_mag_sqr + sin_alpha_i_2*u0*u1*(-
        2*x0 + 2*y0)/vi_mag_sqr + sin_alpha_i_2*u1*u1*(-x1 + y1)/vi_mag_sqr +
         sin_alpha_i_2*u1*u2*(-2*x2 + 2*y2)/vi_mag_sqr -
         sin_alpha_i_2*u2*u2*(-x1 + y1)/vi_mag_sqr)*(-cos_alpha_i_2*u0*u1*(-
        2*x2 + 2*y2)/vi_mag_sqr + cos_alpha_i_2*u1*u2*(-2*x0 +
         2*y0)/vi_mag_sqr - 2*cos_alpha_i*sin_alpha_i*u0*u0*u1*(-x1 +
         y1)/(pow(vi_mag,3)) + 2*cos_alpha_i*sin_alpha_i*u0*u1*u1*(-2*x0 +
         2*y0)/(pow(vi_mag,3)) + 2*cos_alpha_i*sin_alpha_i*u0*u1*(-2*x2 +
         2*y2)/(pow(vi_mag,3)) + 2*cos_alpha_i*sin_alpha_i*pow(u1,3)*(-x1 +
         y1)/(pow(vi_mag,3)) + 2*cos_alpha_i*sin_alpha_i*u1*u1*u2*(-2*x2 +
         2*y2)/(pow(vi_mag,3)) - 2*cos_alpha_i*sin_alpha_i*u1*u2*u2*(-x1 +
         y1)/(pow(vi_mag,3)) - 2*cos_alpha_i*sin_alpha_i*u1*u2*(-2*x0 +
         2*y0)/(pow(vi_mag,3)) - 2*cos_alpha_i*sin_alpha_i*u1*(-x1 + y1)/vi_mag +
         4*sin_alpha_i_2*u0*u0*u1*(-x1 + y1)/(pow(vi_mag,4)) -
         4*sin_alpha_i_2*u0*u1*u1*(-2*x0 + 2*y0)/(pow(vi_mag,4)) +
         sin_alpha_i_2*u0*u1*(-2*x2 + 2*y2)/vi_mag_sqr +
         2*sin_alpha_i_2*u0*(-2*x0 + 2*y0)/vi_mag_sqr -
         4*sin_alpha_i_2*pow(u1,3)*(-x1 + y1)/(pow(vi_mag,4)) -
         4*sin_alpha_i_2*u1*u1*u2*(-2*x2 + 2*y2)/(pow(vi_mag,4)) +
         4*sin_alpha_i_2*u1*u2*u2*(-x1 + y1)/(pow(vi_mag,4)) -
         sin_alpha_i_2*u1*u2*(-2*x0 + 2*y0)/vi_mag_sqr +
         4*sin_alpha_i_2*u1*(-x1 + y1)/vi_mag_sqr + 2*sin_alpha_i_2*u2*(-
        2*x2 + 2*y2)/vi_mag_sqr))/(2*a*a))*exp(-pow(cos_alpha_i_2*(-x2 + y2) +
         cos_alpha_i*sin_alpha_i*u0*(-2*x1 + 2*y1)/vi_mag +
         cos_alpha_i*sin_alpha_i*u1*(2*x0 - 2*y0)/vi_mag -
         sin_alpha_i_2*u0*u0*(-x2 + y2)/vi_mag_sqr + sin_alpha_i_2*u0*u2*(-
        2*x0 + 2*y0)/vi_mag_sqr - sin_alpha_i_2*u1*u1*(-x2 + y2)/vi_mag_sqr +
         sin_alpha_i_2*u1*u2*(-2*x1 + 2*y1)/vi_mag_sqr +
         sin_alpha_i_2*u2*u2*(-x2 + y2)/vi_mag_sqr,2)/(2*b*b) + (-
        pow(cos_alpha_i_2*(-x0 + y0) + cos_alpha_i*sin_alpha_i*u1*(-2*x2 +
         2*y2)/vi_mag - cos_alpha_i*sin_alpha_i*u2*(-2*x1 + 2*y1)/vi_mag +
         sin_alpha_i_2*u0*u0*(-x0 + y0)/vi_mag_sqr + sin_alpha_i_2*u0*u1*(-
        2*x1 + 2*y1)/vi_mag_sqr + sin_alpha_i_2*u0*u2*(-2*x2 +
         2*y2)/vi_mag_sqr - sin_alpha_i_2*u1*u1*(-x0 + y0)/vi_mag_sqr -
         sin_alpha_i_2*u2*u2*(-x0 + y0)/vi_mag_sqr,2) - pow(cos_alpha_i_2*(-
        x1 + y1) - cos_alpha_i*sin_alpha_i*u0*(-2*x2 + 2*y2)/vi_mag +
         cos_alpha_i*sin_alpha_i*u2*(-2*x0 + 2*y0)/vi_mag -
         sin_alpha_i_2*u0*u0*(-x1 + y1)/vi_mag_sqr + sin_alpha_i_2*u0*u1*(-
        2*x0 + 2*y0)/vi_mag_sqr + sin_alpha_i_2*u1*u1*(-x1 + y1)/vi_mag_sqr +
         sin_alpha_i_2*u1*u2*(-2*x2 + 2*y2)/vi_mag_sqr -
         sin_alpha_i_2*u2*u2*(-x1 +
         y1)/vi_mag_sqr,2))/(2*a*a))

    Dpsi_Du2 = K*(-(cos_alpha_i_2*(-
        x2 + y2) + cos_alpha_i*sin_alpha_i*u0*(-2*x1 + 2*y1)/vi_mag +
         cos_alpha_i*sin_alpha_i*u1*(2*x0 - 2*y0)/vi_mag -
         sin_alpha_i_2*u0*u0*(-x2 + y2)/vi_mag_sqr + sin_alpha_i_2*u0*u2*(-
        2*x0 + 2*y0)/vi_mag_sqr - sin_alpha_i_2*u1*u1*(-x2 + y2)/vi_mag_sqr +
         sin_alpha_i_2*u1*u2*(-2*x1 + 2*y1)/vi_mag_sqr +
         sin_alpha_i_2*u2*u2*(-x2 + y2)/vi_mag_sqr)*(cos_alpha_i_2*u0*u2*(-
        2*x1 + 2*y1)/vi_mag_sqr + cos_alpha_i_2*u1*u2*(2*x0 -
         2*y0)/vi_mag_sqr - 2*cos_alpha_i*sin_alpha_i*u0*u0*u2*(-x2 +
         y2)/(pow(vi_mag,3)) + 2*cos_alpha_i*sin_alpha_i*u0*u2*u2*(-2*x0 +
         2*y0)/(pow(vi_mag,3)) - 2*cos_alpha_i*sin_alpha_i*u0*u2*(-2*x1 +
         2*y1)/(pow(vi_mag,3)) - 2*cos_alpha_i*sin_alpha_i*u1*u1*u2*(-x2 +
         y2)/(pow(vi_mag,3)) + 2*cos_alpha_i*sin_alpha_i*u1*u2*u2*(-2*x1 +
         2*y1)/(pow(vi_mag,3)) - 2*cos_alpha_i*sin_alpha_i*u1*u2*(2*x0 -
         2*y0)/(pow(vi_mag,3)) + 2*cos_alpha_i*sin_alpha_i*pow(u2,3)*(-x2 +
         y2)/(pow(vi_mag,3)) - 2*cos_alpha_i*sin_alpha_i*u2*(-x2 + y2)/vi_mag +
         4*sin_alpha_i_2*u0*u0*u2*(-x2 + y2)/(pow(vi_mag,4)) -
         4*sin_alpha_i_2*u0*u2*u2*(-2*x0 + 2*y0)/(pow(vi_mag,4)) -
         sin_alpha_i_2*u0*u2*(-2*x1 + 2*y1)/vi_mag_sqr +
         2*sin_alpha_i_2*u0*(-2*x0 + 2*y0)/vi_mag_sqr +
         4*sin_alpha_i_2*u1*u1*u2*(-x2 + y2)/(pow(vi_mag,4)) -
         4*sin_alpha_i_2*u1*u2*u2*(-2*x1 + 2*y1)/(pow(vi_mag,4)) -
         sin_alpha_i_2*u1*u2*(2*x0 - 2*y0)/vi_mag_sqr + 2*sin_alpha_i_2*u1*(-
        2*x1 + 2*y1)/vi_mag_sqr - 4*sin_alpha_i_2*pow(u2,3)*(-x2 + y2)/(pow(vi_mag,4)) +
         4*sin_alpha_i_2*u2*(-x2 + y2)/vi_mag_sqr)/(2*b*b) + (-
        (cos_alpha_i_2*(-x0 + y0) + cos_alpha_i*sin_alpha_i*u1*(-2*x2 +
         2*y2)/vi_mag - cos_alpha_i*sin_alpha_i*u2*(-2*x1 + 2*y1)/vi_mag +
         sin_alpha_i_2*u0*u0*(-x0 + y0)/vi_mag_sqr + sin_alpha_i_2*u0*u1*(-
        2*x1 + 2*y1)/vi_mag_sqr + sin_alpha_i_2*u0*u2*(-2*x2 +
         2*y2)/vi_mag_sqr - sin_alpha_i_2*u1*u1*(-x0 + y0)/vi_mag_sqr -
         sin_alpha_i_2*u2*u2*(-x0 + y0)/vi_mag_sqr)*(cos_alpha_i_2*u1*u2*(-
        2*x2 + 2*y2)/vi_mag_sqr - cos_alpha_i_2*u2*u2*(-2*x1 +
         2*y1)/vi_mag_sqr + 2*cos_alpha_i*sin_alpha_i*u0*u0*u2*(-x0 +
         y0)/(pow(vi_mag,3)) + 2*cos_alpha_i*sin_alpha_i*u0*u1*u2*(-2*x1 +
         2*y1)/(pow(vi_mag,3)) + 2*cos_alpha_i*sin_alpha_i*u0*u2*u2*(-2*x2 +
         2*y2)/(pow(vi_mag,3)) - 2*cos_alpha_i*sin_alpha_i*u1*u1*u2*(-x0 +
         y0)/(pow(vi_mag,3)) - 2*cos_alpha_i*sin_alpha_i*u1*u2*(-2*x2 +
         2*y2)/(pow(vi_mag,3)) - 2*cos_alpha_i*sin_alpha_i*pow(u2,3)*(-x0 +
         y0)/(pow(vi_mag,3)) + 2*cos_alpha_i*sin_alpha_i*u2*u2*(-2*x1 +
         2*y1)/(pow(vi_mag,3)) - 2*cos_alpha_i*sin_alpha_i*u2*(-x0 + y0)/vi_mag -
         2*cos_alpha_i*sin_alpha_i*(-2*x1 + 2*y1)/vi_mag -
         4*sin_alpha_i_2*u0*u0*u2*(-x0 + y0)/(pow(vi_mag,4)) -
         4*sin_alpha_i_2*u0*u1*u2*(-2*x1 + 2*y1)/(pow(vi_mag,4)) -
         4*sin_alpha_i_2*u0*u2*u2*(-2*x2 + 2*y2)/(pow(vi_mag,4)) +
         2*sin_alpha_i_2*u0*(-2*x2 + 2*y2)/vi_mag_sqr +
         4*sin_alpha_i_2*u1*u1*u2*(-x0 + y0)/(pow(vi_mag,4)) -
         sin_alpha_i_2*u1*u2*(-2*x2 + 2*y2)/vi_mag_sqr +
         4*sin_alpha_i_2*pow(u2,3)*(-x0 + y0)/(pow(vi_mag,4)) + sin_alpha_i_2*u2*u2*(-
        2*x1 + 2*y1)/vi_mag_sqr - 4*sin_alpha_i_2*u2*(-x0 + y0)/vi_mag_sqr) -
         (cos_alpha_i_2*(-x1 + y1) - cos_alpha_i*sin_alpha_i*u0*(-2*x2 +
         2*y2)/vi_mag + cos_alpha_i*sin_alpha_i*u2*(-2*x0 + 2*y0)/vi_mag -
         sin_alpha_i_2*u0*u0*(-x1 + y1)/vi_mag_sqr + sin_alpha_i_2*u0*u1*(-
        2*x0 + 2*y0)/vi_mag_sqr + sin_alpha_i_2*u1*u1*(-x1 + y1)/vi_mag_sqr +
         sin_alpha_i_2*u1*u2*(-2*x2 + 2*y2)/vi_mag_sqr -
         sin_alpha_i_2*u2*u2*(-x1 + y1)/vi_mag_sqr)*(-cos_alpha_i_2*u0*u2*(-
        2*x2 + 2*y2)/vi_mag_sqr + cos_alpha_i_2*u2*u2*(-2*x0 +
         2*y0)/vi_mag_sqr - 2*cos_alpha_i*sin_alpha_i*u0*u0*u2*(-x1 +
         y1)/(pow(vi_mag,3)) + 2*cos_alpha_i*sin_alpha_i*u0*u1*u2*(-2*x0 +
         2*y0)/(pow(vi_mag,3)) + 2*cos_alpha_i*sin_alpha_i*u0*u2*(-2*x2 +
         2*y2)/(pow(vi_mag,3)) + 2*cos_alpha_i*sin_alpha_i*u1*u1*u2*(-x1 +
         y1)/(pow(vi_mag,3)) + 2*cos_alpha_i*sin_alpha_i*u1*u2*u2*(-2*x2 +
         2*y2)/(pow(vi_mag,3)) - 2*cos_alpha_i*sin_alpha_i*pow(u2,3)*(-x1 +
         y1)/(pow(vi_mag,3)) - 2*cos_alpha_i*sin_alpha_i*u2*u2*(-2*x0 +
         2*y0)/(pow(vi_mag,3)) - 2*cos_alpha_i*sin_alpha_i*u2*(-x1 + y1)/vi_mag +
         2*cos_alpha_i*sin_alpha_i*(-2*x0 + 2*y0)/vi_mag +
         4*sin_alpha_i_2*u0*u0*u2*(-x1 + y1)/(pow(vi_mag,4)) -
         4*sin_alpha_i_2*u0*u1*u2*(-2*x0 + 2*y0)/(pow(vi_mag,4)) +
         sin_alpha_i_2*u0*u2*(-2*x2 + 2*y2)/vi_mag_sqr -
         4*sin_alpha_i_2*u1*u1*u2*(-x1 + y1)/(pow(vi_mag,4)) -
         4*sin_alpha_i_2*u1*u2*u2*(-2*x2 + 2*y2)/(pow(vi_mag,4)) +
         2*sin_alpha_i_2*u1*(-2*x2 + 2*y2)/vi_mag_sqr +
         4*sin_alpha_i_2*pow(u2,3)*(-x1 + y1)/(pow(vi_mag,4)) - sin_alpha_i_2*u2*u2*(-
        2*x0 + 2*y0)/vi_mag_sqr - 4*sin_alpha_i_2*u2*(-x1 +
         y1)/vi_mag_sqr))/(2*a*a))*exp(-pow(cos_alpha_i_2*(-x2 + y2) +
         cos_alpha_i*sin_alpha_i*u0*(-2*x1 + 2*y1)/vi_mag +
         cos_alpha_i*sin_alpha_i*u1*(2*x0 - 2*y0)/vi_mag -
         sin_alpha_i_2*u0*u0*(-x2 + y2)/vi_mag_sqr + sin_alpha_i_2*u0*u2*(-
        2*x0 + 2*y0)/vi_mag_sqr - sin_alpha_i_2*u1*u1*(-x2 + y2)/vi_mag_sqr +
         sin_alpha_i_2*u1*u2*(-2*x1 + 2*y1)/vi_mag_sqr +
         sin_alpha_i_2*u2*u2*(-x2 + y2)/vi_mag_sqr,2)/(2*b*b) + (-
        pow(cos_alpha_i_2*(-x0 + y0) + cos_alpha_i*sin_alpha_i*u1*(-2*x2 +
         2*y2)/vi_mag - cos_alpha_i*sin_alpha_i*u2*(-2*x1 + 2*y1)/vi_mag +
         sin_alpha_i_2*u0*u0*(-x0 + y0)/vi_mag_sqr + sin_alpha_i_2*u0*u1*(-
        2*x1 + 2*y1)/vi_mag_sqr + sin_alpha_i_2*u0*u2*(-2*x2 +
         2*y2)/vi_mag_sqr - sin_alpha_i_2*u1*u1*(-x0 + y0)/vi_mag_sqr -
         sin_alpha_i_2*u2*u2*(-x0 + y0)/vi_mag_sqr,2) - pow(cos_alpha_i_2*(-
        x1 + y1) - cos_alpha_i*sin_alpha_i*u0*(-2*x2 + 2*y2)/vi_mag +
         cos_alpha_i*sin_alpha_i*u2*(-2*x0 + 2*y0)/vi_mag -
         sin_alpha_i_2*u0*u0*(-x1 + y1)/vi_mag_sqr + sin_alpha_i_2*u0*u1*(-
        2*x0 + 2*y0)/vi_mag_sqr + sin_alpha_i_2*u1*u1*(-x1 + y1)/vi_mag_sqr +
         sin_alpha_i_2*u1*u2*(-2*x2 + 2*y2)/vi_mag_sqr -
         sin_alpha_i_2*u2*u2*(-x1 + y1)/vi_mag_sqr,2))/(2*a*a))
   
    return array([Dpsi_Du0, Dpsi_Du1, Dpsi_Du2])
