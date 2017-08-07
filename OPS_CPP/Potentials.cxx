def psi(vi, xi, xj, K, a, b):
    u0,u1,u2 = vi
    x0,x1,x2 = xi
    y0,y1,y2 = xj
    
    vi_mag_sqr = u0*u0 + u1*u1 + u2*u2    
    vi_mag = sqrt(vi_mag_sqr)
    sin_alpha_i = sin( 0.5*vi_mag )    
    cos_alpha_i = cos( 0.5*vi_mag )
    sin_alpha_i_2 = sin_alpha_i*sin_alpha_i
    cos_alpha_i_2 = cos_alpha_i*cos_alpha_i

    psi0 = K*exp(-pow(cos_alpha_i_2*(-x2 + y2) + cos_alpha_i*sin_alpha_i*u0*(-2*x1 + 2*y1)/vi_mag + cos_alpha_i*sin_alpha_i*u1*(2*x0 - 2*y0)/vi_mag - sin_alpha_i_2*u0*u0*(-x2 + y2)/vi_mag_sqr + sin_alpha_i_2*u0*u2*(-2*x0 + 2*y0)/vi_mag_sqr - sin_alpha_i_2*u1*u1*(-x2 + y2)/vi_mag_sqr + sin_alpha_i_2*u1*u2*(-2*x1 + 2*y1)/vi_mag_sqr + sin_alpha_i_2*u2*u2*(-x2 + y2)/vi_mag_sqr,2)/(2*b*b) + (-pow(cos_alpha_i_2*(-x0 + y0) + cos_alpha_i*sin_alpha_i*u1*(-2*x2 + 2*y2)/vi_mag - cos_alpha_i*sin_alpha_i*u2*(-2*x1 + 2*y1)/vi_mag + sin_alpha_i_2*u0*u0*(-x0 + y0)/vi_mag_sqr + sin_alpha_i_2*u0*u1*(-2*x1 + 2*y1)/vi_mag_sqr + sin_alpha_i_2*u0*u2*(-2*x2 + 2*y2)/vi_mag_sqr - sin_alpha_i_2*u1*u1*(-x0 + y0)/vi_mag_sqr - sin_alpha_i_2*u2*u2*(-x0 + y0)/vi_mag_sqr,2) - pow(cos_alpha_i_2*(-x1 + y1) - cos_alpha_i*sin_alpha_i*u0*(-2*x2 + 2*y2)/vi_mag + cos_alpha_i*sin_alpha_i*u2*(-2*x0 + 2*y0)/vi_mag - sin_alpha_i_2*u0*u0*(-x1 + y1)/vi_mag_sqr + sin_alpha_i_2*u0*u1*(-2*x0 + 2*y0)/vi_mag_sqr + sin_alpha_i_2*u1*u1*(-x1 + y1)/vi_mag_sqr + sin_alpha_i_2*u1*u2*(-2*x2 + 2*y2)/vi_mag_sqr - sin_alpha_i_2*u2*u2*(-x1 + y1)/vi_mag_sqr,2))/(2*a*a));
    
    return psi0

def phi_p(vi, xi, xj):    
    u0,u1,u2 = vi
    x0,x1,x2 = xi
    y0,y1,y2 = xj
    
    vi_mag_sqr = u0*u0 + u1*u1 + u2*u2
    vi_mag = sqrt(vi_mag_sqr)    
    sin_alpha_i = sin( 0.5*vi_mag )    
    cos_alpha_i = cos( 0.5*vi_mag )
    
    phi_p0 = pow((-x0 + y0)*(2*cos_alpha_i*sin_alpha_i*u1/vi_mag + 2*sin_alpha_i_2*u0*u2/vi_mag_sqr) + (-x1 + y1)*(-2*cos_alpha_i*sin_alpha_i*u0/vi_mag + 2*sin_alpha_i_2*u1*u2/vi_mag_sqr) + (-x2 + y2)*(cos_alpha_i_2 - sin_alpha_i_2*u0*u0/vi_mag_sqr - sin_alpha_i_2*u1*u1/vi_mag_sqr + sin_alpha_i_2*u2*u2/vi_mag_sqr),2);
    
    return phi_p0

def phi_n(vi, vj):    
    u0,u1,u2 = vi
    v0,v1,v2 = vj
    
    vi_mag_sqr = u0*u0 + u1*u1 + u2*u2
    vj_mag_sqr = v0*v0 + v1*v1 + v2*v2
    vi_mag = sqrt(vi_mag_sqr)
    vj_mag = sqrt(vj_mag_sqr)
    sin_alpha_i = sin( 0.5*vi_mag )
    sin_alpha_j = sin( 0.5*vj_mag )
    cos_alpha_i = cos( 0.5*vi_mag )
    cos_alpha_j = cos( 0.5*vj_mag )

    phi_n0 = pow(-2*cos_alpha_i*sin_alpha_i*u0/vi_mag + 2*cos_alpha_j*sin_alpha_j*v0/vj_mag + 2*sin_alpha_i_2*u1*u2/vi_mag_sqr - 2*sin_alpha_j_2*v1*v2/vj_mag_sqr,2) + pow(2*cos_alpha_i*sin_alpha_i*u1/vi_mag - 2*cos_alpha_j*sin_alpha_j*v1/vj_mag + 2*sin_alpha_i_2*u0*u2/vi_mag_sqr - 2*sin_alpha_j_2*v0*v2/vj_mag_sqr,2) + pow(cos_alpha_i_2 - cos_alpha_j_2 - sin_alpha_i_2*u0*u0/vi_mag_sqr - sin_alpha_i_2*u1*u1/vi_mag_sqr + sin_alpha_i_2*u2*u2/vi_mag_sqr + sin_alpha_j_2*v0*v0/vj_mag_sqr + sin_alpha_j_2*v1*v1/vj_mag_sqr - sin_alpha_j_2*v2*v2/vj_mag_sqr,2);

    return phi_n0

def phi_c(vi, vj, xi, xj):    
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

    phi_c0 = pow((-x0 + y0)*(2*cos_alpha_i*sin_alpha_i*u1/vi_mag + 2*cos_alpha_j*sin_alpha_j*v1/vj_mag + 2*sin_alpha_i_2*u0*u2/vi_mag_sqr + 2*sin_alpha_j_2*v0*v2/vj_mag_sqr) + (-x1 + y1)*(-2*cos_alpha_i*sin_alpha_i*u0/vi_mag - 2*cos_alpha_j*sin_alpha_j*v0/vj_mag + 2*sin_alpha_i_2*u1*u2/vi_mag_sqr + 2*sin_alpha_j_2*v1*v2/vj_mag_sqr) + (-x2 + y2)*(cos_alpha_i_2 + cos_alpha_j_2 - sin_alpha_i_2*u0*u0/vi_mag_sqr - sin_alpha_i_2*u1*u1/vi_mag_sqr + sin_alpha_i_2*u2*u2/vi_mag_sqr - sin_alpha_j_2*v0*v0/vj_mag_sqr - sin_alpha_j_2*v1*v1/vj_mag_sqr + sin_alpha_j_2*v2*v2/vj_mag_sqr),2);
    
    return phi_c0
