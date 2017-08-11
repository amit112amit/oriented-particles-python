import src.lib.Potentials as p
import src.lib.JacKernel as k
import src.lib.JacPotX as x
import src.lib.JacPotV as v

import numpy as np;

xi = np.array([0.5, 0.75, 0.85])
vi = np.array([0.3, 0.4, 0.8660254037844386])
xj = np.array([1.5, 1.75, 1.85])
vj = np.array([0.4, 0.8660254037844386, 0.3])

E = 1.0
r = 1.0
s = 6.9314718056

K = 1.0
a = 1.0
b = 1.0

m = p.morse(xi, xj, E, r, s)
ps = p.psi(vi,xi,xj,K,a,b)
pp = p.phi_p(vi, xi, xj)
pn = p.phi_n(vi, vj)
pc = p.phi_c(vi, vj, xi, xj)

dmdx = x.Dmorse_Dxi(xi,xj,E, r, s)
dmdy = x.Dmorse_Dxj(xi,xj,E, r, s)

dsdu = k.Dpsi_Dvi(vi,xi,xj,K,a,b)
dsdx = k.Dpsi_Dxi(vi,xi,xj,K,a,b)
dsdy = k.Dpsi_Dxj(vi,xi,xj,K,a,b)

dpdu = v.Dphi_pDvi(vi,xi,xj)
dpdx = x.Dphi_pDxi(vi,xi,xj)
dpdy = x.Dphi_pDxj(vi,xi,xj)

dndu = v.Dphi_nDvi(vi,vj)
dndv = v.Dphi_nDvj(vi,vj)

dcdu = v.Dphi_cDvi(vi,vj,xi,xj)
dcdv = v.Dphi_cDvj(vi,vj,xi,xj)
dcdx = x.Dphi_cDxi(vi,vj,xi,xj)
dcdy = x.Dphi_cDxj(vi,vj,xi,xj)

str1 = "Morse = {0}".format(m)
print(str1)
str1 = "Psi = {0}".format(ps)
print(str1)
str1 = "Phi_p = {0}".format(pp)
print(str1)
str1 = "Phi_n = {0}".format(pn)
print(str1)
str1 = "Phi_c = {0}".format(pc)
print(str1)

str1 = "dmdx = {0}".format(dmdx)
print(str1)
str1 = "dmdy = {0}".format(dmdy)
print(str1)

str1 = "dsdu = {0}".format(dsdu)
print(str1)
str1 = "dsdx = {0}".format(dsdx)
print(str1)
str1 = "dsdy = {0}".format(dsdy)
print(str1)

str1 = "dpdu = {0}".format(dpdu)
print(str1)
str1 = "dpdx = {0}".format(dpdx)
print(str1)
str1 = "dpdy = {0}".format(dpdy)
print(str1)

str1 = "dndu = {0}".format(dndu)
print(str1)
str1 = "dndv = {0}".format(dndv)
print(str1)

str1 = "dcdu = {0}".format(dcdu)
print(str1)
str1 = "dcdv = {0}".format(dcdv)
print(str1)
str1 = "dcdx = {0}".format(dcdx)
print(str1)
str1 = "dcdy = {0}".format(dcdy)
print(str1)
