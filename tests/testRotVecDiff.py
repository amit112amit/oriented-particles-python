import numpy as np

def p(u):
    U = np.linalg.norm(u)
    A = U*0.5;
    P = np.zeros((3,))
    P[0] = 2*np.sin(A)/U**2*(u[1]*U*np.cos(A) + u[0]*u[2]*np.sin(A))
    P[1] = 2*np.sin(A)/U**2*(-u[0]*U*np.cos(A) + u[1]*u[2]*np.sin(A))
    P[2] = np.cos(A)**2 + np.sin(A)**2/U**2*(u[2]**2 - u[1]**2 - u[0]**2)
    return P

axis = np.array( [0,1,0] )
angle = np.pi/2
u = angle*axis;

M = np.zeros((3,3))
h = 1e-6
for i in range(3):
    for j in range(3):
        u[i] += h
        pp = p(u)
        u[i] -= 2*h
        pm = p(u)
        M[i,j] = 0.5*(pp[j] - pm[j])/h
        u[i] += h

print('Numerical derivative:')
print(M)

U = np.linalg.norm(u)
A = U*0.5;
Q = np.array( [ np.cos(A), u[0]/U*np.sin(A), u[1]/U*np.sin(A),
               u[2]/U*np.sin(A) ] )
q0, q1, q2, q3 = Q
dpdQ = 2*np.array( [[q2, -q1, q0],
                    [q3, -q0, -q1],
                    [q0, q3, -q2],
                    [q1, q2, q3]] )
dQdu = np.zeros((3,4))
dQdu[:,0] = -0.5*np.sin(A)*u/U
dQdu[:,1:] = np.sin(A)*np.eye(3)/U + (0.5*np.cos(A)/U**2 -\
                                      np.sin(A)/U**3)*np.outer(u,u)

dpdu = np.matmul( dQdu, dpdQ )
print('Analytical derivative:')
print(dpdu)
