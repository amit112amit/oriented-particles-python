#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 24 14:06:27 2017
    Derive quaternion multiplication results
@author: amit
"""
import sys
sys.path.insert(0,'../../')

import sympy as sy
from src.derivations import PotentialsSym as ps

"""
First, we will derive result of quaternion multiplication
"""
# Define quaternion q and vector p components
q0,q1,q2,q3,p0,p1,p2,p3 = sy.symbols('q0,q1,q2,q3,p0,p1,p2,p3')

# Define quaternion imaginary numbers as non-commutative
i,j,k = sy.symbols('i,j,k', commutative=False)

# Construct quaternion from components
p_full = p0 + p1*i + p2*j + p3*k
q_full = q0 + q1*i + q2*j + q3*k

# Multiply
ans = sy.expand(p_full*q_full)
print("\nQuaternion multiplication result = ")
print(ans)
print("\n")

# Apply Hamiltonian product rules    
subsRule1 = [(i**2, -1), (j**2, -1), (k**2, -1)]
subsRule2 = [(i*j, k), (j*k, i), (k*i, j)]
subsRule3 = [(j*i, -k), (k*j, -i), (i*k, -j)]
ans = ans.subs(subsRule1)
ans = ans.subs(subsRule2)
ans = ans.subs(subsRule3)

# Gather the components of quaternion in dict d
d = sy.collect(ans,[i,j,k], evaluate=False)
# Check which of i,j,k and the scalar value actually exist

# Return answer in component form
ans_mul = [ d[1], d[i], d[j], d[k] ]

# Another way to multiply quaternions
p_full = sy.Matrix(4,1,[p0,p1,p2,p3])
q_full = sy.Matrix(4,1,[q0,q1,q2,q3])
p1_3 = sy.Matrix(3,1,[p1,p2,p3])
q1_3 = sy.Matrix(3,1,[q1,q2,q3])
Cq = sy.Matrix([[0,-q3,q2],[q3,0,-q1],[-q2,q1,0]] )
Pre = sy.zeros(4,4)
Pre[0,0] = q0
Pre[0,1:4] = -q1_3.T
Pre[1:4,0] = q1_3
Pre[1:4,1:4] = q0*sy.eye(3) - Cq
matmul = Pre*p_full

diff = []
for z in range(4):
    diff.append( sy.simplify(ans_mul[z] - matmul[z]) )
    print( matmul[z] )

print("\nDifference in matrix multiplication = ")
print(diff)
print("\n")

qi_full = sy.Matrix([q0, -q1, -q2, -q3])

"""
print("\nIf p = [p0,p1,p2,p3] and q = [q0,q1,q2,q3] then p*q is")
print(ans_mul)
"""

"""
Now derive equations for quaternion conjugation for UNIT QUATERNIONS
"""
p_full = 0 + p1*i + p2*j + p3*k
q_full = q0 + q1*i + q2*j + q3*k
 #inverse of quaternion q^(-1)
qi_full = q0 - q1*i - q2*j - q3*k

# Conjugation 
ans = sy.expand(q_full*p_full*qi_full)

#print(ans)

# Hamiltonian product rules
subsRule0 = [(i**3, -i), (j**3, -j), (k**3, -k)]
subsRule1 = [(i**2, -1), (j**2, -1), (k**2, -1)]
subsRule2 = [(i*j, k), (j*k, i), (k*i, j)]
subsRule3 = [(j*i, -k), (k*j, -i), (i*k, -j)]
ans1 = ans.subs(subsRule0)
ans2 = ans1.subs(subsRule1)
# We will need to substitute multiple times because
# i**3 gives rise to -i and so on
#for z in range(2):
ans3 = ans2.subs(subsRule2)
ans4 = ans3.subs(subsRule3)
ans5 = ans4.subs(subsRule2)
ans6 = ans5.subs(subsRule3)
# Gather the components of quaternion in dict d
d = sy.collect(ans6,[i,j,k], evaluate=False)

# Return answer in component form
ans7 = [d[i], d[j], d[k] ]
"""
print("\nIf q = [q0,q1,q2,q3] and p = [p1,p2,p3] then q*p_q*q^(-1) is")
print(ans)
"""
# Check conjugation using quatMul(p,q)
p_full = [0,p1,p2,p3]
q_full = [q0,q1,q2,q3]
qi_full = [q0, -q1, -q2, -q3]
prod1 = ps.quatMul(p_full, qi_full, i, j, k)
prod2 = ps.quatMul(q_full, prod1, i, j, k)
#print("Prod2 = ")
prod2_s = []
for i in prod2:
    prod2_s.append( sy.expand(i) )
#print(prod2_s)

# Try matrix multiplication form
q = [q0,q1,q2,q3]
p = [p1,p2,p3]
out = ps.rotateByQuat(q,p)
print("Matrix multiplication result = ")
print(out)

"""
comp = []
for i in range(1,4):
    print("out[{0}] = ".format(i))
    print( sy.expand(out[i]) )
    print("prod2[{0}] = ".format(i))
    print( sy.expand(prod2[i]) )
"""
#print("Comparing the two:")
#print(comp)

a,b,c = sy.symbols('a,b,c')
v = [a,b,c]
u = [sy.Rational(1,2), sy.Rational(1,2),sy.Rational(1,2),sy.Rational(1,2)]
result = ps.conjugation( u,v )
#print("Conjugation result = ")
#print(result)

# Yet another rotation calculation
p1_3 = sy.Matrix(3,1,[p1,p2,p3])
q1_3 = sy.Matrix(3,1,[q1,q2,q3])
Cq = sy.Matrix([[0,-q3,q2],[q3,0,-q1],[-q2,q1,0]] )

r_prime = (q0**2 - q1**2 + q2**2 + q3**2)*p1_3 + 2*q0*(Cq*p1_3) + \
    2*(p1*q1 + p2*q2 + p3*q3)*q1_3

print("Direct multiplication result = ")
rprime = []
for i in range(3):
    rprime.append( sy.expand( r_prime[i] ) )
print(rprime)