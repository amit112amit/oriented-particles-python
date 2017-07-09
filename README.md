# oriented-particles-python
A Python 3 implementation using Numba for Oriented Particles System

Entire code is based on the following paper:
Szeliski, Richard and Tonnesen, D. (1992). Surface Modeling with Oriented Particle Systems. Siggraph ’92, 26(2), 160. https://doi.org/10.1017/CBO9781107415324.004

The code is organized as follows:
    / ->
    
        src ->
        
            derivations -> 
            Contains SymPy code to derive equations for potential energy and jacobian of the energy used in the actual implementation
            
            drivers -> 
            Contains code that makes use of the energy and jacobian implementations for specific assembly of particles
            
            lib -> 
            Contains the actual implementation of potentials and jacobian using Numba for speed improvements
            
        tests -> 
        Contains test functions to check that the code for the jacobian matches numerical derivatives of the energy. There is also some test for rotation using unit quaternions
        
Best,
Amit
