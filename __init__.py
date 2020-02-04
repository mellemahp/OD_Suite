# Constants

### Gravity and earth constants ###
R_E = 6378.1363 # radius of the earth (km)
MU = 3.986004415e+05 # gravitational parameter of earth
J2 = 1.082626925638815e-03 # j2 gravity field coefficient, normalized
J3 = -0.0000025323 # j3 gravity field coefficient

### Rotation rate of the earth ###
# W_E is defined for the ref epoch 3 Oct 1999, 23:11:9.1814
W_E = 7.2921158553e-5 # rotation rate of the earth (radians)
THETA_0 = 0

### Drag model constants ###
H_0 = 88667.0 # reference height (m)
R_0 = 700000.0 + R_E * 1000 # reference radius (m)
RHO_0 = 0.0003614 # kg / km^3
C_D = 2.0 # unitless
A_SAT = 3e-6 # cross sectional area of satellite (km^2)
MASS = 970 # kg
