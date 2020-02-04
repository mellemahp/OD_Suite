# OD_Suite
Suite of tools for solving Orbit Determination problems

# Included
- Simple Stations with noisy measurement simulation

Dynamics: 
- 2 body dynamics 
- gravitational perturbations up to J3
- Simple exponential drag model

Measurement simulation: 
- Range
- Range 
- Rate

Filters: 
- Batch Filter 
- Classic Kalman Filter
- Extended Kalman Filter 
- Square root information Filter 
- Unscented Kalman Filter (parallel implementation)
- 

Other: 
- Dynamic Noise compensation (limited capability)
- Process noise (SNC)
- Backwards smoothing


# Automatic differentiation
In order to avoid the tedious computation of partials for all measurements and dynamics this library instead use automatic 
differentiation to compute the partials. This makes the code very general but reduces the speed of computation. 
