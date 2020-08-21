# Practical Bayesian Linear System Identification using Hamiltonian Monte Carlo
Source code to run the examples provided in the paper at ....

The code related to each example is explained below:

## Section 4
Matlab code for the example of sampling from a donut target using metroplis hastings is in Matlab/MH_donut_example.m

Matlab code for the example of sampling from a donut target using HMC is in Matlab/HMC_donut_example.m

## Section 5.1
Matlab code for the example of jointly sampling the states and parameters of a 1D state space system is given in Matlab/state_space_example.m

## Section 6 Numerical Examples
### Section 6.1 ARX part 1
Matlab/example1_arx_part1.m: Matlab code to generate the data used for system identifcation

example1_arx_part1.py: Python code that estimates the system using HMC 

### Section 6.2 ARX part 2
Matlab/example1_arx_part2.m: Matlab code to generate the data used for system identifcation

example1_arx_part2.py: Python code that estimates the system using HMC 

### Section 6.3 Output Error Model
Matlab/example2_oe.m: Matlab code to generate the data used for system identifcation

example2_oe.py: Python code that estimates the system using HMC 

### Section 6.4 Output Error Model
Matlab/example3_fir.m: Matlab code to generate the data used for system identifcation

example3_fir.py: Python code that estimates the system using HMC 

### Section 6.5 Output Error Model
Matlab/example4_lssm.m: Matlab code to generate the data used for system identifcation

example4_lssm.py: Python code that estimates the system using HMC and saves the estimates for plotting late (this code may takes several hours to run)

example4_lssm_plot.py: Python code to load the estimates and plot the results

### Section 6.6 Output Error Model
Matlab/example5_outlier.m: Matlab code to generate the data used for system identifcation

example5_outliers.py: Python code that estimates the system using HMC 

### Section 6.7 Control design with uncertainty
Matlab/example6_generatesysiddata.m: Matlab code to generate the data used for system identifcation

example6_sysid.py: Python code that estimates the system using HMC and saves the estimates

example6_plotsysid.py: Python code that plots the system identification resutls and converts the samples of state space parameterisation to transfer function paraeterisation to be used in control design

Matlab/example6_controldesign.m: Matlab code that uses the estimates samples converted to Transfer function form by "example6_plotsysid.py" to produce the control design results shown in the paper.

### Section 6.8 Nonlinear inverted pendulum
example7_pendulum.py: Python code that estimates parameters of the QUBE rotary pendulum by QANSAR using HMC from experimentally collected data and saves the results

example7_plot.py: Python code to plot the estimated system results



