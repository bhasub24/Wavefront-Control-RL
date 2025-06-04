To reproduce the results run these 3 jupyter notebooks,
 
standalone_rl - Standalone RL with PPO implementation, 
ga - plain genetic algorithm
ga_with_rl - RL in Genetic algorithm implementation

'phase_mask' folder contains the phase screen used to mimic the transmission matrix
phi_16.npy - phase mask 16*16 for the genetic algorithm and RL in GA case
phi_64.npy - phase mask 64*64 for standalone RL with PPO

The 3 folder named, results, results_ga, results_ga_rl will store the respective plots for the 3 implementations.

Run each notebook with the already existing parameters set.

This helps reproduce the figure 4, 5 and 6 from the report.
