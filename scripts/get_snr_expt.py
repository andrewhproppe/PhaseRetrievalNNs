from PhaseImages import PhaseImages

# Load experimental data set and SVD phase
PI = PhaseImages(acq_time=0.1, date="20230829")
PI.load_expt_data(idx=10)

