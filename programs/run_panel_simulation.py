# run_panel_simulation.py #
# Simulates credit card usage for a panel of users #

import os
import pandas as pd
import numpy as np
import ciso8601
from scipy.stats import bernoulli
from random import randint, randrange, choices, seed
from scipy.stats import lognorm
from tqdm import tqdm

########################################
## ---------- Housekeeping ---------- ##
########################################

# Set wd ----------
wd = os.path.expanduser('~') + '/cubs'
os.chdir(wd)

# File I/O ----------
## << Input >> ##
sim_functions_file_input = 'cubs_simulation/programs/simulation_functions_opt.py'

## << Output >> ##
panel_data_file_output = 'cubs_simulation/data/panel_sim_small.parquet'

# Parameters and seed ----------
N = 100 # number of individuals
T = 365*3 # number of days
seed(6)

####################################
## ---------- Simulate ---------- ##
####################################

# helper functions
exec(open(sim_functions_file_input).read())

# run simulation
panel_daily = SimulatePanel(N=N,T=T)

# sort by id, dt
panel_daily.sort_values(['id', 'dt'], ascending=True, inplace=True)

# save simulations
panel_daily.to_parquet(panel_data_file_output, index=False)
