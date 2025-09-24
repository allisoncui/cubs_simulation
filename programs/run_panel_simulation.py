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
from datetime import datetime
from pathlib import Path

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
# panel_data_file_output = 'cubs_simulation/data/panel_sim_small.parquet'

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

# per run folder
base_data_dir = Path("cubs_simulation/data")
base_data_dir.mkdir(parents=True, exist_ok=True)

run_id = f"run_{datetime.now():%Y%m%d_%H%M%S}_{np.random.randint(1000,9999)}"
run_dir = base_data_dir / run_id
run_dir.mkdir(parents=True, exist_ok=False)

# tag rows with run_id for traceability
panel_daily['run_id'] = run_id

# save simulations (parquet) into the run folder
panel_data_file_output = run_dir / "panel.parquet"
panel_daily.to_parquet(panel_data_file_output, index=False)

print(f"Saved simulation to: {panel_data_file_output}")
print(f"Run ID: {run_id}")