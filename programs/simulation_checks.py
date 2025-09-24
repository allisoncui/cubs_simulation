import os
import pandas as pd
import numpy as np

########################################
## ---------- Housekeeping ---------- ##
########################################

pd.set_option('display.max_columns', 150)

# Set wd ----------
homedir = os.path.expanduser('~')
wd = homedir + '/OneDrive - Columbia Business School/c1-aicollab/cc_sim/'
os.chdir(wd)

# File I/O ---------- #
## << Input >> ##
panel_data_file_input = 'data/panel_sim_small.parquet'

## << Output >> ##

# Parameters ---------- #
sig_digits = 6

# Read data ---------- #
panel_daily = pd.read_parquet(panel_data_file_input)

# Credit card balances identity
panel_daily['cc_bal_id'] = (
    + panel_daily['outstanding_balance']
    - panel_daily.groupby(['id'])['outstanding_balance'].transform('shift',1)
    - panel_daily['cc_consumption_cc']
    - panel_daily['interest_charge']
    + panel_daily['payment_to_card'])

## check
cc_bal_chk = (
    panel_daily.groupby(['id'])
    ['cc_bal_id'].apply(lambda x: all(x[1:].round(sig_digits) == 0))
    .reset_index()
)

# Liquid assets identity
panel_daily['liq_assets_id'] = (
    + panel_daily['liquid_assets']
    - panel_daily.groupby(['id'])['liquid_assets'].transform('shift',1)
    - panel_daily['liquid_credits']
    + panel_daily['cash_consumption']
    + panel_daily['cc_consumption_cash']
    + panel_daily['payment_to_card'])

## check
liq_assets_chk = (
    panel_daily.groupby(['id'])
    ['liq_assets_id'].apply(lambda x: all(x[1:].round(sig_digits) == 0))
    .reset_index()
)

# Available credit identity
panel_daily['credit_avail_id'] = (
    np.where(panel_daily['outstanding_balance']>panel_daily['credit_limit'],
             panel_daily['available_credit'] - 0,
             panel_daily['available_credit'] - (panel_daily['credit_limit'] - panel_daily['outstanding_balance']))
    )

## check
credit_avail_chk = (
    panel_daily.groupby(['id'])
    ['credit_avail_id'].apply(lambda x: all(x[1:].round(sig_digits) == 0))
    .reset_index()
)

# Interest charged in and out of grace periods
panel_daily['grace_period_int'] = (panel_daily['interest_charge'] -
                                   np.where(panel_daily['is_grace_period'] == 0,
                                            panel_daily['interest_charge'],
                                            0)
                                  )

## check
grace_period_int_chk = (
    panel_daily.groupby(['id'])
    ['grace_period_int'].apply(lambda x: all(x[1:].round(sig_digits) == 0))
    .reset_index()
)

# Concatenate checks
checklist = [cc_bal_chk,
             liq_assets_chk,
             credit_avail_chk,
             grace_period_int_chk]

# Checks by user
checks = panel_daily[['id']].drop_duplicates().reset_index(drop=True)
for v in checklist:
    checks = pd.merge(checks, v, on=['id'], how='left')
print(checks.head(10))

checks['all_pass'] = checks.drop(columns=['id']).apply(lambda x: all(x), axis=1)

# All pass?
print(checks['all_pass'].value_counts())
