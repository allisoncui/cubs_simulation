# simulation_functions.py #
# Functions used to run run_panel_simulation.py #

def _lognorm_params_from_mean_sd(mean, sd):
    """Return (mu, sigma) for SciPy lognorm given linear-space mean & sd."""
    var = sd**2
    sigma2 = np.log(1.0 + var/(mean**2)) if mean > 0 else 0.0
    sigma  = np.sqrt(sigma2)
    mu     = np.log(mean) - sigma2/2.0 if mean > 0 else -1e9  # degenerate protection
    return mu, sigma


def genDerivedPars(mu_income_monthly,dt_income_paid,sd_income_monthly,mu_consumption_monthly_target,sd_consumption_monthly_target):
    # Per payment moments
    mu_income = mu_income_monthly / len(dt_income_paid)
    sd_income = np.sqrt((sd_income_monthly**2)/len(dt_income_paid))
    # Target DAILY linear consumption moments
    mu_c_daily = mu_consumption_monthly_target / 30.0
    sd_c_daily = sd_consumption_monthly_target / np.sqrt(30.0)
    # Convert to LOG-space params for SciPy's lognorm (s=sigma, scale=exp(mu))
    mu_c_log, sd_c_log = _lognorm_params_from_mean_sd(mu_c_daily, sd_c_daily)

    derived = pd.DataFrame({
        'mu_income': [mu_income],
        'sd_income': [sd_income],
        'mu_c_log':  [mu_c_log],
        'sd_c_log':  [sd_c_log],
    })
    return derived

def initSimData(T,dt_statement):
    # Create data structure
    data = pd.DataFrame({'dt': pd.date_range('2009-12-31',periods=(T+1))}) #first date is t = 0
    data['day'] = data['dt'].dt.day
    data['month'] = data['dt'].dt.month
    data['year'] = data['dt'].dt.year
    data['billing_cycle_end'] = np.where(data['day'] == dt_statement, 1, 0)
    data['billing_cycle'] = data['billing_cycle_end'].cumsum() + 1
    data['billing_cycle'] = np.where(data['billing_cycle_end'] == 1, data['billing_cycle'] - 1, data['billing_cycle'])
    
    # Determine first full month (month with at least 28 days)
    days_per_monthyr = data.groupby(['month','year']).size().reset_index(name='ndays').sort_values(['year','month'])
    first_full_month = days_per_monthyr[days_per_monthyr['ndays'] >= 28].sort_values(['year','month'])['month'][0]
    first_full_year = days_per_monthyr[days_per_monthyr['ndays'] >= 28].sort_values(['year','month'])['year'][0]
    first_full_dt = str(first_full_year)+'-'+str(first_full_month).rjust(2, '0')+'-01'
    first_full_dt = ciso8601.parse_datetime(first_full_dt)
    return [data,first_full_dt]

def SimulateBalances(T=365,
                     liquid_assets0=1e+04,mu_income_monthly=2000,sd_income_monthly=5,dt_income_paid=[14,28],
                     cash_consumption_share=.5,mu_consumption_monthly_target=100,sd_consumption_monthly_target=5,
                     dt_payment_due=15,dt_statement=23,min_payment_flat=35,fee_late=20,rate_interest=.17,credit_limit=8000,
                     prob_off_date_pay_card=0,prob_fullpay_cond_suffliq=.8,prob_liqmin_cond_suffliq=1
                    ):
    # Initiate data structure
    initSim = initSimData(T,dt_statement)
    data = initSim[0]
    first_full_dt = initSim[1]

    # Generate derived variables
    derived = genDerivedPars(mu_income_monthly,dt_income_paid,sd_income_monthly,mu_consumption_monthly_target,sd_consumption_monthly_target)
    mu_income = derived['mu_income']
    sd_income  = derived['sd_income']
    # Corrected lognormal params
    mu_c_log = derived['mu_c_log']
    sd_c_log = derived['sd_c_log']
    
    for t in range((T+1)):
        dt_today = data['dt'][t]
        day_today = data['day'][t]
        is_paydue_date = (day_today == dt_payment_due) & (dt_today >= first_full_dt)
        is_statement_date = (day_today == dt_statement)
        cycle = data.loc[t,'billing_cycle']
        
        # Day 0
        if t==0:
            data.loc[t,'liquid_assets'] = liquid_assets0
            data.loc[t,'liquid_credits'] = 0
            data.loc[t,'consumption'] = 0
            data.loc[t,'cash_consumption'] = 0
            data.loc[t,'cc_consumption_cc'] = 0
            data.loc[t,'cc_consumption_cash'] = 0
            data.loc[t,'cc_consumption'] = 0
            data.loc[t,'credit_limit'] = credit_limit
            data.loc[t,'available_credit'] = credit_limit
            data.loc[t,'outstanding_balance'] = 0
            data.loc[t,'statement_balance'] = 0
            data.loc[t,'residual_statement_balance'] = 0
            data.loc[t,'payment_to_card'] = 0
            data.loc[t,'interest_charge'] = 0
            data.loc[t,'late_fee'] = 0
            data.loc[t,'is_grace_period'] = 1
            data.loc[t,'is_revolving'] = 0
            
        # Day 1+
        else:
            # 1. Rollover liquid assets, available credit, outstanding/statement balances, credit limit
            data.loc[t,'liquid_assets'] = data.loc[(t-1),'liquid_assets']
            data.loc[t,'credit_limit'] = data.loc[(t-1),'credit_limit']
            data.loc[t,'available_credit'] = data.loc[(t-1),'available_credit']
            data.loc[t,'outstanding_balance'] = data.loc[(t-1),'outstanding_balance']
            data.loc[t,'statement_balance'] = data.loc[(t-1),'statement_balance']
            data.loc[t,'residual_statement_balance'] = data.loc[(t-1),'residual_statement_balance']
            
            # 2. Distribute income
                # Is pay day
            if np.isin(day_today,dt_income_paid):
                data.loc[t,'liquid_credits'] = np.random.normal(mu_income,sd_income,1)
                data.loc[t,'liquid_assets'] = data.loc[t,'liquid_assets'] + data.loc[t,'liquid_credits']
                # Is not pay day
            else:
                data.loc[t,'liquid_credits'] = 0
    
            # 3. Generate random consumption
            # consumption = np.random.normal(mu_consumption,sd_consumption,1)
            # consumption = lognorm.rvs(scale=mu_consumption,s=sd_consumption, size=1, random_state=None)[0]
            consumption = lognorm.rvs(s=float(sd_c_log), scale=float(np.exp(mu_c_log)), size=1)[0]
            cash_consumption = consumption * cash_consumption_share
            cc_consumption = consumption * (1-cash_consumption_share)
    
            # 4. Generate feasible consumption
                # Cover cash expenditure first
            cash_consumption = min(cash_consumption,data.loc[t,'liquid_assets'])
                # Cover credit card expenditure with credit card
            cc_consumption_cc = min(cc_consumption,data.loc[t,'available_credit'])
                # Cover residual credit card expenditure with cash
            cc_consumption_cash = min((cc_consumption - cc_consumption_cc),(data.loc[t,'liquid_assets'] - cash_consumption))
                # Update total consumption
            consumption = cash_consumption + cc_consumption_cc + cc_consumption_cash
            
            # 4. Debit
            data.loc[t,'liquid_assets'] = data.loc[t,'liquid_assets'] - cash_consumption - cc_consumption_cash
            data.loc[t,'outstanding_balance'] = data.loc[t,'outstanding_balance'] + cc_consumption_cc
                # Balance can exceed credit limit due to interest + fees
            data.loc[t,'available_credit'] = min(data.loc[t,'credit_limit'], max(data.loc[t,'credit_limit'] - data.loc[t,'outstanding_balance'], 0.0))
            data.loc[t,'consumption'] = consumption
            data.loc[t,'cash_consumption'] = cash_consumption
            data.loc[t,'cc_consumption_cc'] = cc_consumption_cc
            data.loc[t,'cc_consumption_cash'] = cc_consumption_cash
            data.loc[t,'cc_consumption'] = cc_consumption_cc + cc_consumption_cash
            
            # 5. Compute minimum payment
            # often $25-35 flat fee or 1% (see https://www.experian.com/blogs/ask-experian/how-is-your-credit-card-minimum-payment-calculated/)
            # often computed on statement balance (see https://www.experian.com/blogs/ask-experian/how-is-your-credit-card-minimum-payment-calculated/
            min_payment_percent = 0.01
            # max(flat, 1% of statement balance)
            data.loc[t,'min_payment'] = max(min_payment_flat, min_payment_percent * max(data.loc[t,'statement_balance'], 0.0))
    
            # 6. Payments to card (assume payment due after first full month)
                # Is payment due date
            if (is_paydue_date):
                amt_due = max(data.loc[t,'residual_statement_balance'], 0)
                # Sufficient liquidity for full payment?
                suffliq = data.loc[t,'liquid_assets'] >= amt_due
                if suffliq:
                    # If sufficient liquidity, pay in full with prob `prob_fullpay_cond_suffliq`
                    pay_in_full = bool(bernoulli.rvs(p=prob_fullpay_cond_suffliq,size=1)[0])
                    if pay_in_full:
                        # Pay in full
                        payment = amt_due
                    else:
                        # Pay partial according to Uniform
                        # Possibly take into consideration minimum payment - more likely to pay in min payment range and less outside
                        payment = np.random.uniform(0,1) * amt_due
                # Insufficient liquidity for full payment
                else:
                    # Pay as much as possible with prob `prob_liqmin_cond_suffliq`
                    pay_liquidity_minimizing = bool(bernoulli.rvs(p=prob_liqmin_cond_suffliq,size=1)[0])
                    if pay_liquidity_minimizing:
                        # Pay as much as possible
                        payment = data.loc[t,'liquid_assets']
                    else:
                        # Pay partial according to Uniform
                        payment = np.random.uniform(0,1) * data.loc[t,'liquid_assets']
                # Charge late fee if payment is less than minimum of (minimum payment, amount due)
                # Also lose grace period if payment is missed (https://www.experian.com/blogs/ask-experian/what-happens-when-you-lose-grace-period/)
                if payment < min(data.loc[t,'min_payment'], amt_due):
                    data.loc[t,'late_fee'] = fee_late
                    data.loc[t,'is_grace_period'] = 0
                else:
                    data.loc[t,'late_fee'] = 0
                # Is not payment due date
            else:
                # Cannot be charged late fee
                data.loc[t,'late_fee'] = 0
                # Make payment anyway with probability `prob_off_date_pay_card`
                make_off_duedate_payment = bool(bernoulli.rvs(p=prob_off_date_pay_card,size=1)[0])
                if make_off_duedate_payment:
                    # Pay some fraction of min of liquid assets (to make sure a payment can be made), outstanding balance (so there is something to pay)
                    payment = np.random.uniform(0,1) * (min(data.loc[t,'liquid_assets'], data.loc[t,'outstanding_balance']))
                else:
                    payment = 0
            # CLAMP: never pay more than cash on hand or current outstanding (prevents overpay/negative)
            payment = max(0.0, min(float(payment), float(data.loc[t,'outstanding_balance']),float(data.loc[t,'liquid_assets'])))
            data.loc[t,'payment_to_card'] = payment
            data.loc[t,'liquid_assets'] = data.loc[t,'liquid_assets'] - payment
            data.loc[t,'residual_statement_balance'] = data.loc[t,'residual_statement_balance'] - payment
            data.loc[t,'outstanding_balance'] = data.loc[t,'outstanding_balance'] - payment
            # CLAMP: prevent tiny negatives due to float math / over-clamped payment
            data.loc[t,'outstanding_balance'] = max(data.loc[t,'outstanding_balance'], 0.0)
            # CLAMP: keep available credit within [0, credit_limit]
            data.loc[t,'available_credit'] = min(data.loc[t,'credit_limit'],max(data.loc[t,'credit_limit'] - data.loc[t,'outstanding_balance'], 0.0))    
            # 7. Determine if agent is revolving
                # If payment is due, you are revolving if you did not pay down statement, otherwise not revolving
            if (is_paydue_date):
                if (data.loc[t,'residual_statement_balance'] > 0):
                    # Lose grace period for rest of this billing cycle
                    data.loc[t:,'is_grace_period'] = np.where(data.loc[t:,'billing_cycle'] == data.loc[t,'billing_cycle'], 0, data.loc[t:,'is_grace_period'])
                    # And lose grace period for next billing cycle (https://www.nerdwallet.com/article/credit-cards/credit-card-grace-period#:~:text=When%20your%20credit%20card%20is,never%20pay%20interest%20on%20purchases.)
                    data.loc[t:,'is_grace_period'] = np.where(data.loc[t:,'billing_cycle'] == (data.loc[t,'billing_cycle'] + 1), 0, data.loc[t:,'is_grace_period'])
                    data.loc[t,'is_revolving'] = 1
                elif (data.loc[t,'residual_statement_balance'] <= 0):
                    # recover grace period for next billing cycle (this is an assumption, in practice it might take 2,3,... billing cycles to recover)
                    data.loc[t:,'is_grace_period'] = np.where(data.loc[t:,'billing_cycle'] == (data.loc[t,'billing_cycle'] + 1), 1, data.loc[t:,'is_grace_period'])
                    data.loc[t,'is_revolving'] = 0
                # If payment is not due, you are revolving only if you did not pay down statement while outside grace period
            else:
                # Can only change grace period on due date, unless it recovered for this cycle based on last payment. If it recovered, it will be non-empty for this cycle since it was already set at payment
                data.loc[t,'is_grace_period'] = np.where(np.isnan(data.loc[t,'is_grace_period']), data.loc[t-1,'is_grace_period'], data.loc[t,'is_grace_period'])
                if ( (data.loc[t,'residual_statement_balance'] > 0) & (data.loc[t,'is_grace_period'] == 0) ):
                    data.loc[t,'is_revolving'] = 1
                else:
                    data.loc[t,'is_revolving'] = 0
    
            # 8. If outside of grace period, charge interest (https://www.chase.com/personal/credit-cards/education/interest-apr/when-does-interest-start-to-accrue-on-credit-card)
            # Note: This means you can be charged interest even if not revolving (i.e. paid statement balance in full) if grace period is not recovered
            if (data.loc[t,'is_grace_period'] == 0):
                data.loc[t,'interest_charge'] = data.loc[t,'outstanding_balance']*(rate_interest/365)
                data.loc[t,'outstanding_balance'] = data.loc[t,'outstanding_balance'] + data.loc[t,'interest_charge']
                data.loc[t,'available_credit'] = min(data.loc[t,'credit_limit'],max(data.loc[t,'credit_limit'] - data.loc[t,'outstanding_balance'], 0.0))            
            else:
                data.loc[t,'interest_charge'] = 0
    
            # 9. Record statement balance
            if is_statement_date:
                data.loc[t,'statement_balance'] = data.loc[t,'outstanding_balance']
                data.loc[t,'residual_statement_balance'] = data.loc[t,'statement_balance']
    
    # Give simulation an ID upon exit
    data['id'] = randint(1000000000,9999999999)
    return data

def SimulatePanel(N,T):
    liquid_assets0_grid = lognorm.rvs(scale=1e+05,s=.5, size=N, random_state=None).tolist()
    mu_income_monthly_grid = lognorm.rvs(scale=1.2e+05/12,s=.5, size=N, random_state=None).tolist()
    sd_income_monthly_grid = lognorm.rvs(scale=3,s=.5, size=N, random_state=None).tolist()
    mu_consumption_monthly_target_grid = lognorm.rvs(scale=3000,s=.5, size=N, random_state=None).tolist()
    sd_consumption_monthly_target_grid = lognorm.rvs(scale=3,s=.5, size=N, random_state=None).tolist()
    dt_income_paid_grid = [[14,28]]*int((np.ceil(N*.7))) # Force 70% of users to be paid bi-weekly
    if (N > len(dt_income_paid_grid)):
        dt_income_paid_grid = dt_income_paid_grid + choices([[1],[28],[7,14,21,28]],k=(N-len(dt_income_paid_grid))) # others randomly choose payment schedule
    cash_consumption_share_grid = np.random.uniform(low=0,high=1,size=N).tolist()
    dt_payment_due_grid = choices([12,13,14,15,16,17,18], k = N)
    stmt_offsets = choices([10,9,8,7,6], k=N)   # draw statement offsets once
    dt_statement_grid = [dt_payment_due_grid[i] + stmt_offsets[i] for i in range(N)]
    def interest(x):
        if x < 8e+04:
            return .2
        elif x < 1.6e+05:
            return .16
        else:
            return .13
    rate_interest_grid = [interest(x) for x in liquid_assets0_grid]
    credit_limit_grid = [1e+03*np.ceil((x*12*0.5)/1e+03) for x in mu_income_monthly_grid] #half of annual income rounded to nearest 1K

    frames = []  # collect per-person frames with this array
    for i in tqdm(range(N)):
        # Run simulation for i
        data_d = SimulateBalances(
            T,
            liquid_assets0=liquid_assets0_grid[i],
            mu_income_monthly=mu_income_monthly_grid[i],
            sd_income_monthly=sd_income_monthly_grid[i],
            dt_income_paid=dt_income_paid_grid[i],
            cash_consumption_share=cash_consumption_share_grid[i],
            mu_consumption_monthly_target=mu_consumption_monthly_target_grid[i],
            sd_consumption_monthly_target=sd_consumption_monthly_target_grid[i],
            dt_payment_due=dt_payment_due_grid[i],
            dt_statement=dt_statement_grid[i],
            min_payment_flat=35, fee_late=20,
            rate_interest=rate_interest_grid[i],
            credit_limit=credit_limit_grid[i],
            prob_off_date_pay_card=0.05, prob_fullpay_cond_suffliq=.85, prob_liqmin_cond_suffliq=1
        )

        # Append simulation parameter values
        data_d['par_rate_interest'] = rate_interest_grid[i]
        data_d['par_dt_payment_due'] = dt_payment_due_grid[i]
        data_d['par_dt_statement_due'] = dt_statement_grid[i]
        data_d['par_liquid_assets0'] = liquid_assets0_grid[i]
        data_d['par_mu_income_monthly'] = mu_income_monthly_grid[i]
        data_d['par_sd_income_monthly'] = sd_income_monthly_grid[i]
        data_d['par_dt_income_paid'] = ','.join([str(x) for x in dt_income_paid_grid[i]])
        data_d['par_cash_consumption_share'] = cash_consumption_share_grid[i]
        data_d['par_mu_consumption_monthly_target'] = mu_consumption_monthly_target_grid[i]
        data_d['par_sd_consumption_monthly_target'] = sd_consumption_monthly_target_grid[i]
        data_d['par_dt_payment_due'] = dt_payment_due_grid[i]
        data_d['par_dt_statement'] = dt_statement_grid[i]
        data_d['par_min_payment_flat'] = 35
        data_d['par_fee_late'] = 20
        data_d['par_credit_limit'] = credit_limit_grid[i]
        data_d['par_prob_off_date_pay_card'] = 0.05
        data_d['par_prob_fullpay_cond_suffliq'] = .85
        data_d['par_prob_liqmin_cond_suffliq'] = 1

        # Collect instead of concatenating
        frames.append(data_d)

    # One concat at the end (O(N) instead of O(N^2))
    panel = pd.concat(frames, axis=0, ignore_index=True)
    # Place 'id' first
    panel = panel[['id'] + [x for x in panel.columns if x != 'id']].reset_index(drop=True)
    return panel
