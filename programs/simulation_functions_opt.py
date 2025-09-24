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

    return mu_income, sd_income, mu_c_log, sd_c_log

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


def SimulateBalances_fast(
    T=365,
    liquid_assets0=1e+04, mu_income_monthly=2000, sd_income_monthly=5, dt_income_paid=[14,28],
    cash_consumption_share=.5, mu_consumption_monthly_target=100, sd_consumption_monthly_target=5,
    dt_payment_due=15, dt_statement=23, min_payment_flat=35, fee_late=20, rate_interest=.17, credit_limit=8000,
    prob_off_date_pay_card=0, prob_fullpay_cond_suffliq=.8, prob_liqmin_cond_suffliq=1,
    rng=None
):
    """
    Fast version of SimulateBalances; uses numpy arrays inside the daily loop
    and constructs the DataFrame once at the end
    """
    if rng is None:
        rng = np.random.default_rng()

    # vecotrized calendar
    dt = pd.date_range('2009-12-31', periods=T+1).to_numpy()  # datetime64[ns]
    day = pd.DatetimeIndex(dt).day.values
    month = pd.DatetimeIndex(dt).month.values
    year = pd.DatetimeIndex(dt).year.values

    billing_cycle_end = (day == dt_statement).astype(np.int8)
    billing_cycle = np.cumsum(billing_cycle_end) + 1
    billing_cycle[billing_cycle_end == 1] -= 1
    # first full month start
    first_full_dt = np.datetime64('2010-01-01')
    # derived params with lognormal
    mu_income, sd_income, mu_c_log, sd_c_log = genDerivedPars(mu_income_monthly, dt_income_paid, sd_income_monthly, mu_consumption_monthly_target, sd_consumption_monthly_target)

    # preallocate arrays in numpy
    liquid_assets = np.zeros(T+1, dtype=np.float64)
    liquid_credits = np.zeros(T+1, dtype=np.float64)
    consumption = np.zeros(T+1, dtype=np.float64)
    cash_consumption = np.zeros(T+1, dtype=np.float64)
    cc_consumption_cc = np.zeros(T+1, dtype=np.float64)
    cc_consumption_cash = np.zeros(T+1, dtype=np.float64)
    cc_consumption_total = np.zeros(T+1, dtype=np.float64)
    credit_limit_arr = np.full(T+1, float(credit_limit), dtype=np.float64)
    available_credit = np.zeros(T+1, dtype=np.float64)
    outstanding_balance = np.zeros(T+1, dtype=np.float64)
    statement_balance = np.zeros(T+1, dtype=np.float64)
    residual_statement = np.zeros(T+1, dtype=np.float64)
    payment_to_card = np.zeros(T+1, dtype=np.float64)
    interest_charge = np.zeros(T+1, dtype=np.float64)
    late_fee_arr = np.zeros(T+1, dtype=np.float64)
    is_grace_period = np.zeros(T+1, dtype=np.int8)
    is_revolving = np.zeros(T+1, dtype=np.int8)
    min_payment_arr = np.zeros(T+1, dtype=np.float64)

    # day 0
    liquid_assets[0] = float(liquid_assets0)
    available_credit[0]= float(credit_limit)
    is_grace_period[0] = 1

    # daily loop - read/write to array instead of df here
    min_payment_pct = 0.01
    for t in range(1, T+1):
        # 1) carry forward
        liquid_assets[t] = liquid_assets[t-1]
        available_credit[t] = available_credit[t-1]
        outstanding_balance[t] = outstanding_balance[t-1]
        statement_balance[t] = statement_balance[t-1]
        residual_statement[t] = residual_statement[t-1]
        credit_limit_arr[t] = credit_limit_arr[t-1]

        # 2) income on payday
        if day[t] in dt_income_paid:
            liquid_credits[t] = rng.normal(mu_income, sd_income)
            liquid_assets[t] += liquid_credits[t]

        # 3) desired consumption (lognormal, corrected params)
        cons = rng.lognormal(mean=mu_c_log, sigma=sd_c_log)
        cash_desired = cons * cash_consumption_share
        cc_desired = cons - cash_desired

        # 4) feasible consumption (clamped)
        cash_use = min(cash_desired, liquid_assets[t])
        cc_cc = min(cc_desired, available_credit[t])
        cc_cash = min(cc_desired - cc_cc, max(liquid_assets[t] - cash_use, 0.0))
        cons = cash_use + cc_cc + cc_cash
        # apply consumption debits
        liquid_assets[t] -= (cash_use + cc_cash)
        outstanding_balance[t] += cc_cc
        # available credit clamp
        available_credit[t] = min(credit_limit_arr[t], max(credit_limit_arr[t] - outstanding_balance[t], 0.0))
        # write 
        consumption[t] = cons
        cash_consumption[t] = cash_use
        cc_consumption_cc[t] = cc_cc
        cc_consumption_cash[t] = cc_cash
        cc_consumption_total[t] = cc_cc + cc_cash

        # 5) minimum payment rule
        min_payment_arr[t] = max(min_payment_flat, min_payment_pct * max(statement_balance[t], 0.0))

        # 6) payments
        is_due_day = (day[t] == dt_payment_due) and (dt[t] >= first_full_dt)
        if is_due_day:
            amt_due = max(residual_statement[t], 0.0)
            suffliq = liquid_assets[t] >= amt_due
            if suffliq:
                pay_in_full = (rng.random() < prob_fullpay_cond_suffliq)
                payment = amt_due if pay_in_full else rng.uniform(0.0, 1.0) * amt_due
            else:
                pay_liqmin = (rng.random() < prob_liqmin_cond_suffliq)
                payment = liquid_assets[t] if pay_liqmin else rng.uniform(0.0, 1.0) * liquid_assets[t]

            # late fee & immediate grace toggle for under min payment
            if payment < min(min_payment_arr[t], amt_due):
                late_fee_arr[t] = fee_late
                is_grace_period[t] = 0
            else:
                late_fee_arr[t] = 0.0
        else:
            late_fee_arr[t] = 0.0
            if rng.random() < prob_off_date_pay_card:
                payment = rng.uniform(0.0, 1.0) * min(liquid_assets[t], outstanding_balance[t])
            else:
                payment = 0.0

        # CLAMP payment to outstanding & cash on hand
        payment = max(0.0, min(payment, outstanding_balance[t], liquid_assets[t]))
        # apply payment
        payment_to_card[t] = payment
        liquid_assets[t] -= payment
        residual_statement[t] -= payment
        outstanding_balance[t] -= payment
        outstanding_balance[t] = max(outstanding_balance[t], 0.0)
        # recompute available credit (clamped)
        available_credit[t] = min(credit_limit_arr[t], max(credit_limit_arr[t] - outstanding_balance[t], 0.0))

        # 7) revolving + grace logic
        if is_due_day:
            if residual_statement[t] > 0:
                cyc = billing_cycle[t]
                mask_now  = (billing_cycle[t:] == cyc)
                mask_next = (billing_cycle[t:] == (cyc + 1))
                is_grace_period[t:][mask_now | mask_next] = 0
                is_revolving[t] = 1
            else:
                next_cyc = billing_cycle[t] + 1
                mask_next = (billing_cycle[t:] == next_cyc)
                is_grace_period[t:][mask_next] = 1
                is_revolving[t] = 0
        else:
            # carry/set values; revolve if statement still owed and no grace
            if (residual_statement[t] > 0) and (is_grace_period[t] == 0):
                is_revolving[t] = 1
            else:
                # carry forward grace when not explicitly set today
                if t > 0 and is_grace_period[t] not in (0, 1):
                    is_grace_period[t] = is_grace_period[t-1]
                is_revolving[t] = 0

        # 8) interest accrues if no grace
        if is_grace_period[t] == 0:
            interest = outstanding_balance[t] * (rate_interest / 365.0)
            interest_charge[t] = interest
            outstanding_balance[t] = outstanding_balance[t] + interest
            available_credit[t] = min(credit_limit_arr[t], max(credit_limit_arr[t] - outstanding_balance[t], 0.0))
        else:
            interest_charge[t] = 0.0

        # 9) statement close
        if day[t] == dt_statement:
            statement_balance[t]  = outstanding_balance[t]
            residual_statement[t] = statement_balance[t]

    # build dataframe ONCE
    out = pd.DataFrame({
        'dt': dt,
        'day': day,
        'month': month,
        'year': year,
        'billing_cycle_end': billing_cycle_end,
        'billing_cycle': billing_cycle,
        'liquid_assets': liquid_assets,
        'liquid_credits': liquid_credits,
        'consumption': consumption,
        'cash_consumption': cash_consumption,
        'cc_consumption_cc': cc_consumption_cc,
        'cc_consumption_cash': cc_consumption_cash,
        'cc_consumption': cc_consumption_total,
        'credit_limit': credit_limit_arr,
        'available_credit': available_credit,
        'outstanding_balance': outstanding_balance,
        'statement_balance': statement_balance,
        'residual_statement_balance': residual_statement,
        'payment_to_card': payment_to_card,
        'interest_charge': interest_charge,
        'late_fee': late_fee_arr,
        'is_grace_period': is_grace_period,
        'is_revolving': is_revolving,
        'min_payment': min_payment_arr,
    })

    # give simulation an ID upon exit vectorized
    out['id'] = randint(1000000000, 9999999999)
    return out


def SimulatePanel(N,T):
    liquid_assets0_grid = lognorm.rvs(scale=1e+05,s=.5, size=N, random_state=None).tolist()
    mu_income_monthly_grid = lognorm.rvs(scale=1.2e+05/12,s=.5, size=N, random_state=None).tolist()
    sd_income_monthly_grid = lognorm.rvs(scale=3,s=.5, size=N, random_state=None).tolist()
    # monthly consumption mean
    mu_consumption_monthly_target_grid = lognorm.rvs(scale=3000, s=.5, size=N, random_state=None).tolist()
    # draw coefficient of variation (20%–60%), controls relativity and gives daily consumption a realistic spread
    cv_grid = np.random.uniform(low=0.2, high=0.6, size=N)
    # convert to SD as fraction of mean
    sd_consumption_monthly_target_grid = (cv_grid * np.array(mu_consumption_monthly_target_grid)).tolist()

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
        data_d = SimulateBalances_fast(
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
