# Variable list

| Variable name | Description |
| :---: | :--- |
| `id` | Individual identifier |
| `dt` | Day of observation (decomposed into day, month, year) |
| `billing_cycle` | Integer indicating current billing cycle for dt |
| `is_grace_period` | Indicator returning 1 if user is in a grace period |
| `billing_cycle_end` | Indicator returning 1 if dt is last day of billing cycle |
| `credit_limit` | Card credit limit |
| `par_min_payment_flat` | Card minimum payment |
| `par_rate_interest` | Card APR |
| `par_fee_late` | Card late fee |
| `par_dt_payment_due` | Card payment date |
| `par_dt_statement` | Card statement date |
| `liquid_assets` | Amount of user's cash balance |
| `liquid_credits` | Amount of credits to user's cash account (income) |
| `consumption` | Amount of user's cash and credit spending |
| `cash_consumption` | Amount of user's cash spending on cash-only goods (excluding card payments) |
| `cc_consumption_cash` | Amount of user's cash spending on non-cash-only goods (excluding card payments) |
| `cc_consumption_cc` | Amount of user's credit card spending on non-cash-only goods |
| `cc_consumption` | Amount of user's spending on non-cash-only goods |
| `available_credit` | Amount of card available credit |
| `outstanding_balance` | Amount of card outstanding balance |
| `statement_balance` | Amount of card statement balance |
| `payment_to_card` | Amount of cash payment to credit card |
| `interest_charge` | Amount of interest accrued to credit card balance |
| `late_fee` | Amount of late fee charged |
