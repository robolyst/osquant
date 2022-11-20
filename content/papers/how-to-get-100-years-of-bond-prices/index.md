---
title: "How to get 100 years of bond ETF prices"
blurb: "
    The price of a bond ETF can be calculated from bond yields. This calculation can be used to create a long term performance history of an ETF.
"

date: "2022-11-18"
type: paper
katex: true  # Enable mathematics on the page
markup: "mmark"
authors:
    - Adrian Letchford
draft: true
tags:
    - mathematics
    - finance
---

Managing a macro portfolio of ETFs can be tricky because there isn't much price history. Without enough history, asking questions like "What happens when interest rates hit zero?" become hard to answer. Bond ETFs in particular are tricky as many of them are still relatively new. Take TLT as an example, it only started operating in the early 2000s.

These bond ETFs buy and hold bonds. If you knew what they held, you could calculate their fair price by summing together the value of the bonds they hold. Some bond ETFs continuously hold one type of bond. For example, [iShares 20+ Year Treasury Bond ETF](https://www.ishares.com/us/products/239454/ishares-20-year-treasury-bond-etf) (TLT) holds US Treasury bonds with maturities greater than 20 years. We can use this information to model TLT's price as a function of bond yields without know exactly what TLT holds.

# Modelling bond ETF returns with yields

A simple model of a bond ETF is that every day, the fund buys a bond on the open market, holds it for a day collecting interest and then sells the bond at the end of the day. The fund's return for the day is the interest earned and any capital made or lost on the price of the bond.

$$
\text{ETF return} = \text{interest} + \text{capital gains}
$$

The bond's yield tells us the interest that we earn. To calculate the capital gains, we need to know how to calculate a bond's price. The capital gain is then the return on the price of the bond.

## Bond price

A bond is a cashflow producing instrument. Throughout the holding period, the buyer receives periodic **coupons** and at the end of the bond the buyer receives the **notional** amount.

We're going to use the [present value approach](https://en.wikipedia.org/wiki/Bond_valuation#Present_value_approach) to calculate the price of a bond. 

$$
\begin{aligned}
P &= \sum^{2T}_{i=1} \frac{\frac{C}{2}N}{(1 + \frac{r}{2})^{i}} + \frac{N}{(1 + \frac{r}{2})^{2T}} \\\
&= \frac{CN}{r}\left( 1 - (1 + \frac{r}{2})^{-2T} \right) + N (1 + \frac{r}{2})^{-2T}
\end{aligned}
$$

A key thing to note is that if the coupon equals the rate (\\(C = r\\)) then the price of the bond is equal to the notional \\(P = N\\). This is exactly what happens when a new bond is issued. The price of the bond that the buyer pays is the notional amount and the coupons that they'll receive match the rate at the time of issue.


### Returns

$$
\Delta A = \frac{r_{t-1}}{12}
$$

$$
\begin{aligned}
\Delta P &= \frac{P_{t}}{P_{t-1}} - 1 \\\
&= \frac{r_{t-1}}{r_t} \left( 1 - (1 + \frac{r_t}{2})^{-2T} \right) + (1 + \frac{r_t}{2})^{-2T} - 1
\end{aligned}
$$

This assumes that the bond we buy is at par (\\(C = r_{t-1}\\)) with a notional of $1 (\\(P_{t-1} = N = 1\\)).



# 2002 to present

![](images/tlt.svg)

# 1962 to 2002

![](images/daily_interest_rates.svg)
![](images/daily_index.svg)

# 1925 to 1962

![](images/monthly_interest_rates.svg)
![](images/indexes.svg)

# Putting it all together

![](images/complete_index.svg)