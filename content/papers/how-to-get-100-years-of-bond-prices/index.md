---
title: "How to get 100 years of bond ETF prices"
blurb: "
    The price of a bond ETF can be estimated from bond yields. You can use this technique to create a long term performance history of an ETF.
"

date: "2022-11-22"
type: paper
katex: true  # Enable mathematics on the page
markup: "mmark"
authors:
    - Adrian Letchford
draft: true
tags:
    - finance
---

Managing a macro portfolio of ETFs can be tricky because there isn't much price history. Without enough history, asking questions like "What happens when interest rates hit zero?" become hard to answer. Bond ETFs in particular are tricky as many of them are still relatively new. Take TLT as an example, it only started operating in the early 2000s.

These bond ETFs buy and hold bonds. If you knew what they held, you could calculate their fair price by summing together the value of the bonds they hold. Some bond ETFs continuously hold one type of bond. For example, [iShares 20+ Year Treasury Bond ETF](https://www.ishares.com/us/products/239454/ishares-20-year-treasury-bond-etf) (TLT) holds US Treasury bonds with maturities greater than 20 years. We can use this information to model TLT's price as a function of bond yields without know exactly what TLT holds.

# Modelling bond ETF returns with yields

A previous paper titled [Understanding bond ETF returns]({{<ref "/papers/understanding-bond-etf-returns" >}}) showed that a bond ETF's daily returns can be modelled with:
$$
\text{return}\_t = \frac{r_{t-1}}{260} + \frac{r_{t-1}}{r_t} \left( 1 - (1 + \frac{r_t}{p})^{-pT} \right) + (1 + \frac{r_t}{p})^{-pT} - 1
$$
where:
* \\(r_t\\) is the yield at time \\(t\\)
* \\(p\\) is the number of coupon payments per year
* \\(T\\) is the number of years until maturity

# 2002 to present

The most recent history of TLT can be downloaded from Yahoo Finance ([here](https://uk.finance.yahoo.com/quote/TLT/history?p=TLT)). As TLT pays the bond coupons as dividends, we are using the dividend adjusted price.

![](images/tlt.svg)

# 1962 to 2002

To extend TLT's price beyond 2002, we need to employ the return model from above.

![](images/daily_interest_rates.svg)
![](images/daily_index.svg)

# 1925 to 1962

![](images/monthly_interest_rates.svg)
![](images/indexes.svg)

# Putting it all together

![](images/complete_index.svg)