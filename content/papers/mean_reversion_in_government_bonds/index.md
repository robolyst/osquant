---
title: "Mean reversion in government bonds"
summary: "
Using the Vasicek model, you can calculate the expected return of a government bond spread. Furthermore, you can calculate the expected value of trading a yield curve inversion with ETFs.
"
type: paper
katex: true # Enable mathematics on the page
feature: false
date: "2023-03-27"
authors:
    - Adrian Letchford
categories:
    - mathematics
    - finance
notebook: https://api.observablehq.com/@dradrian-workspace/example-hugo-integration.js?v=3
---

# Interest rate model

We're going to create a model of the long term interest rate \\(r_l(t)\\) and the spread between the long term rate and the short term rate \\(s(t) = r_l(t) - r_s(t)\\). We'll combine these two models to create a model of the short term rate \\(r_s(t)\\). This is a type of [Vasicek model](https://en.wikipedia.org/wiki/Vasicek_model)[^Vasichek1977] known as a two-factor equilibrium Vasicek model [^Souleymanou2021].

For the long term rates we'll use the [30 year US government treasury yields (DGS30)](https://fred.stlouisfed.org/series/DGS30). For the short term rates, we'll use the [5 year US government treasury yields (DGS5)](https://fred.stlouisfed.org/series/DGS5).

The data looks like:
<cell id="rates_plot"></cell>

## Model the long term rate

We want to make as few assumptions as possible about the underlying interest rate. We will model the interest rate of the long term bond \\(r_l(t)\\) as Brownian motion:
$$
d r_l(t) = \sigma_l dW_l(t)
$$

Which means that this processes has normally distributed increments. The conditional moments of the long term rates are:
$$
\begin{aligned}
E[r_l(t) | r_l(0)] &= r_l(0) \\\
\text{var}[r_l(t) | r_l(0)] &= \sigma_l^2 t \\\
\end{aligned}
$$

You can play with the long term interest rate model here:

<cell id="long_rate_model_plot"></cell>

Model parameters:

<cell id="viewof_long_sigma"></cell>
<cell id="viewof_position"></cell>

## Model the spread

We are going to assume that the spread between the long term rate and the short term rate \\(s(t)\\) is mean reverting. For this we will use an [Ornstein–Uhlenbeck process](https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process) that is uncorrelated to the long term rate:
$$
\begin{aligned}
d s(t) &= \theta_s (\mu_s - s(t))dt + \sigma_s dW_s(t) \\\
E[dW_l(t)dW_s(t)] &= 0 \\\
\end{aligned}
$$
The conditional moments of the spread are [^Holy2022]:
$$
\begin{aligned}
E[s(t) | s(0)] &= s(0) e^{-\theta_s t} + \mu_s(1 - e^{-\theta_s t}) \\\
\text{var}[s(t) | s(0)] &= \frac{\sigma_s^2}{2 \theta_s}(1 - e^{-2\theta_s t}) \\\
\end{aligned}
$$

You can play with the spread model here:

<cell id="spread_model_plot"></cell>

Model parameters:

<cell id="viewof_spread_mean"></cell>
<cell id="viewof_spread_speed"></cell>
<cell id="viewof_spread_std"></cell>
<cell id="viewof_position_2"></cell>

## Model the short term rate

Based on the defintion of the spread \\(s(t) = r_l(t) - r_s(t)\\) the short term rate is:
$$
r_s(t) = r_l(t) - s(t)
$$
Which gives us  the moments of the short term rates as functions of the long term rate and spread:

$$
\begin{aligned}
E[r_s(t)|r_s(0)] &= E[r_l(t) | r_l(0)] - E[s(t) | s(0)] \\\
& = r_l(0) - s(0) e^{-\theta_s t} - \mu_s(1 - e^{-\theta_s t}) \\\
\end{aligned}
$$

$$
\begin{aligned}
\text{var}[r_s(t)|r_s(0)] &= \text{var}[r_l(t) | r_l(0)] + \text{var}[s(t) | s(0)] \\\
& = \sigma_l^2 t + \frac{\sigma_l^2}{2 \theta_s}(1 - e^{-2\theta_s t}) \\\
\end{aligned}
$$

The covariance between the long and short term rate is:
$$
\begin{aligned}
\text{cov}(r_l(t), r_s(t) | r_l(0), r_s(0)) &= 
[1, -1]
\left[\begin{matrix}
\text{var}[r_l(t) | r_l(0)] & 0 \\\
0 & \text{var}[s(t) | s(0)] \\\
\end{matrix}\right]
\left[\begin{matrix}
1 \\\
0 \\\
\end{matrix}\right] \\\
&= \text{var}[r_l(t) | r_l(0)] \\\
\end{aligned}
$$


<cell id="long_short_rate_model_plot"></cell>

Model parameters:

<cell id="viewof_long_sigma_2"></cell>
<cell id="viewof_spread_mean_2"></cell>
<cell id="viewof_spread_speed_2"></cell>
<cell id="viewof_spread_std_2"></cell>
<cell id="viewof_position_3"></cell>


# ETF Model

# Estimating parameters

# Estimating portfolio weights

# Results

# Conclusions

# Appendix
## Taylor expansion of bond returns

A previous paper titled [Understanding bond ETF returns]({{< ref "/papers/understanding-bond-etf-returns" >}}) showed that a bond ETF's daily returns can be modelled from bond yields:
$$
\begin{aligned}
R(r_t)  &= \frac{r_{t-1}}{f} + \frac{r_{t-1}}{r_t} \left( 1 - (1 + \frac{r_t}{p})^{-pT} \right) + (1 + \frac{r_t}{p})^{-pT} - 1
\end{aligned}
$$
This is a very difficult equation to work with. If we knew the moments of \\(r_t\\) I'm not sure how you would calculate the moments of \\(R(r_t)\\).

However, this function appears to be almost linear. The chart below shows that this function is very close to a straight line:

<cell id="etf_return_plot"></cell>

This chart suggests a second order [Taylor expansion](https://en.wikipedia.org/wiki/Taylor_series) could be used to simplify \\(R(r_t)\\) and allow us to estimate the moments of \\(R(r_t)\\). I've shown the derivation of the second order Taylor expansion is in the notes that follow. The chart below shows that this is a good approximation for the ETF returns.

<cell id="taylor_expansion_plot"></cell>

### First order Taylor expansion

The first derivative is:
$$
\begin{aligned}
R^\prime(r_t) &=\frac{d}{dr_t} \frac{r_{t-1}}{r_t} - \frac{d}{dr_t} \left[\frac{r_{t-1}}{r_t} (1 + \frac{r_t}{p})^{-pT} \right] + \frac{d}{dr_t}(1 + \frac{r_t}{p})^{-pT} \\\
&=\frac{d}{dr_t} \frac{r_{t-1}}{r_t} - \frac{d}{dr_t} \left[\frac{r_{t-1}}{r_t}\right] (1 + \frac{r_t}{p})^{-pT} -  \frac{r_{t-1}}{r_t} \frac{d}{dr_t}\left[ (1 + \frac{r_t}{p})^{-pT}\right]  + \frac{d}{dr_t}(1 + \frac{r_t}{p})^{-pT} \\\
&=\frac{d}{dr_t} \left[\frac{r_{t-1}}{r_t}\right] \left(1 - (1 + \frac{r_t}{p})^{-pT}\right) + (1 -  \frac{r_{t-1}}{r_t}) \frac{d}{dr_t}\left[ (1 + \frac{r_t}{p})^{-pT}\right] \\\
\end{aligned}
$$

We have:
$$
\begin{equation}
\frac{d}{dr_t} \frac{r_{t-1}}{r_t} = -\frac{r_{t-1}}{r_t^2} \label{A1}\tag{A1}
\end{equation}
$$
and (lazily) using [Wolfram Alpha](https://www.wolframalpha.com/input?i=%281+%2B+x%2Fa%29%5E%28-c%29) I get:
$$
\begin{equation}
\frac{d}{dr_t}(1 + \frac{r_t}{p})^{-pT} = -\frac{pT}{p + r_t} (\frac{p + r_t}{p})^{-pT} \label{A2}\tag{A2}
\end{equation}
$$
giving:
$$
\begin{aligned}
R^\prime(r_t) &= -\frac{r_{t-1}}{r_t^2} \left(1 - (1 + \frac{r_t}{p})^{-pT}\right) - (1 -  \frac{r_{t-1}}{r_t})\frac{pT}{p + r_t} (\frac{p + r_t}{p})^{-pT} \\\
\end{aligned}
$$


The first order Taylor expansion is:
$$
R_1(r_t) = R(r_{t-1}) + \frac{R^\prime(r_{t-1})}{1!}(r_t - r_{t-1})
$$

Because we have taken the Taylor expansion around \\(r_{t-1}\\) instead of 0, some of the terms in \\(R(r_{t-1})\\) and \\(R^\prime(r_{t-1})\\) cancel out giving:
$$
\begin{aligned}
R(r_{t-1})  &= \frac{r_{t-1}}{f}  \\\
R^\prime(r_{t-1}) &= -\frac{1}{r_{t-1}} \left(1 - (1 + \frac{r_{t-1}}{p})^{-pT}\right) \\\
R_1(r_t) &= R(r_{t-1}) + R^\prime(r_{t-1})(r_t - r_{t-1})
\end{aligned}
$$

### Second order Taylor expansion

Taking the second derivative is quite tedious. Starting off:
$$
\begin{aligned}
R^{\prime\prime}(r_t) = \frac{d}{dr_t}R^\prime(r_t) =& -\frac{d}{dr_t} \left[\frac{r_{t-1}}{r_t^2} \left(1 - (1 + \frac{r_t}{p})^{-pT}\right)\right] \\\ 
& - \frac{d}{dr_t} \left[(1 - \frac{r_{t-1}}{r_t})\frac{pT}{p + r_t} (\frac{p + r_t}{p})^{-pT}\right] \\\
\\\
=& -\frac{d}{dr_t} \left[\frac{r_{t-1}}{r_t^2}\right] \left(1 - (1 + \frac{r_t}{p})^{-pT}\right) \\\ 
& -\frac{r_{t-1}}{r_t^2} \frac{d}{dr_t} \left[\left(1 - (1 + \frac{r_t}{p})^{-pT}\right)\right] \\\ 
& - \frac{d}{dr_t} \left[(1 - \frac{r_{t-1}}{r_t})\right]\frac{pT}{p + r_t} (\frac{p + r_t}{p})^{-pT} \\\
& - (1 - \frac{r_{t-1}}{r_t})\frac{d}{dr_t}\left[\frac{pT}{p + r_t} (\frac{p + r_t}{p})^{-pT}\right] \\\
\\\
\end{aligned}
$$
We have:
$$
\frac{d}{dr_t} \frac{r_{t-1}}{r_t^2} = -2\frac{r_{t-1}}{r_t^3}
$$
and using equations \\(\eqref{A1}\\) & \\(\eqref{A2}\\) we can solve the first three terms:
$$
\begin{aligned}
R^{\prime\prime}(r_t) =& \ 2\frac{r_{t-1}}{r_t^3} \left(1 - (1 + \frac{r_t}{p})^{-pT}\right) \\\ 
& -\frac{r_{t-1}}{r_t^2}\frac{pT}{p + r_t} (\frac{p + r_t}{p})^{-pT} \\\ 
& -\frac{r_{t-1}}{r_t^2}\frac{pT}{p + r_t} (\frac{p + r_t}{p})^{-pT} \\\
& - (1 - \frac{r_{t-1}}{r_t})\frac{d}{dr_t}\left[\frac{pT}{p + r_t} (\frac{p + r_t}{p})^{-pT}\right] \\\
\\\
\end{aligned}
$$

The last term we will ignore because we are only interested in taking the second derivative at \\(R^{\prime\prime}(r_{t-1})\\) where the last term equals 0:
$$
R^{\prime\prime}(r_{t-1}) = 2\frac{1}{r_{t-1}^2} \left(1 - (1 + \frac{r_{t-1}}{p})^{-pT}\right) -2\frac{1}{r_{t-1}}\frac{pT}{p + r_{t-1}} (\frac{p + r_{t-1}}{p})^{-pT}
$$
Which gives us the second order Taylor expansion:
$$
\begin{aligned}
R(r_{t-1}) &= \frac{r_{t-1}}{f} \\\
R^\prime(r_{t-1}) &= -\frac{1}{r_{t-1}} \left(1 - (1 + \frac{r_{t-1}}{p})^{-pT}\right) \\\
R^{\prime\prime}(r_{t-1}) &= 2\frac{1}{r_{t-1}^2} \left(1 - (1 + \frac{r_{t-1}}{p})^{-pT}\right) -2\frac{1}{r_{t-1}}\frac{pT}{p + r_{t-1}} (\frac{p + r_{t-1}}{p})^{-pT} \\\ 
R_2(r_t) &= R(r_{t-1}) + R^\prime(r_{t-1})(r_t - r_{t-1}) + R^{\prime\prime}(r_{t-1}) \frac{1}{2}(r_t - r_{t-1})^2
\end{aligned}
$$




{{% citation
    id="Souleymanou2021"
    author="Souleymanou"
    title="Estimation of one- and two-factor Vasicek term structure model of interest rates for the West African Economic and Monetary Union countries"
    publication="International Journal of Business and Social Science"
    year="2021"
    volume="12"
    number="2"
    link="https://ijbssnet.com/journals/Vol_12_No_2_February_2021/8.pdf"
%}}


{{% citation
    id="Vasichek1977"
    author="Oldrich A. Vasicek"
    title="An equilibrium characterization of the term structure"
    publication="Journal of Financial Economics"
    year="1977"
    volume="5"
    link="http://public.kenan-flagler.unc.edu/faculty/lundblac/bcrp/vasicek77.pdf"
%}}

{{% citation
    id="Holy2022"
    author="Vladimír Holý, Petra Tomanová"
    title="Estimation of Ornstein–Uhlenbeck Process Using Ultra-High-Frequency Data with Application to Intraday Pairs Trading Strategy"
    year="2022"
    link="https://arxiv.org/pdf/1811.09312.pdf"
%}}