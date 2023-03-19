---
title: "Mean reversion in government bonds"
summary: "
Using the Vasicek model, you can calculate the expected return of a government bond spread. Furthermore, you can calculate the expected value of trading a yield curve inversion with ETFs.
"
date: "2023-03-21"
type: paper
katex: true # Enable mathematics on the page
plotly: true  # Enable plotly on the page
feature: false
authors:
    - Adrian Letchford
categories:
    - mathematics
    - finance
---


{{<plotly id="interest_rates" data="data.csv" src="interest_rates.js" />}}
{{<plotly id="interest_rates_with_ewm" data="data.csv" src="interest_rates_with_ewm.js" />}}


# Vasicek model

Excellent write up of this model with noise[^Holy2022].

We're going to create a model of the long term interest rate \\(r_l(t)\\) and the spread between the long term rate and the short term rate \\(s(t) = r_l(t) - r_s(t)\\). This is a type of [Vasicek model](https://en.wikipedia.org/wiki/Vasicek_model)[^Vasichek1977] known as a two-factor equilibrium Vasicek model [^Souleymanou2021] .

We model both the interest rate of the long term bond \\(r_l(t)\\) and the spread between the long term rate and the short term rate \\(s(t)\\) as an [Ornstein–Uhlenbeck process](https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process):
$$
\begin{aligned}
d r_l(t) &= \sigma_l dW_l(t) \\\
d s(t) &= \theta_s (\mu_s - s(t))dt + \sigma_s dW_s(t) \\\
E[dW_l(t)dW_s(t)] &= 0 \\\
\end{aligned}
$$

Which means that both of these processes have independently normally distributed increments. The conditional moments of the long term rates are:
$$
\begin{aligned}
E[r_l(t) | r_l(0)] &= r_l(0) \\\
\text{var}[r_l(t) | r_l(0)] &= \sigma_l^2 t \\\
\end{aligned}
$$
and the conditional moments of the spread are [^Holy2022]:
$$
\begin{aligned}
E[s_l(t) | s_l(0)] &= s_l(0) e^{-\theta_s t} + \mu_s(1 - e^{-\theta_s t}) \\\
\text{var}[s_l(t) | s_l(0)] &= \frac{\sigma_l^2}{2 \theta_s}(1 - e^{-2\theta_s t}) \\\
\end{aligned}
$$

Based on the defintion of the spread \\(s(t) = r_l(t) - r_s(t)\\) the short term rate is:
$$
r_s(t) = r_l(t) - s(t)
$$
And the moments of the short term rates are:

$$
\begin{aligned}
E[r_s(t)|r_s(0)] &= E[r_l(t) | r_l(0)] - E[s_l(t) | s_l(0)] \\\
& = r_l(0) - s_l(0) e^{-\theta_s t} - \mu_s(1 - e^{-\theta_s t}) \\\
\end{aligned}
$$

$$
\begin{aligned}
\text{var}[r_s(t)|r_s(0)] &= \text{var}[r_l(t) | r_l(0)] + \text{var}[s_l(t) | s_l(0)] \\\
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
0 & \text{var}[s_l(t) | s_l(0)] \\\
\end{matrix}\right]
\left[\begin{matrix}
1 \\\
0 \\\
\end{matrix}\right] \\\
&= \text{var}[r_l(t) | r_l(0)] \\\
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
