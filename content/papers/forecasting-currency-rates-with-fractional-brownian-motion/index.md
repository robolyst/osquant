---
title: "Forecasting currency rates with fractional brownian motion"
summary: "
blah blah blah
"
type: paper
katex: true
date: "2023-08-02"
authors:
    - Adrian Letchford
categories:
    - mathematics
    - finance
notebook: https://api.observablehq.com/d/60be1865dc6d5a1e.js?v=3
---

# Fractional Brownian motion

Fractional Brownian motion (or fBm for short) is defined as a stochastic Gaussian process \\( X_t \\) that starts at zero \\( X_0 = 0 \\) has an expectation of zero \\( \mathbb{E}[X_t] = 0 \\) and has the following covariance[^Garcin-2021]:
$$
\mathbb{E}[X_t X_s] = \sigma^2 \frac{1}{2}(|t|^{2H} + |s|^{2H} - |t - s|^{2H}) \label{1}\tag{1}
$$
where \\( \sigma \\) is the volatility parameter and \\( H \in (0, 1) \\) is the Hurst exponent.

The Hurst parameter \\( H \\) controls the auto-correlation in the process. When \\(H = 0.5 \\) then you get regular Brownian motion --- where the increments are an uncorrelated Gaussian process. When \\(H \lt 0 \\) then you get a process that is more mean reverting and when \\( H \gt 0.5 \\) you get a process that exhibits sustained trends.

Sometimes the mathematics behind stochastic processes can seem a little mystifying. Here's an interactive example where you can play around with the Hurst exponent to see how the process changes.

<todo>Interactive fBm example</todo>

Normally, when dealing with price data, we like to convert the price series into returns, or the difference between log returns. The covariance above can be reworked into the covariance between increments[^Garcin-2021]:
$$
\mathbb{E}[(X_t - X_s)(X_v - X_u)] = \sigma^2 \frac{1}{2}(|u - t|^{2H} + |v - s|^{2H} - |v - t|^{2H} - |u - s|^{2H})
$$

If we're looking at a h-step ahead covariance of logged prices, this would mean that:
$$
\begin{aligned}
s &= t - 1 \\\
v &= t + h \\\
u &= t + h - 1 \\\
\end{aligned}
$$
so the covariance function becomes:
$$
\mathbb{E}[(X_t - X\_{t - 1})(X\_{t + h} - X\_{t + h - 1})] = \sigma^2 \frac{1}{2}(|h - 1|^{2H} + |h + 1|^{2H} - 2|h|^{2H})
$$

<plot id="incremental_cov_plot"></plot>

# Forecasting fractional Brownian motion

Let \\( \boldsymbol{p}\_{t+1} \\) be a vector of logged currency prices where the last value is the price one step into the future:
$$
\boldsymbol{p}\_{t+1} = \left[ \begin{matrix}
\boldsymbol{p}_t \\\
p\_{t+1} \\\
\end{matrix} \right ]
$$

Modelling \\( \boldsymbol{p}\_{t+1} \\) as a fractional Brownian motion means that \\( \mathbb{\boldsymbol{p}\_{t+1}} = 0 \\) and the covariance matrix follows equation (\\( \ref{1} \\)). The covariance matrix is partitioned as:
$$
\boldsymbol{\Sigma}\_{t+1} = \left[ \begin{matrix}
\boldsymbol{\Sigma}_t & \boldsymbol{\Sigma}\_{t,t+1} \\\
\boldsymbol{\Sigma}\_{t,t+1}^T & \sigma^2\_{t+1} \\\
\end{matrix} \right ]
$$

In the [appendix](#conditional-gaussian-distribution) I show how to calculate a conditional Gaussian distribution. We can calculated the expected value of \\(p\_{t+1} \\) conditioned on \\( \boldsymbol{p}_t\\) as:
$$
\mathbb{E}[p\_{t+1} | \boldsymbol{p}_t] = \boldsymbol{\Sigma}\_{t,t+1}^T\boldsymbol{\Sigma}_t^{-1} \boldsymbol{p}_t
$$

This is a linear combination of past prices. We can calculate the linear weights in Python with:
```python
import numpy as np

def fbm_weights(window, H):
    t = np.arange(1, window+1).reshape(-1, 1)
    s = t.T
    cov = 0.5 * (t**(2*H) + s**(2*H) - np.abs(t - s)**(2*H))
    cov_v = 0.5 * (t**(2*H) + (window+1)**(2*H) - np.abs(t - (window+1))**(2*H))
    weights = cov_v.T @ np.linalg.inv(cov) 
    
    return weights.flatten()
```

The weights look something like this:
```python
weights = fbm_weights(window=3000, H=0.45)
plt.plot(np.log(weights))  # import matplotlib.pyplot as plt
```

An example of 
This is a linear combination of past prices
1. Calculate weights as function of window, volatility and Hurst exponent
2. See if forecasts of H < 0.5 are better than H = 0.5




# Appendix

## Conditional Gaussian distribution

Define \\( \boldsymbol{x} \\) to be a \\( d \\) dimensional Gaussian vector:
$$
\boldsymbol{x} \in \mathcal{R}^d \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})
$$

Partition \\( \boldsymbol{x} \\) into two disjoint sets \\( a \\) and \\( b \\):
$$
\boldsymbol{x} = \left[ \begin{matrix}
\boldsymbol{x}_a \\\
\boldsymbol{x}_b \\\
\end{matrix} \right ]
$$
and also split the mean vector \\( \boldsymbol{\mu} \\) and covariance matrix \\( \boldsymbol{\Sigma} \\) into corresponding partitions:
$$
\boldsymbol{\mu} = \left[ \begin{matrix}
\boldsymbol{\mu}_a \\\
\boldsymbol{\mu}_b \\\
\end{matrix} \right ]
\quad
\boldsymbol{\Sigma} = \left[ \begin{matrix}
\boldsymbol{\Sigma}\_{aa} & \boldsymbol{\Sigma}\_{ab} \\\
\boldsymbol{\Sigma}\_{ba} & \boldsymbol{\Sigma}\_{bb} \\\
\end{matrix} \right ]
$$
Note that because \\( \boldsymbol{\Sigma} \\) is symmetric that means that \\( \boldsymbol{\Sigma}\_{aa} \\) and \\( \boldsymbol{\Sigma}\_{bb} \\) are symmetric and that \\( \boldsymbol{\Sigma}\_{ab} = \boldsymbol{\Sigma}\_{ba}^T \\).

The distribution of \\( \boldsymbol{x}_a \\) conditional on \\( \boldsymbol{x}_b \\) is:
$$
\boldsymbol{x}_a |  \boldsymbol{x}_b \sim \mathcal{N}(\boldsymbol{\mu}\_{a|b}, \boldsymbol{\Sigma}\_{a|b})
$$
where:
$$
\begin{aligned}
\boldsymbol{\mu}\_{a|b} &= \boldsymbol{\mu}_a + \boldsymbol{\Sigma}\_{ab}\boldsymbol{\Sigma}\_{bb}^{-1} (\boldsymbol{x}_b - \boldsymbol{\mu}_b) \\\
\boldsymbol{\Sigma}\_{a|b} &= \boldsymbol{\Sigma}\_{aa} - \boldsymbol{\Sigma}\_{ab}\boldsymbol{\Sigma}\_{bb}^{-1}\boldsymbol{\Sigma}\_{ba}
\end{aligned}
$$

For the curious, you can find a derivation of the Gaussian conditional distribution in section 2.3 of [^Bishop-2006]. I found a [PDF online](https://www.seas.upenn.edu/~cis520/papers/Bishop_2.3.pdf) of this section.

{{% citation
    id="Bishop-2006"
    author="Christopher M. Bishop"
    title="Pattern Recognition and Machine Learning"
    year="2006"
    publisher="Springer"
    link="https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf"
%}}



{{% citation
    id="Garcin-2021"
    author="Matthieu Garcin"
    title="Forecasting with fractional Brownian motion: a financial perspective"
    year="2021"
    publisher="HAL Open Science"
    link="https://hal.science/hal-03230167/document"
%}}
