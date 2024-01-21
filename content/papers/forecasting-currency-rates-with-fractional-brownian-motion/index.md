---
title: "Forecasting currency rates with fractional brownian motion"
summary: "
Fractional Brownian motion is a stochastic process that can model mean reversion. Predicting future values turns out to be a simple linear model. This model has significant predictive power when applied to currencies.
"
type: paper
katex: true
date: "2023-08-09"
authors:
    - Adrian Letchford
categories:
    - mathematics
    - finance
acknowledgements: "All graphs were made with [Observable](https://observablehq.com/). The article's hero image (thumbnail) was made with [Figma](http://figma.com)."
---

<script type="module" src="index_files/libs/quarto-ojs/quarto-ojs-runtime.js"></script>
<link  href="index_files/libs/quarto-ojs/quarto-ojs.css" rel="stylesheet" />

<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjs/12.2.1/math.js"></script>

# Fractional Brownian motion

Fractional Brownian motion is defined as a stochastic Gaussian process $X_t$ that starts at zero $X_0 = 0$ has an expectation of zero $\mathbb{E}[X_t] = 0$ and has the following covariance[^1]:
$$
\mathbb{E}[X_t X_s] = \sigma^2 \frac{1}{2}(|t|^{2H} + |s|^{2H} - |t - s|^{2H}) \label{1}\tag{1}
$$
where $\sigma$ is the volatility parameter and $H \in (0, 1)$ is the Hurst exponent.

The Hurst parameter \$ H \$ controls the auto-correlation in the process. When $H = 0.5$ then you get regular Brownian motion --- where the increments are an uncorrelated Gaussian process. When $H \lt 0.5$ then you get a process that is more mean reverting and when $H \gt 0.5$ you get a process that exhibits sustained trends.

Sometimes the mathematics behind stochastic processes can seem a little mystifying. Here's an interactive example where you can play around with the Hurst exponent to see how the process changes.

<div id="ojs-cell-1"></div>
<div id="ojs-cell-2"></div>

The processes in the plot above were generated based on the method of taking the square root of the covariance matrix. The method is described on the [Wikipedia page](https://en.wikipedia.org/wiki/Fractional_Brownian_motion#Method_1_of_simulation). The details for how to calculate the square root of the covariance matrix can be found in a previous article: {{< xref "square-root-of-covariance-matrix" >}}.

Intuitively, from playing with the Hurst exponent in the chart above, we can see that when $H \gt 0.5$ there are positive auto-correlations and when $H \lt 0.5$ there are negative auto-correlations. We can see this mathematically by reworking the covariance function into the covariance between increments[^2]:
$$
\mathbb{E}[(X_t - X_s)(X_v - X_u)] = \sigma^2 \frac{1}{2}(|u - t|^{2H} + |v - s|^{2H} - |v - t|^{2H} - |u - s|^{2H})
$$

If we're looking at a h-step ahead covariance of a differenced time series, this would mean that:
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

We can see in the chart below that when $H \gt 0.5$ there are positive auto-correlations and when $H \lt 0.5$ there are negative auto-correlations.

<div id="ojs-cell-3"></div>

# Forecasting fractional Brownian motion

Let $\boldsymbol{p}\_{t+1}$ be a vector of logged currency prices where the last value is the price one step into the future:
$$
\boldsymbol{p}\_{t+1} = \left[ \begin{matrix}
\boldsymbol{p}_t \\\
p\_{t+1} \\\
\end{matrix} \right ]
$$

Modelling $\boldsymbol{p}\_{t+1}$ as a fractional Brownian motion means that $\mathbb{E}[\boldsymbol{p}\_{t+1}] = 0$ and the covariance matrix follows equation $\ref{1}$. The covariance matrix is partitioned as:
$$
\boldsymbol{\Sigma}\_{t+1} = \left[ \begin{matrix}
\boldsymbol{\Sigma}_t & \boldsymbol{\Sigma}\_{t,t+1} \\\
\boldsymbol{\Sigma}\_{t,t+1}^T & \sigma^2\_{t+1} \\\
\end{matrix} \right ]
$$

In the [appendix](#conditional-gaussian-distribution) I show how to calculate a conditional Gaussian distribution. We can calculated the expected value of $p\_{t+1}$ conditioned on $\boldsymbol{p}_t$ as:
$$
\mathbb{E}[p\_{t+1} | \boldsymbol{p}_t] = \boldsymbol{\Sigma}\_{t,t+1}^T\boldsymbol{\Sigma}_t^{-1} \boldsymbol{p}_t
$$

This is a linear combination of past prices. We can calculate the linear weights in Python with:

``` python
import numpy as np

def fbm_weights(window, H):
    t = np.arange(1, window+1).reshape(-1, 1)
    s = t.T
    cov = 0.5 * (t**(2*H) + s**(2*H) - np.abs(t - s)**(2*H))
    cov_v = 0.5 * (t**(2*H) + (window+1)**(2*H) - np.abs(t - (window+1))**(2*H))
    weights = cov_v.T @ np.linalg.inv(cov) 
    
    return weights.flatten()
```

Using the parameters `window = 30` and `H = 0.45`, the weights look like:

<div id="ojs-cell-4"></div>

If we have a Pandas DataFrame of prices we can predict the next step ahead price with:

``` python
# prices = pandas.DataFrame of currency prices
weights = fbm_weights(window=100, H=0.45)
predicted_price = prices.rolling(100).apply(lambda x: weights @ x)
```

# Forecasting currency returns

Taking the `predicted_price` DataFrame above, we can calculate the expected return for each currency with:

``` python
predicted_return = predicted_price / price - 1
```

I find that calculating mispricing yields a stronger signal than predicting returns. In this context, if we predict that a currency is expected to return 1% but it actually returns 2%, then we say that the currency is 1% mispriced and we expect it to move lower by 1%.

We can calculate this mispricing with:

``` python
actual_returns = prices.pct_change()
mispricing = predicted_return.shift(1) - actual_returns
```

The `mispricing` DataFrame will contain positive values for currencies that we believe are undervalued and negative values for currencies we believe are overvalued.

We can then normalise these mispricings so that currencies with a higher mispricing have a greater weight in our portfolio. We normalise so that absolute sum of the portfolio weights equals 1:

``` python
portfolio_weights = mispricing.divide(mispricing.abs().sum(1), axis=0)
```

We can get an idea of the strength of this signal by looking at an equity curve without transaction costs:

``` python
portfolio_returns = (actual_returns * portfolio_weights.shift(1)).sum(1)
capital = (1 + portfolio_returns).cumprod()
```

As an example, I use a dataset (the `prices` DataFrame) of 30 minute prices of the following pairs:

``` plaintext
CAD_USD
EUR_USD
JPY_USD
SEK_USD
GBP_USD
SGD_USD
PLN_USD
CHF_USD
AUD_USD
NOK_USD
CZK_USD
NZD_USD
DKK_USD
CNH_USD
HUF_USD
```

which are most of the currencies on Oanda except for `HKD` which is pegged to the `USD` and some of the less traded currencies.

Calculating `capital` on this dataset gives me:

<div id="ojs-cell-5"></div>

This is a fairly basic model, all we've done is derive a linear filter of past prices. Yet, there appears to be a fair amount of predictive power. The problem is that if I were to include transaction costs, all performance disappears. This predictive signal isn't strong enough to overcome transaction costs and is more suitable as a feature in a machine learning model.

# Summary

Here we've made a linear filter of past prices that predicts future returns derived from the fractional Brownian motion stochastic process. This filter demonstrates predictive power, however, not enough to overcome transaction costs. Specifically, the signal does not overcome the spread.

# Appendix

## Conditional Gaussian distribution

Define $\boldsymbol{x}$ to be a $d$ dimensional Gaussian vector:
$$
\boldsymbol{x} \in \mathcal{R}^d \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})
$$

Partition $\boldsymbol{x}$ into two disjoint sets $a$ and $b$:
$$
\boldsymbol{x} = \left[\begin{matrix}
\boldsymbol{x}_a \\\
\boldsymbol{x}_b \\\
\end{matrix} \right ]
$$
and also split the mean vector $\boldsymbol{\mu}$ and covariance matrix $\boldsymbol{\Sigma}$ into corresponding partitions:
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
Note that because $\boldsymbol{\Sigma}$ is symmetric that means that $\boldsymbol{\Sigma}\_{aa}$ and $\boldsymbol{\Sigma}\_{bb}$ are symmetric and that $\boldsymbol{\Sigma}\_{ab} = \boldsymbol{\Sigma}\_{ba}^T$.

The distribution of $\boldsymbol{x}_a$ conditional on $\boldsymbol{x}_b$ is:
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

For the curious, you can find a derivation of the Gaussian conditional distribution in section 2.3 of [^3]. I found a [PDF online](https://www.seas.upenn.edu/~cis520/papers/Bishop_2.3.pdf) of this section.

[^1]:
    {{< citation author="Matthieu Garcin" title="Forecasting with fractional Brownian motion: a financial perspective" year="2021" publisher="HAL Open Science" link="https://hal.science/hal-03230167/document" >}}

[^2]:
    {{< citation author="Matthieu Garcin" title="Forecasting with fractional Brownian motion: a financial perspective" year="2021" publisher="HAL Open Science" link="https://hal.science/hal-03230167/document" >}}

[^3]:
    {{< citation author="Christopher M. Bishop" title="Pattern Recognition and Machine Learning" year="2006" publisher="Springer" link="https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf" >}}


<script type="ojs-module-contents">
eyJjb250ZW50cyI6W3sibWV0aG9kTmFtZSI6ImludGVycHJldCIsImNlbGxOYW1lIjoib2pzLWNlbGwtMSIsImlubGluZSI6ZmFsc2UsInNvdXJjZSI6InZpZXdvZiBIID0gSW5wdXRzLnJhbmdlKFswLjEsIDAuOV0sIHtsYWJlbDogdGV4YEhgLCBzdGVwOiAwLjEsIHZhbHVlOiAwLjV9KTtcbiJ9LHsibWV0aG9kTmFtZSI6ImludGVycHJldCIsImNlbGxOYW1lIjoib2pzLWNlbGwtMiIsImlubGluZSI6ZmFsc2UsInNvdXJjZSI6ImZibXNfcGxvdCA9IHtcblxuICBsZXQgZmJtcyA9IGF3YWl0IEZpbGVBdHRhY2htZW50KFwiZmJtLmNzdlwiKS5jc3Yoe3R5cGVkOiB0cnVlfSk7XG5cbiAgbGV0IGZibXNfcGxvdCA9IFBsb3QucGxvdCh7XG4gICAgICB3aWR0aDogNzAwLFxuICAgICAgaGVpZ2h0OiAzMDAsXG4gICAgICBzdHlsZToge1xuICAgICAgICBmb250U2l6ZTogXCIxNnB4XCIsXG4gICAgICAgIGZvbnRGYW1pbHk6IFwiU291cmNlIFNhbnMgUHJvLCBIZWx2ZXRpY2EsIEFyaWFsXCIsXG4gICAgICAgIGJhY2tncm91bmRDb2xvcjogJ3RyYW5zcGFyZW50JyxcbiAgICAgIH0sXG4gICAgICB5OiB7XG4gICAgICAgIGxhYmVsOiAn4oaRIEZyYWN0aW9uYWwgQnJvd25pYW4gbW90aW9uJyxcbiAgICAgIH0sXG4gICAgICB4OiB7XG4gICAgICAgIGdyaWQ6IHRydWUsXG4gICAgICAgIG5pY2U6IHRydWUsXG4gICAgICB9LFxuICAgICAgbWFya3M6IFtcbiAgICAgICAgUGxvdC5saW5lKGZibXMsIHt4OiBcInhcIiwgeTogSC50b1N0cmluZygpfSksXG4gICAgICAgIFBsb3QuYXhpc1koW10pLFxuICAgICAgXSxcbiAgICAgIG1hcmdpblRvcDogMzUsXG4gICAgICBtYXJnaW5MZWZ0OiAxMCxcbiAgICAgIG1hcmdpbkJvdHRvbTogNDUsXG4gICAgfSk7XG5cbiAgcmV0dXJuIGZibXNfcGxvdDtcbn1cbiJ9LHsibWV0aG9kTmFtZSI6ImludGVycHJldCIsImNlbGxOYW1lIjoib2pzLWNlbGwtMyIsImlubGluZSI6ZmFsc2UsInNvdXJjZSI6Im15X3Bsb3QgPSB7XG5cbiAgZnVuY3Rpb24gaW5jcmVtZW50YWxfY292KG4sIEgpIHtcbiAgICByZXR1cm4gbWF0aC5yYW5nZSgxLCBuKzEpLnRvQXJyYXkoKS5tYXAoKGgpID0+IHtcbiAgICAgIHJldHVybiB7XG4gICAgICAgIHg6IGgsXG4gICAgICAgIHk6IChoIC0gMSkqKigyKkgpICsgKGggKyAxKSoqKDIqSCkgLSAyKihoKSoqKDIqSCksXG4gICAgICAgIEg6IFwiSCA9IFwiICsgSC50b1N0cmluZygpLFxuICAgICAgfVxuICAgIH0pXG4gIH07XG5cbiAgbGV0IGNvdl9kYXRhID0gKFxuICAgIGluY3JlbWVudGFsX2NvdigyMCwgMC41NSlcbiAgICAuY29uY2F0KGluY3JlbWVudGFsX2NvdigyMCwgMC41KSlcbiAgICAuY29uY2F0KGluY3JlbWVudGFsX2NvdigyMCwgMC40NSkpXG4gICk7XG5cbiAgcmV0dXJuIFBsb3QucGxvdCh7XG4gICAgICB3aWR0aDogNzAwLFxuICAgICAgaGVpZ2h0OiAzMDAsXG4gICAgICBzdHlsZToge1xuICAgICAgICBmb250U2l6ZTogXCIxNnB4XCIsXG4gICAgICAgIGZvbnRGYW1pbHk6IFwiU291cmNlIFNhbnMgUHJvLCBIZWx2ZXRpY2EsIEFyaWFsXCIsXG4gICAgICAgIGJhY2tncm91bmRDb2xvcjogJ3RyYW5zcGFyZW50JyxcbiAgICAgIH0sXG4gICAgICB5OiB7XG4gICAgICAgIGdyaWQ6IHRydWUsXG4gICAgICAgIGxhYmVsOiAn4oaRIENvdmFyaWFuY2UnLFxuICAgICAgICBuaWNlOiB0cnVlLFxuICAgICAgfSxcbiAgICAgIHg6IHtcbiAgICAgICAgZ3JpZDogdHJ1ZSxcbiAgICAgICAgbmljZTogdHJ1ZSxcbiAgICAgICAgbGFiZWw6ICfihpIgaCcsXG4gICAgICB9LFxuICAgICAgY29sb3I6IHtcbiAgICAgICAgbGVnZW5kOiB0cnVlLFxuICAgICAgfSxcbiAgICAgIG1hcmtzOiBbXG4gICAgICAgIFBsb3QubGluZShjb3ZfZGF0YSwge3g6IFwieFwiLCB5OiBcInlcIiwgc3Ryb2tlOiBcIkhcIn0pLFxuICAgICAgXSxcbiAgICAgIG1hcmdpblRvcDogMzUsXG4gICAgICBtYXJnaW5MZWZ0OiA1NSxcbiAgICAgIG1hcmdpbkJvdHRvbTogNDUsXG4gIH0pO1xufVxuIn0seyJtZXRob2ROYW1lIjoiaW50ZXJwcmV0IiwiY2VsbE5hbWUiOiJvanMtY2VsbC00IiwiaW5saW5lIjpmYWxzZSwic291cmNlIjoid2VpZ2h0c19wbG90ID0ge1xuXG4gIGxldCB3ZWlnaHRzID0gYXdhaXQgRmlsZUF0dGFjaG1lbnQoXCJ3ZWlnaHRzLmNzdlwiKS5jc3Yoe3R5cGVkOiB0cnVlfSk7XG5cbiAgbGV0IHdlaWdodHNfcGxvdCA9IFBsb3QucGxvdCh7XG4gICAgICB3aWR0aDogNzAwLFxuICAgICAgaGVpZ2h0OiAzMDAsXG4gICAgICBzdHlsZToge1xuICAgICAgICBmb250U2l6ZTogXCIxNnB4XCIsXG4gICAgICAgIGZvbnRGYW1pbHk6IFwiU291cmNlIFNhbnMgUHJvLCBIZWx2ZXRpY2EsIEFyaWFsXCIsXG4gICAgICAgIGJhY2tncm91bmRDb2xvcjogJ3RyYW5zcGFyZW50JyxcbiAgICAgIH0sXG4gICAgICB5OiB7XG4gICAgICAgIGdyaWQ6IHRydWUsXG4gICAgICAgIGxhYmVsOiAn4oaRIFdlaWdodCcsXG4gICAgICAgIG5pY2U6IHRydWUsXG4gICAgICAgIHR5cGU6IFwibG9nXCIsXG4gICAgICAgIHRpY2tGb3JtYXQ6IFwiLjJcIixcbiAgICAgIH0sXG4gICAgICB4OiB7XG4gICAgICAgIGdyaWQ6IHRydWUsXG4gICAgICAgIG5pY2U6IHRydWUsXG4gICAgICAgIGxhYmVsOiAn4oaSIHQnLFxuICAgICAgfSxcbiAgICAgIG1hcmtzOiBbXG4gICAgICAgIFBsb3QubGluZSh3ZWlnaHRzLCB7eDogXCJ0XCIsIHk6IFwiMC40NVwifSksXG4gICAgICBdLFxuICAgICAgbWFyZ2luVG9wOiAzNSxcbiAgICAgIG1hcmdpbkxlZnQ6IDYwLFxuICAgICAgbWFyZ2luQm90dG9tOiA0NSxcbiAgICBjYXB0aW9uOiBgVGhlIHdlaWdodCB2ZWN0b3IgZm9yIHByZWRpY3RpbmcgYSBmcmFjdGlvbmFsIEdhdXNzaWFuIG1vdGlvbiBcbiAgICBwcm9jZXNzLiBIZXJlLCB3aW5kb3c9MzAgYW5kIHRoZSBIdXJzdCBleHBvbmVudCA9IDAuNDUuIFRoZSBtb3N0IFxuICAgIHJlY2VudCBwcmljZXMgaGF2ZSB0aGUgaGlnaGVzdCB3ZWlnaHQgd2hpY2ggZGVjYXlzIHRoZSBmdXJ0aGVyIGJhY2sgdGhlXG4gICAgcHJpY2UgaXMuYFxuICAgIH0pO1xuICBcbiAgcmV0dXJuIHdlaWdodHNfcGxvdDtcbn1cbiJ9LHsibWV0aG9kTmFtZSI6ImludGVycHJldCIsImNlbGxOYW1lIjoib2pzLWNlbGwtNSIsImlubGluZSI6ZmFsc2UsInNvdXJjZSI6ImNhcGl0YWxfcGxvdCA9IHtcbiAgbGV0IGNhcGl0YWwgPSBhd2FpdCBGaWxlQXR0YWNobWVudChcImNhcGl0YWwuY3N2XCIpLmNzdih7dHlwZWQ6IHRydWV9KTtcblxuICBsZXQgY2FwaXRhbF9wbG90ID0gUGxvdC5wbG90KHtcbiAgICB3aWR0aDogNzAwLFxuICAgIGhlaWdodDogMzAwLFxuICAgIHN0eWxlOiB7XG4gICAgICBmb250U2l6ZTogXCIxNnB4XCIsXG4gICAgICBmb250RmFtaWx5OiBcIlNvdXJjZSBTYW5zIFBybywgSGVsdmV0aWNhLCBBcmlhbFwiLFxuICAgICAgYmFja2dyb3VuZENvbG9yOiAndHJhbnNwYXJlbnQnLFxuICAgIH0sXG4gICAgeToge1xuICAgICAgZ3JpZDogdHJ1ZSxcbiAgICAgIGxhYmVsOiAn4oaRIENhcGl0YWwnLFxuICAgICAgbmljZTogdHJ1ZSxcbiAgICAgIHRpY2tGb3JtYXQ6IFwiLjAlXCIsXG4gICAgfSxcbiAgICB4OiB7XG4gICAgICBncmlkOiB0cnVlLFxuICAgICAgbmljZTogdHJ1ZSxcbiAgICB9LFxuICAgIG1hcmtzOiBbXG4gICAgICBQbG90LmxpbmUoY2FwaXRhbCwge3g6IFwidGltZVwiLCB5OiBcIjBcIn0pLFxuICAgIF0sXG4gICAgbWFyZ2luVG9wOiAzNSxcbiAgICBtYXJnaW5MZWZ0OiA2MCxcbiAgICBtYXJnaW5Cb3R0b206IDQ1LFxuICBjYXB0aW9uOiBgRXhhbXBsZSBjYXBpdGFsIHRyYWRpbmcgdGhlIG1pc3ByaWNpbmcgaWRlbnRpZmllZCBieSB0aGVcbiAgICBmcmFjdGlvbmFsIGJyb3duaWFuIG1vdGlvbi4gVGhpcyBpcyB3aXRob3V0IHRyYW5zYWN0aW9uIGNvc3RzLmBcbiAgfSk7XG5cbiAgcmV0dXJuIGNhcGl0YWxfcGxvdDtcbn1cbiJ9LHsibWV0aG9kTmFtZSI6ImludGVycHJldFF1aWV0Iiwic291cmNlIjoic2hpbnlJbnB1dCgnSCcpIn1dfQ==
</script>
<script type="module">
if (window.location.protocol === "file:") { alert("The OJS runtime does not work with file:// URLs. Please use a web server to view this document."); }
window._ojs.paths.runtimeToDoc = "../../..";
window._ojs.paths.runtimeToRoot = "../../../../../..";
window._ojs.paths.docToRoot = "../../..";
window._ojs.selfContained = false;
window._ojs.runtime.interpretFromScriptTags();
</script>
