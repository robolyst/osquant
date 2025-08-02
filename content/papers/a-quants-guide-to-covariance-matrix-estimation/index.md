---
title: "A Quant's Guide to Covariance Matrix Estimation"
summary: "
    In this article, we explore three techniques to improve covariance matrix estimation: evaluating estimates independently of backtests, decoupling variance and correlation, and applying shrinkage for more robust outputs.
"

date: "2025-07-20"
type: paper
mathjax: true
authors:
    - Adrian Letchford
categories:
    - mathematics
    - finance
acknowledgements: "All figures in this article were made with [Figma](http://figma.com)."
---

Estimating a covariance matrix from historical prices is a core task in quantitative finance. The go-to solution is an exponentially weighted moving (EWM) estimate of the covariance matrix. It's a simple, fast, and widely available method.

I use an EWM when I need something simple to get started, but there are a few things we can do to improve the estimates. In this article, we'll explore three key ideas that can significantly improve covariance estimation:

1. **Measurement** - Instead of relying solely on backtest performance to evaluate covariance estimates, we can define objective metrics to assess their statistical fitness in isolation.

1. **Separating variance and correlation estimation** - Variance and correlation behave differently over time and should be modeled independently.

1. **Shrinkage** - By pulling our noisy empirical estimates toward a structured target, shrinkage methods offer a systematic way to reduce estimation error.

This article won't cover the various methods of covariance matrix estimation, such as EWM estimates, GARCH or other methods. We'll focus on the above three ideas to improve the estimates we get from these methods and we'll use EWM estimates as the example.

# Measurement

The most straightforward way to evaluate covariance matrix estimates is through full backtesting. Simply run your complete investment strategy with different covariance estimators and compare the resulting performance metrics.

However, this approach has significant drawbacks. Backtesting is computationally expensive and time-consuming, making it impractical for rapid iteration and research. More importantly, mixing covariance estimation research with portfolio construction can lead to overfitting. When performance improves, you don't know whether the gains come from better covariance estimates or inadvertent optimization of other portfolio components.

To address these issues, we need isolated evaluation methods that test covariance estimators independently of the full investment process.

The fiddly thing is that there is no "best" covariance matrix, we can pick a general purpose metric or something that targets our particular need. In his book, [Gappy](https://x.com/__paleologo) covers a whole bunch of different ways of measuring covariance matrices[^Paleologo2025]. Here, we're going to cover two approaches; minimum-variance portfolios and the log-likelihood.

## Minimum-variance portfolios

The idea here is that we usually want a covariance matrix to minimise the variance of our portfolio. So, we can construct the minimum-variance portfolio (MVP) using different covariance estimators and measure their realized variance. The estimator producing the lowest variance portfolio is considered the better option.

The MVP is given by:
$$
\begin{aligned}
\underset{\boldsymbol{w}\_{t-1}}{\text{min}} &\quad \boldsymbol{w}^T_{t-1}\hat{\boldsymbol{\Omega}}_{t-1}\boldsymbol{w}\_{t-1} \\\
\text{s.t.} &\quad \boldsymbol{1}^T\boldsymbol{w}\_{t-1} = 1
\end{aligned}
$$
Where $\boldsymbol{w}\_{t-1}$ are the portfolio weights calculated at time $t-1$, $\hat{\boldsymbol{\Omega}}\_{t-1}$ is the covariance matrix of asset returns at time $t-1$ and $\boldsymbol{1}$ is a vector of ones. The constraint ensures that the weights sum to one.

The solution is:
$$
\boldsymbol{w}\_{t-1} = \frac{\hat{\boldsymbol{\Omega}}\_{t-1}^{-1}\boldsymbol{1}}{\boldsymbol{1}^T\hat{\boldsymbol{\Omega}}\_{t-1}^{-1}\boldsymbol{1}}
$$
We can intuitively understand this by imagining that the assets are not correlated so that the covariance matrix $\hat{\boldsymbol{\Omega}}$ is a diagonal matrix of just the variances. Then you can see that the portfolio weights are proportional to the inverse of the variances. This means that assets with lower variance will receive larger weights in the portfolio.

Here is some Python code to calculate it:
```python
import numpy as np
import pandas as pd

def mvp(cov: pd.DataFrame) -> pd.Series:
    """
    Returns the minimum-variance portfolio weights
    from a covariance matrix.

    Parameters
    ----------
        cov : DataFrame
            Covariance matrix of asset returns over
            time.
    """
    try:
        icov = np.linalg.pinv(cov.values)
        ones = np.ones(cov.shape[0])
        w = icov @ ones / (ones @ icov @ ones)

    except np.linalg.LinAlgError:
        w = np.full(cov.shape[0], np.nan)
    
    return pd.Series(w, index=cov.columns)

```

Once we have the portfolio weights, we can calculate the portfolio returns. The returns of each asset at time $t$ are given by $\boldsymbol{r}_t$, which is a vector of returns for each asset. The portfolio's return at time $t$ is then:
$$
p_t = \boldsymbol{w}^T\_{t-1}\boldsymbol{r}_t
$$
And the variance of the portfolio returns is:
$$
\text{var}(p_1, p_2, \dots, p_T)
$$
When comparing different covariance estimators, we can calculate the variance of the portfolio returns for each estimator and select the one that produces the lowest variance.

The code for this metric is:
```python
def mvp_metric(
    covs: pd.DataFrame,
    returns: pd.DataFrame,
) -> float:
    """
    Returns the variance of the minimum-variance
    portfolio.

    Parameters
    ----------

    cov : DataFrame
        Covariance matrices of asset returns over
        time. The covariance at time t should be
        the covariance matrix we can use to act on
        returns at time t. Expected to have a column
        or index called 'Date' that contains the date
        of the covariance matrix.

    returns : DataFrame
        Asset returns over time.
    """
    portfolios = covs.groupby('Date').apply(mvp)
    preturns = (portfolios * returns).sum(1, skipna=False)
    var = np.var(preturns)
    return var
```



## Log-likelihood

Assuming returns follow a multivariate Gaussian distribution, we can evaluate how well each estimated covariance matrix explains observed returns using the likelihood function.

The likelihood that a vector of returns $\boldsymbol{r}_t$ was generated from a Gaussian process with mean $\boldsymbol{\mu}\_{t-1}$ and covariance matrix $\hat{\boldsymbol{\Omega}}\_{t-1}$ is given by the [probability density function](https://en.wikipedia.org/wiki/Multivariate_normal_distribution):
$$
\text{P}(\boldsymbol{r}_t | \boldsymbol{\mu}\_{t-1}, \hat{\boldsymbol{\Omega}}\_{t-1}) = \frac{1}{(2\pi)^{n/2}|\hat{\boldsymbol{\Omega}}\_{t-1}|^{1/2}} \exp\left(-\frac{1}{2}(\boldsymbol{r}_t - \boldsymbol{\mu}\_{t-1})^T\hat{\boldsymbol{\Omega}}\_{t-1}^{-1}(\boldsymbol{r}_t - \boldsymbol{\mu}\_{t-1})\right)
$$
We read this as: the likelihood of observing the returns $\boldsymbol{r}_t$ given the mean $\boldsymbol{\mu}\_{t-1}$ and covariance matrix $\hat{\boldsymbol{\Omega}}\_{t-1}$.

We could iterate over all $t$ and multiply the likelihoods together to estimate the fit of our estimates:
$$
\prod_{t} \ \text{P}(\boldsymbol{r}\_t | \boldsymbol{\mu}\_{t-1}, \hat{\boldsymbol{\Omega}}\_{t-1})
$$
However, this can lead to numerical problems in practice. Instead, we convert the product into a sum by taking the logarithm of $\text{P}(\boldsymbol{r}_t | \boldsymbol{\mu}\_{t-1}, \hat{\boldsymbol{\Omega}}\_{t-1})$. This is known as the log-likelihood:
$$
\log L(\boldsymbol{r}_t) = -\frac{1}{2} \left[ \log(|\hat{\boldsymbol{\Omega}}\_{t-1}|) + (\boldsymbol{r}_t - \boldsymbol{\mu}\_{t-1})^T\hat{\boldsymbol{\Omega}}\_{t-1}^{-1}(\boldsymbol{r}_t - \boldsymbol{\mu}\_{t-1}) + n \log(2\pi) \right]
$$
We want the sum of these for all $t$ to be as large as possible.

We can simplify the log-likelihood function:
1. For our purposes, we can assume that $\boldsymbol{\mu}\_{t-1} = 0$.
1. As we are optimising over the same set of $n$ returns, the term $n \log(2\pi)$ has no impact on our results. We can drop it.
1. Multiplying by $-\frac{1}{2}$ only has the effect of flipping the sign, we can drop this too.

The final metric is:
$$
\sum_t \left( \log(|\hat{\boldsymbol{\Omega}}\_{t-1}|) + \boldsymbol{r}_t^T\hat{\boldsymbol{\Omega}}\_{t-1}^{-1}\boldsymbol{r}_t \right)
$$
When comparing different covariance estimators, we can calculate this metric for each estimator and select the one that produces the smallest value.

The code for this metric is:
```python
def ll_metric(
    covs: pd.DataFrame,
    returns: pd.DataFrame
) -> float:
    """
    Returns a log-likelihood based metric of the
    fitness of the covariance matrices.

    Parameters
    ----------

    cov : DataFrame
        Covariance matrices of asset returns over
        time. The covariance at time t should be
        the covariance matrix we can use to act on
        returns at time t. Expected to have a column
        or index called 'Date' that contains the date
        of the covariance matrix.

    returns : DataFrame
        Asset returns over time.
    """

    ll = np.full(len(returns), np.nan)

    for i, date in enumerate(returns.index):

        r = returns.loc[date]
        cov = covs.loc[date].loc[r.index, r.index]

        if cov.isnull().any().any():
            continue

        det = np.linalg.det(cov)
        icov = np.linalg.inv(cov)

        ll[i] = np.log(det) + r @ icov @ r.T

    return np.nanmean(ll)
```

## Example

We'll now walk through an example of how to use these metrics to compare different covariance estimators. Specifically, we'll vary the half-life parameter of an exponentially weighted moving (EWM) covariance matrix and evaluate the results across a small set of ETFs.

Each ETF represents a distinct asset class:

- SPY -- U.S. equities (S&P 500)

- TLT -- Long-term U.S. Treasury bonds

- GLD -- Gold

- GSG -- Broad commodities

- VNQ -- U.S. real estate investment trusts (REITs)

For each half-life value, we'll compute the EWM covariance matrix over time and assess the estimates using two evaluation metrics: the variance of the MVP, and the  log-likelihood based metric.

The results are shown in the figure below:

{{<figure src="mvp_vs_ll.svg" title="Example evaluation" >}}
of EWM covariance matrices using two metrics on ETF returns (data from Yahoo Finance). In both plots, the x-axis shows the half-life used in the exponentially weighted estimate. The left plot reports the variance of the minimum-variance portfolio (MVP), which decreases as the half-life increases, reaching a minimum around 33 days. The right plot shows a log-likelihood-based metric, which also improves (i.e. decreases) with longer half-lives, reaching its lowest point near 24 days.
{{</figure>}}

Both metrics demonstrate that the covariance estimates improve as the half-life increases from 5, but they suggest different optimal half-lives. The MVP metric suggests a half-life of around 33 days, while the log-likelihood metric suggests a half-life of around 24 days. Neither of these metrics is inherently better than the other; they simply measure different aspects of the covariance matrix's performance. Despite targeting different properties of the covariance matrix, both evaluation metrics agree that short half-lives (e.g. 5-10 days) underperform, and that there is a sweet spot between 20 and 35 days.

The code to replicate this example is:

```python
import yfinance as yf

tickers = yf.Tickers('SPY TLT GLD GSG VNQ')
prices = tickers.download(period='30y', interval='1d')
returns = prices['Close'].pct_change().dropna()

# We'll dump the results of each evalulation
# into a list here.
results: list[dict] = []

# I notice that the resulting curve is neater
# if the x-axis is logged. So we'll use a logspace
# to generate the half-lives. 
halflives = np.logspace(
    start=np.log10(5),  # 5-day half-life
    stop=np.log10(100), # 100-day half-life
    num=20,  # 20 data points
)

for halflife in halflives:

    # Shift the returns by one day so that
    # the covariance matrix for time t-1 aligns
    # with the returns at time t.
    covs = (
        returns.shift(1)
        .ewm(halflife=halflife, min_periods=200)
        .cov()
    )

    results.append({
        'halflife': halflife,
        'pvar': mvp_metric(covs, returns),
        'll': ll_metric(covs, returns),
    })
```

# Splitting

Now that we have a way of measuring covariance estimates independently of backtests, we can explore how to improve the estimates themselves. The first key idea is to decouple estimating variance and correlation.


Note that the [correlation](https://en.wikipedia.org/wiki/Correlation) between two variables is:
$$
\text{corr}(X, Y) = \frac{\sigma^2_{X,Y}}{\sigma_X \cdot \sigma_Y}
$$
Where $\sigma^2_{X,Y}$ is the covariance between $X$ and $Y$, $\sigma_X$ is the standard deviation of $X$, and $\sigma_Y$ is the standard deviation of $Y$. We can flip this around to get the covariance in terms of the correlation:
$$
\sigma^2_{X,Y} = \text{corr}(X, Y) \cdot \sigma_X \cdot \sigma_Y
$$
Which means, if we have estimated the variance (standard deviation squared) of each variable, we only need to estimate the correlation between the two variables to get their covariance.

Now, the reason we want to estimate variance and correlation seperately is that they behave differently over time. Variance lies on the range $[0, \infty)$ and exhibits dramatic spikes followed by a decay back to a mean. Correlation, on the other hand, is bounded between -1 and 1 and tends to be more stable over time. See the figure below for an example of the variance and correlation of SPY and TLT over time.

{{<figure src="variance_vs-correlation.svg" title="Variance and correlation behave differently." >}}
The top plot shows the variance of SPY and the middle plot shows the variance of TLT. The bottom plot shows the correlation between SPY and TLT. Notice how the variance of both SPY and TLT exhibit spikes and decay back to a mean over time, while the correlation between teh two ETFs is much more stable.
{{</figure>}}

You can see from the figure that the spikes in variance happen quike quickly. Ideally, we would want to capture these spikes in variance. If we estimate variance with an EWM, we would want to use a short half-life so as not to average out these spikes. The figure also shows that correlation is much more stable, mostly fluctuating around a slowly changing mean. This suggests we can use a longer half-life to capture this slowly changing mean.

We can use the metrics from the previous section to test this idea of decoupling the variance and correlation estimates. We'll use the same ETFs as before, but this time we'll calculate the variance and correlation separately using different half-lives.

The following code calculates the EWM covariance matrix by estimating variance and correlation independently:
```python
def ewm_cov(
    returns: pd.DataFrame,
    var_hl: float,
    corr_hl: float,
    min_periods: int,
) -> pd.DataFrame:
    """
    Returns the exponentially weighted covariance matrix.

    Parameters
    ----------

    returns: DataFrame
        Asset returns over time.
    
    var_hl: float
        Half-life of the variance.

    corr_hl: float
        Half-life of the correlation.

    min_periods: int
        Minimum number of periods to consider for the
        calculation of the covariance.
    """
    vars = returns.ewm(halflife=var_hl, min_periods=min_periods).var()
    corrs = returns.ewm(halflife=corr_hl, min_periods=min_periods).corr()

    for date in returns.index:
        std = np.sqrt(vars.loc[date])
        v = np.outer(std, std)
        corrs.loc[date] = corrs.loc[date].values * v

    return corrs
```

We'll conduct an experiment where we use the ETF returns from earlier, vary the half-lives for variance and correlation and measure both the variance of the MVP and the log-likelihood based metric. For the variance estimates, we'll use the same half-lives as before, for the correlation estimates, we'll use the half-lives `[10, 20, 60]`. The results are shown in the figure below.

{{<figure src="longer_corr_halflife.svg" title="Decoupling variance and correlation." >}}
Both plots show the results of evaluating covariance estimates using two metrics: the variance of the minimum-variance portfolio (MVP) on the left, and a log-likelihood based metric on the right. The x-axis for both plots is the half-life used for estimating variance. The dashed line uses the same half-life for both variance and correlation. The remaining lines fix the half-life for the correlation estimates to one of 10, 20, or 60 days.
{{</figure>}}

The figure shows that as the half-life for correlation increases, the performance of the covariance estimates improves. Specifically, under the MVP metric, the best variance half-life is approximately `18` days, while the best correlation half-life is `60` days. The log-likelihood metric shows a similar trend, with the best variance half-life also approximately `18` days and the best correlation half-life at `60` days.

The figure also shows the results when the half-life for correlation is set to the same value as the variance half-life (the dashed line). This produces worse estimates than when the correlation half-life is longer than the variance half-life.

This example demonstrates the intuition that we should estimate correlation with a longer half-life than variance.

The code to replicate this example is:
```python
from itertools import product

results: list[dict] = []

var_hls = np.logspace(
    start=np.log10(5),
    stop=np.log10(100),
    num=20,
)
corr_hls = [10, 20, 60]

for var_hl, corr_hl in product(var_hls, corr_hls):

    covs = ewm_cov(
        returns=returns.shift(1),
        var_hl=var_hl,
        corr_hl=corr_hl,
        min_periods=200,
    )

    results.append({
        'var_hl': var_hl,
        'corr_hl': corr_hl,
        'pvar': mvp_metric(covs, returns),
        'll': ll_metric(covs, returns),
    })
```


# Shrinkage

Paper on shrinkage [^Lodit2003]. The idea is to pull our noisy empirical estimates toward a structured target. This reduces estimation error and improves the robustness of our covariance matrix.

{{<figure src="shrinkage.svg" title="Variance and correlation behave differently." >}}
Curabitur pulvinar magna sit amet mattis semper. Nulla interdum nunc quis turpis iaculis finibus. Donec purus leo, aliquam at malesuada sit amet, elementum vitae quam. Quisque mi justo, euismod ac leo nec, elementum eleifend purus. Etiam ut ornare velit.
{{</figure>}}

# Conclusion


{{% citation
    id="Ghojogh2023"
    author="Benyamin Ghojogh, Fakhri Karray and Mark Crowley"
    title="Eigenvalue and Generalized Eigenvalue Problems: Tutorial"
    publication="arXiv"
    year="2023"
    link="https://arxiv.org/abs/1903.11240"
%}}


{{% citation
    id="Lodit2003"
    author="Olivier Ledoit and Michael Wolf"
    title="Honey, I Shrunk the Sample Covariance Matrix"
    publication="UPF Economics and Business Working Paper No. 691"
    year="2003"
    link="https://papers.ssrn.com/sol3/papers.cfm?abstract_id=433840"
%}}



{{% citation
    id="Paleologo2025"
    author="Giuseppe A. Paleologo"
    title="The Elements of Quantitative Finance"
    publication="Wiley"
    year="2025"
    link="https://www.wiley.com/en-us/The+Elements+of+Quantitative+Investing-p-9781394265466"
%}}
