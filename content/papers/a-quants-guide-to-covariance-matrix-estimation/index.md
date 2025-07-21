---
title: "A Quant's Guide to Covariance Matrix Estimation"
summary: "
    Tired of noisy, unreliable covariance estimates? We break down three ideas that make a real difference: evaluate estimators on their own merits, decouple variance and correlation, and shrink your way to cleaner, more robust matrices.
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

# The basics

Cover the general model for estimating variance GARCH. Show how it collapses to an EWM. Point out other models in the literature, but say that we're going to stick wtih EWMs for this article.

We want to keep this bit simple and concise, we're really just making sure we're covering all bases.


# Measurement

The most straightforward way to evaluate covariance matrix estimates is through full backtesting. Simply run your complete investment strategy with different covariance estimators and compare the resulting performance metrics.

However, this approach has significant drawbacks. Backtesting is computationally expensive and time-consuming, making it impractical for rapid iteration and research. More importantly, mixing covariance estimation research with portfolio construction introduces confounding factors that can lead to overfitting. When performance improves, you don't know whether the gains come from better covariance estimates or inadvertent optimization of other portfolio components.

To address these issues, we need isolated evaluation methods that test covariance estimators independently of the full investment process. The fiddly thing is that there is no "best" covariance matrix, we can pick a general purpose metric or something that targets our particular need.  [REFERENCE] So and so covers a whole bunch of different ways of measuring covariance matrices. Here, we're going to cover two approaches; minimum-variance portfolios and the log-likelihood.

## Minimum-variance portfolios

The idea here is that we usually want a covariance matrix to minimise the variance of our portfolio. So, we can construct the minimum variance portfolios using different covariance estimators and measure their realized variance. The estimator producing the lowest variance portfolio is considered the better option.

The minimum variance portfolio is given by:
$$
\begin{aligned}
\underset{\boldsymbol{w}\_{t-1}}{\text{min}} &\quad \boldsymbol{w}^T_{t-1}\hat{\boldsymbol{\Sigma}}_{t-1}\boldsymbol{w}\_{t-1} \\\
\text{s.t.} &\quad \boldsymbol{w}^T\_{t-1}\boldsymbol{w}\_{t-1} = 1
\end{aligned}
$$
The solution is given by the eigen-vector corresponding to the smallest eigen-value of $\hat{\boldsymbol{\Sigma}}$ [^Ghojogh2023]. We'll call this portfolio the minimum-variance portfolio. Here is some Python code to calculate it:
```python
import numpy as np
import pandas as pd

def mvp(cov: pd.DataFrame) -> pd.Series:
    """
    Returns the minimum-variance portfolio weights
    from a covariance matrix.

    Args:
        cov: Covariance matrix of asset returns.
    """
    try:
        _, vecs = np.linalg.eigh(cov)
        w = vecs[:, 0]
    except np.linalg.LinAlgError:
        w = np.full(cov.shape[0], np.nan)
    
    return pd.Series(w, index=cov.columns)
```

The portfolio's return at time $t$ is then:
$$
p_t = \boldsymbol{w}^T\_{t-1}\boldsymbol{r}_t
$$
And the variance of the portfolio returns is:
$$
\text{var}(p_1, p_2, \dots, p_T)
$$



## Log-likelihood

Assuming returns follow a multivariate Gaussian distribution, we can evaluate how well each estimated covariance matrix explains observed returns using the log-likelihood function. For a vector of returns $\boldsymbol{r}_t$ and estimated covariance matrix $\hat{\boldsymbol{\Sigma}}\_{t-1}$, the log-likelihood is:
$$
\log L(\boldsymbol{r}_t) = -\frac{1}{2} \left[ \log(|\hat{\boldsymbol{\Sigma}}\_{t-1}|) + (\boldsymbol{r}_t - \boldsymbol{\mu})^T\hat{\boldsymbol{\Sigma}}\_{t-1}^{-1}(\boldsymbol{r}_t - \boldsymbol{\mu}) + n \log(2\pi) \right]
$$
We want the sum of these for all $t$ to be as large as possible. We can simplify this:
1. For our purposes, we can assume that $\boldsymbol{\mu} = 0$.
1. As we are optimising over the same set of $n$ returns, the term $n \log(2\pi)$ has no impact on our results.
1. Multiplying by $-\frac{1}{2}$ only has the effect of flipping the sign, we can drop this too.

The final metric is:
$$
\Sigma_t^T \left( \log(|\hat{\boldsymbol{\Sigma}}\_{t-1}|) + \boldsymbol{r}_t^T\hat{\boldsymbol{\Sigma}}\_{t-1}^{-1}\boldsymbol{r}_t \right)
$$

#### Similarity to the Mahalanobis distance

Blah blah blah

The determinant of the covariance matrix is a measure of the spread of the distribution. By including it, we are penalizing wider distributions. A wider distribution equals a lower distance and a thinner distribution equals larger distances. The inclusion of the determinant protects against simply estimating a wider distribution to decrease the distance.

## Example

Make an example for fitting the half-life to a regular EWM.


{{<figure src="mvp_vs_ll.svg" title="Mean-variance portfolios vs log-likelihood" >}}
Curabitur pulvinar magna sit amet mattis semper. Nulla interdum nunc quis turpis iaculis finibus. Donec purus leo, aliquam at malesuada sit amet, elementum vitae quam. Quisque mi justo, euismod ac leo nec, elementum eleifend purus. Etiam ut ornare velit.
{{</figure>}}


# Splitting

{{<figure src="variance_vs-correlation.svg" title="Variance and correlation behave differently." >}}
Curabitur pulvinar magna sit amet mattis semper. Nulla interdum nunc quis turpis iaculis finibus. Donec purus leo, aliquam at malesuada sit amet, elementum vitae quam. Quisque mi justo, euismod ac leo nec, elementum eleifend purus. Etiam ut ornare velit.
{{</figure>}}


Code to calculate variance and correlation independently:
```python
def ewm_cov(
    returns: pd.DataFrame,
    var_hl: float,
    corr_hl: float,
    min_periods: int,
) -> pd.DataFrame:
    """
    Calculate the exponentially weighted covariance matrix.

    Args:
        returns: DataFrame of asset returns.
        var_hl: Half-life of the variance.
        corr_hl: Half-life of the correlation.
        min_periods: Minimum number of periods to consider.
    """
    vars = returns.ewm(halflife=var_hl, min_periods=min_periods).var()
    corrs = returns.ewm(halflife=corr_hl, min_periods=min_periods).corr()

    for date in returns.index:
        std = np.sqrt(vars.loc[date])
        v = np.outer(std, std)
        corrs.loc[date] = corrs.loc[date].values * v

    return corrs
```

{{<figure src="longer_corr_halflife.svg" title="Better performance when correlation halflife is longer." >}}
Curabitur pulvinar magna sit amet mattis semper. Nulla interdum nunc quis turpis iaculis finibus. Donec purus leo, aliquam at malesuada sit amet, elementum vitae quam. Quisque mi justo, euismod ac leo nec, elementum eleifend purus. Etiam ut ornare velit.
{{</figure>}}

# Shrinkage

# Conclusion


{{% citation
    id="Ghojogh2023"
    author="Benyamin Ghojogh, Fakhri Karray and Mark Crowley"
    title="Eigenvalue and Generalized Eigenvalue Problems: Tutorial"
    publication="arXiv"
    year="2023"
    link="https://arxiv.org/abs/1903.11240"
%}}
