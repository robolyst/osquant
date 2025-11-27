---
title: "Statistical Factor Modelling"
summary: "
    Lorem ipsum dolor sit amet, consectetur adipiscing elit. Integer odio neque, volutpat vel nunc
    ut. Duis maximus massa vitae libero imperdiet feugiat quis a sapien. Quisque sodales neque dui,
    a mollis justo porta eu. Nullam semper ipsum ac ante rhoncus, ac facilisis lacus posuere. Mauris
    pulvinar elementum ligula in mattis. Fusce rhoncus consequat lorem accumsan rhoncus.
"

date: "2025-11-26"
type: paper
mathjax: true
authors:
    - Adrian Letchford
categories:
    - finance
---

When you buy something, and the price changes, what happend? Let's say, for example, you buy shares in a made-up health company called "Orange". This company makes wearable health monitors. The price of Orange goes down by 5%. Why? It could be because the whole market went down, or because the technology sector went down, or because the health sector went down, or because Orange genuinely performed poorly. Each of these possible explanations is called a *factor*.

The idea of a factor model is that every stream of returns can be explained by a number of underlying factors. A factor model roughly says $\boldsymbol{r}_t \approx \boldsymbol{\beta} \boldsymbol{f}_t$ where $\boldsymbol{r}_t$ is a vector of returns at time $t$, $\boldsymbol{\beta}$ is a matrix of factor loadings and $\boldsymbol{f}_t$ is a vector of factor returns. If we can identify these factors, we can understand what is driving returns.

Not only does this model allow us to explain returns, we can compare the returns of different assets in a meaningful way. For example, we could break down the returns of [Apple](https://uk.finance.yahoo.com/quote/AAPL/) and Orange and see how much of their returns are explained by the same factors.

There are a handful of ways to build a factor model. (1) You know the factors $\boldsymbol{f}_t$, and want to estimate the loadings $\boldsymbol{\beta}$. This is a macroeconomic factor model. (2) You know the loadings and you want to estimate the factors. This is a characteristic factor model. (3) You don't know the factors or loadings and you want to estimate both. This is as *statical factor model* [^Conner2007].

The literature on these approaches is vast, but a fantastic overview is given in the paper [Factor Models, Machine Learning, and Asset Pricing](https://www.annualreviews.org/content/journals/10.1146/annurev-financial-101521-104735) [^Giglio2022].

In this article, we're going to explore statistical factor modelling. We will address a few issues that arise in this approach: (1) the method is data-hungry, requiring long historical return series; (2) the factors we get are often not economically meaningful; (3) the factors we get may not be stable out-of-sample.

# Factor models

*For a deep dive into factor models, the best resource on the market is [The Elements of Quantitative Investing](https://www.wiley.com/en-us/The+Elements+of+Quantitative+Investing-p-9781394265466) [^Paleologo2025]. Here, we'll lightly go over everything that will be useful for this article.*

A factor model is formulated as
$$
\boldsymbol{r}_t = \boldsymbol{\alpha} + \boldsymbol{\beta} \boldsymbol{f}_t + \boldsymbol{\epsilon}_t
$$

where
* $\boldsymbol{r}_t$ is a random vector of returns for $N$ assets at time $t$. Generally, the risk free rate is subtracted, making these excess returns.
* $\boldsymbol{\alpha}$ is the *alpha* vector. This is an $N \times 1$ vector of constants that represent the average return of each asset not explained by the factors.
* $\boldsymbol{\beta}$ is the factor loading matrix. This is an $N \times K$ matrix where each row represents an asset and each column represents a factor. The entries represent the sensitivity of each asset to each factor.
* $\boldsymbol{f}_t$ is a $K \times 1$ random vector of factor returns.
* $\boldsymbol{\epsilon}_t$ is a random vector of *idiosyncratic* returns. This is an $N \times 1$ vector representing the portion of each asset's return not explained by the factors.

The alpha vector $\boldsymbol{\alpha}$ is generally assumed to be $\text{E}[\boldsymbol{\alpha}] = 0$. If it is not, then we have an arbitrage opportunity. That is, we can find portfolio weights $\boldsymbol{w}$ such that $\boldsymbol{w}^\top \boldsymbol{\alpha} > 0$ and $\boldsymbol{w}^\top \boldsymbol{\beta} \boldsymbol{f}_t = 0$, meaning we can earn positive returns while having no exposure to the factors. In practice, a model that predicts asset returns is predicting this vector for time $t$, the elusive "alpha" that traders talk about.

Analogously, any model of $\text{E}[\boldsymbol{f}_t]$ is a model of the risk premia associated with each factor. 

The ideal factor returns have a covariance matrix that is the identity matrix, i.e. $\boldsymbol{\Sigma}_f = \boldsymbol{I}$. The immediate implication of this is that the covariance of $\boldsymbol{r}_t$ is
$$
\boldsymbol{\Sigma}_r = \boldsymbol{\beta} \boldsymbol{\Sigma}\_f \boldsymbol{\beta}^\top + \boldsymbol{\Sigma}\_\epsilon = \boldsymbol{\beta} \boldsymbol{\beta}^\top + \boldsymbol{\Sigma}\_\epsilon
$$
This condition turns out to be very easy to enforce as we will see later.

The idiosyncratic returns $\boldsymbol{\epsilon}_t$ are assumed to have expectation of 0 as any mean is captured with the $\boldsymbol{\alpha}$ vector. They are also assumed to be uncorrelated with the factors and generally with eachother. This means that the covariance matrix $\boldsymbol{\Sigma}\_\epsilon$ is a diagonal matrix. This is known as the *idiosyncratic risk* or *specific risk* of each asset. In practice, this assumption does not hold perfectly. Practitioners instead aim for a sparsely populated covariance matrix, meaning that most assets have little correlation with each other after accounting for the factors. Intuitively, if two assets are very similar, then they will have some correlation that is not explained by the factors.

Most factor models aim to have a small number of factors relative to the number of assets, $K \ll N$. For example, the commercially available [MSCI USA Equity Factor Models](https://www.msci.com/downloads/web/msci-com/data-and-analytics/factor-investing/equity-factor-models/MSCI%20USA%20Equity%20Factor%20Model-cfs-en.pdf) factor models have on the order of 50-100 factors for 20,000+ assets. These factor models require vast data resources and research to build. They are expensive to purchase and used by large institutions.

For our purposes, we will focus on building a useful factor model that can be built with at home resources. We will be using the same number of factors as assets, but we will see that we will still gain a large amount of insight into the returns of our assets.

# Why?

I said earlier that factor models allow us to explain returns. Why this is useful is not immediately obvious. Before we jump into the details of building these models, let's explore what we can do with them.

1. You can explain where returns come from.
    1. You can attribute performance. I.e. you can attribute portfolio performance to specific factors or to asset-specific alpha/idio-syncratic.
1. You can explicitly express risk premia.
1. You can seperate risk premia from alpha in your return predictions.
1. You can build better risk models. i.e. because we know where the risk is coming from, we can hedge it. Or, more specifically, we can invest in a single or multiple factors to the exclusion of all others.
1. You can build more diversified portfolios. I.e. you diversify across factors, not across assets all exposed to the same factors.


# Data

We're going to use real data to illustrate the modelling in this article. To keep things simple, we're going to use a selection of ETFs.

Choosing ETFs means we can focus on only a small number of assets but still capture a wide selection of the market. But more importantly, as ETFs tend to be mechanically constructed to track an index, we can extend their historical data backwards by using either the index returns or the returns of related assets before the ETF was created.

In this article, we're going to use the following U.S. based ETFs:

| Ticker  |Inception    | Description |
|:--------|:------------|:------------|
| [IWM](https://www.ishares.com/us/products/239710/ishares-russell-2000-etf)    | 22 May 2000 | Tracks the Russell 2000 --- small-capitalization equities. |
| [QQQ](https://www.invesco.com/qqq-etf/en/home.html)                           |  3 Oct 1999 | Tracks the Nasdaq-100 --- dominated by technology.         |
| [SPY](https://www.ssga.com/us/en/intermediary/etfs/spdr-sp-500-etf-trust-spy) | 22 Jan 1993 | Tracks the S&P 500 --- represents all equities.            |
| [XLC](https://www.sectorspdrs.com/mainfund/xlc)                               | 18 Jun 2018 | Tracks the communication services sector.                  |
| [XLY](https://www.sectorspdrs.com/mainfund/xly)                               | 16 Dec 1998 | Tracks the consumer discretionary sector.                  |
| [XLP](https://www.sectorspdrs.com/mainfund/xlp)                               | 16 Dec 1998 | Tracks the consumer staples sector.                        |
| [XLE](https://www.sectorspdrs.com/mainfund/xle)                               | 16 Dec 1998 | Tracks the energy sector.                                  |
| [XLF](https://www.sectorspdrs.com/mainfund/xlf)                               | 16 Dec 1998 | Tracks the financials sector.                              |
| [XLV](https://www.sectorspdrs.com/mainfund/xlv)                               | 16 Dec 1998 | Tracks the health care sector.                             |
| [XLI](https://www.sectorspdrs.com/mainfund/xli)                               | 16 Dec 1998 | Tracks the industrials sector.                             |
| [XLB](https://www.sectorspdrs.com/mainfund/xlb)                               | 16 Dec 1998 | Tracks the materials sector.                               |
| [XLRE](https://www.sectorspdrs.com/mainfund/xlre)                             |  7 Oct 2015 |  Tracks the real estate sector.                            |
| [XLK](https://www.sectorspdrs.com/mainfund/xlk)                               | 16 Dec 1998 | Tracks the technology sector.                              |
| [XLU](https://www.sectorspdrs.com/mainfund/xlu)                               | 16 Dec 1998 | Tracks the utilities sector.                               |

At first, these ETFs look like they cover unrelated parts of the market. Looking in the figure below, we can see that the energy sector (XLE) is 45% correlated with the real estate sector (XLRE). In fact, all the ETFs are highly correlated suggesting that there are similar underlying factors that they all share. 

{{<figure src="images/etf_correlations.svg" title="ETF correlation matrix." >}}
Shows the correlations between the selected ETFs over the period 2016-01-01 to 2025-11-25. Key thing to note is that all of the ETFs are highly correlated, suggesting that there are common factors driving their returns.
{{</figure>}}

The only issue with these ETFs is that that they have different inception dates and some only have a few years of history. For example, the communication sector ETF (XLC) only has data from 2018 onwards. To get around this, I construct synthetic returns for each ETF before its inception date going back to 1990-01-01. I do this by getting a large set of related stock returns and index returns and regress the first 5 years of ETF returns against these related returns. I then use the resulting regressed returns as the historical returns before the ETF's inception date.

While this is not perfect, it does give us an approximation of the returns you might have realised if you were tracking the sector (or index) before the ETF was created.

The full details of this process are in the [appendix](#appendix).

As an example, here are the returns of the materials sector ETF (XLB) spliced onto the synthetic returns before inception:

{{<figure src="images/synthetic_etf_example.svg" title="Synthetic history example." >}}
The capital over time from investing in the materials sector ETF (XLB). From 1990-01-01 to 1998-12-15, the returns are synthetic, generated by regressing related stock and index returns against the actual ETF returns for the 5 year period starting at inception. After inception, the actual ETF returns are used. The synthetic returns after inception are included in the plot to show how well the synthetic returns approximate the actual returns after inception.
{{</figure>}}

<todo>Later on, we throw around the term "risk premia". This is incorrect as we are not looking at excess returns. Adjust everything by the effective federal funds rate. Can backfill this with the monthly effective federal funds rate and the plain federal funds rate.</todo>

So that you can reproduce what we do here, the full set of returns can be downloaded [here](returns.csv). And you can read them in with:
```python
import pandas as pd

R = pd.read_csv('returns.csv', index_col=0, parse_dates=True)
```

# Statistical factors

Now that we have a long history of returns, we can build a statistical factor model. For this section, we are going to work in-sample. There are unique challenges when working out-of-sample that distract from the core modelling ideas. We will address these challenges later.

## PCA

Principle component analysis [(PCA) is the workhorse](https://www.google.com/search?q=%22PCA+is+the+workhorse%22) of statistical factor modelling. It is robust and deeply understood. Many before me have written excellent explanations of PCA, see for example [this blog post](https://gregorygundersen.com/blog/2022/09/17/pca/). Here, will go through the idea and the theory behind PCA as it relates to factor modelling.

* A short paragraph on what PCA is doing.
* A visualisation of PCA in 2D. Make something of multiple panels in a similar way to quantamagazine.
* Run through the mathematics of PCA.
* Point out explicitly that PCA is just a rotation of the data. We're able to rotate the data to a new basis with certain properties. This is very powerful and will be a recurring idea in factor modelling.
* Show the factor model found with PCA. That is, how do we get the factor loadings from PCA, and show that r = LF (a=0, e=0).
* Run PCA on the ETF returns. Do all the code ourselves, do not use sklearn. This ensure no centering and drives home what we're doing.
* Show that the correlation matrix of the factors is the identity matrix.
* Convert the factor loadings to weights that we can use to interpret the factors.
* Show one of the factors and point out that it's virtually impossible to interpret.

The PCA algorithm in Python with numpy:
```python
import numpy as np

cov = R.cov()
eigvals, vecs = np.linalg.eigh(cov)

# The eigenvalues and eigenvectors are
# in ascending order. Reverse them so
# that the largest eigenvalues are first.
eigvals = eigvals[::-1]
vecs = vecs[:, ::-1]

# Converts returns to factors
iloadings = pd.DataFrame(vecs, index=R.columns)

# Converts factors to returns
loadings = pd.DataFrame(vecs.T, columns=R.columns)

factors = R @ iloadings
```

You can check for yourself that the factors are not correlated by computing their correlation matrix: `factors.corr()`.

Generally when you run PCA, the idea is that you only keep the first $K$ factors that explain most of the variance. Here, however, we are going to keep all the factors. Previous work on factor modelling shows that there are on the order of 100 factors. In our case, we only have 14 assets each of which covers a broad selection of the market. Therefore, we expect that all 14 factors are needed to explain the returns. Additionally, by keeping all the factors, $\boldsymbol{\beta}$ is square and invertible:
$$
\begin{aligned}
\boldsymbol{R}^\top &= \boldsymbol{\beta}\boldsymbol{F}^\top \\\
\boldsymbol{\beta}^{-1}\boldsymbol{R}^\top &= \boldsymbol{F}^\top \\\
\end{aligned}
$$

The factors look like:

![PCA factors](images/pca_factors.svg)

The first factor is often called the "market factor" as it tends to represent the overall market movement. As we are looking at equities, we could interpret the first factor as the equity risk premia. The other factors are generally more difficult to interpret. However, we can see that there may be some hits of other orthogonal risk premia. For example, factor 3 exhibits an upward trend.

{{<figure src="images/pca_asset_component_weights.svg" title="Synthetic history example." >}}
The capital over time from investing in the materials sector ETF (XLB). From 1990-01-01 to 1998-12-15, the returns are synthetic, generated by regressing related stock and index returns against the actual ETF returns for the 5 year period starting at inception. After inception, the actual ETF returns are used. The synthetic returns after inception are included in the plot to show how well the synthetic returns approximate the actual returns after inception.
{{</figure>}}

Some observations:

* The market factor does not have any assets with large weights. This factor contributes fairly evenly to all assets.


## Whitening

This makes the factor covariance matrix the identity matrix.

## Varimax rotation

* Is it possible to keep one of the factors fixed and rotate the others around it? For example, can we keep the market factor fixed and rotate the other factors to be more interpretable?

## Interpretation

# In practice

## Lagged loadings

## Exponential weighting

## Ensuring factor orthogonality

## Interpretation

## Testing

Or ensuring the factor model maintains in-sample properties out-of-sample.

{{% citation
    id="Giglio2022"
    author="Stefano Giglio, Bryan Kelly and Dacheng Xiu"
    title="Factor Models, Machine Learning, and Asset Pricing"
    year="2022"
    publication="Annual Review of Financial Economics"
    volume="14"
    pages="337-68"
    link="https://doi.org/10.1146/annurev-financial-101521-104735"
%}}

{{% citation
    id="Paleologo2025"
    author="Giuseppe A. Paleologo"
    title="The Elements of Quantitative Finance"
    publication="Wiley"
    year="2025"
    link="https://www.wiley.com/en-us/The+Elements+of+Quantitative+Investing-p-9781394265466"
%}}

{{% citation
    id="Conner2007"
    author="Gregory Connor and Robert A. Korajczyk"
    title="Factor Models of Asset Returns"
    year="2007"
    publication="ENCYCLOPEDIA OF QUANTITATIVE FINANCE"
    link="https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1024709"
%}}

