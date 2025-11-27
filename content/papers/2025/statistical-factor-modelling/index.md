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

Not only does this model allow us to explain returns, we can compare the returns of different assets in a meaningful way. For example, we could break down the returns of Apple and Orange and see how much of their returns are explained by the same factors.

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

The alpha vector $\boldsymbol{\alpha}$ is generally assumed to be $0$. If it is not, then we have an arbitrage opportunity. That is, we can find portfolio weights $\boldsymbol{w}$ such that $\boldsymbol{w}^\top \boldsymbol{\alpha} > 0$ and $\boldsymbol{w}^\top \boldsymbol{\beta} \boldsymbol{f}_t = 0$, meaning we can earn positive returns while having no exposure to the factors. In practice, a model that predicts asset returns is predicting this vector for time $t$, the elusive "alpha" that traders talk about.

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

Cons of statistical factor models:
* Requires long historical data. Cannot add new assets with little historical data.

Build long historical return series for a bunch of ETFs.

# Statistical factors

## PCA

1. How PCA works: gives us the loadings and the factors under two constraints:
    1. Factors are uncorrelated
    2. Factors explain maximum variance
1. Issues with interpretation

## Whitening

This makes the factor covariance matrix the identity matrix.

## Varimax rotation

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

