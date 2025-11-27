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

There are a handful of ways to build a factor model. (1) You know the factors $\boldsymbol{f}_t$, and want to estimate the loadings $\boldsymbol{\beta}$. This is *time series regression*. (2) You know the loadings and you want to estimate the factors. This is *cross-sectional regression*. (3) You don't know the factors or loadings and you want to estimate both. This is known as *statical factor modelling*.

The literature on these approaches is vast, but a fantastic overview is given in the paper [Factor Models, Machine Learning, and Asset Pricing](https://www.annualreviews.org/content/journals/10.1146/annurev-financial-101521-104735) [^Giglio2022].

In this article, we're going to explore statistical factor modelling. We will address a few issues that arise in this approach: (1) the method is data-hungry, requiring long historical return series; (2) the factors we get are often not economically meaningful; (3) the factors we get may not be stable out-of-sample.

# Factor models

A factor model is formulated as
$$
\boldsymbol{r}_t =  \boldsymbol{\alpha} + \boldsymbol{\beta} \boldsymbol{f}_t + \boldsymbol{\epsilon}_t
$$
where
* $\boldsymbol{r}_t$ is a random vector of returns for $N$ assets at time $t$. Generally, the risk free rate is subtracted, making these excess returns.
* $\boldsymbol{\alpha}$ is the *alpha* vector. This is an $N \times 1$ vector of constants that represent the average return of each asset not explained by the factors.
* $\boldsymbol{\beta}$ is the factor loading matrix. This is an $N \times K$ matrix where each row represents an asset and each column represents a factor. The entries represent the sensitivity of each asset to each factor.
* $\boldsymbol{f}_t$ is a $K \times 1$ random vector of factor returns.
* $\boldsymbol{\epsilon}_t$ is a random vector of *idiosyncratic* returns. This is an $N \times 1$ vector representing the portion of each asset's return not explained by the factors.


outline:

* Restate the factor model equation. I could probably make the intro equation simpler (drop the alpha term) and a full explainer here.
* Explain each term
* Talk about the assumptions we make about each term
    * E.g. factors are uncorrelated, errors are uncorrelated with factors, etc.
* Provide the names of the terms. I.e. say "idiosyncratic" for the error term, etc.
* point out that if alpha is not zero, then we have an arbitrage opportunity. In general, we assume alpha is zero, but it could be modeled as conditional on some variables.
* A perfect factor model has alpha = 0, E[e] = 0, Cov(e) = diagonal matrix, Cov(f) = identity.
* Most factor models aim for a small number of factors relative to the number of assets.
* Factor are extremely difficult to build. Large institutions either use their vast resources to build the in house or purchase commercially available factor models. I.E. Barra, Axioma, Northfield, etc.

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
