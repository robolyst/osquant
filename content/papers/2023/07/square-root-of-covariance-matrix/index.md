---
title: "Square root of a portfolio covariance matrix"
summary: "
The square root of your portfolio's covariance matrix gives you a powerful way of understanding where your portfolio variance is coming from. Here I show how to calculate the square root and provide an interactive example to explore how it works.
"
type: paper
katex: true # Enable mathematics on the page
date: "2023-07-27"
authors:
    - Adrian Letchford
categories:
    - mathematics
    - finance
notebook: ./notebook.js
# This article was written before the site organised articles under YEAR/MONTH/slug
url: /papers/square-root-of-covariance-matrix
---

You're likely familiar with calculating the variance of a portfolio given a covariance matrix of the portfolio's components. Denote the covariance matrix with \\( \boldsymbol{\Sigma} \\) and the vector of portfolio weights as \\( \boldsymbol{w} \\). The portfolio's variance is:
$$
\sigma^2 = \boldsymbol{w}^T\boldsymbol{\Sigma}\boldsymbol{w}
$$

If we could calculate the square root of \\( \boldsymbol{\Sigma} \\) such that:
$$
\boldsymbol{\Sigma} = \sqrt{\boldsymbol{\Sigma}}^T \sqrt{\boldsymbol{\Sigma}}
$$
then the standard deviation held in each of the components could be written as:
$$
\boldsymbol{\sigma} = \sqrt{\boldsymbol{\Sigma}}\boldsymbol{w}
$$
and the portfolio variance neatly is:
$$
\boldsymbol{\sigma}^T\boldsymbol{\sigma}
= \boldsymbol{w}^T\sqrt{\boldsymbol{\Sigma}}^T\sqrt{\boldsymbol{\Sigma}}\boldsymbol{w}
= \boldsymbol{w}^T\boldsymbol{\Sigma}\boldsymbol{w}
= \sigma^2
$$

In this post, I want to show you how to calculate \\( \sqrt{\boldsymbol{\Sigma}} \\) and explore what the vector of standard deviations \\( \boldsymbol{\sigma} \\) can tell us about our portfolio.

# Positive-semidefinite matrix

For the square root calculation to work, we require that the covariance matrix be positive-semidefinite.

A positive-semidefinite matrix \\( \boldsymbol{A} \\) is a matrix where \\( \boldsymbol{x}^T\boldsymbol{A}\boldsymbol{x} \geq 0 \\) for all \\( \boldsymbol{x} \in \mathcal{R}^n \\).

A covariance matrix is positive-semidefinite as it can be written in the form \\( \boldsymbol{\Sigma} = \boldsymbol{B}^T\boldsymbol{B}\\) which satisfies the positive-semidefinite inequality:
$$
\boldsymbol{x}^T\boldsymbol{B}^T\boldsymbol{B}\boldsymbol{x} \geq 0
$$

# Matrix square root

The square root of a positive-semidefinite matrix can be found by performing an eigen-decomposition on \\( \boldsymbol{\Sigma} \\)[^1]:
$$
\boldsymbol{\Sigma} = \boldsymbol{V}\boldsymbol{D}\boldsymbol{V}^{-1}
$$
Where \\( \boldsymbol{V} \\) is a matrix whose columns are the eigenvectors of \\( \boldsymbol{\Sigma} \\) and \\( \boldsymbol{D} \\) is the diagonal matrix whose elements are the corresponding eigenvalues.

We can write:

$$
\left(\boldsymbol{V}\boldsymbol{D}^{\frac{1}{2}}\boldsymbol{V}^{-1}\right)^2
= \boldsymbol{V}\boldsymbol{D}^{\frac{1}{2}}(\boldsymbol{V}^{-1}\boldsymbol{V})\boldsymbol{D}^{\frac{1}{2}}\boldsymbol{V}^{-1}
= \boldsymbol{V}\boldsymbol{D}^{\frac{1}{2}}\boldsymbol{D}^{\frac{1}{2}}\boldsymbol{V}^{-1}
= \boldsymbol{V}\boldsymbol{D}\boldsymbol{V}^{-1}
= \boldsymbol{\Sigma}
$$

Therefore, we have a square root of \\( \boldsymbol{\Sigma} \\):

$$
\sqrt{\boldsymbol{\Sigma}} = \boldsymbol{V}\sqrt{\boldsymbol{D}}\boldsymbol{V}^{-1}
$$

We can calculate this in Python with:

```python
import numpy as np

def calc_cov_sqrt(cov):
    d, V = np.linalg.eig(cov)
    return V @ np.sqrt(np.diag(d)) @ np.linalg.inv(V)
```

Some useful facts to note:

**Symmetry** - The matrix \\( \sqrt{\boldsymbol{\Sigma}} \\) is symmetric. We know this because we know that \\( \boldsymbol{\Sigma} = \boldsymbol{V}\boldsymbol{D}\boldsymbol{V}^{-1} \\) is symmetric therefore \\( \boldsymbol{V}\sqrt{\boldsymbol{D}}\boldsymbol{V}^{-1} \\) must also be symmetric.

**Positive eigenvalues** - A positive-semidefinite matrix has positive eigenvalues. We can prove this by remembering that \\( \boldsymbol{x}^T\boldsymbol{A}\boldsymbol{x} \geq 0 \\) for all \\( \boldsymbol{x} \\) including eigenvectors. This means that if \\( \boldsymbol{x} \\) is an eigen-vector with corresponding eigenvalue \\( \lambda \\) then we can say \\( \lambda\boldsymbol{x}^T\boldsymbol{x} \geq 0 \\) which means that \\( \lambda \geq 0 \\).

**Only one positive-semidefinite square root** - The square root of a positive number can be one of two values; a negative and a positive value. Therefore, the diagonal matrix \\( \boldsymbol{D} \\) has \\(2^n\\) possible square roots but only one of them has all positive values making it the only positive-semidefinite square root.

<feature>

# Interactive example

Here's an example using two assets. You can play with the variances, correlation and portfolio weights to see how the component standard deviations change.

Create the covariance matrix and square root:

<p>
    <div class="row align-items-center">
        <div class="col-12 col-md-6">
            <cell id="cov_matrix"></cell>
        </div>
        <div class="col-12 col-md-6">
            <cell id="viewof_std1"></cell>
            <cell id="viewof_std2"></cell>
            <cell id="viewof_rho"></cell>
        </div>
    </div>
</p>

Set the portfolio weights:

<p>
    <div class="row align-items-center">
        <div class="col-12 col-md-6">
            <cell id="w_vector"></cell>
        </div>
        <div class="col-12 col-md-6">
            <cell id="viewof_w1"></cell>
            <cell id="viewof_w2"></cell>
        </div>
    </div>
</p>

Component and portfolio std:

<div class="row">
    <div class="col-12 col-md-6">
        <cell id="component_std"></cell>
    </div>
</div>

</feature>

# Interpretation

Each element of the vector \\(\boldsymbol{\sigma} = \sqrt{\boldsymbol{\Sigma}}\boldsymbol{w}\\) tells you how much of your portfolio's standard deviation is held in that component. Squaring and summing the values gives you the portfolio's variance.

## Captures exposure

The correlation between assets is taken into account and the standard deviation is distributed across the correlated assets. For example, if the correlation between the two assets is 1 and we only invested into one asset we'd see the variance distributed between the two assets:
$$
\begin{aligned}
\sigma_1 &= 1 \\\
\sigma_2 &= 1 \\\
\rho &= 1 \\\
w_1 &= 1 \\\
w_2 &= 0 \\\
\text{then} \\ \boldsymbol{\sigma} &= \left[\begin{matrix}0.707\\\0.707\end{matrix}\right]
\end{aligned}
$$

The vector \\(\boldsymbol{\sigma} \\) tells us our exposure to an asset. Notice in the example above that the allocation to asset 2 is 0. Yet, because asset 1 is 100% correlated with asset 2, we have the same amount of exposure to asset 2 as we do to asset 1.

The vector \\(\boldsymbol{\sigma} \\) also tells you the direction of your exposure to a particular asset. If we change the example above so that the correlation is -1 then the portfolio has an effective short position in asset 2:
$$
\begin{aligned}
\sigma_1 &= 1 \\\
\sigma_2 &= 1 \\\
\rho &= -1 \\\
w_1 &= 1 \\\
w_2 &= 0 \\\
\text{then} \\ \boldsymbol{\sigma} &= \left[\begin{matrix}0.707 \\\ -0.707\end{matrix}\right]
\end{aligned}
$$

## Captures correlated risk

We can also see the effects of correlation on how risky our positions are. Take for example a situation where we hold a larger position in a less risky asset:
$$
\begin{aligned}
\sigma_1 &= 0.3 \\\
\sigma_2 &= 0.7 \\\
\rho &= 0 \\\
w_1 &= 0.5 \\\
w_2 &= 0.2 \\\
\text{then} \\ \boldsymbol{\sigma} &= \left[\begin{matrix}0.15 \\\ 0.14\end{matrix}\right] \\\
\sigma &= 0.205
\end{aligned}
$$
The positions \\( w_1 \\) and \\(w_2\\) have been set so that the level of risk in each asset is roughly the same.

Now, if these two assets were correlated, how does that change the riskiness of the positions?
$$
\begin{aligned}
\sigma_1 &= 0.3 \\\
\sigma_2 &= 0.7 \\\
\rho &= 1 \\\
w_1 &= 0.5 \\\
w_2 &= 0.2 \\\
\text{then} \\ \boldsymbol{\sigma} &= \left[\begin{matrix}0.114 \\\ 0.267\end{matrix}\right] \\\
\sigma &= 0.29
\end{aligned}
$$
The level of risk in the first asset lowers from 0.15 to 0.114 but the second asset increases from 0.14 to 0.267. This increase in the second asset is large enough to raise the riskiness of the total portfolio.

# Summary

The square root of a covariance matrix provides a way of measuring the amount of portfolio variance associated with each component. You can quantify exposure, direction and correlated risk.

[^1]: https://en.wikipedia.org/wiki/Square_root_of_a_matrix#Matrices_with_distinct_eigenvalues
