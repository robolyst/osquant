---
title: "Square root of a portfolio covariance matrix"
summary: "
The square root of your portfolio's covariance matrix gives you a powerful way of understanding where your return variance is coming from. Here I show how to calculate the square root and and I created an interactive example to explore how it works.
"
type: paper
katex: true # Enable mathematics on the page
date: "2023-07-27"
authors:
    - Adrian Letchford
categories:
    - mathematics
    - finance
notebook: https://api.observablehq.com/d/f0af168967249909.js?v=3
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

**Only one positive-semidefinite square root** - A positive-semidefinite matrix has \\(2^n\\) square roots but only one of them is also positive-semidefinite. There are two ways of taking the square root of a positive number, a negative number and a positive number. Therefore, the diagonal matrix \\( \boldsymbol{D} \\) has \\(2^n\\) possible square roots but only one of them has all positive values. 

<feature>

# Interactive example

Here's an example using two assets. You can play with the variances, correlation and portfolio weights to see how the component standard deviations change.

### Create the covariance matrix and square root

<div class="row align-items-center">
    <div class="col">
        <cell id="cov_matrix"></cell>
    </div>
    <div class="col">
        <cell id="viewof_std1"></cell>
        <cell id="viewof_std2"></cell>
        <cell id="viewof_rho"></cell>
    </div>
</div>

### Set the portfolio weights

<div class="row align-items-center">
    <div class="col">
        <cell id="w_vector"></cell>
    </div>
    <div class="col">
        <cell id="viewof_w1"></cell>
        <cell id="viewof_w2"></cell>
    </div>
</div>

### Component std

<div class="row">
    <div class="col">
        <cell id="component_std"></cell>
    </div>
</div>

</feature>

# Interpretation






[^1]: https://en.wikipedia.org/wiki/Square_root_of_a_matrix#Matrices_with_distinct_eigenvalues