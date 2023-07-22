---
title: "Recursive least-squares linear regression"
summary: "
Rather than recalculating a least--squares model on each new data point, you can simply update the fitted coefficients. This saves you computational time and allows you to place greater emphasis on recent data.
"
type: paper
katex: true # Enable mathematics on the page
date: "2023-07-22"
authors:
    - Adrian Letchford
categories:
    - mathematics
feature: true
---

*I first learned about this algorithm in the book Kernel Adaptive Filter: A Comprehensive Introduction[^Liu-2010] sometime in 2012 or 2013. This book goes in depth into how to build kernel filters and does a fantastic job of easing you into the mathematics. I highly recommend having a read if you can.*

In my trading algorithms, at each time period, I use a linear regression to predict future returns of each instrument in my portfolio. I previously fit this regression at each time step. However, the solution to the least--squares problem can be rewritten so that the weight vector is updated at each time period without refitting the entire model. This means that my algorithm run significantly faster speeding up back tests.

# Least squares

Define a sequence of training data \\( \\{ \boldsymbol{x}\_{t}, y_{t} \\}\_{t=1}^{i-1} \\) where we want to predict \\( y_{t} \\) with \\( \boldsymbol{x}_t \\). We can use least-squares to define a minimisation problem:

$$
\boldsymbol{w}\_{i-1} = \min_{\boldsymbol{w}} \sum_{t=1}^{i-1} ( y_{t} - \boldsymbol{x}_t^T\boldsymbol{w})^2
$$

Denote:
$$
\begin{aligned}
\boldsymbol{X}\_{i-1} &= [ \boldsymbol{x}_1,  \dots, \boldsymbol{x}\_{T-1} ]^T \\\
\boldsymbol{y}\_{i-1} &= [ y_1,  \dots, y\_{T-1} ]^T \\\
\end{aligned}
$$

Which allows us to rewrite the problem above as:
$$
\boldsymbol{w}\_{i-1} = \min_{\boldsymbol{w}} ||\boldsymbol{y}\_{i-1} - \boldsymbol{X}\_{i-1}\boldsymbol{w}||^2
$$

The solution is:
$$
\begin{aligned}
\boldsymbol{w}_{i-1} = (\boldsymbol{X}\_{i-1}^T \boldsymbol{X}\_{i-1})^{-1} \boldsymbol{X}\_{i-1} \boldsymbol{y}\_{i-1} \label{1}\tag{1}
\end{aligned}
$$

This is used to make a prediction of \\( y_{i} \\):
$$
\boldsymbol{x}_i^T\boldsymbol{w}\_{i-1}
$$

# Recursive least-squares

We can derive a recursive formula for \\( \boldsymbol{w}\_{i-1} \\) so that we do not have to recalculate Eq. \\( \ref{1} \\) at each time step.

Denote:
$$
\begin{aligned}
\boldsymbol{X}\_{i} &= \left[\begin{matrix}\boldsymbol{X}\_{i-1} \\\ \boldsymbol{x}_{i} \end{matrix}\right] \\\
\boldsymbol{y}\_{i} &= \left[\begin{matrix}\boldsymbol{y}\_{i-1} \\\ y\_{i} \end{matrix}\right] \\\
\end{aligned}
$$

And define the inverse correlation matrices:
$$
\begin{aligned}
\boldsymbol{P}\_{i-1} &= (\boldsymbol{X}\_{i-1}^T \boldsymbol{X}\_{i-1})^{-1} \\\
\boldsymbol{P}\_{i} &= (\boldsymbol{X}\_{i}^T \boldsymbol{X}\_{i})^{-1} \\\
\end{aligned}
$$

Observe that:
$$
\begin{aligned}
\boldsymbol{P}\_{i} &= (\boldsymbol{X}\_{i-1}^T \boldsymbol{X}\_{i-1} + \boldsymbol{x}_i\boldsymbol{x}_i^T)^{-1} \\\
\end{aligned}
$$

By using the [Sherman--Morrison formula](https://en.wikipedia.org/wiki/Sherman%E2%80%93Morrison_formula) which is a special case of the [Woodbury matrix identity](https://en.wikipedia.org/wiki/Woodbury_matrix_identity):
$$
(\boldsymbol{A} + \boldsymbol{b}\boldsymbol{c}^T)^{-1} =
\boldsymbol{A}^{-1} - \frac{\boldsymbol{A}^{-1}\boldsymbol{b}\boldsymbol{c}^T\boldsymbol{A}^{-1}}{1 + \boldsymbol{c}^T\boldsymbol{A}^{-1}\boldsymbol{b}}
$$
with the following substitutions:
$$
\boldsymbol{A} = \boldsymbol{X}\_{i-1}^T \boldsymbol{X}\_{i-1}, \quad
\boldsymbol{b} = \boldsymbol{x}_i, \quad
\boldsymbol{c} = \boldsymbol{x}_i
$$

we can derive a recursive formula for \\( \boldsymbol{P}_i \\):
$$
\boldsymbol{P}_i = \left[ \boldsymbol{P}\_{i-1} - \frac{\boldsymbol{P}\_{i-1}\boldsymbol{x}_i\boldsymbol{x}_i^T\boldsymbol{P}\_{i-1}}{1 + \boldsymbol{x}_i^T \boldsymbol{P}\_{i-1} \boldsymbol{x}_i} \right]
$$

This leads us to a recursive formula for \\( \boldsymbol{w}_i \\):
$$
\begin{aligned}
\boldsymbol{w}_i &= \boldsymbol{P}_i \boldsymbol{X}_i \boldsymbol{y}_i \\\
&= \left[ \boldsymbol{P}\_{i-1} - \frac{\boldsymbol{P}\_{i-1}\boldsymbol{x}_i\boldsymbol{x}_i^T\boldsymbol{P}\_{i-1}}{1 + \boldsymbol{x}_i^T \boldsymbol{P}\_{i-1} \boldsymbol{x}_i} \right][\boldsymbol{X}\_{i-1}\boldsymbol{y}\_{i-1} + \boldsymbol{x}_i y_i] \\\
&= \boldsymbol{w}\_{i-1} + \frac{\boldsymbol{P}\_{i-1}\boldsymbol{x}_i}{1 + \boldsymbol{x}_i^T \boldsymbol{P}\_{i-1} \boldsymbol{x}_i} [y_i - \boldsymbol{x}_i^T\boldsymbol{w}\_{i-1}]
\end{aligned}
$$

I've skipped steps in the derivation of \\( \boldsymbol{w}_i \\) above. The missing steps are made up of tedious algebra. If you're curious, the full derivation can be found on page 96 of Kernel Adaptive Filtering[^Liu-2010].

The recursive linear regression algorithm keeps track of \\( \boldsymbol{P} \\) and \\( \boldsymbol{w} \\) and updates them on each data point.

The next question is what should the initial values be? The weight vector \\( \boldsymbol{w} \\) can be set to zero. A common strategy for \\( \boldsymbol{P} \\) is to collect an initial set of data and explicitly calculate \\( \boldsymbol{P} \\). However, as the matrix is updated, it may become rank deficient. This can be solved by regularising \\( \boldsymbol{w} \\).

# L2 regularisation

We can add a L2 regularisation to the minimisation problem:

$$
\boldsymbol{w}\_i = \min_{\boldsymbol{w}} \sum_{t=1}^{i} ( y_{t} - \boldsymbol{x}_t^T\boldsymbol{w})^2 + \lambda ||\boldsymbol{w}||^2
$$

Where \\( \lambda \\) is the regularisation term and is a positive number.

The solution to this problem is:
$$
\boldsymbol{w}_{i} = (\boldsymbol{X}\_{i} \boldsymbol{X}\_{i}^T + \lambda \boldsymbol{I})^{-1} \boldsymbol{X}\_{i} \boldsymbol{y}\_{i}
$$

Where \\( \boldsymbol{I} \\) is the identity matrix. This means that the initial value for \\( \boldsymbol{P} \\) can be set to:
$$
\boldsymbol{P}_0 = \lambda^{-1} \boldsymbol{I}
$$
Note that because we are calculating the inverse of \\( \lambda \\) we cannot set it to 0; it must strictly be greater than 0.

----
**Algorithm 1** - Recursive linear regression with L2 regularisation

---

*Initialise*
$$
\begin{aligned}
\boldsymbol{w}\_0 &= 0 \\\
\boldsymbol{P}\_0 &= \lambda^{-1} \boldsymbol{I} \\\
\end{aligned}
$$

*Computation*

For \\( i \geq 1 \\):
$$
\begin{aligned}
r_i &= 1 + \boldsymbol{x}\_i^T\boldsymbol{P}\_{i-1}\boldsymbol{x}\_i \\\
\boldsymbol{k}_i &= \frac{\boldsymbol{P}\_{i-1}\boldsymbol{x}\_i}{r_i} \\\
e_i &= y_i - \boldsymbol{x}\_i^T \boldsymbol{w}\_{i-1} \\\
\boldsymbol{w}_i &= \boldsymbol{w}\_{i-1} + \boldsymbol{k}_i e_i \\\
\boldsymbol{P}_i &= \boldsymbol{P}\_{i-1} - \boldsymbol{k}_i \boldsymbol{k}_i^T r_i
\end{aligned}
$$

----

In Python, this looks like:

```python
import numpy as np

class L2Regression:
    
    def __init__(self, num_features: int, lam: float):
        self.n = num_features
        self.lam = lam
        self.w = np.zeros(self.n)
        self.P = np.diag(np.ones(self.n) * self.lam)
        
    def update(self, x: np.ndarray, y: float) -> None:
        r = 1 + (x.T @ self.P @ x)
        k = (self.P @ x) / r
        e = y - x @ self.w
        self.w = self.w + k * e
        k = k.reshape(-1, 1)
        self.P = self.P - (k @ k.T) * r
        
    def predict(self, x: np.ndarray) -> float:
        return self.w @ x
```

# Exponential weighting

A problem with the recursive algorithm is that each data point has an equal impact on \\( \boldsymbol{w}_i \\). That is, each data point has the same weight. In practice an exponential weighting is used to put more weight on recent data and less weight on older data.

The weighted regularised least--squares problem is defined as:
$$
\boldsymbol{w}\_{i} = \min_{\boldsymbol{w}} \sum_{t=1}^{i} \beta^{i-t}( y_{t} - \boldsymbol{x}_t^T\boldsymbol{w})^2 + \beta^i \lambda ||\boldsymbol{w}||^2
$$

The parameter \\( \beta \\) is often called the *forgetting factor*. This value is not intuitive to set. I like to use the half--life (\\( h \\)) from which  \\( \beta \\) can be calculated:
$$
\beta = e^{\frac{\log(0.5)}{h}}
$$

Note that the regularisation term is multiplied by \\( \beta^i \\) which means that regularising has less impact with more data points.

The solution is given by:
$$
\boldsymbol{w}_i = (\boldsymbol{X}_i^T \boldsymbol{B}_i \boldsymbol{X}_i + \beta^i \lambda \boldsymbol{I})^{-1} \boldsymbol{X}_i \boldsymbol{B}_i \boldsymbol{y}_i
$$
where:
$$
 \boldsymbol{B}_i = \text{diag}([\beta^{i-1}, \beta^{i-2}, \dots, 1])
$$

Following the same derivation as before gives us the following algorithm:

----
**Algorithm 2** - Exponentially weighted recursive linear regression with L2 regularisation

---

*Initialise*
$$
\begin{aligned}
\boldsymbol{w}\_0 &= 0 \\\
\boldsymbol{P}\_0 &= \lambda^{-1} \boldsymbol{I} \\\
\end{aligned}
$$

*Computation*

For \\( i \geq 1 \\):
$$
\begin{aligned}
r_i &= 1 + \beta^{-1} \boldsymbol{x}\_i^T \boldsymbol{P}\_{i-1} \boldsymbol{x}\_i \\\
\boldsymbol{k}_i &= \beta^{-1} \frac{\boldsymbol{P}\_{i-1}\boldsymbol{x}\_i}{r_i} \\\
e_i &= y_i - \boldsymbol{x}\_i^T \boldsymbol{w}\_{i-1} \\\
\boldsymbol{w}_i &= \boldsymbol{w}\_{i-1} + \boldsymbol{k}_i e_i \\\
\boldsymbol{P}_i &= \beta^{-1} \boldsymbol{P}\_{i-1} - \boldsymbol{k}_i \boldsymbol{k}_i^T r_i
\end{aligned}
$$

----

In Python, this looks like:

```python
import numpy as np

class ExpL2Regression:
    
    def __init__(self, num_features: int, lam: float, halflife: float):
        self.n = num_features
        self.lam = lam
        self.beta = np.exp(np.log(0.5) / halflife)
        self.w = np.zeros(self.n)
        self.P = np.diag(np.ones(self.n) * self.lam)
        
    def update(self, x: np.ndarray, y: float) -> None:
        r = 1 + (x.T @ self.P @ x) / self.beta
        k = (self.P @ x) / (r * self.beta)
        e = y - x @ self.w
        self.w = self.w + k * e
        k = k.reshape(-1, 1)
        self.P = self.P / self.beta - (k @ k.T) * r
        
    def predict(self, x: np.ndarray) -> float:
        return self.w @ x
```

# L1 regularisation

If you wish to adapt a sparse model fit, a L1--norm regularisation is usually used. Adding such a term to the problem definition gives:

$$
\boldsymbol{w}\_{i} = \min_{\boldsymbol{w}} \sum_{t=1}^{i} \beta^{i-t}( y_{t} - \boldsymbol{x}_t^T\boldsymbol{w})^2 + \beta^i \lambda ||\boldsymbol{w}||^2 + \gamma ||\boldsymbol{w}||_1
$$

Eksioglu[^Eksioglu-2011] derived an L1--norm version of the recursive least--squares algorithm. The derivation is quite involved and surmounts to adding an extra term to the weight vector \\( \boldsymbol{w}_i \\).

The extra term is:
$$
\gamma \left(\frac{\beta - 1}{\beta}\right) (\boldsymbol{I} - \boldsymbol{k}_i \boldsymbol{x}_i^T) \boldsymbol{P}\_{i-1} \frac{\text{sign}(\boldsymbol{w}\_{i-1})}{|\boldsymbol{w}\_{i-1}| + \epsilon}
$$

Which gives us the following algorithm:

----
**Algorithm 3** - Exponentially weighted recursive linear regression with L1 & L2 regularisation

---
*Parameters*
$$
\lambda \gt 0, \quad 0 \lt \beta \lt 1, \quad \gamma > 0, \quad \epsilon > 0
$$

*Initialise*
$$
\boldsymbol{w}\_0 = 0, \quad \boldsymbol{P}\_0 = \lambda^{-1} \boldsymbol{I}
$$

*Computation*

For \\( i \geq 1 \\):
$$
\begin{aligned}
r_i &= 1 + \beta^{-1} \boldsymbol{x}\_i^T \boldsymbol{P}\_{i-1} \boldsymbol{x}\_i \\\
\boldsymbol{k}_i &= \beta^{-1} \frac{\boldsymbol{P}\_{i-1}\boldsymbol{x}_i}{r_i} \\\
e_i &= y_i - \boldsymbol{x}\_i^T \boldsymbol{w}\_{i-1} \\\
\boldsymbol{P}_i &= \beta^{-1} \boldsymbol{P}\_{i-1} - \boldsymbol{k}_i \boldsymbol{k}_i^T r_i \\\
\boldsymbol{z}_i &= \gamma \left(\frac{\beta - 1}{\beta}\right) (\boldsymbol{I} - \boldsymbol{k}_i \boldsymbol{x}_i^T) \boldsymbol{P}\_{i-1} \frac{\text{sign}(\boldsymbol{w}\_{i-1})}{|\boldsymbol{w}\_{i-1}| + \epsilon} \\\
\boldsymbol{w}_i &= \boldsymbol{w}\_{i-1} + \boldsymbol{k}_i e_i + \boldsymbol{z}_i \\\
\end{aligned}
$$

----

Two new parameters have been introduced. \\( \gamma \\) is the L1 regularisation parameter. Set this to a positive non-zero value. \\( \epsilon \\) is a small positive value to prevent division by zero when weights are zero.

In Python this looks like:

```python
import numpy as np

class ExpL1L2Regression:
    
    def __init__(self, num_features: int, lam: float, halflife: float, gamma: float, epsilon: float):
        self.n = num_features
        self.lam = lam
        self.gamma = gamma
        self.epsilon = epsilon
        self.beta = np.exp(np.log(0.5) / halflife)
        self.w = np.zeros(self.n)
        self.P = np.diag(np.ones(self.n) * self.lam)
        
    def update(self, x: np.ndarray, y: float) -> None:
        r = 1 + (x.T @ self.P @ x) / self.beta
        k = (self.P @ x) / (r * self.beta)
        e = y - x @ self.w

        k = k.reshape(-1, 1)
        self.P = self.P / self.beta - (k @ k.T) * r

        extra = (
            self.gamma * ((self.beta - 1) / self.beta)
            * (np.eye(self.n) - k @ self.w.reshape(1, -1))
            @ self.P @ (np.sign(self.w) / (np.abs(self.w) + self.epsilon))
        )

        self.w = self.w + k.flatten() * e + extra

        
    def predict(self, x: np.ndarray) -> float:
        return self.w @ x
```

{{% citation
    id="Liu-2010"
    author="Weifeng Liu, José C. Principe, and Simon Haykin"
    title="Kernel Adaptive Filtering: A Comprehensive Introduction"
    year="2010"
    publisher="Wiley"
    link="https://www.wiley.com/en-gb/Kernel+Adaptive+Filtering%3A+A+Comprehensive+Introduction-p-9780470447536"
%}}

{{% citation
    id="Eksioglu-2011"
    author="E.M. Eksioglu"
    title="Sparsity regularised recursive least squares adaptive filtering"
    publication="IET Signal Process"
    year="2011"
    pages="480-487"
    volume="5"
    number="5"
%}}
