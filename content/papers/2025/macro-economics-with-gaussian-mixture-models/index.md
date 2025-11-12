---
title: "Macro Economics with Gaussian Mixture Models"
summary: "
    Lorem ipsum dolor sit amet, consectetur adipiscing elit. Integer odio neque, volutpat vel nunc
    ut. Duis maximus massa vitae libero imperdiet feugiat quis a sapien. Quisque sodales neque dui,
    a mollis justo porta eu. Nullam semper ipsum ac ante rhoncus, ac facilisis lacus posuere. Mauris
    pulvinar elementum ligula in mattis. Fusce rhoncus consequat lorem accumsan rhoncus.
"

date: "2025-11-10"
type: paper
mathjax: true
authors:
    - Adrian Letchford
categories:
    - mathematics
    - finance
---

# Intro
1. GMMs are a type of hidden state model
2. Normally, we estimate expected return, expected variance, expected correlation. In a state space model, we estimate these for some number of states and estimate the expected state, or the probability of each state.

# The basics

## Latent states

We are going to model a market with $d$ assets. The vector of returns at time $t$ is $\boldsymbol{r}_t \in \mathbb{R}^d$. In a hidden state model, we say that the market is in one of $K$ states and that each state is associated with a distribution that generates $\boldsymbol{r}_t$.

There are two notations for the state. We can say that the state is $k$, i.e. we can ask "what is the distribution of $\boldsymbol{r}_t$ given we are in state $k$?". We can also represent the state with a vector $\boldsymbol{z}$ which has $z\_{k} = 1$ if the state is $k$ and $0$ for all other states.

Based on this definition of states, we can write out a few distributions.

**Prior distribution.** The probability of being in state $k$ is:
$$
p(k)
$$

**Emission distribution.** The probability of observing $\boldsymbol{r}_t$ given we are in state $k$ is:
$$
p(\boldsymbol{r}_t | k)
$$
this is often called the *emission* distribution. Later on, we will give this a shape (Gaussian) and parameters.

**Joint distribution.** The probability that we are in state $k$ *and* we observe $\boldsymbol{r}_t$ is (using both notations):
$$
\begin{aligned}
p(k, \boldsymbol{r}_t) &= p(k) p(\boldsymbol{r}_t | k) \\\
\\\
p(\boldsymbol{z}, \boldsymbol{r}_t) &= \prod_k \left[ p(k) p(\boldsymbol{r}_t | k) \right]^{z_k} \\\
\end{aligned}
$$
The information $\boldsymbol{z}$ and $\boldsymbol{r}_t$ are often called the *full set* because at time $t$ the full set of information is the state and the realised returns.

**Mixture distribution.** The probability that we observe $\boldsymbol{r}_t$ is:
$$
p(\boldsymbol{r}_t) = \sum_k^K p(k)p(\boldsymbol{r}_t | k)
$$
Here we've combined or "mixed" the emission distributions into a "mixture" distribution.

**Posterior distribution.** The probability that we are in state $k$ given we observe $\boldsymbol{r}_t$ is derived from the above distributions using Bayes' theorem:
$$
p(k|\boldsymbol{r}_t) = \frac{p(k)p(\boldsymbol{r}_t|k)}{p(\boldsymbol{r}_t)} = \frac{p(k)p(\boldsymbol{r}_t|k)}{\sum_k^K p(k)p(\boldsymbol{r}_t | k)}
$$
This is often viewed as the *responsibility* that state $k$ has for explaining or generating the observation $\boldsymbol{r}_t$ [^Bishop2006].

To do useful things with this latent state model, we need to assume specific distributions for $p(k)$ and $p(\boldsymbol{r}_t | k)$.

## Mixture of Gaussians

We will fix the priors to a single value:
$$
p(k) = \pi_k, \quad \pi_k \ge 0, \quad \sum \pi_k = 1
$$
The emission distribution is set as a Gaussian:
$$
p(\boldsymbol{r}_t | k) = \mathcal{N}(\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)
$$
which means the mixture distribtion is:
$$
p(\boldsymbol{r}_t) = \sum_k^K \pi_k \mathcal{N}(\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)
$$
which is called *a mixture of Gaussians* [^Bishop2006] or a Gaussian mixture model (GMM).

The posterior distribution becomes:
$$
p(k|\boldsymbol{r}_t) = \frac{\pi_k\mathcal{N}(\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)}{\sum_k^K \pi_k\mathcal{N}(\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)}
$$

<todo>Some good visuals of a GMM would work well here.</todo>

## Estimation

Define $\boldsymbol{R} = [\boldsymbol{r}_1, \dots, \boldsymbol{r}_t]$ to be all the observations and $\boldsymbol{Z} = [\boldsymbol{z}_1, \dots, \boldsymbol{z}_t]$ to be the corresponding hidden states. We will fix the number of states to be $K$. We want to estimate the values $\pi_k$, $\boldsymbol{\mu}_k$ and $\boldsymbol{\Sigma}_k$ for each of the states.

The usual method of estimating a model is to maximise the log of the likelihood function of the given set of samples. In this case, the likelihood is:
$$
\prod_t p(\boldsymbol{r}_t) = \prod_t \left[ \sum_k^K p(k)p(\boldsymbol{r}_t | k) \right]
$$
Taking the log gives us this:
$$
\mathcal{L}( \boldsymbol{R} | \boldsymbol{\pi}, \boldsymbol{\mu}, \boldsymbol{\Sigma}) = \sum_t \log \left( \sum_k^K p(k)p(\boldsymbol{r}_t | k) \right)
$$
which is a very complex function to maximise due to the summation within the logarithm. In fact, there isn't a closed form solution to this. We can derive an iterative solution which is called the expectation maximisation (EM) algorithm in general. The EM algorithm is an iterative method of finding a solution to a hidden state model. There is a fair amount of theory which you can read about [^Bishop2006]. Here, we will follow a more information derivation of the solution [^Bishop2006].

Filling in the log-likelihood with our Gaussian mixture distribtuions we get:
$$
\mathcal{L}( \boldsymbol{R} | \boldsymbol{\pi}, \boldsymbol{\mu}, \boldsymbol{\Sigma}) = \sum_t \log \left( \sum_k^K \pi_k p(\boldsymbol{r}_t | k) \right)
$$

<todo>Figure out how much detail we need here.</todo>

## Portfolio construction

For each of the $k$ states, if we have portfolio weights $\boldsymbol{w}$ we can calculate the expected portfolio return as:
$$
E_k[r_t] = \mu_k = \boldsymbol{w}^\top \boldsymbol{\mu}_k
$$
and the expected variance as:
$$
\sigma^2_k = \boldsymbol{w}^\top \boldsymbol{\Sigma}_k \boldsymbol{w}
$$
We can safely assume that the portfolio return between each state is uncorrelated.


Given a set of PDFs $f_i$ and mixing weights $p_i$ the PDF of the mixture is:
$$
f(x) = \sum_i p_i f_i(x)
$$
It follows that any moment of the mixture distribution is the weighted sum of the moments of the individual distributions:
$$
E[X^n] = \sum_i p_i E_i[X^n]
$$

Thus, the expected portfolio return across all states is:
$$
E[r_t] = \mu = \sum_k^K \pi_k E_k[r_t] = \sum_k^K \pi_k \boldsymbol{w}^\top\boldsymbol{\mu}_k
$$
We can collect that last sum on the right hand side into a single mean vector:
$$
\boldsymbol{\mu} = \sum_k^K \pi_k \boldsymbol{\mu}_k
$$
such that the expected portfolio return is:
$$
E[r_t] = \mu = \boldsymbol{w}^\top \boldsymbol{\mu}
$$

The expected portfolio variance across all states is:
$$\begin{aligned}
E[(r_t - E[r_t])^2] &= E[r_t^2] - (E[r_t])^2 \\\
&= \sum_k^K \pi_k E_k[r_t^2] - (E[r_t])^2 \\\
&= \sum_k^K \pi_k \left( \sigma^2_k + (E_k[r_t])^2 \right) - (E[r_t])^2 \\\
&= \sum_k^K \pi_k \sigma^2_k + \sum_k^K \pi_k (E_k[r_t])^2 - (E[r_t])^2 \\\
\end{aligned}
$$
Expanding out the variance and moments:
$$
= \sum_k^K \pi_k \boldsymbol{w}^\top \boldsymbol{\Sigma}_k \boldsymbol{w} + \sum_k^K \pi_k (\boldsymbol{w}^\top \boldsymbol{\mu}_k)^2 - ( \boldsymbol{w}^\top \boldsymbol{\mu} )^2
$$
which is just a sum of quadratic terms. We can collect the terms into a single covariance matrix:
$$
\boldsymbol{\Sigma} = \sum_k^K \pi_k \boldsymbol{\Sigma}_k + \sum_k^K \pi_k \boldsymbol{\mu}_k \boldsymbol{\mu}_k^\top - \boldsymbol{\mu}\boldsymbol{\mu}^\top
$$
such that the expected portfolio variance is:
$$
\sigma^2 = \boldsymbol{w}^\top \boldsymbol{\Sigma} \boldsymbol{w}
$$

In practice, you'll find that the GMM mean and cov matches almost exactly the empirical mean and cov when fitted to financial returns. The only difference is that in the GMM model, we can break the mean and cov down into different states.



## Example

We can grab some returns with:
```python
import yfinance as yf

tickers = yf.Tickers("SPY TLT GLD GSG VNQ IWM")
prices = tickers.download(period="30y", interval="1d")
returns = prices["Close"].pct_change().dropna()
```

![](asset_returns.svg)

We can fit a Gaussian Mixture Model with the help of scikit-learn's `GaussianMixture` class ([docs](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html)).

The only decision we need to make is how many states to use. It's fairly common to go with a small number of states (2-4). We're going to use 3 states as it gives us surprising results that are not immediately obvious. You might think that the model will converge to states that align with a "bull", "bear" and "neutral" market. However, this is not the case, and the actual states are far more interesting.

We can fit the model with:

```python
from sklearn.mixture import GaussianMixture
import pandas as pd

model = GaussianMixture(n_components=3, random_state=42)
model.fit(returns)
```
For now, we are going to do this in-sample and investigate the results.

The fitted mixing coefficients ($\pi_k$) can be found with `model.weights_` and look like this:

|   State 0 |   State 1 |   State 2 |
|-----------|-----------|-----------|
|    71.10% |    25.35% |     3.54% |

In this case, the model is saying that State 0 is the most common state, followed by State 1 and then State 2. Curiously, State 2 is very rare. This state is only active for 3.54% of the time.

We can look at the mean vectors ($\boldsymbol{\mu}_k$) with `model.means_`:

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
    <tr>
      <th>Ticker</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>GLD</th>
      <td>0.0737%</td>
      <td>-0.0424%</td>
      <td>0.0498%</td>
    </tr>
    <tr>
      <th>GSG</th>
      <td>0.0862%</td>
      <td>-0.2092%</td>
      <td>-0.3531%</td>
    </tr>
    <tr>
      <th>IWM</th>
      <td>0.1338%</td>
      <td>-0.2322%</td>
      <td>0.2208%</td>
    </tr>
    <tr>
      <th>SPY</th>
      <td>0.1387%</td>
      <td>-0.2030%</td>
      <td>0.0817%</td>
    </tr>
    <tr>
      <th>TLT</th>
      <td>0.0072%</td>
      <td>0.0408%</td>
      <td>0.0727%</td>
    </tr>
    <tr>
      <th>VNQ</th>
      <td>0.1135%</td>
      <td>-0.2258%</td>
      <td>0.4811%</td>
    </tr>
  </tbody>
</table>

In the table above, we can see that State 0 has positive expected returns for all assets. This looks like a "bull" market. State 1 has negative expected returns for all assets except TLT. This looks like a "bear" market with maybe TLT providing a hedge. State 2 has some large positive expected returns (VNQ, IWM) and some large negative expected returns (GSG).

We can also look at the covariance matrices ($\boldsymbol{\Sigma}_k$) with `model.covariances_`. These are a bit more difficult to visualise, but we can look at the standard deviations (the square root of the diagonal elements):

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>State 0</th>
      <th>State 1</th>
      <th>State 2</th>
    </tr>
    <tr>
      <th>Ticker</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>GLD</th>
      <td>12.17%</td>
      <td>23.44%</td>
      <td>42.55%</td>
    </tr>
    <tr>
      <th>GSG</th>
      <td>16.08%</td>
      <td>30.65%</td>
      <td>57.93%</td>
    </tr>
    <tr>
      <th>IWM</th>
      <td>15.19%</td>
      <td>31.19%</td>
      <td>73.84%</td>
    </tr>
    <tr>
      <th>SPY</th>
      <td>10.36%</td>
      <td>24.63%</td>
      <td>65.36%</td>
    </tr>
    <tr>
      <th>TLT</th>
      <td>11.64%</td>
      <td>19.28%</td>
      <td>33.33%</td>
    </tr>
    <tr>
      <th>VNQ</th>
      <td>13.70%</td>
      <td>34.66%</td>
      <td>110.96%</td>
    </tr>
  </tbody>
</table>

The main thing to notice here is that State 2 has much higher volatilities than the other two states. This suggests that State 2 is a high-volatility state.

The story so far is that we have a "bull" market (State 0), a "bear" market (State 1) and a "tail" market (State 2).

Things get really interesting when we look at the posterior state probabilities for each time period. We can get these into a DataFrame with:

```python
import pandas as pd

posterior = pd.DataFrame(
    data=model.predict_proba(returns),
    index=returns.index,
)
```

Plotting these straight away gives us:

![](posteriors.svg)

This is a little bit hard to see what is going on. Looking at state 2, we can see it clustering around certain periods of time. We can highlight this clustering by smoothing the posterior probabilities with an exponential moving average:

```python
posterior.ewm(halflife=5).mean()
```

Plotting the smoothed posterior probabilities gives us:

![](smoothed_posteriors.svg)

We see two big clusters of State 2 activity and lots of little clusters. The first big one is around the 2008 financial crisis and the second is around the 2020 COVID crash. This makes sense as these were both periods of high volatility---both tail events.

We can also get a good view of how returns behave in each state by multiplying the returns by the posterior probablities. To get a frame of each state's returns for a single asset, we can do:

```python
posterior.multiply(returns['SPY'], axis=0)
```

Looking at state 0:

![](state_0_returns.svg)

The statistics we looked at for state 0 suggested a "bull" market and this plot confirms that. All of the returns are positive. The only one that stands out is TLT (bonds).

State 1 looks like:

![](state_1_returns.svg)

This is also inline with our earlier intuition that state 1 is a "bear" market. Most of the returns are negative, with the exception of TLT which is mostly positive. Similarly to state 0, the returns for TLT do not quite match the rest of this state. In the case of state 1, TLT acts as a hedge with mostly positive returns while the remaining assets negative returns.

Finally, state 2:

![](state_2_returns.svg)

This is the most interesting state. It's a little harder to interpret as there are fewer examples of this state. As such, the return plots are mostly flat (as there are many time periods not in this state). However, we can see that the returns are much more extreme in this state. Similarly to before, this state seems to represent a "tail" market with high volatility and extreme returns.

The conclusion from this example is that a Gaussian Mixture Model can identify economically meaningful latent market states. The model was able to not just identify bull and bear periods, but also periods of tail risk.

However, once the model is fitted, we are limited to the sample wide set of mixing coefficients $\pi_k$. This means we are not able to predict future returns any better than using the empirical mean and covariance. We need to extend this model to be able to predict future states.

# Making predictions

Once we have fitted a GMM to historical returns, we have the following fitted parameters:
1. Mixing coefficients $p(k) = \pi_k$
2. Mean vectors $\boldsymbol{\mu}_k$
3. Covariance matrices $\boldsymbol{\Sigma}_k$

The next thing we want to do is make a prediction about the next time period's return. Specifically, we want to estimate the expected mean and covariance for the next time period. If we were to combine the mixing coefficients, mean vectors and covariance matrices as before, we would get the same empirical mean and covariance as the historical data. Instead, we want to combine some extra information to get time-varying parameters.

Now, let's say that we have some extra information that we know at the decision point before $\boldsymbol{r}_t$ is realised. Let's denote it with the vector $\boldsymbol{x}_t$. Also, we assume that $\boldsymbol{x}_t$ is a multivariate Gaussian. We can combine $\boldsymbol{r}_t$ and $\boldsymbol{x}_t$ together into a new vector
$$
\boldsymbol{y}_t = \left[\begin{matrix}
    \boldsymbol{r}_t \\\
    \boldsymbol{x}_t
    \end{matrix}\right]
$$
and fit a new GMM to the combined data $\boldsymbol{y}_t$ giving us:
$$
p(\boldsymbol{y}_t) = \sum_k^K p(k)\phi(\boldsymbol{y}_t, \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)
$$
Making a forecast means calculating the distribution:
$$
p(\boldsymbol{r}_t | \boldsymbol{x}_t) = \sum_k^K p(k | \boldsymbol{x}_t)\phi(\boldsymbol{r}_t, \boldsymbol{\mu}_k | \boldsymbol{x}_t, \boldsymbol{\Sigma}_k | \boldsymbol{x}_t)
$$

# Exponentially weighted estimates
All information is important, recent information is more important. We want to discount prices far into the past more than recent prices.

# Reflecting current market volatility
We're fitting over long periods of time. This means that the current volatility is not reflected in our forecasts.

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

For the curious, you can find a derivation of the Gaussian conditional distribution in section 2.3 of [^Bishop2006]. I found a [PDF online](https://www.seas.upenn.edu/~cis520/papers/Bishop_2.3.pdf) of this section.


{{% citation
    id="Bishop2006"
    author="Christopher M. Bishop"
    title="Pattern Recognition and Machine Learning"
    year="2006"
    publication="Springer"
    link="https://www.microsoft.com/en-us/research/wp-content/uploads/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf"
%}}
