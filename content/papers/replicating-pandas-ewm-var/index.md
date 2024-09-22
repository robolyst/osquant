---
title: "Replicating Pandas exponentially weighted variance"
summary: "
    Learn why calculating an exponentially weighted variance doesn't yield a correct estimation of variance.
"

date: "2024-09-17"
type: paper
mathjax: true
authors:
    - Adrian Letchford
categories:
    - mathematics
---

You are most likely familiar with the idea of calculating averages with an exponential weighting. The idea is that you have a higher weight to more recent information. The weights for an exponentially weighted average look like:
$$
w_t = (1 - \alpha)^t
$$
for $t \in [0, \dots, T]$. And the exponentially weighted average of a series $X_t$ looks like:
$$
\bar{X}_T = \frac{1}{\sum_t w_t} \sum_t w_t X_t
$$

You can easily calculate an exponentially weighted moving average in Pandas with:
```python
df.ewm(alpha=0.1).mean()
```

If you calculate the exponentially weighted average yourself you will find that it matches the result Pandas gives you. However, we're about to see that if we try doing this with the variance, we will get a poor estimate. This is because of something called [estimation bias](https://en.wikipedia.org/wiki/Bias_of_an_estimator).

# What is bias?

An estimator's bias is the difference between the estimator's expected value and the true value of the parameter being estimated--in this case the variance. A biased estimator is one where this difference is not zero and an unbiased estimator is one where this difference is zero.

Let's try measuring variance and see what happens.

The variance of a random variable $X$ is:
$$
\sigma^2 = E[(X - \mu)^2]
$$

If we had a sample of $n$ values of $X$ we could try to estimate the variance by replacing the expectancy $E[\cdot]$ with the average value across the samples:
$$
\frac{1}{n} \sum_i \left(X_i - \mu \right)^2
$$
and then replace $\mu$ with the sample's mean:
$$
\begin{aligned}
    \bar{X} &= \frac{1}{n}\sum_i X_i \\\
    \hat{\sigma}^2 &= \frac{1}{n} \sum_i \left(X_i - \bar{X} \right)^2 \\\
\end{aligned}
$$

We can write a quick simulation in Python to see how well our estimator ($\hat{\sigma}^2$) works:

```python
from scipy.stats import norm

num_simulations = 10_000_000
n = 5  # The number of samples

# This will give us an array where each row is a simulation
# and the columns are the sample values.
simulations = norm.rvs(loc=0, scale=1, size=(num_simulations, n))

# This gives us the estimated mean from each simulation.
avg = simulations.mean(axis=1).reshape(-1, 1)

# Use our estimator to estimate the variance from
# each of the simulations.
estimates = ((simulations - avg)**2).sum(1) / n
```

We now have 10 million estimates of variance where the true variance is $1$. And what is our average estimate? Calculating the mean:
```python
estimates.mean()
```
gives us about 0.8! Our estimator is off by 0.2. This is bias.

# Where does bias come from?

Let's go back to the definition of variance and how we turned it into a sample estimate. To form our estimator, we swapped $\mu$ for the sample mean:
$$
\frac{1}{n} \sum_i \left(X_i - \mu \right)^2 \quad\Rightarrow\quad \frac{1}{n} \sum_i \left(X_i - \bar{X} \right)^2
$$

This is where the bias was introduced. The mean ($\bar{X}$) of a small sample will be closer to those samples than the population's mean ($\mu$).

Figure 1. shows an example with 100 random dots where the center/mean is 0. Five of these dots were selected randomly and their mean is shown with a black cross. The mean of these 5 samples is the closest point to these five samples. By definition, the sample mean is closer than the population's mean. Therefore:
$$
\frac{1}{n} \sum_i \left(X_i - \bar{X} \right)^2 \quad\lt\quad \frac{1}{n} \sum_i \left(X_i - \mu \right)^2
$$
Our estimator will underestimate the variance of $X$!

{{<figure src="images/example.svg" title="Figure 1: Illustration of bias." width="medium">}}
This is a plot of 100 random dots where the average is (0, 0). Five dots have been selected randomly and highlighted. The mean of these 5 random dots is shown with the black cross.
{{</figure>}}

In fact, if you repeat the Python simulation above but replace the sample mean with 0 (the population mean):
```python
avg = 0
```
then the average of the sample variance will be 1:
```python
estimates.mean()
```
By knowing the population mean we can get an unbiased estimate of the variance from a set of samples. In practice, we do not know the population mean. Luckily, we can quantify bias and correct for it.


# Quantifying bias

Thus far, we have seen that $\hat{\sigma}^2$ is a biased estimate of the population variance. We discovered this by simulating many values for $\hat{\sigma}^2$ and taking the average. This simulation showed that:
$$
E[\hat{\sigma}^2] \ne \sigma^2
$$

We now want to move away from simulations and calculate the exact value of $E[\hat{\sigma}^2]$. We can do this by expanding it out. We start with:
$$
E[\hat{\sigma}^2] = E \left[ \frac{1}{n} \sum_i \left(X_i - \bar{X} \right)^2 \right]
$$
We can say that $\bar{X} = \mu - (\mu - \bar{X})$ which means we can expand out to:
$$
E[\hat{\sigma}^2] = E \left[ \frac{1}{n} \sum_i \left((X_i - \mu)- (\bar{X} - \mu) \right)^2 \right]
$$
with some algebra, we can expand out the power of two:
$$
\begin{aligned}
E[\hat{\sigma}^2] &= E \left[ \frac{1}{n} \sum_i \left((X_i - \mu)^2 - 2(\bar{X} - \mu)(X_i - \mu) + (\bar{X} - \mu)^2\right) \right] \\\
&= E \left[ \frac{1}{n} \sum_i (X_i - \mu)^2 - 2(\bar{X} - \mu) \frac{1}{n} \sum_i(X_i - \mu) +  \frac{1}{n} \sum_i(\bar{X} - \mu)^2 \right] \\\
&= E \left[ \frac{1}{n} \sum_i (X_i - \mu)^2 - 2(\bar{X} - \mu) \frac{1}{n} \sum_i(X_i - \mu) +  (\bar{X} - \mu)^2 \right] \\\
\end{aligned}
$$

Now, note that:
$$
\frac{1}{n} \sum_i(X_i - \mu) = \frac{1}{n} \sum_i X_i - \frac{1}{n} \sum_i \mu  = \frac{1}{n} \sum_i X_i - \mu = \bar{X} - \mu 
$$
which means that:
$$
\begin{aligned}
E[\hat{\sigma}^2] &= E \left[ \frac{1}{n} \sum_i (X_i - \mu)^2 - 2(\bar{X} - \mu)^2 + (\bar{X} - \mu)^2 \right] \\\
&= E \left[ \frac{1}{n} \sum_i (X_i - \mu)^2 - (\bar{X} - \mu)^2 \right] \\\
&= E \left[ \frac{1}{n} \sum_i (X_i - \mu)^2 \right] - E \left[ (\bar{X} - \mu)^2 \right] \\\
\end{aligned}
$$

The nice thing here is that:
$$
E \left[ \frac{1}{n} \sum_i (X_i - \mu)^2 \right] = \sigma^2
$$
which means:
$$
\begin{aligned}
E[\hat{\sigma}^2] &= \sigma^2 - E \left[ (\bar{X} - \mu)^2 \right] \\\
\end{aligned}
$$

The term $E \left[ (\bar{X} - \mu)^2 \right]$ is the variance of the sample mean. We know from [Bienaymé's identity
](https://en.wikipedia.org/wiki/Variance#Sum_of_uncorrelated_variables) that this is equal to:
$$
E \left[ (\bar{X} - \mu)^2 \right] = \frac{1}{n}\sigma^2
$$
which gives us:
$$
\begin{aligned}
E[\hat{\sigma}^2] &= \sigma^2 - \frac{1}{n}\sigma^2 = (1 - \frac{1}{n}) \sigma^2 \\\
\end{aligned}
$$

Think back to  our Python simulation; the number of samples was $n=5$, the true variance was $\sigma^2 = 1$ and the estimated variance came to $\hat{\sigma}^2 = 0.8$. If we plug $n$ and $\sigma^2$ into the above we get the biased answer:
$$
E[\hat{\sigma}^2] = (1 - \frac{1}{n}) \sigma^2 = (1 - \frac{1}{5}) \times 1 = 0.8
$$

# Unbiased estimator

Now that we know the exact value of $E[\hat{\sigma}^2]$ we can figure out how to correct $\hat{\sigma}^2$ so that it is an unbiased estimator of $\sigma^2$.

The correction term is:
$$
\frac{n}{n-1}
$$

We can see that this works by playing this through:
$$
\begin{aligned}
E[\frac{n}{n-1} \hat{\sigma}^2] &= \frac{n}{n-1} E[\hat{\sigma}^2] \\\
&= \frac{n}{n-1}(1 - \frac{1}{n}) \sigma^2 \\\
&= \frac{n(1 - \frac{1}{n})}{n-1} \sigma^2 \\\
&= \frac{n - 1}{n-1} \sigma^2 \\\
&= \sigma^2 \\\
\end{aligned}
$$
Therefore, an unbiased estimator of $\sigma^2$ from a set of samples is:
$$
\begin{aligned}
\frac{n}{n-1} \hat{\sigma}^2 &= \frac{n}{n-1} \frac{1}{n} \sum_i \left(X_i - \bar{X} \right)^2 \\\
&= \frac{1}{n-1} \sum_i \left(X_i - \bar{X} \right)^2 \\\
\end{aligned}
$$

# Unbiased weighted estimator

Now to expand the above to cover the case when the samples are weighted.

The weighted sample mean is:
$$
\bar{X} = \frac{1}{\sum_i w_i} \sum_i w_i X_i
$$
and the weighted variance:
$$
\hat{\sigma}^2 = \frac{1}{\sum_i w_i} \sum_i w_i\left(X_i - \bar{X} \right)^2 
$$

Following the exact same expansion procedure as before, we end up with:
$$
\begin{aligned}
E[\hat{\sigma}^2] &= \sigma^2 - E \left[ (\bar{X} - \mu)^2 \right] \\\
\end{aligned}
$$

The variance of the mean turns out to be:
$$
\begin{aligned}
E \left[ (\bar{X} - \mu)^2 \right] &= \text{var}(\bar{X}) \\\
&= \text{var}\left(\frac{1}{\sum w_i} \sum w_i X_i \right) \\\
&= \frac{1}{(\sum w_i)^2} \sum \text{var} (w_i X_i) \\\
&= \frac{1}{(\sum w_i)^2} \sum w_i^2 \text{var} (X_i) \\\
&= \frac{\sum w_i^2}{(\sum w_i)^2} \sigma^2 \\\
\end{aligned}
$$

This gives us:
$$
\begin{aligned}
E[\hat{\sigma}^2] &= \sigma^2 - \frac{\sum w_i^2}{(\sum w_i)^2} \sigma^2 \\\
&= \left(1 - \frac{\sum w_i^2}{(\sum w_i)^2} \right)\sigma^2 \\\
\end{aligned}
$$

The bias correction term is then:
$$
b = \frac{(\sum w_i)^2}{(\sum w_i)^2 - \sum w_i^2}
$$

which means the unbiased weighted estimate of variance is:
$$
b \hat{\sigma}^2
$$


# Replicating Pandas exponentially weighted variance

We now have all the tools we need to replicate the exponentially weighted variance from Pandas.

```python
import numpy as np
import pandas as pd

N = 1000

# Create some fake data
df = pd.DataFrame()
df['data'] = np.random.randn(N)

# Set a halflife for the EWM and convert
# to alpha for the calculations.
halflife = 10
a = 1 - np.exp(-np.log(2)/halflife)  # alpha

# This is the ewm from Pandas
df['var_pandas'] = df.ewm(alpha=a).var()

# Initialize variable
varcalc = np.zeros(len(df))

# Calculate exponential moving variance
for i in range(0, N):

    x = df['data'].iloc[0:i+1].values

    # Weights
    n = len(x)
    w = (1-a)**np.arange(n-1, -1, -1) # This is reverse order to match Series order

    # Calculate exponential moving average
    ewma = np.sum(w * x) / np.sum(w)

    # Calculate bias
    bias = np.sum(w)**2 / (np.sum(w)**2 - np.sum(w**2))

    # Calculate exponential moving variance with bias
    varcalc[i] = bias * np.sum(w * (x - ewma)**2) / np.sum(w)

df['var_calc'] = varcalc
```

which gives us a DataFrame that looks like:

{{<figure src="images/dataframe.png" width="small">}}{{</figure>}}
