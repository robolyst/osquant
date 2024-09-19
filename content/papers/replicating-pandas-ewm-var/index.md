---
title: "Replicating Pandas EWM variance and covariance"
summary: "
    Lorem ipsum dolor sit amet, consectetur adipiscing elit. Integer odio neque, volutpat vel nunc
    ut. Duis maximus massa vitae libero imperdiet feugiat quis a sapien. Quisque sodales neque dui,
    a mollis justo porta eu. Nullam semper ipsum ac ante rhoncus, ac facilisis lacus posuere. Mauris
    pulvinar elementum ligula in mattis. Fusce rhoncus consequat lorem accumsan rhoncus.
"

date: "2024-09-17"
type: paper
mathjax: true
authors:
    - Adrian Letchford
categories:
    - mathematics
acknowledgements: "All figures in this article were made with [Figma](http://figma.com)."
---

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nulla at varius turpis. Ut nec purus efficitur, dictum urna eget, molestie ante. Vestibulum sed scelerisque mi. Cras sed lorem ac ante rhoncus egestas quis sed augue. Fusce eget venenatis felis, lobortis blandit felis. Nunc feugiat eu neque ac fringilla. Curabitur in felis facilisis, dignissim enim eget, posuere massa. Aliquam gravida ut mi in aliquet. Fusce convallis at tortor sodales lobortis. Fusce sem enim, cursus quis purus rutrum, tincidunt accumsan est.

# Exponentially weighted moving average

This is easy, but we'll use this to create the structure of how we'll explore the harder stuff.

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nulla at varius turpis. Ut nec purus efficitur, dictum urna eget, molestie ante. Vestibulum sed scelerisque mi. Cras sed lorem ac ante rhoncus egestas quis sed augue. Fusce eget venenatis felis, lobortis blandit felis. Nunc feugiat eu neque ac fringilla. Curabitur in felis facilisis, dignissim enim eget, posuere massa. Aliquam gravida ut mi in aliquet. Fusce convallis at tortor sodales lobortis. Fusce sem enim, cursus quis purus rutrum, tincidunt accumsan est. Sed egestas eleifend ligula, et scelerisque enim maximus sit amet. Fusce a purus a est tristique hendrerit. Donec scelerisque in dolor quis gravida. Pellentesque pretium nisi purus, sed feugiat ligula euismod a.

Mauris in iaculis risus. Nam auctor blandit velit. Nulla condimentum diam diam, vitae tincidunt velit bibendum id. Nullam aliquet eros ac tortor vestibulum, in blandit velit molestie. Maecenas varius luctus bibendum. Nulla facilisi. Pellentesque eleifend nibh laoreet nibh auctor, sed aliquam turpis sagittis. Donec ultrices, dui ut gravida bibendum, lacus odio hendrerit lacus, sit amet consequat arcu mi nec nisl. Sed laoreet nibh nisl, accumsan dignissim nisl consectetur porta.

Donec a lectus non ipsum aliquet malesuada sit amet in nulla. Nulla vitae dictum erat. Nullam ut turpis et enim euismod porta. Donec ut ornare nisl. Integer efficitur neque vitae enim rutrum auctor. Morbi sed metus orci. Maecenas consectetur at velit ac consectetur. Ut tincidunt augue eget eros efficitur, vel lacinia orci porta. Morbi volutpat vulputate ullamcorper. Ut ornare at ante imperdiet fermentum.

# Exponentially weighted moving variance

For the exponentialy weighted moving average, we took the formula for the expected value, plugged in sample values and we got an estimate of the mean. We're about to see that if we try this with the variance, we will get a very poor estimate. This is because of something called [estimation bias](https://en.wikipedia.org/wiki/Bias_of_an_estimator).

## What is bias?

An estimator's bias is the difference between the estimator's expected value and the true value of the parameter being estimated--in this case the variance. A biased estimator is one where this difference is not zero and an unbiased estimator is one where this difference is zero.

Let's try measuring variance and see what happens.

The variance of a random variable $X$ is:
$$
\sigma^2 = E[(X - \mu)^2]
$$

If we had a sample of $n$ values of $X$ we could try to estimate the variace by replacing $\mu$ with the sample's mean and replacing the expectancy $E$ with the average value across the samples:
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

So now we have 10 million estimates of variance where the true variance is $1$. And what is our average estimate? Calculating the mean:
```python
estimates.mean()
```
gives us about 0.8! Our estimator is off by 0.2. This is bias.

## Quantifying bias

Work through the maths to show what the bias is and how to correct it

## Replicating Pandas exponentially weighted variance

Aliquam erat volutpat. Donec vel massa vitae nulla varius vestibulum quis scelerisque purus. Etiam magna lorem, faucibus non urna vel, dignissim pulvinar justo. Phasellus sollicitudin ex in tortor blandit tristique. Cras ornare dui eget lorem viverra ultricies. Aenean feugiat fringilla sapien, vitae tincidunt orci. Cras non est id nisl porttitor ultrices. Donec accumsan odio venenatis vulputate fermentum. Etiam maximus eu augue quis auctor.

Nulla varius erat a orci mollis, ut euismod magna efficitur. Sed a condimentum neque. Suspendisse vulputate, ante ut suscipit rutrum, libero velit eleifend est, non vestibulum odio nisi et libero. Nullam dignissim ultricies elit, et aliquet felis dictum sit amet. Donec mollis nisi sed leo tempus placerat. Nulla efficitur nisl erat, sagittis scelerisque orci congue sit amet. Integer tristique erat sit amet urna faucibus, vel efficitur elit ultricies. Nunc id vehicula felis. In lobortis vitae mauris nec pulvinar.

* Someone made an example here: https://stackoverflow.com/questions/40754262/pandas-ewm-std-calculation
* https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Reliability_weights

# Exponentially weighted moving standard deviation

# Exponentially weighted moving covariance

# Summary