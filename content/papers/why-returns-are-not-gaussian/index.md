---
title: "Why Returns Are Not Gaussian"
summary: "
I want to share with you my intuition as to why asset returns are not Gaussian. This will not be a proof, but rather some points discussing why this phenomenon makes sense.
"

date: "2023-03-12"
type: paper
mathjax: true # Enable mathematics on the page
plotly: true  # Enable plotly on the page
authors:
    - Adrian Letchford
categories:
    - finance
---

For decades, researchers have noted properties of returns that are consistent across different assets. These properties have become known as *stylised facts.* They are measurable, but, researchers have not been able to prove that they must be true.

The common stylised facts we are interested in here are [^Cont2001]:

1. Returns are not Gaussian.
2. Returns have fat tails.
3. Returns become more Gaussian on higher time frames and less Gaussian on lower time frames.

I want to show you why these three properties actually make sense, even if we cannot prove that they are true. I’ll explain the first property based on the definition of a Gaussian distribution. The second and third can be shown if some intuitive assumptions are made.

I’ll start by breaking down the Gaussian distribution.

# What is a Gaussian distribution?

The probability density function of a Gaussian distribution is depicted in the figure below and described by the following equation:

$$
f(x) = \frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{1}{2}(\frac{x-\mu}{\sigma})^2}
$$

{{<plotly id="gaussian" width="500" height="200">}}
    var x = [-4.0, -3.919, -3.838, -3.758, -3.677, -3.596, -3.515, -3.434, -3.353, -3.273, -3.192, -3.111, -3.03, -2.95, -2.869, -2.788, -2.707, -2.626, -2.546, -2.465, -2.384, -2.303, -2.222, -2.141, -2.061, -1.98, -1.899, -1.818, -1.737, -1.657, -1.576, -1.495, -1.414, -1.333, -1.252, -1.172, -1.091, -1.01, -0.9293, -0.8485, -0.7677, -0.6869, -0.6061, -0.5253, -0.4444, -0.3636, -0.2828, -0.202, -0.1212, -0.0404, 0.0404, 0.1212, 0.202, 0.2828, 0.3636, 0.4444, 0.5253, 0.6061, 0.6869, 0.7677, 0.8485, 0.9293, 1.01, 1.091, 1.172, 1.252, 1.333, 1.414, 1.495, 1.576, 1.657, 1.737, 1.818, 1.899, 1.98, 2.061, 2.141, 2.222, 2.303, 2.384, 2.465, 2.546, 2.626, 2.707, 2.788, 2.869, 2.95, 3.03, 3.111, 3.192, 3.273, 3.353, 3.434, 3.515, 3.596, 3.677, 3.758, 3.838, 3.919, 4.0];

    var y = [0.0001, 0.0002, 0.0003, 0.0003, 0.0005, 0.0006, 0.0008, 0.0011, 0.0014, 0.0019, 0.0024, 0.0032, 0.004, 0.0052, 0.0065, 0.0082, 0.0102, 0.0127, 0.0156, 0.0191, 0.0233, 0.0281, 0.0338, 0.0403, 0.0477, 0.0562, 0.0657, 0.0764, 0.0882, 0.1012, 0.1153, 0.1305, 0.1468, 0.164, 0.1821, 0.2008, 0.22, 0.2395, 0.2591, 0.2783, 0.2971, 0.3151, 0.332, 0.3475, 0.3614, 0.3734, 0.3833, 0.3909, 0.396, 0.3986, 0.3986, 0.396, 0.3909, 0.3833, 0.3734, 0.3614, 0.3475, 0.332, 0.3151, 0.2971, 0.2783, 0.2591, 0.2395, 0.22, 0.2008, 0.1821, 0.164, 0.1468, 0.1305, 0.1153, 0.1012, 0.0882, 0.0764, 0.0657, 0.0562, 0.0477, 0.0403, 0.0338, 0.0281, 0.0233, 0.0191, 0.0156, 0.0127, 0.0102, 0.0082, 0.0065, 0.0052, 0.004, 0.0032, 0.0024, 0.0019, 0.0014, 0.0011, 0.0008, 0.0006, 0.0005, 0.0003, 0.0003, 0.0002, 0.0001];

    var traces = [
        {
            x: x,
            y: y,
            mode: 'lines',
            fill: 'tozeroy',
            fillcolor: 'rgba(255, 0, 0, 0.05)',
            line: {
                color: 'red',
                width: 3,
            }
        },
        {
            x: [0, 0],
            y: [0, 0.4],
            mode: 'line',
            line: {
                dash: 'dot',
                width: 3,
                color: '#777'
            },
            marker: {
                opacity: 0
            }
        },
        {
            x: [-1.4, 1.4],
            y: [0.15, 0.15],
            mode: 'line',
            line: {
                dash: 'dot',
                width: 3,
                color: '#777'
            },
            marker: {
                opacity: 0
            }
        }
    ];

    var layout = {
        showlegend: false,
        yaxis: {
            visible: false,
        },
        xaxis: {
            visible: true,
            showgride: false,
            tickmode: 'array',
            tickvals: [0],
            ticktext: ['$\\mu$'],
            zeroline: false,
            gridcolor: 'transparent',
        },
        margin: {
            l: 0,
            r: 0,
            b: 20,
            t: 0,
            pad: 0,
        },
        annotations: [
            {
            x: 1.8,
            y: 0.15,
            xref: 'x',
            yref: 'y',
            text: '$\\sigma$',
            showarrow: false,
            arrowhead: 0,
            ax: 10,
            ay: -10
            }
        ]
    };

    var config = {
        staticPlot: true
    };
{{</plotly>}}

The Gaussian distribution turns up everywhere, and with good reason. It describes the average of \\(n\\) independently and identically distributed random samples from any distribution as \\(n\\) tends to infinity:

$$
\frac{X_1 + \dots + X_n}{n} \sim \mathcal{N}(\mu, \sigma^2)
$$

We know this is true from the [central limit theorem](https://en.wikipedia.org/wiki/Central_limit_theorem).

We can actually make the Gaussian distribution a bit more general. A Gaussian variable multiplied by some constant is still Gaussian. This means we can multiply an average value by the number of samples to turn it into a sum:

$$
X_1 + \dots + X_n \sim n\mathcal{N}(\mu, \sigma^2) = \mathcal{N}(n\mu, n\sigma^2)
$$

A Gaussian distribution models the sum of \\(n\\) independently and identically distributed random values as \\(n\\) tends to infinity.

We can see this in action by taking \\(n\\) samples from a uniform distribution, summing, and plotting the distribution of those sums:

{{<figure src="images/gaussian_sample_sizes.svg" title="Different sample sizes for a Gaussian distribution." >}}
The distribtion of the sum of \\(n\\) samples from a uniform distribution. The red line is the best fit Gaussian distribution. As \\(n\\) increases, the distribution becomes more Gaussian.
{{</figure>}}

# Returns are not Gaussian

This is the first stylized fact is that returns are not Gaussian. It turns out, returns do not meet the assumptions of a Gaussian distribution.

For this discussion, we are going to think in terms of logged prices. That is, when I say "return" I mean change in logged price:

$$
\text{return at time } t = \frac{p_t - p_{t-1}}{p_{t-1}} \approx \log(p_t) - \log(p_{t-1})
$$

This means we can think of returns as summing together rather than multiplying together.

A daily return is the sum of 8 hourly returns (8 trading hours in a day). Since we know that a sum of \\(n = 8\\) values is approximately Gaussian, it makes sense that the daily returns ought to be Gaussian. To see why this isn't the case, we need to break down a day's return to its atomic level. Each hourly return is the sum of minutely returns which are the sum of per second returns. If we continue this logic, we get to the atomic level: ticks.

Each day's return is a sum of tick returns. The key thing to note is that each day has a different number of ticks. Bringing this back to the Gaussian distribution, each day is the sum of a different number of ticks. The \\(n\\) is different on each day. The Gaussian distribution assumes that \\(n\\) is the same for each sample. Therefore, returns on any time scale do not meet the assumptions of a Gaussian distribution.

# Returns have fat tails

 We need to make two assumptions to show that fat tails make sense. First that ticks, $T$, are drawn from a Gaussian distribution. Second that returns, $R$, are a sum of $N$ ticks where $N$ is drawn from a Poisson distribution. This is written as:
$$
\begin{aligned}
T &\sim \mathcal{N}(\mu, \sigma^2) \\\
N &\sim \text{Poisson}(\lambda) \\\
R &\sim \mathcal{N}(N\mu, N\sigma^2) \\\
\end{aligned}
$$

This is a type of [compound Poisson distribution](https://en.wikipedia.org/wiki/Compound_Poisson_distribution).

By making ticks Gaussian, we guarantee that their sum is also Gaussian. Which means that any deviation from a Gaussian distribution will have to come from summing together $N \sim \text{Poisson}$ values.

A distribution's tails are measured with the fourth standard [moment](https://en.wikipedia.org/wiki/Moment_(mathematics)) commonly known as [*kurtosis*](https://en.wikipedia.org/wiki/Kurtosis). A Gaussian distribution has a kurtosis of 3. If a distribution has a kurtosis larger than 3, then it has fatter tails than a Gaussian distribution. Therefore, we need to show that $\text{kurtosis}(R) > 3$.

Using the [method of moments](https://en.wikipedia.org/wiki/Method_of_moments_(statistics)) in [the appendix](#deriving-kurtosis) we show that:
$$
\text{kurtosis}(R) = 3 + \frac{E[T^4]}{\lambda E[T^2]^2}
$$

Both $E[T^4]$ and $E[T^2]$ are positive values. Therefore, $\text{kurtosis}(R) > 3$ which means that a Poisson sum of ticks is not Gaussian and has fatter tails.

# More Gaussian at higher time frames

The last stylised fact I want to touch on is the observation that returns become more Gaussian on higher time frames. That is, weekly returns look more Gaussian than hourly returns.

Higher time frames mean that returns are sums of more ticks, or, larger $N$ in the compound Poisson model we're using. Getting larger values for $N$ means the expected value $E[N] = \lambda$ is larger. This is equivalent to larger values for $\lambda$.

As $\lambda$ gets bigger and bigger, $\text{kurtosis}(R)$ asymptotes towards 3:
$$
\lim_{\lambda \to \infty} \text{kurtosis}(R) = \lim_{\lambda \to \infty} \left[ 3 + \frac{E[T^4]}{\lambda E[T^2]^2} \right]= 3
$$

A kurtosis of $3$ is the same as a Gaussian distribution. This means, under the compound Poisson model, as the expected number of ticks increases returns become more Gaussian.

# Conclusions

Three of the common stylized facts about returns appear to make sense:

1. **Returns are not Gaussian** makes sense because a Gaussian distribution models the sum of \\(n\\) things but returns are the sum of varying amounts of ticks.
2. **Returns have fat tails** makes sense because we get fat tails when each sample is the sum of a different number of things.
3. **More Gaussian at higher time frames** makes sense because when each sample is the sum of a different number of things, kurtosis asymptotes to 3 as the number of things increases to infinity. A Gaussian distribution has a kurtosis of 3.

These are not the only explanations for returns not being Gaussian. For example, overnight shocks can cause the opening price to be wildly different from the previous day's close. The model in this paper does not include such shocks.

# Appendix

## Deriving kurtosis

There's a paper which shows the statistics of a compound Poisson distribution[^Bening2022]. However, I'm not sure how they derived the values. Here, I will use the method of moments to derive the kurtosis of a Poisson sum of Gaussians. The derivation just uses algebra, though it is a little tedious.

The kurtosis of $R$ is:

$$
\text{kurtosis}(R) = E\left[\left(\frac{R - \mu_R}{\sigma}\right)^4\right]
=  \frac{E[(R - \mu_R)^4]}{E[(R - \mu_R)^2]^2}
$$

The value $E[(R - \mu_R)^4]$ is known as the fourth central moment and $E[(R - \mu_R)^2]$ is the second central moment or [*variance*](https://en.wikipedia.org/wiki/Variance). Both of these can be expanded into raw moments:

$$
\begin{aligned}
E[(R - \mu_R)^2] &= E[R^2] - E[R]^2 \\\
E[(R - \mu_R)^4] &= E[R^4] - 4 E[R]E[R^3] + 6 E[R]^2E[R^2] - 3E[R]^4 \\\
\end{aligned}
$$

Each of the raw moments of $R$ can be written as functions of the raw moments of $T$ and $N$. To do this, we need a table of the raw moments of a Gaussian distribution ($T$)[^5] and a Poisson distribution ($N$) [^3] [^2]:

|   | Gaussian raw moments                         |
|---|:---------------------------------------------|
| 1 | $E[T] = \mu$                                 |
| 2 | $E[T^2] = \mu^2 + \sigma^2$                  |
| 3 | $E[T^3] = \mu^3 + 3\mu\sigma^2$              |
| 4 | $E[T^4] = \mu^4 +6\mu^2\sigma^2 + 3\sigma^4$ |

|   | Poisson raw moments                                      |
|---|:---------------------------------------------------------|
| 1 | $E[N] = \lambda$                                         |
| 2 | $E[N^2] = \lambda^2 + \lambda$                           |
| 3 | $E[N^3] = \lambda^3 + 3\lambda^2 + \lambda$              |
| 4 | $E[N^4] = \lambda^4 + 6\lambda^3 + 7\lambda^2 + \lambda$ |

We note that $E[R^n]$ can be expanded with[^1]:
$$
E[R^n] = E[E[R^n|N]]
$$
Where $E[R^n|N]$ is a raw moment of a Gaussian distribution. Expanding the first raw moment we get:
$$
E[R] = E[E[R|N]] = E[N]\mu = \lambda\mu
$$
The second raw moment:
$$
\begin{aligned}
E[R^2] &= E[E[R^2|N]] \\\
       &= E[N^2]\mu^2 + E[N]\sigma^2 \\\
       &= (\lambda^2 + \lambda)\mu^2 + \lambda\sigma^2 \\\
       &= \lambda^2\mu^2 + \lambda(\mu^2 + \sigma^2) \\\
       &= \lambda^2\mu^2 + \lambda E[T^2] \\\
\end{aligned}
$$
The third raw moment:
$$
\begin{aligned}
E[R^3] &= E[E[R^3|N]] \\\
       &= E[N^3]\mu^3 + 3 E[N^2]\mu\sigma^2  \\\
       &= (\lambda^3 + 3\lambda^2 + \lambda)\mu^3 + 3 (\lambda^2 + \lambda)\mu\sigma^2 \\\
       &= \lambda^3\mu^3 + 3\lambda^2\mu^3 + \lambda\mu^3 + 3\lambda^2\mu\sigma^2 + 3\lambda\mu\sigma^2 \\\
       &= \lambda^3\mu^3 + 3\lambda^2\mu^3 + 3\lambda^2\mu\sigma^2 + \lambda E[T^3] \\\
       &= \lambda^3\mu^3 + 3\lambda^2\mu(\mu^2 + \sigma^2) + \lambda E[T^3] \\\
       &= \lambda^3\mu^3 + 3\lambda^2\mu E[T^2] + \lambda E[T^3] \\\
\end{aligned}
$$

The fourth raw moment:
$$
\begin{aligned}
E[R^4] &= E[E[R^4|N]] \\\
       &= E[N^4]\mu^4 +6E[N^3]\mu^2\sigma^2 + 3 E[N^2]\sigma^4 \\\
       &= (\lambda^4 + 6\lambda^3 + 7\lambda^2 + \lambda)\mu^4 +6(\lambda^3 + 3\lambda^2 + \lambda)\mu^2\sigma^2 + 3 (\lambda^2 + \lambda)\sigma^4 \\\
       &= \lambda^4\mu^4 + 6\lambda^3\mu^4 + 7\lambda^2\mu^4 + \lambda\mu^4 + 6\lambda^3\mu^2\sigma^2 + 18\lambda^2\mu^2\sigma^2 + 6\lambda\mu^2\sigma^2 + 3 \lambda^2\sigma^4 + 3\lambda\sigma^4 \\\
       &= \lambda^4\mu^4 + 6\lambda^3\mu^4 + 7\lambda^2\mu^4 + 6\lambda^3\mu^2\sigma^2 + 18\lambda^2\mu^2\sigma^2  + 3 \lambda^2\sigma^4 + \lambda\mu^4 + 6\lambda\mu^2\sigma^2 + 3\lambda\sigma^4 \\\
       &= \lambda^4\mu^4 + 6\lambda^3\mu^4 + 7\lambda^2\mu^4 + 6\lambda^3\mu^2\sigma^2 + 18\lambda^2\mu^2\sigma^2  + 3 \lambda^2\sigma^4 + \lambda(\mu^4 + 6\mu^2\sigma^2 + 3 \sigma^4) \\\
       &= \lambda^4\mu^4 + 6\lambda^3\mu^4 + 7\lambda^2\mu^4 + 6\lambda^3\mu^2\sigma^2 + 18\lambda^2\mu^2\sigma^2  + 3 \lambda^2\sigma^4 + \lambda E[T^4] \\\
       &= \lambda^4\mu^4 + 7\lambda^2\mu^4 + 6\lambda^3\mu^2 E[T^2] + 18\lambda^2\mu^2\sigma^2  + 3 \lambda^2\sigma^4 + \lambda E[T^4] \\\
\end{aligned}
$$

Putting these raw moments into the second central moment we get:
$$
\begin{aligned}
E[(R - \mu_R)^2] &= E[R^2] - E[R]^2 \\\
                 &= \lambda^2\mu^2 + \lambda E[T^2] - \lambda^2\mu^2 \\\
                 &= \lambda E[T^2] \\\
\end{aligned}
$$

Similarly, putting these raw moments into the fourth central moment, we get:
$$
\begin{aligned}
E[(R - \mu_R)^4] &= E[R^4] - 4 E[R]E[R^3] + 6 E[R]^2E[R^2] - 3E[R]^4 \\\
                 &= E[R^4] - 4 \lambda\mu E[R^3] + 6 \lambda^2 \mu^2 E[R^2] - 3\lambda^4\mu^4 \\\
\end{aligned}
$$
Expanding out \\(E[R^2]\\):
$$
\begin{aligned}
    &= E[R^4] - 4 \lambda\mu E[R^3] + 6 \lambda^2 \mu^2 (\lambda^2\mu^2 + \lambda E[T^2]) - 3\lambda^4\mu^4 \\\
    &= E[R^4] - 4 \lambda\mu E[R^3] + 6 (\lambda^4\mu^4 + \lambda^3 \mu^2 E[T^2]) - 3\lambda^4\mu^4 \\\
    &= E[R^4] - 4 \lambda\mu E[R^3] + 6 \lambda^4\mu^4 + 6\lambda^3 \mu^2 E[T^2] - 3\lambda^4\mu^4 \\\
    &= E[R^4] - 4 \lambda\mu E[R^3] + 3 \lambda^4\mu^4 + 6\lambda^3 \mu^2 E[T^2] \\\
\end{aligned}
$$
Expanding out \\(E[R^3]\\):
$$
\begin{aligned}
    &= E[R^4] - 4 \lambda\mu (\lambda^3\mu^3 + 3\lambda^2\mu E[T^2] + \lambda E[T^3]) + 3 \lambda^4\mu^4 + 6\lambda^3 \mu^2 E[T^2] \\\
    &= E[R^4] - 4 (\lambda^4\mu^4 + 3\lambda^3\mu^2 E[T^2] + \lambda^2\mu E[T^3]) + 3 \lambda^4\mu^4 + 6\lambda^3 \mu^2 E[T^2] \\\
    &= E[R^4] - 4 \lambda^4\mu^4 - 12\lambda^3\mu^2 E[T^2] - 4\lambda^2\mu E[T^3] + 3 \lambda^4\mu^4 + 6\lambda^3 \mu^2 E[T^2] \\\
    &= E[R^4] - \lambda^4\mu^4 - 12\lambda^3\mu^2 E[T^2] - 4\lambda^2\mu E[T^3] + 6\lambda^3 \mu^2 E[T^2] \\\
    &= E[R^4] - \lambda^4\mu^4 - 6\lambda^3\mu^2 E[T^2] - 4\lambda^2\mu E[T^3]  \\\
    &= E[R^4] - \lambda^4\mu^4 - 6\lambda^3\mu^2 E[T^2] - 4\lambda^2\mu E[T^3]  \\\
\end{aligned}
$$
Expanding out \\(E[R^4]\\):
$$
\begin{aligned}
                    &= \lambda^4\mu^4 + 7\lambda^2\mu^4 + 6\lambda^3\mu^2 E[T^2] + 18\lambda^2\mu^2\sigma^2  + 3 \lambda^2\sigma^4 + \lambda E[T^4] \\\
                    & \quad - \lambda^4\mu^4 - 6\lambda^3\mu^2 E[T^2] - 4\lambda^2\mu E[T^3]  \\\
                    &=  7\lambda^2\mu^4 + 6\lambda^3\mu^2 E[T^2] + 18\lambda^2\mu^2\sigma^2  + 3 \lambda^2\sigma^4 + \lambda E[T^4] \\\
                    & \quad - 6\lambda^3\mu^2 E[T^2] - 4\lambda^2\mu E[T^3]  \\\
                    &=  7\lambda^2\mu^4 + 18\lambda^2\mu^2\sigma^2  + 3 \lambda^2\sigma^4 + \lambda E[T^4] - 4\lambda^2\mu E[T^3]  \\\
                    &=  7\lambda^2\mu^4 + 18\lambda^2\mu^2\sigma^2  + 3 \lambda^2\sigma^4 + \lambda E[T^4] - 4\lambda^2\mu (\mu^3 + 3\mu\sigma^2)  \\\
                    &=  7\lambda^2\mu^4 + 18\lambda^2\mu^2\sigma^2  + 3 \lambda^2\sigma^4 + \lambda E[T^4] - 4\lambda^2\mu^4 - 12\lambda^2\mu^2\sigma^2  \\\
                    &=  7\lambda^2\mu^4 + 6\lambda^2\mu^2\sigma^2  + 3 \lambda^2\sigma^4 + \lambda E[T^4] - 4\lambda^2\mu^4  \\\
                    &=  3\lambda^2\mu^4 + 6\lambda^2\mu^2\sigma^2  + 3 \lambda^2\sigma^4 + \lambda E[T^4]  \\\
E[(R - \mu_R)^4]    &=  3\lambda^2 E[T^2]^2 + \lambda E[T^4]  \\\
\end{aligned}
$$

And putting these central moments back into kurtosis:
$$
\begin{aligned}
\text{kurtosis}(R) &= \frac{E[(R - \mu_R)^4]}{E[(R - \mu_R)^2]^2} \\\
                   &= \frac{3\lambda^2 E[T^2]^2 + \lambda E[T^4]}{\lambda^2 E[T^2]^2} \\\
                   &= \frac{3\lambda^2 E[T^2]^2}{\lambda^2 E[T^2]^2} + \frac{\lambda E[T^4]}{\lambda^2 E[T^2]^2} \\\
                   &= 3 + \frac{E[T^4]}{\lambda E[T^2]^2} \\\
\end{aligned}
$$

If we assume that $\mu = 0$ then:
$$
\begin{aligned}
\text{kurtosis}(R|\mu = 0) &= 3 + \frac{E[T^4]}{\lambda E[T^2]^2} \\\
                           &= 3 + \frac{3 \sigma^4}{\lambda \sigma^4} \\\
                           &= 3 + \frac{3}{\lambda} \\\
\end{aligned}
$$

{{% citation
    id="Cont2001"
    author="Rama Cont"
    title="Empirical properties of asset returns: stylized facts and statistical issues"
    publication="Quantitative Finance"
    year="2001"
    pages="223-236"
    volume="1"
    link="http://rama.cont.perso.math.cnrs.fr/pdf/empirical.pdf"
%}}

{{% citation
    id="Bening2022"
    author="Vladimir Bening, Victor Korolev"
    title="Comparing Compound Poisson Distributions by Deficiency: Continuous-Time Case"
    publication="Mathematics"
    year="2022"
    volume="10"
    number="24"
    link="https://www.mdpi.com/2227-7390/10/24/4712"
%}}

[^1]: [First and second raw moments of Poisson sum of random variables](https://math.stackexchange.com/a/1646479). Answer on Stack Exchange.

[^2]: [Derivation of the third moment of Poisson distribution using Stein-Chen identity](https://math.stackexchange.com/questions/1075558/derivation-of-the-third-moment-of-poisson-distribution-using-stein-chen-identity). Question on Stack Exchange.

[^3]: [Poisson Distribution](https://mathworld.wolfram.com/PoissonDistribution.html) by Wolfram Alpha.

[^5]: [Raw Gaussian moments](https://math.stackexchange.com/a/4030443). Answer on Stack Exchange.
