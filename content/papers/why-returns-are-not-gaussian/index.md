---
title: "Why Returns Are Not Gaussian"
summary: "
I want to share with you my intuition as to why asset returns are not Gaussian. This will not be a proof, but rather a reasonable explanation for this phenomenon.
"

date: "2023-01-08"
type: paper
mathjax: true # Enable mathematics on the page
plotly: true  # Enable plotly on the page
authors:
    - Adrian Letchford
draft: true
categories:
    - finance
---

For decades, researchers have noted properties of returns that are consistent across different assets. These properties have become known as *stylised facts.* They are measurable, but, researchers have not been able to prove that they must be true.

The common stylised facts we are interested in here are [^Cont2001]:

1. Returns are not Gaussian.
1. Returns have fat tails.
1. Returns become more Gaussian on higher time frames and less Gaussian on lower time frame.

I want to show you why these three properties actually make sense, even if we cannot prove that they are true. I’ll explain the first property based on the definition of a Gaussian distribution. The second and third can be proven if some intuitive assumptions are made.

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

The Gaussian distribution pops up everywhere, and with good reason. It describes the average of \\(n\\) independently and identically distributed random samples from any distribution as \\(n\\) tends to infinity:

$$
\frac{X_1 + \dots + X_n}{n} \sim \mathcal{N}(\mu, \sigma^2)
$$

We know this is true from the [central limit theorem](https://en.wikipedia.org/wiki/Central_limit_theorem). I’m not going to write out the proof here, you can read about the central limit theorem and proofs on [Wikipedia](https://en.wikipedia.org/wiki/Central_limit_theorem).

We can actually make the Gaussian distribution a bit more general. A Gaussian variable multiplied by some constant is still Gaussian. This means we can multiply an average value by the number of samples to turn it into a sum:

$$
\begin{aligned}
X_1 + \dots + X_n &\sim n\mathcal{N}(\mu, \sigma^2) = \mathcal{N}(n\mu, n^2\sigma^2)
\end{aligned}
$$

A Gaussian distribution models the sum of \\(n\\) independently and identically distributed random values as \\(n\\) tends to infinity.

What we will do is visualise a simulation of this for a couple of different distributions:

<todo>
Distributions: uniform, beta with high tails, exponential and show for n = 1, 10, 100, 1000
</todo>

# Returns are not Gaussian

More precisely, returns do not meet the assumptions of a Gaussian distribution.

For this discussion, we are going to think in terms of logged prices. That is, when I say "return" I mean change in logged price:
$$
\text{return at time } t = \frac{p_t - p_{t-1}}{p_{t-1}} \approx \log(p_t) - \log(p_{t-1})
$$
This means we can think of returns as summing together rater than multiplying together.

Think about what a daily return is. You could say that a daily return is the sum of 8 hourly returns. Since we know that a sum of \\(n = 8\\) values (8 trading hours in a day) is approximately Guassian, it makes sense that the daily returns ought to be Guassian. To see why this isn't the case, we need to break down a day's return to its atomic level. Each hourly return is the sum of minutely returns which are the sum of per second returns. If we continue this logic, we get to the atomic level: ticks.

<todo>
Graphic illustrating this breakdown
</todo>

Each day's return is a sum of tick returns. The key thing to note is that each day has a different number of ticks. Bringing this back to the Gaussian distribution, each day is the sum of a different number of ticks. The \\(n\\) is different on each day. The Gaussian distribution assumes that \\(n\\) is the same for each sample. Therefore, returns on any time scale do not meet the assumptions of a Gaussian distribution.

# Returns have fat tails

Now to make our first assumption: Tick changes are Gaussian.

- Assume that ticks are Gaussian and the number of ticks are Poisson
- Calculate the kurtosis
- The kurtosis is greater than 3. This means the distribution has fatter tails than a Gaussian distribution. We say the distribution is leptokurtic.

# More Gaussian at higher time frames

- Higher time frames are sums of more ticks.
- Under the Gaussian-Poisson model, this means n increases
- As n increases, kurtosis gets closer to 3.
- 3 is the same as a Gaussian

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
