---
title: "Conditional Value at Risk"
summary: "
    A practical crash course on conditional value at risk. Why it beats value at risk, how to estimate it from real data, and how to optimise portfolios with it. Complete with working code.
"

date: "2025-09-05"
type: paper
mathjax: true
authors:
    - Adrian Letchford
categories:
    - mathematics
    - finance
hover_color: "#FF9696"
---

[Value at Risk](https://en.wikipedia.org/wiki/Value_at_risk) (VaR) is the industry's go-to portfolio risk metric. But, it's a cutoff completely ignoring tail risk. It tells you how often you'll breach a threshold, not how bad losses are when you do. [Conditional Value at Risk](https://en.wikipedia.org/wiki/Expected_shortfall)  (CVaR) looks at that damage. It measures the average of your worst days.

In this article we recap VaR, build intuition for CVaR, estimate it from historical returns, and use it as a constraint in a portfolio optimiser. You get reusable Python to compute and plot CVaR and plug it into your workflow. By the end you'll know what CVaR means, how to measure it sensibly, and how to use it in real portfolio decisions.


# Measuring risk

VaR answers a frequency question: what is the ***minimum loss*** during the worst X% of outcomes? While CVaR answers a severity question: what is the ***average loss*** during the worst X% of outcomes? We'll discuss VaR's shortcomings and show how CVaR fills them before turning to code and an example.


## Value at risk

Value at Risk (VaR) is a measure of the maximum amount of money you could lose on a regular day. Given a time horizon (e.g. 1 day) and a regular-day frequency (e.g. 95%), VaR tells you the maximum loss you can expect on those regular days.

If 95% of days are "regular" and we expect to lose at most \\$1m, we say "95% of the time you lose less than \\$1m." About 1 day in 20 you could lose \\$1m or more. You might see this written as "1-day 95% VaR of \\$1m."

A better interpretation of VaR is that it is the *minimum* loss you will see on your worst days.

{{<figure src="var_description.svg" width="medium" >}}{{</figure>}}

VaR has some fairly serious shortcomings; tail blindness, failure to be subadditive and awkwardness for optimisation.

**Tail blindness** &nbsp;&nbsp; The critical thing to understand about VaR is that it does not tell you how much you could lose on bad days. It only tells you the maximum loss you can expect to see on regular days. It ignores the losses entirely once you're past the threshold. A breach of this threshold will *always* be worse than the VaR figure. It could be a little worse, or it could be catastrophically worse.

**Fails subadditivity** &nbsp;&nbsp;  We want diversification to reduce risk. If we combine two portfolios, the risk of the combined portfolio should be no greater than the sum of the risks of the individual portfolios. This is known as [subadditivity](https://en.wikipedia.org/wiki/Subadditivity) and is a desirable property for a risk metric [^coherent]. VaR does not satisfy this property. See the [appendix](#appendix-var-is-not-subadditive) for an example.

**Awkward for optimisation** &nbsp;&nbsp; Minimising VaR is a non-convex, unstable problem. I have not seen a portfolio optimisation using VaR as a constraint or objective.

The Conditional Value at Risk (CVaR) addresses these issues.

## Conditional value at risk

*Conditional Value at Risk (CVaR) is also known as Expected Shortfall, Mean Excess Loss, Mean Shortfall, or Tail VaR [^Uryasev2000].*

Rather than telling you the maximum loss you can expect on regular days, CVaR tells you the *average loss* on the worst days. This gives a clearer view of tail risk.

{{<figure src="cvar_description.svg" width="medium" >}}{{</figure>}}

This change from minimum loss (VaR) to average loss (CVaR) addresses the tail blindness problem. A breach of the VaR threshold will, on average, be equal to the CVaR figure. Also, the metric is subadditive making it inline with our intuition that diversification should reduce risk. And, while on first pass the CVaR is not convex, it can be reformulated as a convex problem that can be incorporated into a portfolio optimisation[^Rockafellar1999] as we will see later.

## Estimation

We'll use the symbol $\alpha$ to denote the risk level. Carrying on from the examples above, we'll use $\alpha = 0.95$ to denote a 95% VaR or CVaR.

To estimate $\text{VaR}(\alpha)$ and $\text{CVaR}(\alpha)$, we use a set of scenarios of possible return vectors. This is a way of representing the distribution of returns without a parametric model. For simplicity, we'll use all historical returns as the scenarios.

Let's say we have a vector of portfolio weights $\boldsymbol{w}$ and vectors of asset returns $\boldsymbol{r}_t$ where each $t$ is a different time period over some historical window. These $\boldsymbol{r}_t$s are our scenarios. We estimate the risk metrics at the $\alpha$ level as follows:

1. Calculate the portfolio returns $R_t = \boldsymbol{w}^\top \boldsymbol{r}_t$ for each time $t$.
1. Calculate the $(1 - \alpha)$ quantile of the returns. This is the VaR.
3. Calculate the average of the worst returns (returns equal to or less than VaR). This is the CVaR.

## Example

Let's look at an example of what CVaR looks like in practice.

We're going to look at an equally risk weighted portfolio of the following ETFs:

- SPY -- U.S. equities (S&P 500)
- TLT -- Long-term U.S. Treasury bonds
- GLD -- Gold
- GSG -- Broad commodities
- VNQ -- U.S. real estate investment trusts (REITs)

You can grab these prices with the following Python code:
```python
import yfinance as yf

tickers = yf.Tickers("SPY TLT GLD GSG VNQ")
prices = tickers.download(period="30y", interval="1d")
returns = prices["Close"].pct_change().dropna()
```

And then use an exponentially weighted estimate of volatility to determine the portfolio weights at each time step:
```python
vols = returns.ewm(halflife=21, min_periods=252).std()

# Larger vol should have smaller weight.
# We invert the volatility
inv_vols =  1 / vols

weights = inv_vols.divide(inv_vols.sum(1), axis=0).dropna()
```
We've used a half-life of 21 days (about a month) to estimate volatility. The minimum period of 252 days (about a year) ensures we have a reasonable estimate before calculating weights.

The weights on a date (each row in the `weights` DataFrame) are the weights known at the end of that day as they need that day's returns to estimate volatility. This means that today's weights are used to trade tomorrow's return. We need to remember to shift by one when multiplying with returns.

The equity curve (barring costs and other frictions) can be calculated with the following code and is shown in the figure below:

```python
portfolio_returns = (returns * weights.shift(1)).sum(1)
portfolio_equity = (1 + portfolio_returns).cumprod()
```

{{<figure src="portfolio_returns.svg" title="Risk parity portfolio">}}
A portfolio of SPY, TLT, GLD, GSG and VNQ with risk parity weights. The 21-day exponentially weighted volatility is used to determine the weights. The portfolio is rebalanced daily. Costs and other frictions are not taken into account.
{{</figure>}}

We can estimate VaR and CVaR using the historical prices as described above:
```python
import pandas as pd

risk_level = 0.95

var = pd.Series(index=weights.index)
cvar = pd.Series(index=weights.index)

for date, w in weights.iterrows():

    # We use all the known history as the scenarios
    scenarios = returns.loc[:date]
    scenario_returns = (scenarios * w).sum(1)

    threshold = scenario_returns.quantile(1 - risk_level)
    worst_returns = scenario_returns[scenario_returns <= threshold]

    var.loc[date] = threshold
    cvar.loc[date] = worst_returns.mean()
```

The VaR and CVaR over time look like:

{{<figure src="var_and_cvar.svg" title="VaR vs CVaR">}}
Using the same portfolio as the previous figure, we estimate the 1-day 95% VaR and CVaR using all historical returns as scenarios. The CVaR is always worse than the VaR.
{{</figure>}}

The main point to take away from this graph is that the CVaR is always worse than the VaR. From an average loss perspective, CVaR captures the tail risk whereas VaR completely ignores it.

# Procyclical estimates

We now want to do a sanity check to see if our estimates are reasonable. For the VaR estimate (which is the 5% quantile of returns), we want to see if it correlates with the actual 5% quantile of future portfolio returns. For the CVaR estimate (which is the average of the worst 5% of returns), we want to see if it correlates with the actual average of the worst 5% of future portfolio returns.

**VaR check** &nbsp; Bucket the VaR estimates into 10 evenly sized buckets. Bucket 1 has the 10% worst estimates while bucket 10 has the 10% best. For each bucket, calculate the **5% quantile of future returns**. We expect a roughly linearly increasing relationship.

**CVaR check** &nbsp; Repeat for CVaR. For each bucket, calculate the **average of the worst 5% of future returns.** Expect a roughly linearly increasing relationship.

Here is the code:
```python
import numpy as np

# We want future returns, so shift back by 1.
df = pd.concat([var, cvar, portfolio_returns.shift(-1)], axis=1)
df.columns = ['var', 'cvar', 'portfolio_returns']

# Calcualte sanity check for VaR
buckets = np.ceil(df['var'].rank(pct=True) * 10)
y = df.groupby(buckets)['portfolio_returns'].quantile(0.05)
x = df.groupby(buckets)['var'].mean()

# Calcualte sanity check for CVaR
buckets = np.ceil(df['cvar'].rank(pct=True) * 10)
y = df.groupby(buckets)['portfolio_returns'].apply(
    lambda x: x[x <= x.quantile(0.05)].mean()
)
x = df.groupby(buckets)['cvar'].mean()
```

And we get the resulting graph:

{{<figure src="simple_sanity_check.svg" title="Sanity check risk estimates">}}
(left) The Var(95%) estimates are bucketed into 10 evenly sized buckets. For each bucket, the 5% quantile of future portfolio returns is calculated. We do not see a roughly linear increasing relationship. (right) The CVaR(95%) estimates are bucketed into 10 evenly sized buckets. For each bucket, the average of the worst 5% of future portfolio returns is calculated. We do not see a roughly linear increasing relationship.
{{</figure>}}

Which is roughly the opposite of what we want! The worst VaR estimates have the highest 5% quantile. Except for the 10th bucket (noise), the 5% quantile decreases where it should increase. Similarly, we do not see the expected increasing relationship for CVaR.

This tells us the CVaR estimates are poor. Quite bad, actually.

What is happening? When the market is volatile, the historical returns have more extreme values. When the market is calm, the historical returns have less extreme values. This means that the risk metrics increase their estimates DURING (as opposed to before) a volatile market, and decrease their estimates during a calm market. This has the effect of estimating high risk as the market is moving into a calm period and estimating low risk as the market is moving into a volatile period.

This behaviour is called [*procyclical*](https://en.wikipedia.org/wiki/Procyclical_and_countercyclical_variables)[^Murphy2014].

{{<figure src="procyclical.svg" title="Example of procyclical behaviour">}}
SPY during the 2025 US tarrif episode. The left axis shows the SPY price drop and rebound. The right axis shows the CVaR(95) estimated with the previous two years of history. CVaR did not get worse as the market became volatile and dropped. Once the market calmed and prices started to recover, the estimated risk remained elevated.
{{</figure>}}

If we were to use these estimates in a portfolio optimisation, we would lower our risk when we should be increasing it and vice versa. We'd make the portfolio worse, not better.

To fix this we normalise historical returns by their volatility[^Perez2015], removing the procyclical behaviour. We then re-scale VaR and CVaR by current volatility to put them back into the right units. We use a long half-life to de-volatilise, and a short half-life to re-volatilise to current conditions.

We can use the following function to calculate the volatility adjusted VaR and CVaR estimates:

```python
def risk_from_weights(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    risk_level: float = 0.95,
    long_vols_half_life: float = 252,
    short_vols_half_life: float = 21 * 3,
    min_periods: int = 252,
) -> tuple[pd.Series, pd.Series]:
    """
    Compute the Value at Risk (VaR) and Conditional
    Value at Risk (CVaR) for a given set of portfolio
    weights and historical returns.

    Parameters
    ----------

    weights : pd.DataFrame
        DataFrame of portfolio weights. Rows
        are time periods and columns are assets.

    returns : pd.DataFrame
        DataFrame of historical asset returns.
        Same shape as `weights`.

    risk_level : float
        The VaR and CVaR risk level. E.g., 0.95
        for 95% CVaR.

    long_vols_half_life : float
        Half-life for the long-term volatility
        estimate (in days).

    short_vols_half_life : float
        Half-life for the short-term volatility
        estimate (in days).

    min_periods : int
        Minimum number of periods required to
        compute risk measures.

    Returns
    -------
    VaR : pd.Series
        A Series containing the VaR values, with the same
        index as the input returns DataFrame.

    CVaR : pd.Series
        A Series containing the CVaR values, with the same
        index as the input returns DataFrame.
    """

    long_vols = returns.ewm(halflife=long_vols_half_life).std()
    short_vols = returns.ewm(halflife=short_vols_half_life).std()

    scaled = returns / long_vols
    var = pd.Series(index=returns.index)
    cvar = pd.Series(index=returns.index)

    for date, w in tqdm(weights.iterrows(), total=len(weights)):

        # Create the scenarios from the normalised
        # historical prices.
        history = scaled.loc[:date].dropna()
        vol = short_vols.loc[date]
        scenarios = history * vol

        if len(scenarios) < min_periods:
            continue

        w = weights.loc[date]

        losses = scenarios @ w  # Negative is a loss
        var[date] = losses.quantile(1 - risk_level)
        cvar[date] = losses[losses <= var[date]].mean()

    return var, cvar
```

These improved estimates look like this:

{{<figure src="var_and_cvar_adjusted.svg" title="Volatility adjusted metrics">}}
Using the same risk parity portfolio as before, we estimate the 1-day 95% VaR and CVaR using volatility adjusted historical returns as scenarios.
{{</figure>}}

Re-running the sanity check gives us much better results:

{{<figure src="sanity_check.svg" title="Sanity check adjusted risk estimates">}}
(left) The volatility adjusted Var(95%) estimates are bucketed into 10 evenly sized buckets. For each bucket, the 5% quantile of future portfolio returns is calculated. We see a roughly linear increasing relationship. (right) The volatility adjusted CVaR(95%) estimates are bucketed into 10 evenly sized buckets. For each bucket, the average of the worst 5% of future portfolio returns is calculated. We see a roughly linear increasing relationship.
{{</figure>}}

Which is much closer to what we want to see. As the estimate for both VaR and CVaR get worse, the actual returns also get worse. In the worst 10% of samples, the returns are not as similarly worse. For our purposes, this will work ok. Refining the risk models any further is outside of the scope of this article.


# Portfolio optimisation

The CVaR is not a convex function we can use in a portfolio optimisation. At least, in it's current form presented above, it is not. Rockafellar and Uryasev showed that we can reformulate the CVaR as a convex problem[^Rockafellar1999].

The derivation is clever and makes use of two tricks. Understanding it will teach you some key concepts in optimisation. We're going to work through the derivation here and show the final optimisation problem at the end.

## Derivation

The derivation involves rewriting the CVaR into a form that is easier to work with, then applying two tricks to get it into a convex form that can plug into a portfolio optimisation.

### Rewriting the CVaR

Let's say that we have $N$ scenarios of returns denoted by $r_i$ and we define the loss for scenario $i$ as $l_i = -r_i$. In this derivation, a loss is positive, profit is negative.

We will write the VaR at level $\alpha$ as $\text{VaR}_{\alpha}(l)$. The CVaR is then:
$$
\text{CVaR}\_{\alpha}(l) = E[l \ | \ l \geq \text{VaR}\_{\alpha}(l)]
$$
We replace the expectation with a sample average. $\text{VaR}\_{\alpha}(l)$ is the $\alpha$<sup>th</sup> sample quantile, so $(1 - \alpha) N$ scenarios lie in the tail. Thus:
$$
\text{CVaR}\_{\alpha}(l) = \frac{1}{(1 - \alpha) N} \sum\_{i=1}^N l_i \cdot \\{ l_i \geq \text{VaR}\_{\alpha}(l) \\}
$$
Where $\\{ \cdot \\}$ is the indicator function, which is 1 if the condition is true and 0 otherwise.

We can replace the indicator function with a max function as follows:
$$
\text{CVaR}\_{\alpha}(l) = \text{VaR}\_{\alpha}(l) + \frac{1}{(1 - \alpha) N} \sum\_{i=1}^N \max(l_i - \text{VaR}\_{\alpha}(l), 0)
$$
We'll use a simpler notation for the max between a variable and 0:
$$
\text{CVaR}\_{\alpha}(l) = \text{VaR}\_{\alpha}(l) + \frac{1}{(1 - \alpha) N} \sum\_{i=1}^N |l_i - \text{VaR}\_{\alpha}(l)|\_+
$$
This gives us a form we can make convex. For this we'll need two tricks.

### Trick 1: VaR as an optimisation

We're going to replace the value for VaR with an unknown variable $\tau$ to give us:
$$
F(\tau) = \tau + \frac{1}{(1 - \alpha) N} \sum\_{i=1}^N |l_i - \tau|\_+
$$

Something remarkable happens here. If we minimise $F(\tau)$ with respect to $\tau$, the value of $\tau$ that minimises $F(\tau)$ is exactly $\text{VaR}\_{\alpha}(l)$:
$$
\text{VaR}\_{\alpha}(l) = \underset{\tau}{\text{argmin}} \ F(\tau)
$$
To me, this is quite astonishing. Before proving it, let's see a practical example. We'll use the negative of the SPY returns for the loss values $l_i$, and plot $F(\tau)$ for a range of $\tau$ at $\alpha = 95\%$. The $\tau$ that minimises $F(\tau)$ is exactly the 95% VaR of the losses.

{{<figure src="trick1.svg" title="Minimising F">}}
Here we show the value for $F(\tau)$ for a range of $\tau$ values on SPY returns. The minimum value is at the 95% VaR of the losses (the negative of SPY returns).
{{</figure>}}

We prove this is true by finding the minimum of $F(\tau)$. Differentiate and set to zero. The derivative is:
$$
\frac{d}{d\tau}F(\tau) = 1 - \frac{1}{(1 - \alpha) N} \sum\_{i=1}^N \\{ l_i \geq \tau \\}
$$
Note that the indicator function is always either 0 or 1. That means this function is *monotonic* which means there is one and only one point where the derivative is zero minimising the function. Setting the derivative to zero and rearranging we get:
$$
\frac{1}{N}\sum\_{i=1}^N \\{ l_i \geq \tau \\} = 1 - \alpha
$$
The left hand side is the fraction of losses greater than $\tau$ and the right hand side is also the fraction of losses greater than the VaR at level $\alpha$. Therefore, the $\tau$ that minimises $F(\tau)$ is exactly the $\text{VaR}\_{\alpha}(l)$.

This allows us to rewrite the CVaR as a minimisation problem:
$$
\text{CVaR}\_{\alpha}(l) = \min\_{\tau} \  \tau + \frac{1}{(1 - \alpha) N} \sum\_{i=1}^N |l_i - \tau|\_+
$$

This minimisation is a convex problem (as shown by the derivative). However, we still have the max function to deal with.

### Trick 2: Max function as a linear problem

To handle the max function, we replace each $|l_i - \tau|\_+$ with a [slack variable](https://en.wikipedia.org/wiki/Slack_variable) $u_i$ with constraints:
$$
\begin{align}
u_i & \geq l_i - \tau \\\
u_i & \geq 0 \\\
\end{align}
$$
The first constraint says that $u_i$ must be greater than the excess loss $l_i - \tau$ which could be negative. The second constraint says that $u_i$ must be at least 0 (not negative). Together, they imply:
$$
u_i \geq |l_i - \tau|\_+
$$
And if we minimise $u_i$ along with $\tau$, the optimal solution will find the smallest $u_i$ that satisfies these constraints. Which means, in the solution:
$$
u_i = |l_i - \tau|\_+
$$

Putting this together, we write CVaR as:
$$
\text{CVaR}\_{\alpha}(l) = \min\_{\tau, u_i} \ \tau + \frac{1}{(1 - \alpha) N} \sum\_{i=1}^N u_i
$$
Subject to:
$$
\begin{align}
u_i &\geq l_i - \tau \\\
u_i &\geq 0 \\\
\end{align}
$$

This is a convex linear program.

## Optimisation problem

Now, we can build out a portfolio optimisation problem. The objective is to maximise expected returns subject to a portfolio variance constraint and a CVaR constraint. Additionally, we'll have long only weights and no leverage.

**Parameters and variables**

* $\boldsymbol{w}$ - the portfolio weights we are trying to find.
* $\boldsymbol{\mu}$ - the expected asset returns.
* $\boldsymbol{\Sigma}$ - the asset return covariance matrix.
* $\boldsymbol{r}_i$ - the asset returns in scenario $i$ for $i = 1, \ldots, N$.
* $\boldsymbol{w}^\top \boldsymbol{r}_i$ - the portfolio return under scenario $i$.
* $l_i = -\boldsymbol{w}^\top \boldsymbol{r}_i$ - the loss in scenario $i$ (positive means you lost money).
* $\alpha$ - the CVaR level (e.g. 0.95 for 95% CVaR).
* $\kappa$ - the maximum allowed CVaR (risk limit). This will be in percentage terms (e.g. 0.1 for 10% average loss).
* $\sigma$ - the maximum allowed portfolio standard deviation (risk limit).

**Problem**

$$
\begin{align}
\underset{\boldsymbol{w}}{\text{maximise}} \quad& \boldsymbol{w}^\top \boldsymbol{\mu} \\\
\text{s.t.} \quad
& \boldsymbol{w} \geq 0 & \textit{Long only} \\\
& \boldsymbol{w}^\top \boldsymbol{1} \leq 1 & \textit{No leverage} \\\
& \boldsymbol{w}^\top \boldsymbol{\Sigma} \boldsymbol{w} \leq \sigma^2 & \textit{Variance limit} \\\
& \tau + \frac{1}{(1 - \alpha) N} \sum\_{i=1}^N u_i \leq \kappa & \textit{CVaR limit} \\\
& u_i \geq -\boldsymbol{w}^\top \boldsymbol{r}_i - \tau \\\
& u_i \geq 0
\end{align}
$$

We can wrap this up in a Python function using [cvxpy](https://www.cvxpy.org/en/stable/) as follows:
```python
import numpy as np
import cvxpy as cp

def optimise(
    expected_returns: np.ndarray,
    expected_cov: np.ndarray,
    scenarios: np.ndarray,
    risk_level: float,
    max_avg_risk: float,
    max_vol: float,
) -> np.ndarray:
    """
    Solves a portfolio optimisation problem with a
    portfolio variance and Conditional Value-at-Risk
    (CVaR) risk constraint.

    Given expected asset returns, scenario returns, a
    CVaR confidence level, and a risk limit, this
    function finds the optimal long-only portfolio
    weights that maximise expected return subject to
    a CVaR constraint and no leverage.

    Parameters
    ----------
    expected_returns : np.ndarray
        Array of expected returns for each asset
        (shape: [M], where M is number of assets).

    expected_cov : np.ndarray
        Array of expected covariances for each asset
        (shape: [M, M], where M is number of assets).

    scenarios : np.ndarray
        Array of scenario returns
        (shape: [N, M], where N is number of
        scenarios).

    risk_level : float
        Confidence level for CVaR (e.g., 0.95 for
        95% CVaR).

    max_avg_risk : float
        Maximum allowed CVaR (risk limit).
        
    max_vol : float
        Maximum allowed portfolio standard deviation.

    Returns
    -------
    np.ndarray
        Optimal portfolio weights (shape: [M]) that
        maximise expected return under the constraints.
    """

    # Number of scenarios, number of assets
    N, M = scenarios.shape

    # These are the weights we want to find
    w  = cp.Variable(M)

    # These are the auxiliary variables for CVaR
    # using the Rockafellar-Uryasev formulation.
    tau = cp.Variable()
    u = cp.Variable(N, nonneg=True)

    # CVaR expression
    cvar = tau + (1/((1-risk_level)*N)) * cp.sum(u)

    # Objective: maximise expected return
    objective = cp.Maximize(expected_returns @ w)

    # Constraints - the constraint for `u` to be
    # greater or equal to 0 is handled by
    # the nonneg=True argument above.
    constraints = [
        # long-only
        w >= 0,
        # No leverage
        cp.sum(w) <= 1,
        # volatility constraint
        w @ expected_cov @ w <= max_vol**2,
        # risk limit
        cvar <= max_avg_risk,
        u >= -(scenarios @ w) - tau,
    ]

    prob = cp.Problem(objective, constraints)
    
    prob.solve(solver=cp.CLARABEL)

    return w.value
```

# Example

We're going to run through a simple example of using this optimiser to manage risk in a portfolio. We'll look at how the portfolio performs with different CVaR limits.

We can use the same ETFs as before and we'll do the following:

- **Expected return** - The expected returns will be the exponentially weighted historical return with a half-life of 63 days (about 3 months).
- **Covariance** - The expected covariance will be the exponentially weighted historical covariance with a half-life of 63 days (about 3 months).
- **Max volatility** - We'll set the maximum portfolio volatility to 10% annualised.
- **Risk level** - We'll fix the CVaR level to  $\alpha = 0.95$.
- **Risk limit** - We'll look at three different limits $\kappa \in [1.0, 0.05, 0.025]$. We include the 1.0 limit to show the portfolio without an active CVaR constraint.

We'll use the following code to find the weights over time:
```python
def rolling_weights(
    returns: pd.DataFrame,
    expected_returns_half_life: float = 252 * 3,
    expected_covs_half_life: float = 252 * 3,
    risk_level: float = 0.95,
    max_avg_risk: float = 0.01,
    long_vols_half_life: float = 252,
    short_vols_half_life: float = 21 * 3,
    min_periods: int = 252,
    max_vol: float = 0.005,
) -> pd.DataFrame:
    """
    Finds the optimal portfolio weights over time in
    a rolling fashion.

    Parameters
    ----------
    returns : pd.DataFrame
        DataFrame of historical asset returns.
        Rows are time periods and columns are assets.
    
    expected_returns_half_life : float
        Half-life for the expected returns
        estimate (in days).
    
    expected_covs_half_life : float
        Half-life for the expected covariances
        estimate (in days).

    risk_level : float
        The CVaR risk level. E.g., 0.95 for 95%
        CVaR.
    
    max_avg_risk : float
        The maximum average risk (CVaR) allowed
        in the portfolio.
    
    long_vols_half_life : float
        Half-life for the long-term volatility
        estimate (in days).
        
    short_vols_half_life : float
        Half-life for the short-term volatility
        estimate (in days).
    
    min_periods : int
        Minimum number of periods required to
        compute risk measures.
    
    max_vol : float
        The maximum portfolio volatility allowed.    
    """
    
    expected_returns = returns.ewm(halflife=expected_returns_half_life).mean()
    expected_covs = returns.ewm(halflife=expected_covs_half_life).cov()

    long_vols = returns.ewm(halflife=long_vols_half_life).std()
    short_vols = returns.ewm(halflife=short_vols_half_life).std()

    scaled = returns / long_vols

    weights = pd.DataFrame(index=returns.index, columns=returns.columns)

    for date, w in tqdm(returns.iterrows(), total=len(returns)):

        # Create the scenarios from the normalised
        # historical prices.
        history = scaled.loc[:date].dropna()
        vol = short_vols.loc[date]
        scenarios = history * vol

        eret = expected_returns.loc[date]
        ecov = expected_covs.loc[date]

        if len(scenarios) < min_periods:
            continue

        weights.loc[date] = optimise(
            expected_returns=eret.values,
            expected_cov=ecov.values,
            scenarios=scenarios.values,
            risk_level=risk_level,
            max_avg_risk=max_avg_risk,
            max_vol=max_vol,
        )

    return weights
```

The resulting equity curves and CVaR estimates are in the following figure:

{{<figure src="optimisation_example.svg" title="Optimisation example">}}
A portfolio of SPY, TLT, GLD, GSG and VNQ with weights optimised with the `optimise` code. The CVaR constraint is added as an extra constraint and the risk limit is varied between 1.0 (no limit), 0.05 (5% average loss) and 0.025 (2.5% average loss). The portfolio is rebalanced daily. You can see that the constraint is limiting the estimated CVaR.

{{</figure>}}

Looking at the figure above, the optimiser has reduced estimated CVaR with little impact on the equity curve. This suggests that the CVaR constraint is effective at reducing tail risk without significantly impacting returns. Assuming, of course, that the scenarios are a good representation of future returns.

# Summary

CVaR is a diversification-friendly, convex risk measure that addresses VaR's shortcomings. It tells you how bad losses are on the worst days.

In this article, we estimated CVaR from historical returns, stabilised it via volatility normalisation, sanity-checked it, reformulated it for linear programming, and built an optimiser tested on ETFs.

Use the code snippets throughout to reproduce results and build your own CVaR-based optimiser.


# Appendix: VaR is not subadditive

Say we have some risk metric $\rho(\cdot)$. We say that it is subadditive if, for any two portfolios $X_1$ and $ X_2$:
$$
\rho(X_1 + X_2) \leq \rho(X_1) + \rho(X_2)
$$
which is to say that the risk of the combined portfolio is no greater than the sum of the risks of the individual portfolios.

To demonstrate that VaR is not subadditive, we will consider two **independent** loans $X_1$ and $X_2$ such that each loan loses **\\$1** with a probability of **10%**, and **\\$0** otherwise. The two loans are independent. We'll look at the 90% VaR of each loan individually and then combined.

Single loan:


| Loss | Probability |
|------|------------:|
| \$0  | 90%         |
| \$1  | 10%         |

The 90% VaR of a single loan is \\$0, since 90% of the time the loss will be no more than \\$0. That gives us:
$$
\begin{align}
\rho(X_1) = 0 \\\
\rho(X_2) = 0 \\\
\end{align}
$$

Combined portfolio (two loans):

| Loss | Probability                        |
|------|-----------------------------------:|
| \$0  | $0.9 \times 0.9 = 81\\%$           |
| \$1  | $0.9 \times 0.1 \times 2 = 18\\%$  |
| \$2  | $0.1 \times 0.1 = 1\\%$            |

The cumulative probability at \\$0 is 81% (< 90%), and at \\$1 it's 99% (> 90%). Therefore, the 90% VaR of the combined position is \\$1, since 90% of the time the loss will be no more than \\$1. That gives us:
$$
\rho(X_1 + X_2) = 1
$$
which fails the subadditivity property since:
$$
\rho(X_1 + X_2) = 1 \nleq 0 = \rho(X_1) + \rho(X_2)
$$


{{% citation
    id="Jorion1999"
    author="Philippe Jorion"
    title="Risk Management Lessons from Long-Term Capital Management"
    year="1999"
    link="https://ssrn.com/abstract=169449"
%}}

{{% citation
    id="Rockafellar1999"
    author="R. Tyrrell Rockafellar and Stanislav Uryasev"
    title="Optimization of Conditional Value-at-Risk"
    year="1999"
    link="https://sites.math.washington.edu/~rtr/papers/rtr179-CVaR1.pdf"
%}}

{{% citation
    id="Uryasev2000"
    author="Stanislav Uryasev"
    title="Conditional Value-at-Risk: Optimization Algorithms and Applications"
    year="2000"
    publication="Financial Engineering News"
    number="14"
    link="https://uryasev.ams.stonybrook.edu/wp-content/uploads/2011/11/FinNews.pdf"
%}}

{{% citation
    id="Perez2015"
    author="Pedro Gurrola-Perez and David Murphy"
    title="Filtered historical simulation Value-at-Risk models and their competitors. Working Paper No. 525."
    year="2015"
    publication="Bank of England"
    link="https://www.bankofengland.co.uk/-/media/boe/files/working-paper/2015/filtered-historical-simulation-value-at-risk-models-and-their-competitors.pdf"
%}}

{{% citation
    id="Murphy2014"
    author="David Murphy, Michalis Vasios and Nick Vause"
    title="An investigation into the procyclicality of risk-based initial margin models. Financial Stability Paper No. 29."
    year="2014"
    publication="Bank of England"
    link="https://www.bankofengland.co.uk/-/media/boe/files/financial-stability-paper/2014/an-investigation-into-the-procyclicality-of-risk-based-initial-margin-models.pdf"
%}}

[^coherent]: An ideal risk metric is said to be [coherent](https://en.wikipedia.org/wiki/Coherent_risk_measure) if it satisfies a list of properties. See the Wikipedia page for more details.