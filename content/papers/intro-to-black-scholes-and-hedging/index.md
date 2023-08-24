---
title: "Intro to Black-Scholes, implied volatility and hedging"
summary: "
A break down of how the black-scholes option pricing model works, what implied volatility is and how you can use the model to hedge risks.
"

date: "2023-08-19"
type: paper
mathjax: true # Enable mathematics on the page
plotly: true  # Enable plotly on the page
authors:
    - Adrian Letchford
categories:
    - mathematics
---

I'm a little embarrased to admit this, but I was recently in a quant interview and the interviewer quickly realised that I didn't know the Black-Scholes formula! That was definitely a moment when imposter syndrome became reality. To fix the situation, I've written up the Black-Scholes model here; being as succinct and practical as I can.

This write up deals with the ideas and mathematics behind the Black-Scholes model. I assume you know what an option contract is and you know the difference between a [call](https://en.wikipedia.org/wiki/Call_option) and a [put](https://en.wikipedia.org/wiki/Put_option) opition.

I hope this helps you as much as it helped me.

# Black-Scholes setup

The Black-Scholes model answers the question: what should the price of an option be?

The idea behind the Black-Scholes model is to perfectly hedge the option with a position in the underlying. Assuming the hedge breaks even, you can solve for the option's price giving us the Black-Scholes formula.

There are a few different [types of option contracts](https://en.wikipedia.org/wiki/Option_style). Black-Scholes handles European style options. We'll start with a **European** style **call** option on a stock that **does not pay** dividends.

A European call option gives the purcahser the right to buy a fixed number of shares at a fixed price on a fixed dated.

If you were to buy a call option, and the price of the stock goes up, then the value of the call option also goes up. To hedge this call option so that your total position value doesn't change you would need to sell some number of the stock. We can write such a portfolio like this:
$$
\Pi = V - \Delta S
$$
Where \\( V \\) is the price of the option, \\( S \\) is the price of the stock and \\( \Delta \\) is the number of shares we've sold to hedge the position.

For this portfolio to be "correctly hedged" we do not want it's value to change if the price of the option changes or the price of the stock changes. However, we do need to take the time value of money into account. A given amount of value ought to increase over time by the risk free rate. This means that our correctly hedged portfolio should increase in value by the risk free rate. We write this as:
$$
d\Pi = r \Pi dt = dV - \Delta dS
$$
where \\( r\\) is the risk free rate and \\( dt\\) is the amount of time that passes. Since we know the value of the portfolio \\( \Pi \\):
$$
\begin{align}
r (V - \Delta S) dt = dV - \Delta dS \label{1} \tag{1}
\end{align}
$$


The intuition here is that the return on the total position (option + stock) should be equal to the risk free rate. This is what we call an arbitrage free assumption.

The full list of assumptions requried to solve for \\(V\\) are:

1. There exists a risk free asset with a fixed rate of return for the life of the option.
1. The market is arbitrage free.
1. You can borrow and lend any amount of money at the risk free rate.
1. You can buy or sell any amount of the stock. This includes short selling and fractional amounts.
1. There are no transaction costs (no bid-ask spread or commissions).
1. The instantaneous log return of the underlying's price is a geometric Brownian motion with constant drift and volatility.

# Deriving the Black-Scholes equation

We're assuming that the price of the stock \\( S \\) follows a [geometric Brownian motion](https://en.wikipedia.org/wiki/Geometric_Brownian_motion) process with a fixed drift and volatility. This means that the logged prices are a Brownian motion. We write this as:
$$
dS = \mu S dt + \sigma S d W
$$
where \\( \mu \\) is the mean of the stock's returns, \\( \sigma \\) is the volatility and \\(dW\\) is the change in a Brownian motion.

As for \\( dV \\), all we know at the moment is that \\( V \\) is a function of \\( S \\) (the price of the stock) and \\( t \\) (time). As luck would have it, very smart people have already figured out what a function of a stochastic process and time looks like. Itô's lemma tells us [^ito]. Deriving Itô's lemma takes a bit of time. But all it says is that such a function can be expanded in a similar way to a Taylor series where the higher order terms are zero.

Applying this expansion to \\( dV \\) we get:
$$
dV = \frac{\delta V}{\delta t} dt + \frac{\delta V}{\delta S} dS + \frac{1}{2} \sigma^2S^2\frac{\delta^2 V}{\delta S^2} dt
$$

Substituting this into Eq. (\\( \ref{1} \\)) gives us:
$$
r (V - \Delta S) dt = \left(\frac{\delta V}{\delta t} + \frac{1}{2} \sigma^2S^2\frac{\delta^2 V}{\delta S^2} \right) dt + \left(\frac{\delta V}{\delta S} - \Delta \right) dS
$$

If we want the number of shares sold \\( \Delta \\) to exactly offset any change in \\( V \\), then we can imply that:
$$
\Delta = \frac{\delta V}{\delta S}
$$
substituting this in and cancelling out the \\( dt \\) term we get:
$$
\begin{align}
r \left( V - \frac{\delta V}{\delta S} S \right) = \frac{\delta V}{\delta t} + \frac{1}{2} \sigma^2S^2\frac{\delta^2 V}{\delta S^2} \label{2}\tag{2}
\end{align}
$$

This is the famous Black-Scholes equation. You can substitute \\( V \\) for a \\( C \\) in the case of a call option or \\( P \\) for a put option.

Before looking at the solution to this equation, it is helpful to get an appreciation for what it is saying.

# Intuition behind the equation

Equation (\\(\ref{2}\\)) looks pretty intimidating. The left hand side is just the amount of money we should make if we just invested in the risk free asset. The left right hand side says that this should equal the amount of money gained/lossed over time plus any money gained/lossed by the price movement in the stock.

<todo>graphic highlighting each bit in the equation</todo>


# Solving the Black-Scholes equation

The solution to the Black-Scholes equation above (\\(\ref{2}\\)) gives us the Black-Scholes model[^Natenberg2015]:
$$
\begin{align}
C &= S \mathcal{N}(d_1) - E e^{-rt} \mathcal{N}(d_2) \label{3}\tag{3} \\\
\\\
d_1 &= \frac{\log(S/E) + [r + (\sigma^2 / 2)]t}{\sigma \sqrt{t}} \\\
d_2 &= d_1 - \sigma \sqrt{t}
\end{align}
$$
where:
- \\( C = \\) the theoretical value of a **European** style **call** option.
- \\( S = \\) the price of a stock that **does not pay** dividends.
- \\( E = \\) the exercise price.
- \\( t = \\) the time to expiration in years.
- \\( \sigma = \\) the annual standard deviation of the stock price returns.
- \\( r = \\) the annual risk free rate.
- \\( \mathcal{N} = \\) the cumulative normal distribution function.

## Put options

Let's now price a **European** style **put** option on a stock that **does not pay** dividends.

A European put option gives the purcahser the right to sell a fixed number of shares at a fixed price on a fixed dated.

To find the price of a put option, we need to use an idea call **put-call parity.**

Imagine you purchased a call option and sold a put option at the same strike. If the stock is above the strike at expiration, you would exercise the call option buying the shares at the strike. If the stock is below the strike, the purchaser will exercise resulting in you buying the shares at the strike. This portfolio of a long call option and a short put option is equivalent to a single forward contract at the same strike and expiry (assuming no arbitrage).

This means that the value of the call minus the value of the put must equal the present value of the forward price of the underlying minus the exercise price
$$
C - P = \frac{Se^{rt} - E}{e^{rt}} = S - Ee^{-rt}
$$
<todo>I don't intuitively understand this 👆🏻</todo>

This equivalence is what put-call parity refers to.

If we take the put-call parity and plug in the price of a call option, we get:
$$
\begin{align}
C - P &= S - Ee^{-rt} \\\
P &= C - S + Ee^{-rt} \\\
P &= S \mathcal{N}(d_1) - E e^{-rt} \mathcal{N}(d_2) - S + Ee^{-rt} \\\
P &=E e^{-rt} (1 - \mathcal{N}(d_2)) - S (1 - \mathcal{N}(d_1)) \label{4}\tag{4} \\\
\end{align}
$$

<todo>This has an interpretation section that might be good: http://www.timworrall.com/fin-40008/bscholes.pdf</todo>


# Options with Dividends

So far, we've only been looking at options on stocks that do not pay a dividend. Let's now price a **European** style **call** option on a stock that **does pay** dividends.

When considering dividends, rather than account for a fixed amount, we work with the dividend yield. Let's say that the stock pays a dividend equal to the yield \\( y \\). This means that over the time \\( dt \\), \\(y S dt \\) dividends are received. All else being equal, that the stock will decrease in value by \\(y S dt \\). The stochastic model for the stock becomes:
$$
dS = \mu S dt + \sigma S d W - y S dt = (\mu - y) S dt + \sigma S d W
$$

Similarly to before, we construct a portfolio \\( \Pi = \Delta S - V \\) where \\( V \\) is the price of the option and \\( \Delta \\) will be picked so that the option is perfectly hedged. Because the stock pays a dividend, this portfolio will increase in value by  \\( \Delta y S dt \\):
$$
d\Pi = \Delta dS - dV + \Delta y S dt
$$

If we replace \\( dV \\) with it's expansion from Itô's lemma:
$$
d\Pi = \Delta dS - \frac{\delta V}{\delta t} dt - \frac{\delta V}{\delta S} dS - \frac{1}{2} \sigma^2S^2\frac{\delta^2 V}{\delta S^2} dt + \Delta y S dt
$$

As before, if we pick \\( \Delta = \frac{\delta V}{\delta S} \\) to exactly offset any change in \\( V \\):
$$
d\Pi = - \frac{\delta V}{\delta t} dt - \frac{1}{2} \sigma^2S^2\frac{\delta^2 V}{\delta S^2} dt + \frac{\delta V}{\delta S} y S dt
$$

Since the value of the portfolio is risk free we have:
$$
d\Pi = r \Pi dt = r(\frac{\delta V}{\delta S} S - V)dt = - \frac{\delta V}{\delta t} dt - \frac{1}{2} \sigma^2S^2\frac{\delta^2 V}{\delta S^2} dt + \frac{\delta V}{\delta S} y S dt
$$
rearranging:
$$
\frac{\delta V}{\delta t} + \frac{1}{2} \sigma^2S^2\frac{\delta^2 V}{\delta S^2} + (r - y)\frac{\delta V}{\delta S} S  - rV = 0
$$

We can see tha that this equation is nearly identical to the Black-Scholes equation except that the risk free rate on the value of the stock has been offset by the dividend.

The value of a call option on a stock with dividends can be calculated with:
$$
\begin{align}
C &= e^{-yt}S \mathcal{N}(d_1) - E e^{-rt} \mathcal{N}(d_2) \\\
\\\
d_1 &= \frac{\log(S/E) + [r - y + (\sigma^2 / 2)]t}{\sigma \sqrt{t}} \\\
d_2 &= d_1 - \sigma \sqrt{t}
\end{align}
$$
This has been modified so that the forward value of the stock is adjusted by the dividend yeild.

# Implied Volatility

The inputs to the option model above are the underlying price, the exercise price, the time to expiration, the risk free rate and the expected volatility. All of these quantities are known except for the expected volatility.

In fact, if you look at market prices, then the current option price is also known. If you were to plug in the current option price and solve for the volatility \\( \sigma \\) you would get the market's expectation for volatility. This is called implied volatility---the volatility implied by the market price.

Unfortunately, there is no closed form solution for the implied volatility. However, the Black-Scholes model shows us that the price of an option is monotonic in \\( \sigma \\). This means that we can use any root finding method to solve for \\( \sigma \\) in:
$$
f(\sigma, S, E, t, r) - C = 0
$$
where \\( f(\cdot) \\) is the theoretical of a call option.

# Greeks

The "greeks" are a set of derivatives measuring how sensitive an option is to underlying changes. These values are used in hedging. A trader is able to isolate the various factors that impact an option's price and hedge away those risks.

## Delta

We've met delta before. Delta \\( \Delta \\) measures the change of the option value with respect to the underlying's price:
$$
\Delta = \frac{\delta V}{\delta S}
$$

For a long call option, delta will be between 0.0 and 1.0. Delta will be zero if the option is far out of the money or one if deep in the money. This is the same for a short put positions. For both a long put and short call, delta is between 0.0 and -1.0.

THe total delta of a portfolio of options on the same underlying can be calculated by summing the deltas for each individual option.

The delta of the underlying is always one, so a trader could "delta-hedge" a portfolio of options by buying or shorting the number of shares indicated by the sum of deltas.

We learned before that a long call and a short put position at the same strike is equilavent to a forward contract. A forward contract has a delta of one. We can then say that the delta of a call minus the delta of a put equals one:
$$
\Delta C - \Delta P = 1
$$

## Vega

Vega[^vega] \\( \mathcal{V} \\) measures the change in option value relative to changes in the underlying's volatility:
$$
\mathcal{V} = \frac{\delta V}{\delta \sigma}
$$

## Theta

Theta \\( \Theta \\) measures the change in option value relative to the passage of time:
$$
\Theta = \frac{\delta V}{\delta t}
$$
Theta is almost always negative for a long option position. This is known as "time decay". The value of the option decreases over time. For a short option position, theta is positive.

You can break down the value of an option into two parts, the intrinsic value and the time value. The intrinsic value of an option is the money you would make if you exercised the option imediately. So a call option whose strike is $10 below the current price has an intrisic value of $10. The remaining value of the option is the time value--the value in being able to wait to exercise.

## Rho
Rho \\( \rho \\) measures the change in option value relative to the changes in the risk free interest rate:
$$
\rho = \frac{\delta V}{\delta r}
$$
An option's value is least sensitive to the risk free rate making this a little used greek.

## Gamma
Gamma \\( \Gamma \\) measures the change in the delta with respect to the underlying's price:
$$
\Gamma = \frac{\delta \Delta}{\delta S} = \frac{\delta^2 V}{\delta S^2}
$$

For a long options position gamma is positive. This is true for both calls and puts.

Most options have opposite sign theta and gamma. So a long call has negative theta and positive gamma.

When a trader is delta hedging a portfolio, they may also try and get their net gamma position to zero. This ensures that the hedge remains effective over a larger range of price movements.

A positive gamma means you will benefit from price movements. A negative gamma means you will be hurt from price movements.


## Table of greeks

this includes annual dividend yield: https://en.wikipedia.org/wiki/Greeks_(finance)#Formulae_for_European_option_Greeks


|                      | Call                     | Put                          |
| ---------------------|--------------------------|------------------------------|
| Delta \\( \Delta \\) | \\( \mathcal{N}(d_1) \\) | \\( \mathcal{N}(d_1) - 1 \\) |
| Gamma \\( \Gamma \\) <td colspan=2> \\( \frac{ \mathcal{N}^{\prime}(d_1)}{S \sigma \sqrt{t}} \\) |
| Vega \\( \mathcal{V} \\) <td colspan=2> \\( S \mathcal{N}^{\prime}(d_1) \sqrt{t} \\) |
| Theta \\( \Theta \\) | \\( - \frac{S\mathcal{N}^{\prime}(d_1) \sigma}{2\sqrt{t}} - r E e^{-rt}\mathcal{N}(d_2) \\) | \\( - \frac{S\mathcal{N}^{\prime}(d_1) \sigma}{2\sqrt{t}} + r E e^{-rt}\mathcal{N}(-d_2) \\) |
| Rho \\( \rho \\) | \\( E t e^{-rt}\mathcal{N}(d_2) \\) | \\( -E t e^{-rt}\mathcal{N}(-d_2) \\) |

# Portfolio analysis

In a simplified world, traders who are trading stocks only have to worry about one risk, the price changing. However, as we've learned, the value of an option contract changes based on a number of different factors. These factors are mainly the price of the underlying, the expected volatility, the risk free rate and the passage of time.

The greeks above all measure an option's sensitivity to changes in each of these factors and are a way of measuring and tracking risks.

These risk measures (delta, gamma, theta, vega and rho) are all additive. This means that if you have a complex portfolio of call and put options at varying strikes and expiries (but the same underlying) then you can sum together their greeks to estimate the risks of the whole portfolio.

# Delta-hedging


Acknowledgements

The derivation of the Black-Scholes equation was inspired by [this article](https://www.linkedin.com/pulse/option-pricing-how-its-related-simulating-temperature-yafus-siddiqui/).

The derivation of dividends was inspired by [this article](https://www.math.tamu.edu/~mike.stecher/425/Sp12/optionsForDividendStocks.pdf).



[^ito]: Itô's lemma. If \\( X_t \\) is a stochastic process with infinitesimal variance \\(v(X_t)\\) and if \\( u(X_t, t) \\) is a function with enough derivatives then \\( u(X_t, t) \\) is another stochastic process that satisfies:
$$
d u(X_t, t) = \frac{\delta u(X_t, t)}{\delta t} dt + \frac{\delta u(X_t, t)}{\delta X_t} d X_t + \frac{1}{2} \frac{\delta^2 u(X_t, t)}{\delta X_t^2} v(X_t)dt
$$
A good explaination of Itô's lemma can be found [here](https://math.nyu.edu/~goodman/teaching/StochCalc2018/notes/Lesson4.pdf).

[^vega]: "Vega" is not a greek letter! [Wikipedia](https://en.wikipedia.org/wiki/Greeks_(finance)#Vega) suggests that it is a variation on the greek letter nu (\\( \nu \\)) which looks like a "v" and "ega" was added to the end to make it sound like the greek letters beta, eta and theta.

{{% citation
    id="Natenberg2015"
    author="Sheldon Natenberg"
    title="Option Volatility & Pricing: Advanced Trading Strategies and Techniques"
    year="2015"
    publisher="McGraw-Hill Education"
    link="https://www.mheducation.co.uk/option-volatility-and-pricing-advanced-trading-strategies-and-techniques-2nd-edition-9780071818773-emea"
%}}
