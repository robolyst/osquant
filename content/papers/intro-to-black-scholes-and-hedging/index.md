---
title: "Intro to Black-Scholes and option hedging"
summary: "
A break down of how the black-scholes option pricing model works and how you can use it to hedge risks.
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

*I'm a little embarrased to admit this, but I was recently in a quant interview and the interviewer quickly realised that I didn't know the Black-Scholes formula by heart! That was definitely a moment when imposter syndrome became reality. To fix the situation, I've written up a piece here on the the Black-Scholes model and how to use it to hedge. I hope this helps you as much as it helps me.*

# Black-Scholes basics

The Black-Scholes model answers the question: what should the price of an option be?

The idea behind the Black-Scholes model is to perfectly hedge the option with a position in the underlying. If the hedge should break even, you can solve for the option's price giving us the Black-Scholes formula.

There are a few different types of option contracts. We'll start with the most basic one; a **European** style **call** option on a stock that **does not pay** dividends.

A European call option gives the purcahser the right to buy a fixed number of shares at a fixed price on a fixed dated.

If you were to buy a call option, and the price of the stock goes up, then the value of the call option also goes up. To hedge this call option so that your total position value doesn't change you would need to sell some number of the stock. We can write such a portfolio like this:
$$
\Pi = V - \Delta S
$$
Where \\( V \\) is the price of the option, \\( S \\) is the price of the stock and \\( \Delta \\) is the number of shares we've sold to hedge the position.

For this portfolio to be "correctly hedged" we do not want it's value to change if the price of the option changes or the price of the stock changes. However, this is not entirely true. We do need to take the time value of money into account. A given amount of value ought to increase over time by the risk free rate. This means that our correctly hedged portfolio should increase in value by the risk free rate. We write this as:
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

We're assuming that the price of the stock \\( S \\) follows a [geometric Brownian motion](https://en.wikipedia.org/wiki/Geometric_Brownian_motion) process with a fixed drift and volatility. This means that the logged prices are a Brownian motion. I find it a little easier to say that the changes in the logged prices are Gaussian with a fixed mean and variance. We write this as:
$$
dS = \mu S dt + \sigma S d W
$$
where \\( \mu \\) is the mean of the stock's returns, \\( \sigma \\) is the volatility and \\(dW\\) is the change in a Brownian motion (standard Gaussian).

As for \\( dV \\), all we know at the moment is that V is a function of the price of the stock \\( S \\) and time \\( t \\). As luck would have it, very smart people have already figured out what a function of a stochastic process and time looks like. Itô's lemma tells us [^ito]. Deriving this takes a bit of time, so we're just going to consider it magical and jump straight into the formula for \\( dV \\).

Using Itô's lemma we can break down \\( dV \\):
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
- \\( t = \\) time to expiration in years.
- \\( \sigma = \\) the annual standard deviation of the stock price returns.
- \\( r = \\) the annual risk free rate.
- \\( \mathcal{N} = \\) the cumulative normal distribution function.

## Types of options contracts

So far, we have only looked at a **European** style **call** option on a stock that **does not pay** dividends. An option can be either European or America, a call or a put and the underlying stock may or may not pay a dividend. This gives us 8 types of options:

| Style    | Side | Dividends |
|----------|------|-----------|
| European | call | No        |
| European | put  | No        |
| American | call | No        |
| American | put  | No        |
| European | call | Yes       |
| European | put  | Yes       |
| American | call | Yes       |
| American | put  | Yes       |

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







Acknowledgements

The derivation of the Black-Scholes equation was inspired by [this article](https://www.linkedin.com/pulse/option-pricing-how-its-related-simulating-temperature-yafus-siddiqui/).




[^ito]: Itô's lemma. If \\( X_t \\) is a stochastic process with infinitesimal variance \\(v(X_t)\\) and if \\( u(X_t, t) \\) is a function with enough derivatives then \\( u(X_t, t) \\) is another stochastic process that satisfies:
$$
d u(X_t, t) = \frac{\delta u(X_t, t)}{\delta t} dt + \frac{\delta u(X_t, t)}{\delta X_t} d X_t + \frac{1}{2} \frac{\delta^2 u(X_t, t)}{\delta X_t^2} v(X_t)dt
$$
A good explaination of Itô's lemma can be found [here](https://math.nyu.edu/~goodman/teaching/StochCalc2018/notes/Lesson4.pdf).

{{% citation
    id="Natenberg2015"
    author="Sheldon Natenberg"
    title="Option Volatility & Pricing: Advanced Trading Strategies and Techniques"
    year="2015"
    publisher="McGraw-Hill Education"
    link="https://www.mheducation.co.uk/option-volatility-and-pricing-advanced-trading-strategies-and-techniques-2nd-edition-9780071818773-emea"
%}}
