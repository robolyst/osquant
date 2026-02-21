---
title: "Entering Quantitative Commodity Fundamentals"
summary: "
This article explores using data to uncover latent streams of returns, otherwise known as factors. We jump into the world of rotations to make sense of the factors and show how to ensure stability when used for real-world trading.
"

date: "2026-02-21"
type: paper
mathjax: true
authors:
    - Daniel Nunns
categories:
    - finance
    - mathematics
---

# Entering Quantitative Commodity Fundamentals

Trading commodities can be done in various ways. One can use a black-box model to trade signals, another can be speculating on fundamentals, and other players have physical flow to hedge in markets. If you belong to one of the two latter categories, you generally want to model a system and optimise for a profit to understand necessary trades.

In this article, we'll be exploring some basic fundamentals about the oil market, and introducing [AMPL](https://ampl.com/) to optimise a ficticious distillation facility. Check the [appendix](#package-installation) for instructions on package installation.

## Background

Running an oil operation is conceptually rather straightforward. You bring in some crude oil, you process it, you get a bunch of products you can then sell on to various markets. However, as with anything in the real-world, there are complexities coming in the form of grades of oil, regulations in certain markets, and seasonality of product demand, to only name a few.

Because oil and its refined products are such a large market, naturally exchanges exist to trade these products, allowing them to find their fair values over time, solving the supply-and-demand structure. How then, as a refiner, will you optimise your facility to maximise profit given production constraints?

Crude oil comes in various 'grades', which refers to its specific gravity (light/medium/heavy) and sulfur content (sweet/sour). Essentially, these measures relate to the proportion of different distillates received when refining, with light-sweet usually attracting the highest premium given the low sulfur content (less corrosive on equipment) and lighter hydrocarbons (yielding higher value products like diesel and gasoline).

To better understand the grades of oil and how they're used, [*Oil 101*](https://www.goodreads.com/book/show/6377613-oil-101) by Morgan Downey gives a great overview of oil, beginning with its history and explaining various stages of production, consumption, and how the markets operate. To gain insight into the various oil producing regions, the [Platts Periodic Table of Oil](https://www.spglobal.com/commodity-insights/en/news-research/infographics/content-design-infographics/platts-periodic-table-of-oil) provides an interactive infographic which also describes the type of crude oil from each region.

There's a tonne of theory we could dive into. Instead, I'll be highlighting the important concepts required to optimise the refinery.

### Fractional Distillation

One form of distillation is 'fractional distillation', whereby a fractioning column is heated with a particular grade of oil inside. Distillates are grouped together, predominantly on the lengths of their hydrocarbon chains, and these lengths are separated in the column with lighter hydrocarbons rising to the surface. The 'fractioning' process is then taking the hydrocarbons at various levels to create the products within specification.

Because the hydrocarbons don't separate discretely, i.e. it's a continuous gradient of their length, one must choose how much at each level to fraction to create various products. This becomes important when the relative pricing of two similar fuels changes over time, as this forms the crux of the distillate optimisation.

For further information on fractional distillation, *Oil 101* covers many concepts, though the theory is widely available across the web.

### Distillate Products

Once crude has been refined, the lighter hydrocarbon products are usually more volatile, and thus less easy to store, or may cost more to produce with required additives. As such, gasoline is usually produced inline with the demand at any one time, i.e. we need not worry about optimising our production with respect to storage.

## Crude & Distillates

We will be taking crude oil and producing two fuels, heating oil and gasoline. We also wish to understand how much of each we'll be producing over the next 24 months to ensure we're operationally efficient into the future, and can hedge our exposure to prices of the crude and distillates.

Because our crude and distillates trade on a liquid exchange (CME) as futures, we have good fair valuations of these products per-unit over the next two years.

The data can be loaded in with the below. The data used can be [downloaded here](link), or otherwise can be obtained following the [instructions](#obtaining-futures-data).

```python
import pandas as pd

"""
CL: Crude Oil (USD/bbl)
HO: Heating Oil (USD/gallon)
RB: RBOB Gasoline (USD/gallon)
"""

df = pd.read_csv("commodity_futures_prices.csv", index_col=0, parse_dates=True)

df
```

Plotting their current valuation, we see the following...

<!-- TODO Expand  -->

```python
import plotly.graph_objs as go

fig = go.Figure()

fig.add_trace(
    go.Scatter(x=df.index, y=df["CL"], mode="lines", name="Crude Oil (USD/bbl)", line=dict(shape="hv")),
)
fig.add_trace(
    go.Scatter(x=df.index, y=df["HO"], mode="lines", name="Heating Oil (USD/gal)", line=dict(shape="hv"), yaxis="y2"),
)
fig.add_trace(
    go.Scatter(x=df.index, y=df["RB"], mode="lines", name="Gasoline (USD/gal)", line=dict(shape="hv"), yaxis="y2"),
)

fig.update_layout(
    title="Futures Contract Prices for Fuels (Next 18 Months)",
    xaxis_title="Date",
    yaxis_title="Crude Oil Price (USD/bbl)",
    yaxis2=dict(
        title="Product Prices (USD/gallon)",
        overlaying="y",
        side="right",
        showgrid=False,
        zeroline=False,
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.3,
        xanchor="center",
        x=0.5,
    ),
)
```

<!-- TODO Insert graph with everything, split y-axis for bbl vs gals -->

We can note some characteristics already:

- The quotation units are different. Crude oil is in USD per barrel (42 U.S. gallons per barrel), whilst the distillates are in USD per U.S. gallon.
- Crude oil prices are in **backwardation**, meaning the spot price (i.e. the price right now for oil) is higher than that of in the future. The reverse situation is known as contango. More about these phenomena can be understood on the [CME's guide](https://www.investopedia.com/articles/07/contango_backwardation.asp).
- For the two distillates of interest, we see heating oil consistently attracts a higher premium.
- The gasoline has a seasonal shape, attracting a higher premium in the U.S.-based summer months, where generally more driving occurs leading to a higher demand.

## OSQ Fuels

OSQ Fuels, a subsidiary of the Open Source Quant Group, has purchased a refinery.

We'll first define our refining problem with simple constraints to see how this translates into the AMPL syntax.

Translating the linear model into AMPL is rather straightforward given its syntax. The documentation [introduction](https://dev.ampl.com/ampl/introduction.html) goes over the basics which we'll be applying here, namely parameters (`param`), sets (`set`), variables (`var`), objectives and constraints.

### Sets

The model is built around a single set, $T$, which represents a collection of **time points**. The number of these points is defined by the parameter $N$, which will be $18$ for each month in our optimisation.

```ampl
# The number of months within the optimisation
param N integer > 0;

# Create the set (or array) of time points within the optimisation
set T = 1..N;
```

### Parameters

The direct inputs into the model will be:

- $T$: Set of time points, indexed by $t = 1, \dots, N$.
- $P^{C,B}_{t}$: Crude oil price at time $t$ (USD/barrel).
- $P^{H}_{t}$: Heating oil price at time $t$ (USD/gallon).
- $P^{G}_{t}$: Gasoline price at time $t$ (USD/gallon).

To simplify calculations, some parameters are converted to a per-gallon basis:

- $P^{C}_{t}$: Crude oil price at time $t$ (USD/gallon), where $P^{C}_{t} = P^{C,B}_{t} \div 42$.

```ampl
param crude_price {T};                 # Crude USD/bbl at time t
param heating_oil_price {T};           # Heating oil USD/gal at time t
param gasoline_price {T};              # Gasoline USD/gal at time t

# Refinery characteristics and economic constraints
param maximum_barrels_per_month >= 0;  # Maximum barrels processed per month

# Conversions
param maximum_crude_per_month_gal = maximum_barrels_per_month * 42;
param crude_price_gal {t in T} = crude_price[t] / 42;
```

Notice here the `maximum_barrels_per_month` is a single number (as it does not vary over time) and also includes a constraint, a sense check to make our model more foolproof.

Conversions are also defined to normalise the units for the optimisation.

### Variables

Decision variables are defined internally within the model. These are the objects which AMPL adjusts in order to optimise the output based on the obective function.

The model's decision variables represent the quantities of crude oil to be processed and the quantities of various fuels to be produced at each time point.

- $I_{t}$: Gallons of crude oil used at time $t$.
- $O^{H}_{t}$: Gallons of heating oil production at time $t$.
- $O^{G}_{t}$: Gallons of gasoline production at time $t$.
- $O^{R}_{t}$: Gallons of residual material at time $t$.

```ampl
var crude_used_gal {t in T} >= 0;          # Crude used at time t in gallons
var heating_oil_production {t in T} >= 0;  # Heating oil production at time t
var gasoline_production {t in T} >= 0;     # Gasoline production at time t
var residual {t in T} >= 0;                # Residual at time t
```

### Objective

The objective is to maximise the total profit, which is calculated as the total revenue from selling refined fuels minus the total cost of purchasing and processing crude oil. This is summed over all time points $t$.

$$\max \quad \text{profit} = \sum_{t \in T} \left( P^{H}_{t} O^{H}_{t} + P^{G}_{t} O^{G}_{t} - P^{C}_{t} I^{G}_{t} \right)$$

```ampl
maximize profit: sum{t in T} (-1 * crude_price_gal[t] * crude_used_gal[t]
                + heating_oil_price[t] * heating_oil_production[t]
                + gasoline_price[t] * gasoline_production[t]);
```

Note the residual is not sold. Consider this as wasted product.

### Constraints

The model includes several constraints to ensure the refinery's operations are realistic and adhere to specified production limits and ratios.

**1. Maximum Crude Usage:**

The amount of crude oil processed at each time point cannot exceed the maximum monthly capacity.

$$I^{G}_{t} \leq M^{G} \quad \forall t \in T$$

```ampl
subject to Maximum_Crude_Used {t in T}:
    crude_used_gal[t] <= maximum_crude_per_month_gal;
```

**2. Product Production Ratios:**

These constraints ensure that the production of specific fuels does not exceed a certain percentage of the total crude oil used. This is a simplified expression of the aforementioned 'fractioning'.

- Heating Oil: $O^{H}_{t} \leq 0.6 \times I^{G}_{t} \quad \forall t \in T$
- Gasoline: $O^{G}_{t} \leq 0.6 \times I^{G}_{t} \quad \forall t \in T$

```ampl
subject to Heating_Oil_Ratio {t in T}:
    heating_oil_production[t] <= 0.6 * crude_used_gal[t];

subject to Gasoline_Ratio {t in T}:
    gasoline_production[t] <= 0.6 * crude_used_gal[t];
```

**3. Combined Product Ratios:**

These constraints specify limits on the combined production of certain fuel types. This is again a simplified expression of having to choose between fuels when they may contain similar hydrocarbon lengths.

- Heating Oil and Gasoline: $O^{H}_{t} + O^{G}_{t} \leq 0.9 \times I^{G}_{t} \quad \forall t \in T$

```ampl
subject to Gasoline_Heating_Oil_Ratio {t in T}:
    gasoline_production[t] + heating_oil_production[t] <= 0.9 * crude_used_gal[t];
```

**4. Material Balance:**

The total quantity of all refined products (heating oil, gasoline, and residual) cannot exceed the total crude oil used at each time point.

$$O^{H}_{t} + O^{G}_{t} + O^{R}_{t} \leq I^{G}_{t} \quad \forall t \in T$$

```ampl
subject to Total_Production {t in T}:
    heating_oil_production[t] + gasoline_production[t] + residual[t] <= crude_used_gal[t];
```

**5. Residual Balance:**

More a technicality of the model, but for the residual to properly calculate so we can read the output, it needs to be constrained as the 'leftover' from production. Otherwise, given the above contraint for material balance, the residual is illdefined and the solver would likely leave this at zero.

$$ O^{R}_{t} \geq I^{G}_{t} - \left( O^{H}_{t} + O^{G}_{t} \right) \quad \forall t \in T$$

```ampl
subject to Residual_Definition {t in T}:
    residual[t] >= crude_used_gal[t] - (heating_oil_production[t] + gasoline_production[t]);
```

### Full AMPL Definition

<!-- TODO Make this a dropdown? -->

```ampl
# =================================== SETS ===================================

# The number of months within the optimisation
param N integer > 0;

# Create the set (or array) of time points within the optimisation
set T = 1..N;

# ================================ PARAMETERS ================================

param crude_price {T};                 # Crude USD/bbl at time t
param heating_oil_price {T};           # Heating oil USD/gal at time t
param gasoline_price {T};              # Gasoline USD/gal at time t

# Refinery characteristics and economic constraints
param maximum_barrels_per_month >= 0;  # Maximum barrels processed per month

# Conversions
param maximum_crude_per_month_gal = maximum_barrels_per_month * 42;
param crude_price_gal {t in T} = crude_price[t] / 42;

# ================================= VARIABLES =================================

var crude_used_gal {t in T} >= 0;          # Crude used at time t in gallons
var heating_oil_production {t in T} >= 0;  # Heating oil production at time t
var gasoline_production {t in T} >= 0;     # Gasoline production at time t
var residual {t in T} >= 0;                # Residual at time t

# ============================ OBJECTIVE FUNCTION =============================

maximize profit: sum{t in T} (-1 * crude_price_gal[t] * crude_used_gal[t]
                + heating_oil_price[t] * heating_oil_production[t]
                + gasoline_price[t] * gasoline_production[t]);

# ================================ CONSTRAINTS ================================

# Maximum production
subject to Maximum_Crude_Used {t in T}:
    crude_used_gal[t] <= maximum_crude_per_month_gal;

# Contrain the ratios of distillates
subject to Heating_Oil_Ratio {t in T}:
    heating_oil_production[t] <= 0.6 * crude_used_gal[t];

subject to Gasoline_Ratio {t in T}:
    gasoline_production[t] <= 0.6 * crude_used_gal[t];

subject to Gasoline_Heating_Oil_Ratio {t in T}:
    gasoline_production[t] + heating_oil_production[t] <= 0.9 * crude_used_gal[t];

# Ensure the total production does not exceed the crude used
subject to Total_Production {t in T}:
    heating_oil_production[t] + gasoline_production[t] + residual[t] <= crude_used_gal[t];

# Define the residual as the difference between crude used and total production
subject to Residual_Definition {t in T}:
    residual[t] >= crude_used_gal[t] - (heating_oil_production[t] + gasoline_production[t]);
```

### Results

<!-- TODO -->

To run this, assuming a lowly 10 barrels a month being processed...

```python
# Create an AMPL instance
ampl = AMPL()

# Set the solver to use
ampl.set_option("solver", "cbc")

ampl.eval(
    r"""
        ...
    """
)

ampl.get_parameter("N").set(len(df))
ampl.get_parameter("crude_price").set_values(df["CL"].values)
ampl.get_parameter("heating_oil_price").set_values(df["HO"].values)
ampl.get_parameter("gasoline_price").set_values(df["RB"].values)
ampl.get_parameter("maximum_barrels_per_month").set(10)

ampl.solve()
```

<!-- TODO Put production graph in -->

A somewhat anticlimatic output, it's exactly what we expect, the heating oil was always the higher premium fuel, so the facility is consistently optimised to produce that.

For a sanity check, we can confirm the volumes align (sum of outputs = input), and the relative ratios of each fuel produce align with the constraints we defined earlier.

### Introducing Processing Costs

As mentioned previously, sometimes additives need to be incorporated to distillate outputs either for stability, environmental or efficiency reasons. To model the per-gallon costs of production for heating oil and gasoline, two parameters will be introduced, $C^{H}$ and $C^{G}$.

```ampl
param heating_oil_production_cost >= 0;
param gasoline_production_cost >= 0;
```

Incorporating this into a new objective function is simple.

$$\max \quad \text{profit} = \sum_{t \in T} \left( \left( P^{H}_{t} - C^{H} \right) O^{H}_{t} + \left( P^{G}_{t} - C^{G} \right) O^{G}_{t} - P^{C}_{t} I^{G}_{t} \right)$$

```ampl
        maximize profit: sum{t in T} (-1 * crude_price_gal[t] * crude_used_gal[t]
                       + (heating_oil_price[t] - heating_oil_production_cost) * heating_oil_production[t]
                       + (gasoline_price[t] - gasoline_production_cost) * gasoline_production[t]);
```

<!-- TODO Put the new results in here -->

Now it's clear the facility should be swapping between the two distillates given the changing demand profiles over the year.

## Conclusion

This article should have provided some context on quantitative commodities modelling which includes everything from understanding the physical context, through to implementing the likes of linear optimisation and interpreting the outputs.

If the model were to be extended for real-world context, some interesting features might include:

- Shipping charges (usage of pipelines/tankers to transport crude/products).
- Accounting for other distillate products which are produced.
- Accounting for any environmental policies for certain regions (parts of the U.S. require certain fuel additives, for example).
- Expansion/reduction of hydrocarbon densities, which means the model needs to account for volume adjustments.

A couple of much more in-depth examples of a refinery optimisation can be found on the AMPL website:

- [Oil refinery production optimization](https://ampl.com/colab/notebooks/oil-refinery-production-optimization.html)
- [Extra material: Refinery production and shadow pricing with CVXPY](https://ampl.com/mo-book/notebooks/05/refinery-production.html)

## Appendix

### Obtaining Futures Data

```python
import yfinance as yf
from datetime import date
from dateutil.relativedelta import relativedelta
import holidays
import pandas as pd

MONTH_TO_CODE: dict[int, str] = {
    1: "F",
    2: "G",
    3: "H",
    4: "J",
    5: "K",
    6: "M",
    7: "N",
    8: "Q",
    9: "U",
    10: "V",
    11: "X",
    12: "Z",
}

CRUDE_TEMPLATE: str = "CL{month_code}{year_code}.NYM"  # USD/bbl
HEATING_OIL_TEMPLATE: str = "HO{month_code}{year_code}.NYM"  # USD/gal
RBOB_GASOLINE_TEMPLATE: str = "RB{month_code}{year_code}.NYM"  # USD/gal

HOLIDAY_CALENDAR: holidays.HolidayBase = holidays.financial_holidays("NYSE")  # same as CME

MONTHS_FORWARD_TO_QUERY: int = 18

crude_tickers: list[str] = []
heating_oil_tickers: list[str] = []
rbob_gasoline_tickers: list[str] = []

# Find the next delivery month
starting_month = date.today() + relativedelta(months=1, day=1)

for i in range(MONTHS_FORWARD_TO_QUERY):
    current_month = starting_month + relativedelta(months=i)
    month_code: str = MONTH_TO_CODE[current_month.month]
    year_code: str = str(current_month.year)[-2:]

    crude_tickers.append(CRUDE_TEMPLATE.format(month_code=month_code, year_code=year_code))
    heating_oil_tickers.append(HEATING_OIL_TEMPLATE.format(month_code=month_code, year_code=year_code))
    rbob_gasoline_tickers.append(RBOB_GASOLINE_TEMPLATE.format(month_code=month_code, year_code=year_code))

print(f"Crude tickers: {crude_tickers}")
print(f"Heating Oil tickers: {heating_oil_tickers}")
print(f"RBOB Gasoline tickers: {rbob_gasoline_tickers}")

# Find the T-1 date
query_date: date = HOLIDAY_CALENDAR.get_nth_working_day(date.today(), -1)
query_period: str = "1d"

t_minus_1_data = yf.Tickers(crude_tickers + heating_oil_tickers + rbob_gasoline_tickers).history(
    period=query_period, start=query_date
)

# Manipulate the data for the one futures snapshot where the index is the delivery month, and the columns represent the contracts of interests' prices.
close_data: pd.DataFrame = t_minus_1_data["Close"].T
close_data.columns = ["close"]
close_data["symbol"] = close_data.index.str[:2]
close_data["month_code"] = close_data.index.str[2]
close_data["year_code"] = close_data.index.str[3:5].astype(int)
close_data["month"] = close_data["month_code"].map({v: k for k, v in MONTH_TO_CODE.items()})
close_data["year"] = close_data["year_code"] + 2000
close_data["date"] = pd.to_datetime(close_data[["year", "month"]].assign(day=1))
df = close_data.pivot(columns="symbol", index="date", values="close").sort_index()

# Save to CSV
df.to_csv("commodity_futures_prices.csv")
```

## Package Installation

<!-- TODO -->

https://dev.ampl.com/ampl/python/modules.html
