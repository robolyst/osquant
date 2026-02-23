---
title: "Crudely Optimising an Oil Refinery"
summary: "
This article scratches the surface of how oil is used and traded within the world's energy complex. We introduce linear optimisation utilising AMPL to understand the optimal outputs from a ficticious refinery.
"

date: "2026-02-22"
type: paper
mathjax: true
hover_color: '#f5c875'
authors:
    - Daniel Nunns
categories:
    - finance
    - mathematics
---

Trading commodities can be done in various ways. One can use a black-box model to trade signals, another can speculate on fundamentals, and others participate in physical flows to hedge in markets. If you belong to the latter two categories, you generally want to model a system and optimise profit to understand necessary trades.

In this article, we'll explore basic fundamentals of the oil market and introduce [AMPL](https://ampl.com/) to optimise a fictitious distillation facility.

# Background

Running an oil operation is conceptually straightforward: you bring in crude oil, process it, and sell the resulting products to various markets. In the real world, however, complications arise from oil grades, regional regulations, and seasonal demand, among other factors.

Because oil and refined products form a large market, exchanges exist to trade them, allowing fair-values to be discovered, solving the supply-and-demand structure. How, then, as a refiner, do you optimise your facility to maximise profit given production constraints?

Crude oil comes in various 'grades', which refers to its specific gravity (light/medium/heavy) and sulfur content (sweet/sour). Essentially, these measures relate to the proportion of different distillates received when refining, with light-sweet usually attracting the highest premium given the low sulfur content (less corrosive on equipment) and lighter hydrocarbons (yielding higher value products like diesel and gasoline).

To better understand the grades of oil and how they're used, [*Oil 101*](https://www.goodreads.com/book/show/6377613-oil-101) by Morgan Downey gives a great overview of oil, beginning with its history and explaining various stages of production, consumption, and how the markets operate. To gain insight into the various oil producing regions, the [Platts Periodic Table of Oil](https://www.spglobal.com/commodity-insights/en/news-research/infographics/content-design-infographics/platts-periodic-table-of-oil) provides an interactive infographic which also describes the type of crude oil from each region.

There is a lot of theory we could explore; instead, this article highlights the important concepts required to optimise the refinery.

## Fractional Distillation

One form of refining (i.e. processing oil into higher value products) is 'fractional distillation', whereby a fractioning column (think 'big tank') is heated with a particular grade of oil inside. Distillates are grouped together, predominantly on the lengths of their hydrocarbon chains, and these lengths are separated in the column with lighter hydrocarbons rising to the surface. The 'fractioning' process is then taking the hydrocarbons at various levels to create the refined products within specification.

Because hydrocarbons don't separate discretely - they form a continuous gradient by chain length - one must choose how much to fractionate at each level to create various products. This becomes important when the relative pricing of two similar fuels changes over time, as it forms the crux of refining optimisation.

For more on fractional distillation, see *Oil 101*; the theory is also widely available online.

## Distillate Products

Once crude has been refined, lighter hydrocarbon products - like fuels - are usually more volatile, harder to store, and may cost more to produce because of required additives. Consequently, gasoline is often produced in line with demand, so storage typically plays a smaller role in production optimisation.

# The Crude & Distillates Market

We will be taking crude oil and producing two fuels: heating oil and gasoline. We also wish to understand how much of each we'll be producing over the next 24 months to ensure we're operationally efficient into the future, and can hedge our exposure to prices of the crude and distillates.

Because crude oil and its distillates trade on a liquid exchange (CME) as futures, we have good fair-valuations of these products per-unit over the next two years.

The data can be loaded with the code below. The data used can be [downloaded here](data/commodity_futures_prices.csv), or freshly obtained following the [instructions](#obtaining-futures-data).

```python
import pandas as pd

"""
CL: Crude Oil (USD/bbl)
HO: Heating Oil (USD/gallon)
RB: RBOB Gasoline (USD/gallon)
"""

df = pd.read_csv(
    "commodity_futures_prices.csv",
    index_col=0,
    parse_dates=True,
)
```

Plotting their current valuations shows the following:

{{< figure src="img/product_prices.svg" title="Normalised Futures Prices" >}}
For the given mark date, the prices of crude oil, heating oil and gasoline all normalised to USD per gallon.
{{< /figure >}}

Some characteristics are already apparent:

- The quoted units differ: crude oil is in USD per barrel (42 U.S. gallons per barrel), while distillates are in USD per U.S. gallon. The graph above normalised units to USD/gal.
- Crude oil prices are in **backwardation**, meaning the spot price is higher than prices for later delivery. The reverse situation is contango. See this [Investopedia guide](https://www.investopedia.com/articles/07/contango_backwardation.asp) for more detail. In this case, there is little incentive to store crude and refine it later.
- For the two distillates of interest, heating oil consistently attracts a higher premium.
- Gasoline shows a seasonal pattern, with higher prices in US summer months when driving increases.

# OSQ Fuels Optimisation

OSQ Fuels, a subsidiary of the Open Source Quant Group, has purchased a refinery.

We'll first define our refining problem with simple constraints to show how this translates into AMPL syntax.

Translating the linear model into AMPL is rather straightforward. The documentation [introduction](https://dev.ampl.com/ampl/introduction.html) covers the basics we apply here, namely parameters (`param`), sets (`set`), variables (`var`), objectives and constraints.

## Sets

The model is built around a single set, $T$, which represents a collection of **time points**. The number of these points is defined by the parameter $N$, which will be $18$, representing each month in our optimisation.

```ampl
# The number of months within the optimisation
param N integer > 0;

# Create the set (or array) of time points within the optimisation
set T = 1..N;
```

## Parameters

The direct inputs into the model will be:

- $P_{t}^{C,B}$: Crude oil price at time $t$ (USD/barrel).
- $P_{t}^{H}$: Heating oil price at time $t$ (USD/gallon).
- $P_{t}^{G}$: Gasoline price at time $t$ (USD/gallon).

To simplify calculations, the crude oil is converted to a per-gallon basis, where:

$$P_{t}^{C} = P_{t}^{C,B} \div 42$$

```ampl
param crude_price {T};        # Crude USD/bbl at time t
param heating_oil_price {T};  # Heating oil USD/gal at time t
param gasoline_price {T};     # Gasoline USD/gal at time t

# Refinery characteristics and economic constraints
# - Maximum barrels processed per month
param maximum_barrels_per_month >= 0;

# Conversions
param maximum_crude_per_month_gal = maximum_barrels_per_month * 42;
param crude_price_gal {t in T} = crude_price[t] / 42;
```

Note that the `maximum_barrels_per_month` is a single (time-invarient) number and includes a lower-bound constraint as a sanity check.

Conversions are also defined to normalise the units for the optimisation.

## Variables

Decision variables are defined internally within the model. These are the objects which AMPL adjusts in order to optimise the output based on the objective function.

The model's decision variables represent the quantities of crude oil to be processed and the quantities of various fuels to be produced at each time point.

- $I_{t}$: Gallons of crude oil used at time $t$.
- $O_{t}^{H}$: Gallons of heating oil production at time $t$.
- $O_{t}^{G}$: Gallons of gasoline production at time $t$.
- $O_{t}^{R}$: Gallons of residual material at time $t$.

```ampl
# Crude used at time t in gallons
var crude_used_gal {t in T} >= 0;

# Heating oil production at time t
var heating_oil_production {t in T} >= 0;

# Gasoline production at time t
var gasoline_production {t in T} >= 0;

# Residual at time t
var residual {t in T} >= 0;
```

## Objective

The objective is to maximise the total profit, which is calculated as the total revenue from selling refined fuels minus the total cost of purchasing and processing crude oil. This is summed over all time points $t$.

$$\max \quad \text{profit} = \sum_{t \in T} \left( P_{t}^{H} O_{t}^{H} + P_{t}^{G} O_{t}^{G} - P_{t}^{C} I_{t} \right)$$

```ampl
maximize profit: sum{t in T} (
    -1 * crude_price_gal[t] * crude_used_gal[t]
    + heating_oil_price[t] * heating_oil_production[t]
    + gasoline_price[t] * gasoline_production[t]
);
```

The residual is not sold — consider it wasted product.

## Constraints

The model includes several constraints to ensure the refinery's operations are realistic and adhere to specified production limits and ratios.

**1. Maximum Crude Usage:**

The amount of crude oil processed at each time point cannot exceed the maximum monthly capacity.

$$I_{t} \leq M^{G} \quad \forall t \in T$$

```ampl
subject to Maximum_Crude_Used {t in T}:
    crude_used_gal[t] <= maximum_crude_per_month_gal;
```

**2. Product Production Ratios:**

These constraints ensure that the production of specific fuels does not exceed a certain percentage of the total crude oil used. This is a simplified expression of the aforementioned 'fractioning'.

- Heating Oil: $O_{t}^{H} \leq 0.6 \times I_{t} \quad \forall t \in T$
- Gasoline: $O_{t}^{G} \leq 0.6 \times I_{t} \quad \forall t \in T$

```ampl
subject to Heating_Oil_Ratio {t in T}:
    heating_oil_production[t] <= 0.6 * crude_used_gal[t];

subject to Gasoline_Ratio {t in T}:
    gasoline_production[t] <= 0.6 * crude_used_gal[t];
```

**3. Combined Product Ratios:**

These constraints specify limits on combined production of certain fuel types. This is again a simplified expression of having to choose between fuels when they may contain similar hydrocarbon lengths.

- Heating Oil and Gasoline: $O_{t}^{H} + O_{t}^{G} \leq 0.9 \times I_{t} \quad \forall t \in T$

```ampl
subject to Gasoline_Heating_Oil_Ratio {t in T}:
    (gasoline_production[t] + heating_oil_production[t]
        <= 0.9 * crude_used_gal[t]);
```

**4. Material Balance:**

The total quantity of all refined products (heating oil, gasoline, and residual) cannot exceed the total crude oil used at each time point.

$$O_{t}^{H} + O_{t}^{G} + O_{t}^{R} \leq I_{t} \quad \forall t \in T$$

```ampl
subject to Total_Production {t in T}:
    (heating_oil_production[t] + gasoline_production[t] + residual[t]
        <= crude_used_gal[t]);
```

**5. Residual Balance:**

To ensure the residual is defined, constrain it as the leftover from production. Otherwise, given the material-balance constraint, the residual is ill-defined and the solver would likely leave it at zero.

$$ O_{t}^{R} \geq I_{t} - \left( O_{t}^{H} + O_{t}^{G} \right) \quad \forall t \in T$$

```ampl
subject to Residual_Definition {t in T}:
    (residual[t] >=
        crude_used_gal[t]
        - (heating_oil_production[t] + gasoline_production[t]));
```

The full AMPL definition can be [seen in the appendix](#full-ampl-definition).

## Results

To run the model, the following setup can be used whereby we create the optimisation interface, choose the solver ([CBC](https://github.com/coin-or/Cbc) in this case), load the parameters and solve. For instructions on installing AMPL, visit the [AMPL documentation](https://dev.ampl.com/ampl/python/modules.html).

Note the maximum processed barrels per month is set to 10. This is extremely low in real-world terms but makes the results easier to digest.

```python
from amplpy import AMPL

# Create an AMPL instance
ampl = AMPL()

# Set the solver to use
ampl.set_option("solver", "cbc")

ampl.eval(
    r"""
        FULL AMPL MODEL DEFINITION HERE
    """
)

ampl.get_parameter("N").set(len(df))
ampl.get_parameter("crude_price").set_values(df["CL"].values)
ampl.get_parameter("heating_oil_price").set_values(df["HO"].values)
ampl.get_parameter("gasoline_price").set_values(df["RB"].values)
ampl.get_parameter("maximum_barrels_per_month").set(10)

ampl.solve()
```

To extract the output variables as a series, you can use, for example:

```python
ampl.get_variable(
    "gasoline_production"
).get_values().to_pandas().reset_index(drop=True)
```

Plotting each optimised variable over time shows the facility's most profitable action.

{{< figure src="img/production_plan.svg" title="Optimal Facility Output" >}}
Given the futures contract prices, the facility output (in gallons) is continuously optimising for heating oil output.
{{< /figure >}}

A somewhat anticlimactic result, but it's exactly what we expect: heating oil had the higher premium, so the facility is optimised to produce it.

For a sanity check, we can confirm the volumes align (sum of outputs = input) and that the relative production ratios satisfy the constraints defined earlier.

## Introducing Processing Costs

As mentioned previously, sometimes additives need to be incorporated to distillate outputs either for stability, environmental or efficiency reasons. To model the per-gallon costs of production for heating oil and gasoline, two parameters will be introduced, $C^{H}$ and $C^{G}$.

```ampl
param heating_oil_production_cost >= 0;
param gasoline_production_cost >= 0;
```

Incorporating this into a new objective function is simple.

$$\max \quad \text{profit} = \sum_{t \in T} \left( \left( P_{t}^{H} - C^{H} \right) O_{t}^{H} + \left( P_{t}^{G} - C^{G} \right) O_{t}^{G} - P_{t}^{C} I_{t} \right)$$

```ampl
maximize profit: sum{t in T} (
    -1 * crude_price_gal[t] * crude_used_gal[t]
    + ((heating_oil_price[t] - heating_oil_production_cost)
           * heating_oil_production[t])
    + ((gasoline_price[t] - gasoline_production_cost)
           * gasoline_production[t])
);
```

If we set these costs to 0.40 USD/gal for heating oil, and 0.10 USD/gal for gasoline, and then plot the effective premium for the futures contracts, we see the most valuable output is changing over time.

{{< figure src="img/product_prices_adjusted.svg" title="Adjusted Fuel Premiums based on Processing Cost" >}}
Adjusting the futures price by the modelled cost, the effective sell price shows we should be optimising different fuels over time.
{{< /figure >}}

It's clear our facility should be adjusting the distillate offtake over time, based on which has the higher effective premium. Running the optimisation again provides the optimal facility result.

{{< figure src="img/production_plan_with_costs.svg" title="Optimal Production given Processing Costs" >}}
By incorporating production costs, the optimal facility production varies by the relative demand throughout the year.
{{< /figure >}}

Now, it's clear the facility should be swapping between the two distillates given the changing demand profiles over the year.

# Conclusion

This article should have provided some context on quantitative oil refining modelling which includes everything from understanding the physical context, through to implementing the likes of linear optimisation and interpreting the outputs.

If the model were to be extended for real-world context, some interesting features might include:

- Shipping charges (usage of pipelines/tankers to transport crude/products).
- Accounting for other distillate products which are produced (petroleum gas, napthta, paraffin, jet fuel, diesel, fuel oil, bitumen, etc.).
- Accounting for any environmental policies for certain regions (parts of the U.S. require certain fuel additives, for example).
- Expansion/reduction of hydrocarbon densities (based on the variable heat within the physical chamber), which means the model needs to account for volume adjustments.

A couple of much more in-depth examples of a refinery optimisation can be found on the AMPL website:

- [Oil refinery production optimization](https://ampl.com/colab/notebooks/oil-refinery-production-optimization.html)
- [Extra material: Refinery production and shadow pricing with CVXPY](https://ampl.com/mo-book/notebooks/05/refinery-production.html)

# Appendix

## Obtaining Futures Data

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

## Full AMPL Definition

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
