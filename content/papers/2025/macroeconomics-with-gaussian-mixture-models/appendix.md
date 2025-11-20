# Appendix

## Creating long term prices

The dataset of prices is available [here](broad_etf_prices.csv).

### SPY

The price series for SPY was created by splicing together the prices for [SPY](https://stooq.com/q/?s=spy.us) and the prices for the [S&P 500 index (SPX)](https://stooq.com/q/?s=^spx). The prices were fetched from [Stooq](https://stooq.com/).

### GLD

The price series for GLD was created by splicing together the prices for [GLD](https://finance.yahoo.com/quote/GLD/), the price for the [front month gold futures contract](https://finance.yahoo.com/quote/GC=F/) and a [long term historical gold price futures series](https://stooq.com/q/?s=gc.f). The ETF prices and front month futures prices were fetched from [Yahoo Finance](https://finance.yahoo.com/) while the historical gold futures prices were fetched [Stooq](https://stooq.com/).

### GSG

The price series for GSG was created by splicing together the prices for [GSG](https://finance.yahoo.com/quote/GSG/) and the price for the Index of [S&P GSCI Commodity Total Return](https://www.investing.com/indices/sp-gsci-commodity-total-return). The ETF prices were fetched from [Yahoo Finance](https://finance.yahoo.com/) while the historical index prices were fetched from [Investing.com](https://www.investing.com/).

### TLT

The price series for TLT was created by building a synthetic index based on US treasuries downloaded from FRED and the [TLT](https://finance.yahoo.com/quote/TLT) prices. The treasury data was downloaded from [FRED](https://fred.stlouisfed.org/). The TLT prices were fetched from [Yahoo Finance](https://finance.yahoo.com/). The details of the synthetic index construction can be found in a previous article: [How to get 100 years of bond ETF prices]({{< ref "how-to-get-100-years-of-bond-prices" >}}).
