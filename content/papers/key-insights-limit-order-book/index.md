---
title: "Key insights: Imbalance in the order book"
summary: "
    I summarise key insights from a few papers studying the limit order book. You'l learn how to measure volume imblanace in the limit order book and how well it predicts price moves.
"

date: "2023-09-18"
type: paper
mathjax: true # Enable mathematics on the page
katex: false # Enable mathematics on the page
plotly: false  # Enable plotly on the page
authors:
    - Adrian Letchford
draft: true
categories:
    - finance
---

# Limit order book basics

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nulla at varius turpis. Ut nec purus efficitur, dictum urna eget, molestie ante. Vestibulum sed scelerisque mi. Cras sed lorem ac ante rhoncus egestas quis sed augue. Fusce eget venenatis felis, lobortis blandit felis. Nunc feugiat eu neque ac fringilla. Curabitur in felis facilisis, dignissim enim eget, posuere massa. Aliquam gravida ut mi in aliquet. Fusce convallis at tortor sodales lobortis. Fusce sem enim, cursus quis purus rutrum, tincidunt accumsan est. Sed egestas eleifend ligula, et scelerisque enim maximus sit amet. Fusce a purus a est tristique hendrerit. Donec scelerisque in dolor quis gravida. Pellentesque pretium nisi purus, sed feugiat ligula euismod a.

Mauris in iaculis risus. Nam auctor blandit velit. Nulla condimentum diam diam, vitae tincidunt velit bibendum id. Nullam aliquet eros ac tortor vestibulum, in blandit velit molestie. Maecenas varius luctus bibendum. Nulla facilisi. Pellentesque eleifend nibh laoreet nibh auctor, sed aliquam turpis sagittis. Donec ultrices, dui ut gravida bibendum, lacus odio hendrerit lacus, sit amet consequat arcu mi nec nisl. Sed laoreet nibh nisl, accumsan dignissim nisl consectetur port.

# Volume imbalance

The key idea when analysing the limit order book is to identify whether the market as a whole wants to buy more or sell more. This idea is call **volume imbalance**. If 

Volume imbalance at time \\(t\\) is defined as [^Cartea-2018] [^Lipton-2013] [^Cartea-2015]:
$$
\rho_t = \frac{V_t^b - V_t^a}{V_t^b + V_t^a}, \ \ -1 \leq \rho_t \leq 1
$$
where \\(V_t^b\\) is the volume at time \\(t\\) posted at the best bid and \\(V_t^a\\) the volume at the best ask. We can interpret a \\(\rho_t\\) close to 1 as signalling strong buying pressure and a \\(\rho_t\\) close to -1 as strong selling pressure. This only takes into account the volume posted at the best bid and ask, the L1 order book.

Lipton et. al. [^Lipton-2013] compared the price imbalance with the subsequent price move normalised by the spread. The following figure is copied from their paper:

{{<figure width="small" src="imbalance_vs_spread_move.png" title="Volume imbalance vs price move.">}}
This figure shows the bucketed volume imblanace vs the average future price move normalised by the spread. The dataset is the flow of orders for [Vodafone (VOD.L)](https://finance.yahoo.com/quote/VOD.L/) during the first quarter of 2012. There appears to be a linear relationship between the level 1 order imbalance and future price moves. However, on average, the future price move is within the bid-ask spread.
{{</figure>}}

The volume imbalance \\(\rho_t\\) can be segmented into three intervals for easy labelling:

| Interval                                         | Label      |
|--------------------------------------------------|------------|
| \\(-1 \geq \rho_t \gt -\frac{1}{3} \\)           | Sell-heavy |
| \\(-\frac{1}{3} \geq \rho_t \geq \frac{1}{3} \\) | Neutral    |
| \\(\frac{1}{3} \gt \rho_t \geq 1 \\)             | Buy-heavy  |

Key insighte from [^Cartea-2018]:

{{<figure src="volume_imbalance_performance_trans.png" title="Predictive power of volume imbalance." >}}
**(a)** The tick by tick order book for [Intel (INTC)](https://finance.yahoo.com/quote/INTC) was analysed over the period January 2014 to December 2014. For each arriving market order (MO), the volume imbalance was recorded and segmented with the number of ticks that the mid price changed by over the next 10 milliseconds. The chart shows the distribution of segments & mid price changes. We can see that when the order book is buy-heavy then positive changes are much more likely than negative changes and vice versa for a sell-heavy order book. **(b)** These results are replicated for [Oracle (ORCL)](https://finance.yahoo.com/quote/ORCL).
{{</figure>}}


<!-- An introduction to Limit Order Books -->
<!-- https://www.machow.ski/posts/2021-07-18-introduction-to-limit-order-books/#stop-order -->

{{% citation
    id="Cartea-2018"
    author=" Álvaro Cartea, Ryan Francis Donnelly, and Sebastian Jaimungal"
    title="Enhancing Trading Strategies with Order Book Signals"
    publication="Applied Mathematical Finance"
    year="2018"
    pages="1--35"
    volume="25"
    number="1"
    link="https://ora.ox.ac.uk/objects/uuid:006addde-3a03-4d75-89c1-04b59026e1c0/download_file?file_format=application%2Fpdf&safe_filename=Imbalance_AMF_resubmit.pdf&type_of_work=Journal+article"
%}}

{{% citation
    id="Lipton-2013"
    author="Alexander Lipton, Umberto Pesavento, and Michael G Sotiropoulos"
    title="Trade arrival dynamics and quote imbalance in a limit order book"
    publication="arXiv"
    year="2013"
    link="https://arxiv.org/pdf/1312.0514.pdf"
%}}

{{% citation
    id="Cartea-2015"
    author=" Álvaro Cartea, Sebastian Jaimungal, and J. Penalva"
    title="Algorithmic and high-frequency trading."
    publication="Cambridge University Press"
%}}
