---
title: "Visual style guide"
summary: "
    Lorem ipsum dolor sit amet, consectetur adipiscing elit. Integer odio neque, volutpat vel nunc
    ut. Duis maximus massa vitae libero imperdiet feugiat quis a sapien. Quisque sodales neque dui,
    a mollis justo porta eu. Nullam semper ipsum ac ante rhoncus, ac facilisis lacus posuere. Mauris
    pulvinar elementum ligula in mattis. Fusce rhoncus consequat lorem accumsan rhoncus.
"

date: "1900-01-01"
type: paper
katex: true
plotly: true
authors:
    - Adrian Letchford
categories:
    - Guide
draft: true
---

# Text styles

Duis maximus massa vitae libero imperdiet feugiat quis a sapien. Quisque sodales neque dui, a mollis justo porta eu. Nullam semper ipsum ac ante rhoncus, ac facilisis lacus posuere. Mauris pulvinar elementum ligula in mattis. Fusce rhoncus consequat lorem accumsan rhoncus.

## Links

Duis maximus massa vitae [libero](/#) imperdiet feugiat quis a sapien. Quisque sodales neque dui, a mollis justo porta eu. Nullam semper ipsum ac ante rhoncus, ac facilisis lacus posuere.Suspendisse sed [sapien placerat velit](/#) pulvinar condimentum. Curabitur lacinia id metus id hendrerit. Curabitur luctus enim id nibh ullamcorper fringilla. Aenean id molestie metus. Donec [vestibulum]/#) ipsum arcu, sed commodo lacus lacinia sed. Aenean vestibulum leo non condimentum vulputate.

## Bold text

Nulla ultrices gravida interdum. Nam nec est quam. In porta mi in mi facilisis, in suscipit nisl dictum. Aenean sapien nibh, **convallis eget faucibus** sit amet, efficitur blandit odio. Vestibulum libero orci, egestas sit amet risus et, luctus tempor diam.

## Lists

Duis maximus massa vitae libero imperdiet feugiat quis a sapien. Quisque sodales neque dui, a mollis justo porta eu. Nullam semper ipsum ac ante rhoncus, ac facilisis lacus posuere. Mauris pulvinar elementum ligula in mattis. Fusce rhoncus consequat lorem accumsan rhoncus.

* First item
* second item

Duis maximus massa vitae libero imperdiet feugiat quis a sapien. Quisque sodales neque dui, a mollis justo porta eu. Nullam semper ipsum ac ante rhoncus, ac facilisis lacus posuere. Mauris pulvinar elementum ligula in mattis. Fusce rhoncus consequat lorem accumsan rhoncus.

1. First item
2. Second item

# Callout

Nulla ultrices gravida interdum. Nam nec est quam. In porta mi in mi facilisis, in suscipit nisl dictum. Aenean sapien nibh, convallis eget faucibus sit amet, efficitur blandit odio. Vestibulum libero orci, egestas sit amet risus et, luctus tempor diam.

<callout>
Quisque mi justo, euismod ac leo nec, elementum eleifend purus.
</callout>

Curabitur pulvinar magna sit amet mattis semper. Nulla interdum nunc quis turpis iaculis finibus. Donec purus leo, aliquam at malesuada sit amet, elementum vitae quam. Quisque mi justo, euismod ac leo nec, elementum eleifend purus. Etiam ut ornare velit.

# Maths

## Inline maths

Curabitur pulvinar magna sit amet mattis semper. Nulla interdum \\(a = \sum x^2\\) nunc quis turpis iaculis finibus. Donec purus leo, aliquam at malesuada sit amet, elementum vitae quam. This is a financial number $100 ok and this is another $100. Curabitur pulvinar magna sit amet mattis semper.

## Math blocks

Curabitur pulvinar magna sit amet mattis semper. Nulla interdum nunc quis turpis iaculis finibus. Donec purus leo, aliquam at malesuada sit amet, elementum vitae quam. Quisque mi justo, euismod ac leo nec, elementum eleifend purus. Etiam ut ornare velit.

$$
\left( \frac{C}{p_{\text{now}}} +m \sum_{y=\text{now}}^{\text{now}+T} \frac{(1 + \phi)^y}{p_y} \right) p_{\text{now} + T} = A
$$

Curabitur pulvinar magna sit amet mattis semper. Nulla interdum nunc quis turpis iaculis finibus. Donec purus leo, aliquam at malesuada sit amet, elementum vitae quam. Quisque mi justo, euismod ac leo nec, elementum eleifend purus. Etiam ut ornare velit.

# Code

## Python

```python
for i in range(10):
    print(i**2)
```

## JSON

```json
{
    "something": 6,
}
```

# Table

| Investment   | Worst day return  |
|--------------|-------------------|
| Investment A | -0.6%             |
| Investment B | -0.9%             |

# Images

You can create an image with standard mardown:
![Example image](images/example_investments.svg)

You can use a Hugo shortcode to create a figure with a caption:
{{<figure src="images/example_investments.svg" title="Example figure." >}}
Curabitur pulvinar magna sit amet mattis semper. Nulla interdum nunc quis turpis iaculis finibus. Donec purus leo, aliquam at malesuada sit amet, elementum vitae quam. Quisque mi justo, euismod ac leo nec, elementum eleifend purus. Etiam ut ornare velit.
{{</figure>}}

## Pro tips

When creating charts use SVGs. These will render as crisp as possible on all devices. And, as OSQuant evolves, our style may change. SVGs are editable while other formats are not.

In python you can use [matplotlib](https://matplotlib.org/) to create charts. If you want to enhance them after you've generated the SVGs, [Figma](https://www.figma.com/) is a great and free tool.

If you want to go for that handwritten notebook style, we use [Excalidraw](https://excalidraw.com/) to create these with a consistent style.

# Interactive Plots

There is a shortcode `plotly` for creating Plotly.js plots from CSV files. Wherever possible, use this option for showing data. Plotting data with code as part of your papers means we can keep the style OS Quant consistent and up to date.

You can chart data from a CSV file:

{{<plotly id="tesla-stock-plot" data="data/TSLA.csv">}}
    var traces = [{
        x: unpack(data, 'Date'),
        close: unpack(data, 'Close'),
        high: unpack(data, 'High'),
        low: unpack(data, 'Low'),
        open: unpack(data, 'Open'),
        increasing: {line: {color: 'green'}, fillcolor:'green'},
        decreasing: {line: {color: 'red'}, fillcolor:'red'},
        type: 'candlestick',
    }];
    var layout = {
        margin: {
            l: 50,
            r: 0,
            b: 30,
            t: 35,
            pad: 10,
        },
        dragmode: 'zoom',
        showlegend: false,
        xaxis: {
            rangeslider: {
                visible: false
            },
        }
    };
{{</plotly>}}

Or hard code everything:
{{<plotly id="hard-code-example">}}
    var trace1 = {
        x: [1, 2, 3, 4],
        y: [10, 15, 13, 17],
        mode: 'markers',
        name: 'Name of Trace 1',
        marker: {
            color: 'rgb(219, 64, 82)',
            size: 12
        }
    };

    var trace2 = {
        x: [2, 3, 4, 5],
        y: [16, 5, 11, 9],
        mode: 'lines',
        name: 'Name of Trace 2',
        line: {
            color: 'rgb(55, 128, 191)',
            width: 3
        }
    };

    var trace3 = {
        x: [1, 2, 3, 4],
        y: [12, 9, 15, 12],
        mode: 'lines+markers',
        name: 'Name of Trace 3',
        marker: {
            color: 'rgb(128, 0, 128)',
            size: 8
        },
        line: {
            color: 'rgb(128, 0, 128)',
            width: 1
        }
    };

    var traces = [trace1, trace2, trace3];

    var layout = {
        margin: {
            l: 50,
            r: 0,
            b: 30,
            t: 35,
            pad: 10,
        },
    };
{{</plotly>}}

# Footnotes & references

Donec auctor lacus est, sit amet dapibus erat porta ac. Proin facilisis, dui quis lacinia convallis, eros nunc rhoncus orci, at tempor enim magna nec enim. Nulla facilisi [^Cavallo2012]. Etiam ut ex dignissim, porttitor augue ac, congue ex. Ut sit amet tellus gravida, venenatis nisl et, luctus eros. Nam massa lacus, ornare ac pulvinar eget, consequat vitae nisl. Lorem ipsum dolor sit amet, consectetur adipiscing elit [^2]. Integer ut ornare tellus.

Quisque dictum, odio a tristique pretium, velit sem egestas quam, eu ultrices [^Shiller2000] enim risus at leo. Nulla tortor ante, commodo et euismod consectetur, dictum ullamcorper nisi [^Vapnik1998].

[^Cavallo2012]: Cavallo, J. V., & Fitzsimons, G. M. (2012). [Goal competition, conflict, coordination, and completion: How intergoal dynamics affect self-regulation.](https://psycnet.apa.org/record/2011-26825-009) In H. Aarts & A. J. Elliot (Eds.), Frontiers of social psychology. Goal-directed behavior (p. 267–299). Psychology Press.

[^2]: If you type this phrase into Google you will get many hits. I do not know who said it first. There are many people using it without reference. I think I might have first read it in a <a href="https://www.forbes.com/sites/moneywisewomen/2012/10/26/you-can-have-anything-you-want/?sh=78b73db1d8e4" target="_blank">Forbes article</a>.

[^Shiller2000]: Economist Robert J. Shiller wrote a book called *Irrational Exuberance* published in 2000 showing hundreds of years worth of data. He demonstrates that housing prices do not always go up.

{{% citation
    id="Vapnik1998"
    author="Vapnik, Vladimir"
    title="Statistical learn theory"
    year="1998"
    publisher="Wiley & Sons"
    address="New York"
    link="https://www.wiley.com/en-gb/Statistical+Learning+Theory-p-9780471030034"
%}}
