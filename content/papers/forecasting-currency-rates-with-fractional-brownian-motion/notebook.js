function _1(md){return(
md`# Fractional brownian motion for currency rate forecasting`
)}

function _math(require){return(
require('mathjs')
)}

function _3(md){return(
md`## Incremental cov`
)}

function _incremental_cov(math){return(
function incremental_cov(n, H) {
  return math.range(1, n+1).toArray().map((h) => {
    return {
      x: h,
      y: (h - 1)**(2*H) + (h + 1)**(2*H) - 2*(h)**(2*H),
      H: "H = " + H.toString(),
    }
  })
}
)}

function _cov_data(incremental_cov){return(
incremental_cov(20, 0.55).concat(incremental_cov(20, 0.5)).concat(incremental_cov(20, 0.45))
)}

function _incremental_cov_plot(Plot,cov_data){return(
Plot.plot({
    width: 700,
    height: 300,
    style: {
      fontSize: "16px",
      fontFamily: "Source Sans Pro, Helvetica, Arial",
      backgroundColor: 'transparent',
    },
    y: {
      grid: true,
      label: '↑ Covariance',
      nice: true,
    },
    x: {
      grid: true,
      nice: true,
      label: '→ h',
    },
    color: {
      legend: true,
      // range: [colors['shortrate'], colors['longrate']],
    },
    marks: [
      Plot.line(cov_data, {x: "x", y: "y", stroke: "H"}),
    ],
    marginTop: 35,
    marginLeft: 55,
    marginBottom: 45,
  })
)}

function _7(md){return(
md`# Weights`
)}

function _weights(FileAttachment){return(
FileAttachment("weights@1.csv").csv({typed: true})
)}

function _weights_plot(Plot,weights){return(
Plot.plot({
    width: 700,
    height: 300,
    style: {
      fontSize: "16px",
      fontFamily: "Source Sans Pro, Helvetica, Arial",
      backgroundColor: 'transparent',
    },
    y: {
      grid: true,
      label: '↑ Weight',
      nice: true,
      type: "log",
      tickFormat: ".2",
    },
    x: {
      grid: true,
      nice: true,
      label: '→ t',
    },
    marks: [
      Plot.line(weights, {x: "t", y: "0.45"}),
    ],
    marginTop: 35,
    marginLeft: 60,
    marginBottom: 45,
  caption: "The weight vector for predicting a fractional Gaussian motion process. Here, window=30 and the Hurst exponent = 0.45. The most recent prices have the highest weight which decays the further back the price is."
  })
)}

function _10(md){return(
md`# Capital`
)}

function _capital(FileAttachment){return(
FileAttachment("capital@3.csv").csv({typed: true})
)}

function _capital_plot(Plot,capital){return(
Plot.plot({
    width: 700,
    height: 300,
    style: {
      fontSize: "16px",
      fontFamily: "Source Sans Pro, Helvetica, Arial",
      backgroundColor: 'transparent',
    },
    y: {
      grid: true,
      label: '↑ Capital',
      nice: true,
      tickFormat: ".0%",
    },
    x: {
      grid: true,
      nice: true,
    },
    marks: [
      Plot.line(capital, {x: "time", y: "0"}),
    ],
    marginTop: 35,
    marginLeft: 60,
    marginBottom: 45,
  caption: "Example capital trading the mispricing identified by the fractional brownian motion. This is without transaction costs."
  })
)}

function _13(md){return(
md`## fBm example`
)}

function _fbms(FileAttachment){return(
FileAttachment("fbms@1.csv").csv({typed: true})
)}

function _H(Inputs,tex){return(
Inputs.range([0.1, 0.9], {label: tex`H`, step: 0.1, value: 0.5})
)}

function _fbms_plot(Plot,fbms,H){return(
Plot.plot({
    width: 700,
    height: 300,
    style: {
      fontSize: "16px",
      fontFamily: "Source Sans Pro, Helvetica, Arial",
      backgroundColor: 'transparent',
    },
    y: {
      label: '↑ Fractional Brownian motion',
    },
    x: {
      grid: true,
      nice: true,
    },
    marks: [
      Plot.line(fbms, {x: "x", y: H.toString()}),
      Plot.axisY([]),
    ],
    marginTop: 35,
    marginLeft: 10,
    marginBottom: 45,
  })
)}

export default function define(runtime, observer) {
  const main = runtime.module();
  function toString() { return this.url; }
  const fileAttachments = new Map([
    ["weights@1.csv", {url: new URL("./files/b055990362097548e1347a56c4cd9cb318d9f3829ce24b60cc1ba1df6b63f924ecd40508caf08e3b74b2cd6305634c9ee89c0daef06c0f92c5b1bb783129ebfe.csv", import.meta.url), mimeType: "text/csv", toString}],
    ["capital@3.csv", {url: new URL("./files/520f1bc96836fb8afef4f6ecdfeb653ea78d35c649010e8636d4b8ae7eee0ab2d8c4289598b1ecc1e27019c02c3f36424cede66ea4579b1d92007d33c64ff977.csv", import.meta.url), mimeType: "text/csv", toString}],
    ["fbms@1.csv", {url: new URL("./files/de940d080c9ef35c0f213e7531a07fb7e330a657f4f5e1e2b438aa74cb91b459ce90bd463b68054a9d0b703548af39487d621cb8e3993ee226604f53673a9b69.csv", import.meta.url), mimeType: "text/csv", toString}]
  ]);
  main.builtin("FileAttachment", runtime.fileAttachments(name => fileAttachments.get(name)));
  main.variable(observer()).define(["md"], _1);
  main.variable(observer("math")).define("math", ["require"], _math);
  main.variable(observer()).define(["md"], _3);
  main.variable(observer("incremental_cov")).define("incremental_cov", ["math"], _incremental_cov);
  main.variable(observer("cov_data")).define("cov_data", ["incremental_cov"], _cov_data);
  main.variable(observer("incremental_cov_plot")).define("incremental_cov_plot", ["Plot","cov_data"], _incremental_cov_plot);
  main.variable(observer()).define(["md"], _7);
  main.variable(observer("weights")).define("weights", ["FileAttachment"], _weights);
  main.variable(observer("weights_plot")).define("weights_plot", ["Plot","weights"], _weights_plot);
  main.variable(observer()).define(["md"], _10);
  main.variable(observer("capital")).define("capital", ["FileAttachment"], _capital);
  main.variable(observer("capital_plot")).define("capital_plot", ["Plot","capital"], _capital_plot);
  main.variable(observer()).define(["md"], _13);
  main.variable(observer("fbms")).define("fbms", ["FileAttachment"], _fbms);
  main.variable(observer("viewof H")).define("viewof H", ["Inputs","tex"], _H);
  main.variable(observer("H")).define("H", ["Generators", "viewof H"], (G, _) => G.input(_));
  main.variable(observer("fbms_plot")).define("fbms_plot", ["Plot","fbms","H"], _fbms_plot);
  return main;
}
