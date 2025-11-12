function _1(md){return(
md`# Mean reversion in government bonds`
)}

function _colors()
{
  return {
    longrate: "#4e79a7",
    shortrate: "#f28e2c",
    spread: "#e15759",
  };
}


async function _data(FileAttachment){return(
await FileAttachment("data.csv").csv({typed: true})
)}

function _trimmed_data(data,position){return(
data.filter((d) => d.date >= data.at(Math.min(-position, -1000))['date'])
)}

function _scaled_data(trimmed_data){return(
trimmed_data.map((row) => {
  return {
    ...row,
    TLT: row['TLT'] / trimmed_data.at(0)['TLT'],
    SHY: row['SHY'] / trimmed_data.at(0)['SHY'],
    IEF: row['IEF'] / trimmed_data.at(0)['IEF'],
  }
})
)}

function _point_estimates(scaled_data,position,d3,long_sigma,spread_speed,spread_mean,spread_std)
{
  let start_date = scaled_data.at(-position)['date'];
  let current_long = scaled_data.at(-position)['DGS30'];
  let current_short = scaled_data.at(-position)['DGS5'];
  let current_spread = scaled_data.at(-position)['spread'];
  let current_tlt = scaled_data.at(-position)['TLT'];
  let current_shy = scaled_data.at(-position)['SHY'];
  let horizon = 365;

  let estimate = d3.range(horizon).map((t) => {
    let long_mean = current_long;
    let long_var = long_sigma**2 * t;
    let long_std = Math.sqrt(long_var);
    
    let e_spread_mean = current_spread * Math.exp(-spread_speed * t) + spread_mean * (1 - Math.exp(-spread_speed * t));
    let e_spread_var = spread_std**2 * (1 - Math.exp(-spread_speed * t)) / (2 * spread_speed);
    let e_spread_std = Math.sqrt(e_spread_var);

    let short_mean = long_mean - e_spread_mean;
    let short_var = long_var + e_spread_var;
    let short_std = Math.sqrt(short_var);

    return {
      date: new Date(start_date.getTime() + t * 24*60*60000),
      long_mean: long_mean,
      long_upper: long_mean + long_std,
      long_lower: long_mean - long_std,

      spread_mean: e_spread_mean,
      spread_upper: e_spread_mean + e_spread_std,
      spread_lower: e_spread_mean - e_spread_std,

      short_mean: short_mean,
      short_upper: short_mean + short_std,
      short_lower: short_mean - short_std,
    }
  });

  return estimate;
}


function _7(md){return(
md`## Interest rates`
)}

function _interest_rates_plot(data,Plot,colors)
{
  let flatten = [["DGS30", data], ["DGS3", data]].flatMap(([symbol, data]) => data.map(d => ({
    symbol,
    date: d['date'],
    rate: d[symbol]
  })));
  
  return Plot.plot({
    width: 700,
    height: 300,
    background: 'red',
    style: {
      fontSize: "16px",
      fontFamily: "Source Sans Pro, Helvetica, Arial",
      backgroundColor: 'transparent',
    },
    y: {
      grid: true,
      tickFormat: ".0%",
      label: '↑ Interest rate',
      nice: true,
    },
    x: {
      grid: true,
      nice: true,
    },
    color: {
      legend: true,
      range: [colors['shortrate'], colors['longrate']],
    },
    marks: [
      Plot.line(flatten, {x: "date", y: "rate", stroke: "symbol"}),
    ],
    marginTop: 35,
  });
}


function _9(md){return(
md`# Interest rate model`
)}

function _interest_rate_model_plot(trimmed_data,Plot,colors,point_estimates,spread_mean,html)
{
  let flatten = [["DGS30", trimmed_data], ["DGS3", trimmed_data]].flatMap(([symbol, data]) => data.map(d => ({
    symbol,
    date: d['date'],
    rate: d[symbol]
  })));
  
  let rates = Plot.plot({
    width: 700,
    height: 300,
    background: 'red',
    style: {
      fontSize: "16px",
      fontFamily: "Source Sans Pro, Helvetica, Arial",
      backgroundColor: 'transparent',
    },
    y: {
      grid: true,
      tickFormat: ".0%",
      label: '↑ Interest rate',
      nice: true,
    },
    x: {
      grid: true,
      nice: true,
    },
    color: {
      legend: true,
      range: [colors['shortrate'], colors['longrate']],
    },
    marks: [
      Plot.line(flatten, {x: "date", y: "rate", stroke: "symbol"}),
      Plot.areaY(point_estimates, {x: "date", y1: "long_lower", y2: "long_upper", fill: colors['longrate'], fillOpacity: 0.3}),
      Plot.line(point_estimates, {x: "date", y: "long_mean", stroke: colors['longrate']}),
      Plot.areaY(point_estimates, {x: "date", y1: "short_lower", y2: "short_upper", fill: colors['shortrate'], fillOpacity: 0.3}),
      Plot.line(point_estimates, {x: "date", y: "short_mean", stroke: colors['shortrate']}),
    ],
    marginTop: 35,
  });
  
  let spread = Plot.plot({
    width: 700,
    height: 200,
    background: 'red',
    style: {
      fontSize: "16px",
      fontFamily: "Source Sans Pro, Helvetica, Arial",
      backgroundColor: 'transparent',
    },
    y: {
      grid: true,
      tickFormat: ".0%",
      label: '↑ Spread: DGS30 - DGS3',
      nice: true,
    },
    x: {
      grid: true,
      nice: true,
    },
    color: {
      legend: true,
    },
    marks: [
      Plot.line(trimmed_data, {x: "date", y: "spread", stroke: colors['spread']}),
      Plot.areaY(point_estimates, {x: "date", y1: "spread_lower", y2: "spread_upper", fill: colors['spread'], fillOpacity: 0.3}),
      Plot.line(point_estimates, {x: "date", y: "spread_mean", stroke: colors['spread']}),
      Plot.ruleY([spread_mean]),
    ],
    marginTop: 35,
  });

  return html`${[rates, spread]}`;
}


function _long_sigma(Inputs,tex){return(
Inputs.range([0, 0.002], {label: tex`\sigma_l`, step: 0.00001, value: 0.001})
)}

function _spread_mean(Inputs,tex){return(
Inputs.range([-0.005, 0.015], {label: tex`\mu_s`, step: 0.00001})
)}

function _spread_speed(Inputs,tex){return(
Inputs.range([0.00001, 0.02], {label: tex`\theta_s`, step: 0.00001})
)}

function _spread_std(Inputs,tex){return(
Inputs.range([0, 0.002], {label: tex`\sigma_s`, step: 0.00001})
)}

function _position(Inputs){return(
Inputs.range([1, 1000], {label: 'position', step: 1, value: 1})
)}

function _16(md){return(
md`# Parameters`
)}

function _flatten(){return(
function flatten(data, fields, rename={}) {
  let flat = fields.flatMap((field) => data.map(d => ({
    name: rename[field] || field,
    date: d['date'],
    value: d[field]
  })));

  let dropna = flat.filter(d => d['value'] !== null);

  return dropna;
}
)}

function _parameters_plot(Plot,colors,flatten,data,html)
{
  let shared = {
    width: 700,
    height: 250,
    style: {
      fontSize: "16px",
      fontFamily: "Source Sans Pro, Helvetica, Arial",
      backgroundColor: 'transparent',
    },
    x: {
      grid: true,
      nice: true,
    },
    marginLeft: 55,
  };
  

  let spread_plot = Plot.plot({
    ...shared,
    y: {
      grid: true,
      tickFormat: ".0%",
      label: null,
      nice: true,
    },
    color: {
      legend: true,
      range: [colors['spread'], 'black'],
    },
    marks: [
      Plot.line(
        flatten(data, ['spread', 'mu_spread'], {mu_spread: 'spread estimated mean'}),
        {x: "date", y: "value", stroke: "name"},
      ),
    ],
  });

  let variance_plot = Plot.plot({
    ...shared,
    y: {
      grid: true,
      tickFormat: ".0e",
      label: null,
      nice: true,
    },
    color: {
      legend: true,
      range: [colors['longrate'], colors['spread']],
    },
    marks: [
      Plot.line(
        flatten(data, ['sigma_long', 'sigma_spread'], {sigma_long: 'DGS30 estimated std', sigma_spread: 'spread estimated std'}),
        {x: "date", y: "value", stroke: "name"},
      ),
    ],
  });

  let theta_plot = Plot.plot({
    ...shared,
    y: {
      grid: true,
      tickFormat: ".3f",
      label: "↑ Theta",
      nice: true,
    },
    marks: [
      Plot.line(flatten(data, ['theta_spread'], {theta_spread: 'theta'}), {x: "date", y: "value", stroke: colors['spread']}),
    ],
    marginTop: 35,
  });

  return html`${[spread_plot, variance_plot, theta_plot]}`;
}


function _expected_return_plot(Plot,colors,flatten,data){return(
Plot.plot({
    width: 700,
    height: 300,
    background: 'red',
    style: {
      fontSize: "16px",
      fontFamily: "Source Sans Pro, Helvetica, Arial",
      backgroundColor: 'transparent',
    },
    y: {
      grid: true,
      tickFormat: ".0e",
      label: '↑ Expected return',
      nice: true,
    },
    x: {
      grid: true,
      nice: true,
    },
    color: {
      legend: true,
      range: [colors['shortrate'], colors['longrate']],
    },
    marks: [
      Plot.line(
        flatten(data, ['etf_long_mean', 'etf_short_mean'], {etf_long_mean: 'TLT', etf_short_mean: 'SHY'}),
        {x: "date", y: "value", stroke: "name"},
      ),
    ],
    marginTop: 35,
    marginLeft: 50,
  })
)}

function _trade_performance_plot(Plot,colors,flatten,data){return(
Plot.plot({
    width: 700,
    height: 300,
    background: 'red',
    style: {
      fontSize: "16px",
      fontFamily: "Source Sans Pro, Helvetica, Arial",
      backgroundColor: 'transparent',
    },
    y: {
      grid: true,
      tickFormat: ".0%",
      label: '↑ Equity',
      nice: true,
    },
    x: {
      grid: true,
      nice: true,
    },
    color: {
      legend: true,
      range: [colors['longrate'], colors['spread']],
    },
    marks: [
      Plot.line(
        flatten(data, ['trade_performance', 'base_performance'], {trade_performance: 'Model', base_performance: 'Equal weights'}),
        {x: "date", y: "value", stroke: "name"},
      ),
    ],
    marginTop: 35,
    marginLeft: 50,
  })
)}

export default function define(runtime, observer) {
  const main = runtime.module();
  function toString() { return this.url; }
  const fileAttachments = new Map([
    ["data.csv", {url: new URL("./data.csv", import.meta.url), mimeType: "text/csv", toString}]
  ]);
  main.builtin("FileAttachment", runtime.fileAttachments(name => fileAttachments.get(name)));
  main.variable(observer()).define(["md"], _1);
  main.variable(observer("colors")).define("colors", _colors);
  main.variable(observer("data")).define("data", ["FileAttachment"], _data);
  main.variable(observer("trimmed_data")).define("trimmed_data", ["data","position"], _trimmed_data);
  main.variable(observer("scaled_data")).define("scaled_data", ["trimmed_data"], _scaled_data);
  main.variable(observer("point_estimates")).define("point_estimates", ["scaled_data","position","d3","long_sigma","spread_speed","spread_mean","spread_std"], _point_estimates);
  main.variable(observer()).define(["md"], _7);
  main.variable(observer("interest_rates_plot")).define("interest_rates_plot", ["data","Plot","colors"], _interest_rates_plot);
  main.variable(observer()).define(["md"], _9);
  main.variable(observer("interest_rate_model_plot")).define("interest_rate_model_plot", ["trimmed_data","Plot","colors","point_estimates","spread_mean","html"], _interest_rate_model_plot);
  main.variable(observer("viewof long_sigma")).define("viewof long_sigma", ["Inputs","tex"], _long_sigma);
  main.variable(observer("long_sigma")).define("long_sigma", ["Generators", "viewof long_sigma"], (G, _) => G.input(_));
  main.variable(observer("viewof spread_mean")).define("viewof spread_mean", ["Inputs","tex"], _spread_mean);
  main.variable(observer("spread_mean")).define("spread_mean", ["Generators", "viewof spread_mean"], (G, _) => G.input(_));
  main.variable(observer("viewof spread_speed")).define("viewof spread_speed", ["Inputs","tex"], _spread_speed);
  main.variable(observer("spread_speed")).define("spread_speed", ["Generators", "viewof spread_speed"], (G, _) => G.input(_));
  main.variable(observer("viewof spread_std")).define("viewof spread_std", ["Inputs","tex"], _spread_std);
  main.variable(observer("spread_std")).define("spread_std", ["Generators", "viewof spread_std"], (G, _) => G.input(_));
  main.variable(observer("viewof position")).define("viewof position", ["Inputs"], _position);
  main.variable(observer("position")).define("position", ["Generators", "viewof position"], (G, _) => G.input(_));
  main.variable(observer()).define(["md"], _16);
  main.variable(observer("flatten")).define("flatten", _flatten);
  main.variable(observer("parameters_plot")).define("parameters_plot", ["Plot","colors","flatten","data","html"], _parameters_plot);
  main.variable(observer("expected_return_plot")).define("expected_return_plot", ["Plot","colors","flatten","data"], _expected_return_plot);
  main.variable(observer("trade_performance_plot")).define("trade_performance_plot", ["Plot","colors","flatten","data"], _trade_performance_plot);
  return main;
}
