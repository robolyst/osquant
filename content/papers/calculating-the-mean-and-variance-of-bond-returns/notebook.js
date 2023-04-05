function _1(md){return(
md`# Taylor expansion of bond returns`
)}

function _R(){return(
function R(r1, f, steps) {
  return r1 / f * steps;
}
)}

function _Rprime(){return(
function Rprime(r1, p, T) {
  return - 1 / r1 * (1 - (1 + r1 / p)**(- p * T));
}
)}

function _Rprime2(Rprime){return(
function Rprime2(r1, p, T) {
  return - 2 * Rprime(r1, p, T) / r1 - 2 / r1 * p * T / (p + r1) * ((p + r1) / p)**(-p * T);
}
)}

function _C0(R,Rprime,Rprime2){return(
function C0(r1, f, p, T, steps) {
  return R(r1, f, steps) - Rprime(r1, p, T) * r1 + Rprime2(r1, p, T) * 0.5 * r1**2;
}
)}

function _C1(Rprime,Rprime2){return(
function C1(r1, p, T) {
  return Rprime(r1, p, T) - Rprime2(r1, p, T) * r1;
}
)}

function _C2(Rprime2){return(
function C2(r1, p, T) {
  return Rprime2(r1, p, T) * 0.5;
}
)}

function _etf_ret(){return(
function etf_ret(r1, r, f, p, T) {
  let compound = (1 + r / p)**(-p * T);
  return r1 / f + r1/r * (1 - compound) + compound - 1;
}
)}

function _etf_ret_d1(){return(
function etf_ret_d1(r1, f, p, T) {
  let compound = (1 + r1 / p)**(-p * T);
  return - 1 / r1 * (1 - compound);
}
)}

function _etf_ret_d2(){return(
function etf_ret_d2(r1, f, p, T) {
  let compound = (1 + r1 / p)**(-p * T);
  let blegh = 1 / r1 * (p * T) / (p + r1) * ((p + r1) / p)**(-p*T);
  return 2 * 1 / r1**2 * (1 - compound) - 2 * blegh;
}
)}

function _etf_ret_1(etf_ret_d1){return(
function etf_ret_1(r1, r, f, p, T) {
  return r1 / f + etf_ret_d1(r1, f, p, T) * (r - r1);
}
)}

function _etf_ret_2(etf_ret_d1,etf_ret_d2){return(
function etf_ret_2(r1, r, f, p, T) {
  return r1 / f + etf_ret_d1(r1, f, p, T) * (r - r1) + etf_ret_d2(r1, f, p, T) * 0.5 * (r - r1)**2;
}
)}

function _bond_function(d3,etf_ret,etf_ret_1,etf_ret_2)
{
  let delta = d3.range(-0.01, 0.011, 0.001);
  let r1 = 0.05;

  let data = delta.map((delta) => {
    return {
      x: delta,
      y: etf_ret(r1, r1 + delta, 260, 2, 30),
      y1: etf_ret_1(r1, r1 + delta, 260, 2, 30),
      y2: etf_ret_2(r1, r1 + delta, 260, 2, 30),
    }
  });

  return data;
}


function _etf_return_plot(Plot,bond_function){return(
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
      label: 'Estimated ETF return',
      nice: true,
    },
    x: {
      grid: true,
      label: null,
    },
    marks: [
      Plot.line(bond_function, {x: "x", y: "y", strokeWidth: 4}),
    ],
    marginTop: 35,
    marginLeft: 50,
    caption: "Shows the ETF return where the rate/yield is the previous rate plus the x axis.",
  })
)}

function _taylor_expansion_order_1_plot(Plot,bond_function){return(
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
      label: 'First order Taylor expansion',
      nice: true,
    },
    x: {
      grid: true,
      label: null,
      nice: true,
    },
    marks: [
      Plot.line(bond_function, {x: "x", y: "y", strokeWidth: 4}),
      Plot.line(bond_function, {x: "x", y: "y1", stroke: "red", strokeWidth: 3}),
    ],
    marginTop: 35,
    marginLeft: 50,
    caption: "The red line is the first order Taylor expansion.",
  })
)}

function _taylor_expansion_plot(Plot,bond_function){return(
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
      label: 'Second order Taylor expansion',
      nice: true,
    },
    x: {
      grid: true,
      label: null,
    },
    marks: [
      Plot.line(bond_function, {x: "x", y: "y", strokeWidth: 4}),
      Plot.line(bond_function, {x: "x", y: "y2", stroke: "red", strokeWidth: 3}),
    ],
    marginTop: 35,
    marginLeft: 50,
    caption: "The red line is the second order Taylor expansion.",
  })
)}

export default function define(runtime, observer) {
  const main = runtime.module();
  main.variable(observer()).define(["md"], _1);
  main.variable(observer("R")).define("R", _R);
  main.variable(observer("Rprime")).define("Rprime", _Rprime);
  main.variable(observer("Rprime2")).define("Rprime2", ["Rprime"], _Rprime2);
  main.variable(observer("C0")).define("C0", ["R","Rprime","Rprime2"], _C0);
  main.variable(observer("C1")).define("C1", ["Rprime","Rprime2"], _C1);
  main.variable(observer("C2")).define("C2", ["Rprime2"], _C2);
  main.variable(observer("etf_ret")).define("etf_ret", _etf_ret);
  main.variable(observer("etf_ret_d1")).define("etf_ret_d1", _etf_ret_d1);
  main.variable(observer("etf_ret_d2")).define("etf_ret_d2", _etf_ret_d2);
  main.variable(observer("etf_ret_1")).define("etf_ret_1", ["etf_ret_d1"], _etf_ret_1);
  main.variable(observer("etf_ret_2")).define("etf_ret_2", ["etf_ret_d1","etf_ret_d2"], _etf_ret_2);
  main.variable(observer("bond_function")).define("bond_function", ["d3","etf_ret","etf_ret_1","etf_ret_2"], _bond_function);
  main.variable(observer("etf_return_plot")).define("etf_return_plot", ["Plot","bond_function"], _etf_return_plot);
  main.variable(observer("taylor_expansion_order_1_plot")).define("taylor_expansion_order_1_plot", ["Plot","bond_function"], _taylor_expansion_order_1_plot);
  main.variable(observer("taylor_expansion_plot")).define("taylor_expansion_plot", ["Plot","bond_function"], _taylor_expansion_plot);
  return main;
}
