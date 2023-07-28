function _1(md){return(
md`# 2x2 matrix example`
)}

function _math(require){return(
require('mathjs@7.0.0/dist/math.js')
)}

function _sqt_cov_mat(math){return(
function sqt_cov_mat(matrix) {
  let eigs = math.eigs(matrix);
  let V = eigs.vectors;
  let D = math.diag(eigs.values);

  return math.multiply(V, math.multiply(math.sqrt(D), math.inv(V)))
}
)}

function _cov(std1,std2,rho){return(
[
  [std1**2, std1*std2*rho],
  [std1*std2*rho, std2**2],
]
)}

function _scov(sqt_cov_mat,cov){return(
sqt_cov_mat(cov)
)}

function _w(w1,w2){return(
[w1, w2]
)}

function _std(math,scov,w){return(
math.multiply(scov, w)
)}

function _portfolio_std(math,std){return(
math.sqrt(math.multiply(std, std))
)}

function _std1(Inputs,tex){return(
Inputs.range([0, 1], {label: tex`\sigma_1`, step: 0.1, value: 1.0})
)}

function _std2(Inputs,tex){return(
Inputs.range([0, 1], {label: tex`\sigma_2`, step: 0.1, value: 1.0})
)}

function _rho(Inputs,tex){return(
Inputs.range([-1, 1], {label: tex`\rho`, step: 0.1, value: 0.0})
)}

function _cov_matrix(tex,cov,scov,md){return(
md`${tex`
\begin{aligned}
\boldsymbol{\Sigma} &= \left[\begin{matrix}
${cov[0][0].toLocaleString()} & ${cov[0][1].toLocaleString()} \\
${cov[1][0].toLocaleString()} & ${cov[1][1].toLocaleString()} \\
\end{matrix}\right]\\
\sqrt{\boldsymbol{\Sigma}} &= \left[\begin{matrix}
${scov[0][0].toLocaleString()} & ${scov[0][1].toLocaleString()} \\
${scov[1][0].toLocaleString()} & ${scov[1][1].toLocaleString()} \\
\end{matrix}\right]
\end{aligned}
`}`
)}

function _w1(Inputs,tex){return(
Inputs.range([-1, 1], {label: tex`w_1`, step: 0.1, value: 1.0})
)}

function _w2(Inputs,tex){return(
Inputs.range([-1, 1], {label: tex`w_2`, step: 0.1, value: 0.0})
)}

function _w_vector(tex,w,md){return(
md`${tex`\boldsymbol{w} = \left[\begin{matrix}${w[0]}\\${w[1]}\end{matrix}\right] `}`
)}

function _component_std(tex,std,portfolio_std,md){return(
md`${tex`
\begin{aligned}
\sqrt{\boldsymbol{\Sigma}}\boldsymbol{w} &= \left[\begin{matrix}${std[0].toLocaleString()}\\${std[1].toLocaleString()}\end{matrix}\right] \\
\sigma = \sqrt{\sigma^2} &= ${portfolio_std.toLocaleString()}
\end{aligned}
`}`
)}

function _17(Plot,std){return(
Plot.barY([{x: '1', y: std[0]}, {x: '2', y: std[1]}], {x: "x", y: "y"}).plot()
)}

export default function define(runtime, observer) {
  const main = runtime.module();
  main.variable(observer()).define(["md"], _1);
  main.variable(observer("math")).define("math", ["require"], _math);
  main.variable(observer("sqt_cov_mat")).define("sqt_cov_mat", ["math"], _sqt_cov_mat);
  main.variable(observer("cov")).define("cov", ["std1","std2","rho"], _cov);
  main.variable(observer("scov")).define("scov", ["sqt_cov_mat","cov"], _scov);
  main.variable(observer("w")).define("w", ["w1","w2"], _w);
  main.variable(observer("std")).define("std", ["math","scov","w"], _std);
  main.variable(observer("portfolio_std")).define("portfolio_std", ["math","std"], _portfolio_std);
  main.variable(observer("viewof std1")).define("viewof std1", ["Inputs","tex"], _std1);
  main.variable(observer("std1")).define("std1", ["Generators", "viewof std1"], (G, _) => G.input(_));
  main.variable(observer("viewof std2")).define("viewof std2", ["Inputs","tex"], _std2);
  main.variable(observer("std2")).define("std2", ["Generators", "viewof std2"], (G, _) => G.input(_));
  main.variable(observer("viewof rho")).define("viewof rho", ["Inputs","tex"], _rho);
  main.variable(observer("rho")).define("rho", ["Generators", "viewof rho"], (G, _) => G.input(_));
  main.variable(observer("cov_matrix")).define("cov_matrix", ["tex","cov","scov","md"], _cov_matrix);
  main.variable(observer("viewof w1")).define("viewof w1", ["Inputs","tex"], _w1);
  main.variable(observer("w1")).define("w1", ["Generators", "viewof w1"], (G, _) => G.input(_));
  main.variable(observer("viewof w2")).define("viewof w2", ["Inputs","tex"], _w2);
  main.variable(observer("w2")).define("w2", ["Generators", "viewof w2"], (G, _) => G.input(_));
  main.variable(observer("w_vector")).define("w_vector", ["tex","w","md"], _w_vector);
  main.variable(observer("component_std")).define("component_std", ["tex","std","portfolio_std","md"], _component_std);
  main.variable(observer()).define(["Plot","std"], _17);
  return main;
}
