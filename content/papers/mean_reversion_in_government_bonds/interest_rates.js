var long = unpack(data, 'DGS30');
var short = unpack(data, 'DGS5');
var dates = unpack(data, 'date');


var traces = [
    {
        x: dates,
        y: long,
        mode: 'lines',
        line: {
            width: 3,
        },
        name: '30 year',
    },
    {
        x: dates,
        y: short,
        mode: 'lines',
        line: {
            width: 3,
            simplify: false,
        },
        name: '5 year',
    },
];

var layout = {
    margin: {
        l: 50,
        r: 0,
        b: 50,
        t: 35,
        pad: 0,
    },
    dragmode: 'zoom',
    showlegend: false,
    xaxis: {
        rangeslider: {
            visible: false
        },
    },
    yaxis: {
        tickformat: ',.0%',
    },
};