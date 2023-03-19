var spread = unpack(data, 'spread');
var dates = unpack(data, 'date');

var sliderSteps = [];
var steps = [1, 10, 20, 50, 100, 200, 500, 1000];
for (const i in steps) {
    sliderSteps.push({
        method: 'animate',
        label: steps[i],
        args: [[`spread_EWM(${steps[i]})`], {
            mode: 'immediate',
            transition: {duration: 300},
            frame: {duration: 300, redraw: false},
        }]
    });
}

var frames = [];
for (const i in steps) {
    frames.push({
    name: `spread_EWM(${steps[i]})`,
    data: [
        {
            x: dates,
            y: spread,
            line: {
                width: 3,
            },
            name: '30 year',
        },
        {
            x: dates,
            y: unpack(data, `spread_EWM(${steps[i]})`),
            line: {
                width: 3,
            },
            name: 'EWM',
        }
    ]
    });
}

var traces = [
    {
        x: dates,
        y: spread,
        mode: 'lines',
        line: {
            width: 3,
        },
        name: '30 year',
    },
    {
        x: dates,
        y: unpack(data, `spread_EWM(${1})`),
        mode: 'lines',
        line: {
            width: 3,
            simplify: false,
        },
        name: 'EWM',
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
    sliders: [{
        pad: {l: 50, t: 55, r: 50},
        currentvalue: {
            visible: false,
        },
        steps: sliderSteps
    }]
};