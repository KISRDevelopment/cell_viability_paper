function main()
{
    fetch('interpretation-example.json')
    .then(response => response.json())
    .then(data => plot_interpretation(data));
}

function plot_interpretation(data)
{
    const labels = data.labels.map((l) => l.replace('(Sum)', '').replace('(sum)', ''))

    var data = [
        {
          type: 'bar',
          orientation: 'h',
          x: data.components,
          y: labels,
          marker: {
            color: 'green'
          },
          name: 'expenses'
        }]
        var layout = {
            title: '<b>Logistic Regression Model Components</b>',

          height: 1500,
          width: 1000,
          autosize: true,
            font: {
              size: 22,
              
              color: '#7f7f7f'
            },
            yaxis: {
                showgrid: true,
                automargin: true,
                tickmode: 'linear',
            },
            xaxis: {
                showgrid: true,
                automargin: true,
                tickangle: 45
            }
          };

    Plotly.newPlot('myDiv', data, layout);
}

window.onload = main;
