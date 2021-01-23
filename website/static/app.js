function main()
{
    attach_handlers();
}

function attach_handlers()
{
    const trs = document.querySelectorAll('tbody tr');
    trs.forEach((tr) => {
        tr.onclick = gi_selected;
    });

}

function gi_selected()
{
    document.querySelectorAll('tbody tr').forEach((tr) => tr.classList.remove('selected'));
    
    this.classList.add('selected');
    
    const giId = this.dataset.gi_id;
    fetch('/interpret/' + giId)
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
            color: '#6e0099'
          },
        }]
        var layout = {
            title: '<b>Logistic Regression Model Components</b>',

          height: 1500,
          width: 1000,
            font: {
              size: 22,
              
              color: 'black'
            },
            yaxis: {
                showgrid: true,
                automargin: true,
                tickmode: 'linear',
            },
            xaxis: {
                showgrid: true,
                automargin: true,
                range: [-3, 3]
            }
          };

    Plotly.newPlot('interpretation', data, layout);
}

window.onload = main;
