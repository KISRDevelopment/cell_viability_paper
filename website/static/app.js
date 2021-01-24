function main()
{
    init_canvas();
    attach_handlers();
}

function init_canvas()
{
    const canvas = document.getElementById('pointer');
    const parent = canvas.parentNode;
    canvas.width = parent.offsetWidth;
    canvas.height = parent.offsetHeight;
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
    
    const searchResultsCol = document.getElementById('searchResultsCol');
    const refBox = searchResultsCol.getBoundingClientRect();

    const canvas = document.getElementById('pointer');
    
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const rect = this.getBoundingClientRect();
    const middle = (rect.top + rect.bottom) / 2 - refBox.top;
    
    ctx.strokeStyle = "#949999";
    ctx.beginPath();
    ctx.moveTo(0, middle);
    ctx.lineTo(canvas.width, 0);
    ctx.stroke();
    
    ctx.beginPath();
    ctx.moveTo(0, middle);
    ctx.lineTo(canvas.width, canvas.height);
    ctx.stroke();

    const giId = this.dataset.gi_id;
    fetch('/interpret/' + giId)
    .then(response => response.json())
    .then(data => plot_interpretation(data));
}

function plot_interpretation(data)
{
    const labels = data.labels.map((l) => l.replace('(Sum)', '').replace('(sum)', ''))
    const colors = data.components.map((v) => v < 0 ? 'red' : 'blue');

    var data = [
        {
          type: 'bar',
          orientation: 'h',
          x: data.components,
          y: labels,
          marker: {
            color: colors
          },
        }]
        var layout = {
            title: '<b>Model Components</b>',

          height: 1500,
          width: 400,
            font: {
              size: 12,
              
              color: 'black'
            },
            yaxis: {
                showgrid: true,
                automargin: false,
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
