const PLOT_HEIGHT_PIX = 1500;
const PLOT_WIDTH_PX = 400;

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
    canvas.height = PLOT_HEIGHT_PIX;
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
    
    const grd = ctx.createLinearGradient(0, 0, canvas.width, 0);
    grd.addColorStop(0, "#f6e0ff");
    grd.addColorStop(1, "white");

    const region = new Path2D();
    region.moveTo(0, middle);
    region.lineTo(canvas.width, 0);
    region.lineTo(canvas.width, PLOT_HEIGHT_PIX);
    region.closePath();

    ctx.save();

    ctx.clip(region);

    ctx.fillStyle = grd;
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    ctx.restore();


    const giId = this.dataset.gi_id;
    fetch('/interpret/' + giId)
    .then(response => response.json())
    .then(data => plot_interpretation(data));
}

function plot_interpretation(data)
{
    const labels = data.labels.map((l) => l.replace('(Sum)', '').replace('(sum)', ''))
    const colors = data.components.map((v) => v < 0 ? 'red' : 'blue');

    const z = data.components.reduce((a, b) => a + b, 0);

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
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',

          height: PLOT_HEIGHT_PIX,
          width: PLOT_WIDTH_PX,
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
