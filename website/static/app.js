const PLOT_HEIGHT_PIX = 1500;
const PLOT_WIDTH_PX = 400;

function main()
{

    const plots = document.querySelectorAll('.js-interp');
  
    plots.forEach((p) => p.style.display = 'none');

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

    const tabLinks = document.querySelectorAll('.js-tab-link');
    const plots = document.querySelectorAll('.js-interp');
    
    tabLinks.forEach((tl) => {

      const targetId = tl.dataset.tab;
      console.log(targetId);

      tl.onclick = () => {
        plots.forEach((p) => p.style.display = 'none');
        tabLinks.forEach((e) => e.classList.remove('selected'));

        const targetPlot = document.getElementById(targetId);
        targetPlot.style.display = 'block';

        tl.classList.add('selected');
      }

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
    .then(data => interpret(data));
}

function interpret(data)
{
  plot_interpretation('gene_a_features', data.components.gene_a);
  plot_interpretation('gene_b_features', data.components.gene_b);
  plot_interpretation('joint_features', data.components.joint);
  plot_interpretation('z_features', data.components.z);
  populate_pubs('publications', data.pubs);
  //plots[0].style.display = 'block';
}

function populate_pubs(elmId, d)
{
    const elm = document.getElementById(elmId);

    elm.innerHTML = "";

    const ul = document.createElement('ul');
    elm.appendChild(ul);

    d.forEach((p) => {

        const li = document.createElement('li');
        ul.appendChild(li);
        
        li.innerHTML = p.identifier;
    });
}
function plot_interpretation(elm, d)
{
    let labels = d.labels;
    let features = d.features;

    const colors = features.map((v) => v < 0 ? 'red' : 'blue');

    var data = [
        {
          type: 'bar',
          orientation: 'h',
          x: features,
          y: labels,
          marker: {
            color: colors
          },
        }]
        var layout = {
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

    Plotly.newPlot(elm, data, layout);
}

window.onload = main;
