
function main()
{
    const nostartups = document.querySelectorAll('.js-nostartup');
    
    const gi_details_modal = document.getElementById('gi_details_modal');
    gi_details_modal.querySelector('.js-close').onclick = () => {
        gi_details_modal.style.display = 'none';
        document.body.classList.remove('modal-open');
    };

    let curr_form = null;

    const paginator = new Paginator(
        document.querySelector('.js-search-results-paginator'), 50);


    const btnSearch = document.getElementById('btnSearch');
    btnSearch.onclick = function()
    {
        const form = gather_form();
        console.log(form)
        form['page'] = 0;

        call_api('./gi_pairs', form, function(res) {
            paginator.setRows(res.rows);
            curr_form = form;
            nostartups.forEach((e) => e.style.visibility = 'visible');
        });
    }
    const colors = ['#ffb700', '#0095ff', '#b2ff00', '#d000ff', '#7d8282'];
    create_tabs(document.getElementById('gi_details_modal'), colors);
}

function create_tabs(selectorElm, colors)
{
    const tabs = selectorElm.querySelectorAll('.js-tabs > div');
    let currentElm = null;
    tabs.forEach((tab, i) => {
        const targetId = tab.dataset.for;
        const targetElm = document.getElementById(targetId);

        targetElm.style.display = 'none';
        
        tab.onclick = function() {
            if (currentElm !== null)
                currentElm.style.display = 'none';
            
            targetElm.style.display = 'block';
            currentElm = targetElm;

            select_card(selectorElm, tab, colors[i]);
        }
    });
}


function select_card(gi_details_modal, card, color)
{

    const expansion = gi_details_modal.querySelector('.js-expansion');
    expansion.innerHTML = "";
    
    const rect = card.getBoundingClientRect();
    const parentElmRect = card.parentElement.getBoundingClientRect();

    const canvas = document.createElement('canvas');
    expansion.appendChild(canvas);
    canvas.width = expansion.offsetWidth;
    canvas.height = expansion.offsetHeight;

    const ctx = canvas.getContext('2d');
    const originX = rect.x + rect.width / 2 - parentElmRect.x;

    ctx.beginPath();
    ctx.moveTo(originX, 0);
    ctx.lineTo(canvas.width, canvas.height);
    ctx.lineTo(0, canvas.height);
    ctx.closePath();
    ctx.clip();

    var grd = ctx.createLinearGradient(0, 0, 0, canvas.height);
    grd.addColorStop(0, color);
    grd.addColorStop(1, "white");

    ctx.fillStyle = grd;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
}


function call_api(url, data, callback)
{
    fetch(url, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
    })
    .then(response => response.json())
    .then(data => callback(data))
    .catch((error) => {
        console.error('Error:', error);
    });

}
/*
    Collects the search form field values into a dictionary that can
    be transported as JSON to the server
*/
function gather_form()
{
    const inputs = document.querySelectorAll('.js-input');
    const form = {};
    inputs.forEach((input) => {
        if (input.type === 'checkbox')
        {
            form[input.name] = input.checked;
        }
        else 
        {
            if (input.name === "threshold")
                form[input.name] = parseFloat(input.value);
            else if (input.name === "species_id")
                form[input.name] = parseInt(input.value);
            else
                form[input.name] = input.value;
        }
        
    });
    return form;
}

/*
    processes shortest path length
*/
function showSpl(v)
    {
        if (typeof(v) === 'undefined')
            return "";
        
        let spl = v;
        if (spl === 1e5)
            spl = '∞';
        else
            spl = Math.round(spl);
        
        return spl;
}
/*
    Populates GI Search Results
*/
function populate_gi_pairs(rows)
{
    const searchResults = document.getElementById('searchResults');
    searchResults.innerHTML = "";

    rows.forEach((row) => {

        const tr = createElement('tr', searchResults);
        

        const tds = createElements('td', tr, 8);
        tds[0].innerHTML = row.gene_a_locus_tag;
        tds[1].innerHTML = row.gene_a_common_name;
        tds[2].innerHTML = row.gene_b_locus_tag;
        tds[3].innerHTML = row.gene_b_common_name;
        tds[4].innerHTML = showSpl(row.spl);
        tds[5].innerHTML = row.prob_gi.toFixed(2);
        tds[6].innerHTML = (row.reported_gi === 1) ? '<strong>Yes</strong>' : 'No';
        tds[7].innerHTML = "<span class='btn-details'>Details</span>";

        tds[7].onclick = search_result_clicked;
        tds[7].dataset.species_id = row.species_id;
        tds[7].dataset.a_id = row.gene_a_id;
        tds[7].dataset.b_id = row.gene_b_id;
        tds[7].classList.add('clickable');

    });
}

function search_result_clicked()
{
    call_api('./gi', {
        "species_id" : parseInt(this.dataset.species_id),
        "gene_a_id" : parseInt(this.dataset.a_id),
        "gene_b_id" : parseInt(this.dataset.b_id)
    }, 
    function(res) {
        populate_gi_details(res);
        
    });

}

class Paginator
{
    constructor(elm, rowsPerPage)
    {
        this._elm = elm;
        this._pagination = null;
        this._rowsPerPage = rowsPerPage;

        const lastPageElm = this._elm.querySelector('.js-last-page');
        const nextPageElm = this._elm.querySelector('.js-next-page');
        
        this._rows = null;
        this._currPage = 0;

        lastPageElm.onclick = () => {
            if (this._currPage > 0)
            {
                this._currPage -= 1;
                this.updateRows();
            }
            
        }

        nextPageElm.onclick = () => {
            if (this._currPage < this._totalPages-1)
            {
                this._currPage += 1;
                this.updateRows();
            }
            
        }
    }

    setRows(rows)
    {

        this._currPage = 0;
        this._rows = rows;
        this._totalPages = Math.ceil(rows.length / this._rowsPerPage);
        
        this.updateRows();

        const totalRecordsElm = this._elm.querySelector('.js-total-records');
        totalRecordsElm.innerHTML = rows.length;

        const currPageElm = this._elm.querySelector('.js-curr-page');
        currPageElm.innerHTML = 1;

        const totalPagesElm = this._elm.querySelector('.js-total-pages');
        totalPagesElm.innerHTML = this._totalPages;
        
    }

    updateRows() {
        const page = this._currPage;
        const rows = this._rows.slice(page * this._rowsPerPage, (page+1) * this._rowsPerPage);
        populate_gi_pairs(rows);

        const currPageElm = this._elm.querySelector('.js-curr-page');
        currPageElm.innerHTML = this._currPage + 1;
    }
}

function createElement(tag, parent)
{
    const elm = document.createElement(tag);
    parent.appendChild(elm);
    return elm;
}

function createElements(tag, parent, n)
{
    const elements = [];
    for (let i = 0; i < n; ++i)
    {
        elements.push(createElement(tag, parent))
    }
    return elements;
}

/*
    Plot GI details
*/
function populate_gi_details(data)
{
    const modal = document.getElementById('gi_details_modal');
    modal.style.display = 'block';

    const tabContents = document.querySelectorAll('.js-scrollable > div');
    tabContents.forEach((tc) => tc.style.display = 'block');


    document.body.classList.add('modal-open');
    
    const geneAName = `${data.gene_a_locus_tag} (${data.gene_a_common_name})`;
    const geneBName = `${data.gene_b_locus_tag} (${data.gene_b_common_name})`;

    const keys = ["gene_a", "gene_b", "joint", "z"];
    const titles = [geneAName + " features", geneBName + " features", "Joint features", "Model components"];
    const colors = ['#ffb700', '#0095ff', '#b2ff00', '#d000ff'];
    keys.forEach((k, i) => {
        plot("plot_" + k, data[k], true, titles[i], colors[i], colors[i]);
    });

    document.getElementById('card_gene_a').innerHTML = geneAName;
    document.getElementById('card_gene_b').innerHTML = geneBName;
    document.getElementById('card_prob_gi').innerHTML = data.prob_gi.toFixed(2);
    document.getElementById('card_pubs').innerHTML = (data.pubs.length > 0) ? 'Yes' : 'No';
    
    modal.querySelector('.js-scrollable').scrollTop = 0;
    tabContents.forEach((tc) => tc.style.display = 'none');
    const tabs = modal.querySelectorAll('.js-tabs > div');
    tabs[0].click();

    const pubsContent = document.getElementById("content_pubs");
    pubsContent.innerHTML = "";
    const ul = createElement('ul', pubsContent);
    data.pubs.forEach((p) => {
        const li = createElement('li', ul);
        li.innerHTML = p;
    });
}

/*
    Plots the model's components
*/
function plot(elm, d, show_x, title, pos_color, neg_color)
{
    let labels = d.labels;
    let features = d.features;

    const colors = features.map((v) => v < 0 ? neg_color : pos_color);

    var data = [
        {
          type: 'bar',
          orientation: 'v',
          x: labels,
          y: features,
          marker: {
            color: colors
          },
        }]
        var layout = {
            title: title,
            margin: {
                l: 40,
                r: 40,
                b: 0,
                t: 50,
                pad: 0
            },
            paper_bgcolor: 'white',
            plot_bgcolor: 'white',
            height: 200,
            font: {
              size: 16,
              
              color: 'black'
            },
            yaxis: {
                showgrid: true,
                tickmode: 'linear',

                range: [-3, 3]
            },
            xaxis: {
                showgrid: true,
                tickmode: 'linear',
                automargin: false,
                tickangle: 90,
                showticklabels: show_x
            }
          };
    

    Plotly.newPlot(elm, data, layout);
    if (show_x)
    {
        const cl = document.getElementById(elm).querySelector('.cartesianlayer');
        const rect = cl.getBoundingClientRect();
        document.getElementById(elm).style.height = rect.height + 'px';
    
    }
    

}

window.onload = main;