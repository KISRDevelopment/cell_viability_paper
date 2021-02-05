
function main()
{
    const nostartups = document.querySelectorAll('.js-nostartup');
    
    const gi_details_modal = document.getElementById('gi_details_modal');
    gi_details_modal.querySelector('.js-close').onclick = () => {
        gi_details_modal.style.display = 'none';
        document.body.classList.remove('modal-open');
    };

    let curr_form = null;

    const paginator = new Paginator(document.querySelector('.js-search-results-paginator'), function(page) {
        curr_form['page'] = page;

        call_api('http://127.0.0.1:5000/gi_pairs', curr_form, function(res) {
            populate_gi_pairs(res);
        });
    });


    const btnSearch = document.getElementById('btnSearch');
    btnSearch.onclick = function()
    {
        const form = gather_form();
        form['page'] = 0;

        call_api('http://127.0.0.1:5000/gi_pairs', form, function(res) {
            populate_gi_pairs(res);
            paginator.update(res.pagination);
            curr_form = form;
            nostartups.forEach((e) => e.style.visibility = 'visible');
        });
    }

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
    Populates GI Search Results
*/
function populate_gi_pairs(res)
{
    const searchResults = document.getElementById('searchResults');
    searchResults.innerHTML = "";

    res.rows.forEach((row) => {

        const tr = createElement('tr', searchResults);
        tr.dataset.gi_id = row.gi_id;

        const tds = createElements('td', tr, 6);
        tds[0].innerHTML = row.gene_a_locus_tag;
        tds[1].innerHTML = row.gene_a_common_name;
        tds[2].innerHTML = row.gene_b_locus_tag;
        tds[3].innerHTML = row.gene_b_common_name;
        tds[4].innerHTML = row.prob_gi.toFixed(2);
        tds[5].innerHTML = (row.reported_gi === 1) ? '<strong>Yes</strong>' : 'No';

        tr.onclick = search_result_clicked;
    });
}

function search_result_clicked()
{
    const gi_id = this.dataset.gi_id;
    call_api('http://127.0.0.1:5000/gi', { "gi_id" : gi_id }, 
    function(res) {
        populate_gi_details(res);
        
    });

}

class Paginator
{
    constructor(elm, callback)
    {
        this._elm = elm;
        this._pagination = null;
        this._callback = callback;

        const lastPageElm = this._elm.querySelector('.js-last-page');
        const nextPageElm = this._elm.querySelector('.js-next-page');

        lastPageElm.onclick = () => {
            if (this._pagination.page > 0)
            {
                this._pagination.page -= 1;
                callback(this._pagination.page);
                this.update(this._pagination);
            }
            
        }

        nextPageElm.onclick = () => {
            if (this._pagination.page < this._pagination.pages - 1)
            {
                this._pagination.page += 1;
                callback(this._pagination.page);
                this.update(this._pagination);
            }
            
        }
    }

    update(pagination)
    {
        const totalRecordsElm = this._elm.querySelector('.js-total-records');
        totalRecordsElm.innerHTML = pagination.n_rows;

        const currPageElm = this._elm.querySelector('.js-curr-page');
        currPageElm.innerHTML = pagination.page + 1;

        const totalPagesElm = this._elm.querySelector('.js-total-pages');
        totalPagesElm.innerHTML = pagination.pages;

        this._pagination = pagination;
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
    document.body.classList.add('modal-open');
    
    const geneAName = `${data.gene_a_locus_tag} (${data.gene_a_common_name})`;
    const geneBName = `${data.gene_b_locus_tag} (${data.gene_b_common_name})`;

    const keys = ["gene_a", "gene_b", "joint", "z"];
    const titles = [geneAName + " features", geneBName + " features", "Joint features", "Model components"];

    keys.forEach((k, i) => {
        plot("plot_" + k, data.components[k], i === 1 || i === 3, titles[i]);
    });

    document.getElementById('card_gene_a').innerHTML = geneAName;
    document.getElementById('card_gene_b').innerHTML = geneBName;
    document.getElementById('card_prob_gi').innerHTML = data.prob_gi.toFixed(2);
    document.getElementById('card_pubs').innerHTML = (data.pubs.length > 0) ? 
        data.pubs.map((p) => p.identifier).join(', ') : "None";
    
    modal.querySelector('.js-scrollable').scrollTop = 0;
}

/*
    Plots the model's components
*/
function plot(elm, d, show_x, title)
{
    let labels = d.labels;
    let features = d.features;

    const colors = features.map((v) => v < 0 ? 'red' : 'blue');

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