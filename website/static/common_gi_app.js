const ENTRIES_PER_PAGE = 50;

function main()
{
    let current_results = null;
    const nostartups = document.querySelectorAll('.js-nostartup');
    
    const paginator = new Paginator(document.querySelector('.js-search-results-paginator'), showPage);
    const loading = document.getElementById('loadingScreen');
    const loadingDisplay = loading.style.display;
    loading.style.display = 'none';

    function showPage(page)
    {
        const paged_results = current_results.rows.slice(page * ENTRIES_PER_PAGE, (page+1) * ENTRIES_PER_PAGE);
        populate_results(paged_results, current_results.full_names);
    }

    const btnSearch = document.getElementById('btnSearch');
    btnSearch.onclick = function()
    {
        const form = gather_form();
        loading.style.display = loadingDisplay;

        call_api('../common_interactors', form, function(res) {
            current_results = res;
            showPage(0);
            paginator.update({
                "n_rows" : res.rows.length,
                "pages" : numPages(res.rows.length, ENTRIES_PER_PAGE),
                "page" : 0
            });
            loading.style.display = 'none';
            
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
function populate_results(res, full_names)
{
    function showProb(v)
    {
        if (typeof(v) === 'undefined')
            return "";
        
        return `${v[0].toFixed(2)}`;
    }

    function showSpl(v)
    {
        if (typeof(v) === 'undefined')
            return "";
        
        let spl = v[1];
        if (spl === 1e5)
            spl = '∞';
        else
            spl = Math.round(spl);
        
        return spl;
    }

    const searchResults = document.getElementById('searchResults');
    searchResults.innerHTML = "";
    
    res.forEach((row) => {

        const tr = createElement('tr', searchResults);
    
        const tds = createElements('td', tr, 10);
        tds[0].innerHTML = row.interactor[0];
        tds[1].innerHTML = row.interactor[1];


        tds[2].innerHTML = showSpl(row.gene_a);
        tds[3].innerHTML = showSpl(row.gene_b);
        tds[4].innerHTML = showSpl(row.gene_c);
        tds[5].innerHTML = showSpl(row.gene_d);

        tds[6].innerHTML = showProb(row.gene_a);
        tds[7].innerHTML = showProb(row.gene_b);
        tds[8].innerHTML = showProb(row.gene_c);
        tds[9].innerHTML = showProb(row.gene_d);

    });
    
    const cols = ['gene_a', 'gene_b', 'gene_c', 'gene_d'];
    cols.forEach((c) => {
        const col = document.getElementById("col_" + c);

        if (c in full_names)
        {
            col.innerHTML = `${full_names[c][0]} (${full_names[c][1]})`;
        }
        else 
        {
            col.innerHTML = "";
        }
    });

    cols.forEach((c) => {
        const col = document.getElementById("col_" + c + "_spl");

        if (c in full_names)
        {
            col.innerHTML = `${full_names[c][0]} (${full_names[c][1]})`;
        }
        else 
        {
            col.innerHTML = "";
        }
    });
}

function numPages(n_rows, entries_per_page)
{
    let pages = Math.floor(n_rows / entries_per_page);
    if (n_rows % entries_per_page > 0)
        pages += 1;
    return pages; 
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

window.onload = main;