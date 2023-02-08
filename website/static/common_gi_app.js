const ENTRIES_PER_PAGE = 50;

function main()
{
    let current_results = null;
    const nostartups = document.querySelectorAll('.js-nostartup');
    
    const paginator = new Paginator(document.querySelector('.js-search-results-paginator'), ENTRIES_PER_PAGE);

    const loading = document.getElementById('loadingScreen');
    const loadingDisplay = loading.style.display;
    loading.style.display = 'none';

    const btnSearch = document.getElementById('btnSearch');
    btnSearch.onclick = function()
    {
        const form = gather_form();
        loading.style.display = loadingDisplay;

        call_api('./common_interactors', form, function(res) {
            paginator.setRows(res.rows);
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
function populate_results(rows)
{
    function showProb(v)
    {
        if (typeof(v) === 'undefined')
            return "";
        
        return `${v.toFixed(2)}`;
    }

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

    const searchResults = document.getElementById('searchResults');
    searchResults.innerHTML = "";
    
    rows.forEach((row) => {

        const tr = createElement('tr', searchResults);
    
        const tds = createElements('td', tr, 10);
        tds[0].innerHTML = row.ci_locus_tag;
        tds[1].innerHTML = row.ci_common_name;

        for (let i = 0; i < 4; ++i)
        {
            if (i >= row.interaction_props.length)
                break;
            tds[2+i].innerHTML = showSpl(row.interaction_props[i].spl);
            tds[6+i].innerHTML = showProb(row.interaction_props[i].gi_prob);
        }
    });
    
    if (rows.length === 0)
        return ;
    
    const firstRow = rows[0];

    const cols = ['gene_a', 'gene_b', 'gene_c', 'gene_d'];

    cols.forEach((c, i) => {
        const giCol = document.getElementById("col_" + c);
        const splCol = document.getElementById("col_" + c + '_spl');
        
        giCol.innerHTML = "";
        splCol.innerHTML = "";

        if (i >= firstRow.interaction_props.length)
            return;
        const gene_name = `${firstRow.interaction_props[i].gene_locus_tag} (${firstRow.interaction_props[i].gene_common_name})`;
        giCol.innerHTML = gene_name;
        splCol.innerHTML = gene_name;
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
        populate_results(rows);

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

window.onload = main;