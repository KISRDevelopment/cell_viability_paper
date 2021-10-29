from flask import Flask, request, send_from_directory, render_template, g, current_app, jsonify
import db_layer 
import lrm 
import numpy as np 
import waitress
import sys 

app = Flask(__name__)

DB_PATH = "db.sqlite"
ENTRIES_PER_PAGE = 50

def init():
    app.teardown_appcontext(close_db)

@app.route('/gi', methods=['POST'])
def gi():

    rp = request.json 

    SPECIES_MODELS = {
        1: "./models/yeast_gi_hybrid_mn.npz",
        2: './models/pombe_gi_mn.npz',
        3: './models/human_gi_mn.npz',
        4: './models/dro_gi_mn.npz'
    }
    db = get_db()

    row = db.get_gi(rp['gi_id'])

    if row:
        m = lrm.LogisticRegressionModel(SPECIES_MODELS[row['species_id']])
        components = m.interpret(row)
        return jsonify({
            "components" : components,
            "pubs" : row['pubs'],
            "prob_gi" : row['prob_gi'],
            "reported_gi" : row['observed'] and row['observed_gi'],
            "gene_a_locus_tag" : row['gene_a_locus_tag'],
            "gene_b_locus_tag" : row['gene_b_locus_tag'],
            "gene_a_common_name" : row['gene_a_common_name'],
            "gene_b_common_name" : row['gene_b_common_name'],
            
        })
    else:
        return jsonify({})


@app.route('/common_interactors', methods=['POST'])
def common_interactors():
    
    rp = request.json 
    db = get_db()

    interactor_dicts = {}
    common_targets = []
    full_names = {}
    for k in ['gene_a', 'gene_b', 'gene_c', 'gene_d']:
        if rp[k] is None or rp[k] == "":
            continue 
        
        gene_row = db.get_gene(rp['species_id'], rp[k])
        if not gene_row:
            continue 
        
        gene_id = gene_row['gene_id']
        full_names[k] = [gene_row['locus_tag'], gene_row['common_name']]

        interactors = db.get_interactors(rp['species_id'], gene_id, rp['threshold'], rp['published_only'])
        interactor_dicts[k] = interactors

        targets = set(interactors.keys())
        if common_targets == []:
            common_targets = targets
        else:
            common_targets = targets.intersection(common_targets)

    results = []
    for t in common_targets:
        row = {
            "interactor" : t
        }

        for k, interactors in interactor_dicts.items():
            row[k] = interactors[t]
        results.append(row)


    return jsonify({
        "rows" : results,
        "full_names" : full_names
    })

@app.route('/gi_pairs', methods=['POST'])
def gi_pairs():

    rp = request.json 

    db = get_db()

    rows, n_rows = db.get_pairs(rp['species_id'], 
        rp['threshold'], 
        rp['gene_a'], 
        rp['gene_b'], 
        rp['page'], 
        rp['published_only'])

    for r in rows:
        r['reported_gi'] = r['observed'] and r['observed_gi']
        
    pagination = paginate(n_rows, ENTRIES_PER_PAGE, rp['page'])

    return jsonify({
        "pagination" : pagination,
        "request_params" : rp,
        "rows" : rows
    })


def paginate(n_rows, page_size, page):
    
    pages = int(np.ceil(n_rows / page_size))
    
    next_page = page + 1
    if next_page >= pages:
        next_page = -1 
    
    prev_page = page - 1
    if prev_page < 0:
        prev_page = -1 
    
    return {
        "n_rows" : n_rows,
        "pages" : pages,
        "page" : page,
        "next_page" : next_page,
        "prev_page" : prev_page
    }
def get_db():
    if 'db' not in g:
        g.db = db_layer.DbLayer(DB_PATH, ENTRIES_PER_PAGE)

    return g.db

def close_db(e=None):
    db = g.pop('db', None)
    if db is not None:
        db.close()

if __name__ == "__main__":
    init()

    url_prefix = ''
    if len(sys.argv) > 1:
        url_prefix = sys.argv[1]
    
    waitress.serve(app, host='0.0.0.0', url_prefix=url_prefix, port=8090)
