from flask import Flask, request, send_from_directory, render_template, g, current_app, jsonify
import db_layer 
import lrm 
import numpy as np 

app = Flask(__name__)

DB_PATH = "db.sqlite"
ENTRIES_PER_PAGE = 50

def init():
    app.teardown_appcontext(close_db)

@app.route('/')
def index():

    species_id = int(request.args.get('species_id', -1))
    threshold = float(request.args.get('threshold', 0))
    gene_a = request.args.get('gene_a', '')
    gene_b = request.args.get('gene_b', '')
    page = int(request.args.get('page', 0))
    published_only = 'published_only' in request.args
    print(published_only)

    db = get_db()

    rows, n_rows = db.get_pairs(species_id, threshold, gene_a, gene_b, page, published_only)

    for r in rows:
        r['reported_gi'] = r['observed'] and r['observed_gi']
        
    pagination = paginate(n_rows, ENTRIES_PER_PAGE, page)

    return render_template('index.html', 
        rows=rows, 
        species_id=species_id, 
        threshold=threshold, 
        n_rows=n_rows, 
        pagination=pagination,
        gene_a=gene_a, 
        gene_b=gene_b,
        published_only='checked' if published_only else '')

@app.route('/interpret/<int:gi_id>', methods=['GET'])
def interpret(gi_id):
    threshold = float(request.args.get('threshold', 0.5))

    SPECIES_MODELS = {
        1: "./models/yeast_gi_hybrid_mn.npz",
        2: './models/pombe_gi_mn.npz',
        3: './models/human_gi_mn.npz',
        4: './models/dro_gi_mn.npz'
    }
    db = get_db()

    row = db.get_gi(gi_id, threshold)

    if row:
        m = lrm.LogisticRegressionModel(SPECIES_MODELS[row['species_id']])
        components = m.interpret(row)
        return jsonify({
            "components" : components,
            "pubs" : row['pubs'],
            "common" : row['common']
        })
    else:
        return jsonify({})

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

    interactors_a = db.get_interactors(rp['species_id'], rp['gene_a_id'], rp['threshold'])
    interactors_b = db.get_interactors(rp['species_id'], rp['gene_b_id'], rp['threshold'])

    a_targets = set(interactors_a.keys())
    b_targets = set(interactors_b.keys())

    common_targets = a_targets.intersection(b_targets)

    result = [
        (t, interactors_a[t], interactors_b[t]) for t in common_targets
    ]

    return jsonify(result)

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

init()
