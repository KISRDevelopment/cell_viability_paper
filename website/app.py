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
    
    db = get_db()

    rows, n_rows = db.get_pairs(species_id, threshold, gene_a, gene_b, page)

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
        gene_b=gene_b)

@app.route('/interpret/<int:gi_id>', methods=['GET'])
def interpret(gi_id):
    SPECIES_MODELS = {
        1: "./models/yeast_gi_hybrid_mn.npz",
        2: './models/pombe_gi_mn.npz',
        3: './models/human_gi_mn.npz',
        4: './models/dro_gi_mn.npz'
    }
    db = get_db()

    row = db.get_gi(gi_id)

    if row:
        m = lrm.LogisticRegressionModel(SPECIES_MODELS[row['species_id']])
        components = m.get_z_components(row)
        return jsonify(components)
    else:
        return jsonify({})

def paginate(n_rows, page_size, page):
    
    pages = int(np.ceil(n_rows / page_size))
    
    next_page = page + 1
    if next_page >= pages:
        next_page = -1 
    
    prev_page = page - 1
    if prev_page < 0:
        prev_page = -1 
    
    return {
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
