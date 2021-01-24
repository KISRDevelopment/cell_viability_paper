from flask import Flask, request, send_from_directory, render_template, g, current_app, jsonify
import sqlite3
from flask_caching import Cache
import queries 

cache = Cache(config={'CACHE_TYPE' : 'simple'})
app = Flask(__name__)


DB_PATH = "db.sqlite"

ENTRIES_PER_PAGE = 50

def init():
    app.teardown_appcontext(close_db)
    cache.init_app(app)

def dict_factory(cursor, row):
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d

@app.route('/')
def index():
    species_id = 0
    threshold = 0
    page = 0
    gene = ''

    if request.args:
        species_id = int(request.args.get('species_id', -1))
        threshold = float(request.args.get('threshold', 0))
        page = int(request.args.get('page', 0))
        gene = request.args.get('gene', '')
    
    if species_id == -1:
        rows = []
        n_rows = 0
        page = 0
        gene = ''
    else:
        
        conn = get_db()
        
        # c = conn.cursor()
        # c.execute(SPECIES_QUERY, (species_id, threshold, ENTRIES_PER_PAGE, page * ENTRIES_PER_PAGE))
        # rows = c.fetchall()

        rows = queries.get_pairs(conn, species_id, threshold, gene, page, ENTRIES_PER_PAGE)

        for r in rows:
            r['reported_gi'] = r['observed'] and r['observed_gi']
        n_rows = calculate_rows(species_id, threshold, gene)

    pagination = paginate(n_rows, ENTRIES_PER_PAGE, page)

    return render_template('index.html', rows=rows, species_id=species_id, threshold=threshold, n_rows=n_rows, pagination=pagination)

@app.route('/interpret/<int:gi_id>', methods=['GET'])
def interpret(gi_id):
    SPECIES_MODELS = {
        1: "./models/yeast_gi_hybrid_mn.npz",
        2: './models/pombe_gi_mn.npz',
        3: './models/human_gi_mn.npz',
        4: './models/dro_gi_mn.npz'
    }
    conn = get_db()
    conn.row_factory = dict_factory

    row = queries.get_gi(conn, gi_id)

    if row:
        m = queries.LogisticRegressionModel(SPECIES_MODELS[row['species_id']])
        components = m.get_z_components(row)
        return jsonify(components)
    else:
        return jsonify({})
    

@cache.memoize(timeout=None)
def calculate_rows(species_id, threshold, gene):
    conn = get_db()

    return queries.count_pairs(conn, species_id, threshold, gene)

def paginate(n_rows, page_size, page):
    
    pages = n_rows // page_size
    if n_rows % page_size > 0:
        pages += 1
    
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
        g.db = sqlite3.connect(
            DB_PATH,
            detect_types=sqlite3.PARSE_DECLTYPES
        )
        g.db.row_factory = queries.dict_factory

    return g.db

def close_db(e=None):
    db = g.pop('db', None)

    if db is not None:
        db.close()

init()
