from flask import Flask, request, send_from_directory, render_template, g, current_app
import sqlite3
from flask_caching import Cache

cache = Cache(config={'CACHE_TYPE' : 'simple'})
app = Flask(__name__)


DB_PATH = "db.sqlite"

COUNT_QUERY = "select count(gi_id) as n_rows from genetic_interactions  where species_id = ? and prob_gi > ?"
SPECIES_QUERY = "select a.gene_name  gene_a, b.gene_name gene_b, g.observed, g.observed_gi, g.prob_gi from genetic_interactions g join genes a on g.gene_a_id = a.gene_id join genes b on g.gene_b_id = b.gene_id where g.species_id = ? and g.prob_gi > ? limit ? offset ?"

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

    if request.args:
        species_id = int(request.args.get('species_id', -1))
        threshold = float(request.args.get('threshold', 0))
        page = int(request.args.get('page', 0))
    
    if species_id == -1:
        rows = []
        n_rows = 0
        page = 0
    else:
        
        conn = get_db()
        conn.row_factory = dict_factory
        c = conn.cursor()
        c.execute(SPECIES_QUERY, (species_id, threshold, ENTRIES_PER_PAGE, page * ENTRIES_PER_PAGE))
        rows = c.fetchall()

        n_rows = calculate_rows(species_id, threshold)

    pagination = paginate(n_rows, ENTRIES_PER_PAGE, page)

    return render_template('index.html', rows=rows, species_id=species_id, threshold=threshold, n_rows=n_rows, pagination=pagination)

@cache.memoize(timeout=None)
def calculate_rows(species_id, threshold):
    conn = get_db()
    c = conn.cursor()
    c.execute(COUNT_QUERY, (species_id, threshold))
    n_rows = c.fetchone()['n_rows']
    return n_rows

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
        g.db.row_factory = sqlite3.Row

    return g.db

def close_db(e=None):
    db = g.pop('db', None)

    if db is not None:
        db.close()

init()
