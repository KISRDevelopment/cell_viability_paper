from flask import Flask, request, send_from_directory, render_template
import sqlite3

app = Flask(__name__)

DB_PATH = "db.sqlite"

SPECIES_QUERY = "select a.gene_name  gene_a, b.gene_name gene_b, g.observed, g.observed_gi, g.prob_gi from genetic_interactions g join genes a on g.gene_a_id = a.gene_id join genes b on g.gene_b_id = b.gene_id where a.species_id = b.species_id and a.species_id = ? and g.prob_gi > ? limit ? offset ?"

def dict_factory(cursor, row):
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d

@app.route('/')
def index():

    if request.args:
        species_id = int(request.args.get('species_id', -1))
        threshold = float(request.args.get('threshold', 0))

    if species_id == -1:
        rows = []
    else:
        print(species_id)
        print(threshold)
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = dict_factory
        c = conn.cursor()
        c.execute(SPECIES_QUERY, (species_id, threshold, 50, 0))
        rows = c.fetchall()

        conn.close()

    return render_template('index.html', rows=rows, species_id=species_id, threshold=threshold)