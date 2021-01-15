import sqlite3
import pandas as pd
import numpy as np 
import os 

DB_PATH = "website/db.sqlite"

os.remove(DB_PATH)

#
# create database
#
with open('website_db_schema.sql', 'r') as f:
    schema = f.read()
conn = sqlite3.connect(DB_PATH)
c = conn.cursor()
c.executescript(schema)
conn.commit()

#
# populate species tables
#
def populate_from(conn, path, species_id):
    print("Populating from %s" % path)
    df = pd.read_csv(path)

    ix = pd.isnull(df['b_id']) | pd.isnull(df['a_id'])
    if np.sum(ix) > 0:
        print("Warning: found %d entries with no a_id or b_id" % np.sum(ix))
        df = df[~ix]

    list_a_id = list(df['a_id'])
    list_b_id = list(df['b_id'])

    genes = set(list_a_id).union(list_b_id)
    
    # insert into genes table
    c = conn.cursor()
    rows = [(species_id, g) for g in genes]
    c.executemany("INSERT INTO genes(species_id, gene_name) VALUES(?,?)", rows)
    conn.commit()
    c.execute("SELECT gene_name, gene_id FROM genes")
    results = list(c.fetchall())
    n_to_id = {e[0]:e[1] for e in results}
    
    list_prob_gi = list(df['prob_gi'])
    list_obs = list(df['observed'])
    list_interacting = list(df['interacting'])

    rows = [(species_id, n_to_id[a], n_to_id[b], obs == 1, inter == 1, prob) for a,b,obs,inter,prob in 
        zip(list_a_id, list_b_id, list_obs, list_interacting, list_prob_gi)]

    c.executemany("INSERT INTO genetic_interactions(species_id, gene_a_id, gene_b_id, observed, observed_gi, prob_gi) VALUES(?, ?, ?, ?, ?, ?)", rows)
    conn.commit()
    
populate_from(conn, '../results/preds/yeast_gi_hybrid_mn', 1)

populate_from(conn, '../results/preds/pombe_gi_mn', 2)

populate_from(conn, '../results/preds/human_gi_mn', 3)

populate_from(conn, '../results/preds/dro_gi_mn', 4)
