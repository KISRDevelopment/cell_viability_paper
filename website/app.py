import json
from flask import Flask, request, send_from_directory, render_template, g, current_app, jsonify
from flask_compress import Compress
import db_layer 
import lrm 
import numpy as np 
import waitress
import sys 

from numpyencoder import NumpyEncoder
app = Flask(__name__)
app.json_encoder = NumpyEncoder
Compress(app)

ENTRIES_PER_PAGE = 50
DB = db_layer.DbLayer()

def init():
    pass

@app.route('/gi', methods=['POST'])
def gi():

    rp = request.json 

    return DB.get_gi(rp['species_id'], 
        rp['gene_a_id'],
        rp['gene_b_id'])


@app.route('/common_interactors', methods=['POST'])
def common_interactors():
    
    rp = request.json 

    genes = [rp['gene_a'], rp['gene_b'], rp['gene_c'], rp['gene_d']]
    genes = [g for g in genes if g is not None and g != '']

    if len(genes) == 0:
        return jsonify({ "rows" : [] })

    common_interactors = DB.get_common_interactors(
        rp['species_id'],
        genes,
        rp['threshold'],
        rp['published_only'],
        rp['max_spl']
    )
    
    return jsonify({ "rows" : common_interactors })

@app.route('/gi_pairs', methods=['POST'])
def gi_pairs():

    rp = request.json 


    if rp['gene_a'] == '':
        rp['gene_a'] = None 
    if rp['gene_b'] == '':
        rp['gene_b'] = None 
        
    pairs,_ = DB.get_pairs(rp['species_id'], 
        rp['threshold'], 
        rp['gene_a'], 
        rp['gene_b'], 
        rp['published_only'], rp['max_spl'])

    return jsonify({ "rows" : pairs })

@app.route('/gi_triplets', methods=['POST'])
def gi_triplets():

    rp = request.json 


    if rp['gene_a'] == '':
        rp['gene_a'] = None 
    if rp['gene_b'] == '':
        rp['gene_b'] = None 
    if rp['gene_c'] == '':
        rp['gene_c'] = None 
    
    triplets = DB.get_triplets(
        rp['threshold'], 
        rp['gene_a'], 
        rp['gene_b'], 
        rp['gene_c'],
        rp['published_only'], rp['max_spl'])

    return jsonify({ "rows" : triplets })

@app.route('/tgi', methods=['POST'])
def tgi():

    rp = request.json 

    return DB.get_tgi(rp['gene_a_id'], rp['gene_b_id'], rp['gene_c_id'])


if __name__ == "__main__":
    init()

    url_prefix = ''
    if len(sys.argv) > 1:
        url_prefix = sys.argv[1]
    
    waitress.serve(app, host='0.0.0.0', url_prefix=url_prefix, port=8090)
