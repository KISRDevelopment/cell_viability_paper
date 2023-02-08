from flask import Flask, request, send_from_directory, render_template, g, current_app, jsonify
from flask_compress import Compress
import db_layer 
import waitress
import sys 

import datetime 
import json

# setup logging
# https://stackoverflow.com/questions/30135091/write-thread-safe-to-file-in-python 
import logging
import secrets 
from functools import wraps 
import os 
os.makedirs("./tmp", exist_ok=True)
logpath = "./tmp/log.log"
logger = logging.getLogger('applog')
logger.setLevel(logging.INFO)
logger.propagate = False
ch = logging.FileHandler(logpath)
ch.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(ch)

from numpyencoder import NumpyEncoder
app = Flask(__name__)
app.json_encoder = NumpyEncoder
Compress(app)

DB = db_layer.DbLayer()

#
# Decorator to track visitors
#
def track(route):
    @wraps(route)
    def wrapper(*args, **kwargs):

        # to track unique visitors (up to a cookie)
        # we use a unique token in a cookie
        unique_token = request.cookies.get('unique_token')
        store_cookie = False
        if unique_token is None:
            unique_token = secrets.token_hex(16)
            store_cookie = True 

        # log the request's data
        curr_time = datetime.datetime.now(datetime.timezone.utc)
        logger.info(json.dumps({
            "user_agent" : request.headers['User-Agent'],
            "remote_addr" : get_remote_ip(),
            "unique_token" : unique_token,
            "time" : str(curr_time),
            "route" : route.__name__
        }))

        # execute wrapped function
        response = route(*args, **kwargs)

        # store cookie
        if store_cookie:
            response.set_cookie('unique_token', unique_token)

        return response
    
    return wrapper

def get_remote_ip():
    if request.environ.get('HTTP_X_FORWARDED_FOR') is None:
        return request.environ['REMOTE_ADDR']
    else:
        return request.environ['HTTP_X_FORWARDED_FOR']


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
        
    pairs = DB.get_pairs(rp['species_id'], 
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

@app.route('/search_tgi')
@track 
def search_tgi():
    response = send_static('search_tgi.html')
    return response

@app.route('/search_common_gi')
@track 
def search_common_gi():
    response = send_static('search_common_gi.html')
    return response

@app.route('/')
@track
def index():
    response = send_static('search_gi.html')
    return response

@app.route('/<path:path>')
def send_static(path):
    r = send_from_directory('./static/', path)

    # css and html can change so we want to push latest
    # version to clients all the time
    # check: https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Cache-Control
    if path.endswith('.css') or path.endswith('.html') or path.endswith('.js'):
        r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    else:
        r.headers["Cache-Control"] = "must-revalidate, max-age=86400"
    return r
if __name__ == "__main__":
    init()

    url_prefix = ''
    if len(sys.argv) > 1:
        url_prefix = sys.argv[1]
    
    waitress.serve(app, host='0.0.0.0', url_prefix=url_prefix, port=8090)
