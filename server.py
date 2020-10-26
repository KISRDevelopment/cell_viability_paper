import flask 
import os 
import pandas as pd
import json
import numpy as np 
import models.gi_mn 

app = flask.Flask(__name__)

@app.route('/get_input/<int:a_id>/<int:b_id>', methods=['GET'])
def get_input(a_id, b_id):

    df = pd.DataFrame({ "a_id" : [a_id], "b_id" : [b_id] })
    
    features = []
    for proc in processors:
        features.append(proc.transform(df))
                
    batch_F = np.hstack(features).tolist()

    return flask.jsonify(batch_F)
    
@app.route('/<path:filename>')
def static_handler(filename):
    return flask.send_from_directory('www/', filename)

if __name__ == '__main__':
    # from waitress import serve
    # import sys
    # import logging

    # logger = logging.getLogger('waitress')
    # logger.setLevel(logging.INFO)

    # url_prefix = ''
    # if len(sys.argv) > 1:
    #     url_prefix = sys.argv[1]

    # serve(app, url_prefix=url_prefix, port=8091)

    cfg_path = "cfgs/models/yeast_gi_mn.json"
    with open(cfg_path, 'r') as f:
        cfg = json.load(f)
    
    processors, _ = models.gi_mn.load_features(cfg)
    app.run(host='0.0.0.0', port=5000)
    