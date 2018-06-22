 
# This is the file that implements a flask server to do inferences

import numpy as np
import pandas as pd
import os
import flask
import StringIO
from sklearn.cluster import AffinityPropagation



# function that takes similarity matrix and output group clustering
    
def predict(in_obj, damp=0.65, max_itern=2000):
    page_labels = in_obj.columns
    af = AffinityPropagation( damping=damp,max_iter=max_itern).fit(in_obj)
    af_labels = af.labels_
    unique, counts = np.unique(page_labels, return_counts=True)
    dict(zip(unique, counts))
    
    out_dict = {"groups":[], 
                "pages": []}
    
    out_dict["groups"].extend([a for a in range(0, len(np.unique(af_labels)))])
    out_dict["pages"].extend([list(np.asarray(page_labels)[af_labels==a]) for a in range(0, len(np.unique(af_labels)))])

    out_df = pd.DataFrame.from_dict(out_dict)

    return(out_df)


# The flask app for serving predictions
app = flask.Flask(__name__)

@app.route('/ping', methods=['GET'])
def ping():
    return flask.Response(response='\n', status=200, mimetype='application/json')



@app.route('/invocations', methods=['POST'])
def transformation():
    
    # Convert from CSV to pandas
    if flask.request.content_type == 'text/csv':
        data = flask.request.data.decode('utf-8')
        s = StringIO.StringIO(data)
        data = pd.read_csv(s,index_col=0)

    else:
        return flask.Response(response='This predictor only supports CSV data', status=415, mimetype='text/plain')

    # run the input through our prediction function 
    data_process2 = predict(data)

    # Convert from numpy back to CSV
    out = StringIO.StringIO()
    
    pd_recs = data_process2.to_csv(out, index=False)
    result = out.getvalue()

    return flask.Response(response=result, status=200, mimetype='text/csv')


