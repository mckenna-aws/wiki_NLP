 
# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

import wikipedia as wiki
from flask import request
import requests as reqs
import numpy as np
import pandas as pd
import json
import os
import flask
import gensim 
import StringIO
import sys


#function to convert raw text to gensim-tagged document 
    
def convert(indat, tokens_only=False): 
    for i, line in enumerate(indat):        
        if tokens_only:
            yield gensim.utils.simple_preprocess(line)
        else:
            yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line), [i])   


#function to search wikipedia and get full text articles.  Limit var controls how many articles to pull
                
def get_wiki_data(search, limit=20): 
    url = 'https://en.wikipedia.org/w/api.php?action=opensearch&search=' + str(search)  + '&namespace=0&format=json&limit=' + str(limit)
    resp = reqs.request('GET', url)
    pages = json.loads(resp.text)[1]
    pages = [x.encode('utf-8') for x in pages]
    full_text = []
    labels = []
        
    for i in range(len(pages)):
            
        if 'disambiguation' not in pages[i]:
            try: 
                full_text.append ( wiki.page(pages[i]).content)
                labels.append (pages[i])
            except: 
                continue
                
    return ({'full_text':full_text, 'labels':labels})

#function that inputs documents/labels and output similarity matrix 
def predict(intext): 
        
    full_text = intext['full_text']
    labels    = intext['labels']

    train_corpus = list( convert(full_text) )
    test_corpus  = list( convert(full_text  , tokens_only=True))
    
    model = gensim.models.doc2vec.Doc2Vec(size=100, min_count=15, iter=100)
    model.build_vocab(train_corpus) 
    
    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.iter)
    
    out_mat = np.zeros( (len(labels), len(labels)))
    
    for j in range(0,len(labels)): 
        this_inferred_vector = model.infer_vector(test_corpus[j]) 
        this_sim = np.asarray(model.docvecs.most_similar([this_inferred_vector], topn=len(model.docvecs)))
        this_sim = this_sim[this_sim[:, 0].argsort()]
        out_mat[:, j] =  this_sim[:,1]
            
    # normalize similarity vector
    out_mat_norm = (out_mat - out_mat.mean(axis=0)) / out_mat.std(axis=0)
    end_df = pd.DataFrame(out_mat_norm)
    end_df.columns = labels
    
    return( end_df )


#define flask endpoints

app = flask.Flask(__name__)

@app.route('/ping', methods=['GET'])
def ping():
    return flask.Response(response='\n', status=200, mimetype='application/json')

@app.route('/invocations', methods=['POST'])
def transformation():
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """
    
    my_var = request.data.decode('utf-8')
    indat = 'guitar'
    data_process1 = get_wiki_data(my_var)
    data_process2 = predict(data_process1)
    
    #output results 
    out = StringIO.StringIO()
    
    new = data_process2.to_csv(out, index=False)
    result = out.getvalue()

    return flask.Response(response=result, status=200, mimetype='text/csv')

