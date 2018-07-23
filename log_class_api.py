from flask import Flask, jsonify, request
app = Flask(__name__)

import pickle
import urllib.parse
from keras.preprocessing.sequence import pad_sequences

from tensorflow import keras

model = keras.models.load_model('32_struts_model.h5', compile=False)
max_len = model.input.shape[1]

# If you run this code on the other computer, you might need to remove commentout below.
# Sometimes mode.predict function does not load correctly.
# import numpy as ap
# X = np.zeros((10, max_len))
# model.predict(X, batch_size=32)

@app.route('/preds/', methods=['GET'])
def preds():
    # loading
    with open('32_stuts_token.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    reqstr = request.args.get('str', '')
    reqstr = urllib.parse.quote(reqstr, safe='')
    reqstr = [reqstr]
    response = jsonify()
    tokenizer.fit_on_texts(reqstr)
    req_mat = tokenizer.texts_to_sequences(reqstr)
    req_mat = pad_sequences(req_mat, maxlen=int(max_len))
    prediction = model.predict(req_mat)

    # if normal
    if prediction[0][0] <= 0.2:
        response.status_code = 201
    # if attack
    elif prediction[0][0] > 0.2:
        response.status_code = 202

    # save
    with open('request.log', mode='a') as f:
        f.write(str(response.status_code) + str(prediction) + ',' + str(reqstr) + '\n')

    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0')
