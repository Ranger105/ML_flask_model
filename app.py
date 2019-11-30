
#app.py
#structure of code.
'''
>>Load pickled model
>>Name flask app
>>create a route that recieves JSON inputs,
used the trained model to make a prediction,
and returns that prediction in a JSON format,
which can be accessed through API endpoint.
'''

#app using flask
import pandas as pd
from flask import Flask, jsonify, request
import pickle

#load model
model = pickle.load(open('model.pkl','rb'))

#app
app = Flask(__name__)

#routes
@app.route('/', methods=['POST'])

def predict():

    data = request.get_json(force = True)
    print('data',data)
    #convert data into dataframe
    data.update((x,[y]) for x, y in data.items())
    data_df = pd.DataFrame.from_dict(data)
    print('data_df',data_df)

    #predictions
    result = model.predict(data_df)

    #send back to browser
    output = {'results': int (result[0])}

    #return data
    return jsonify(results = output)

if __name__=='__main__':
    app.run(port = 5000, debug = True)

