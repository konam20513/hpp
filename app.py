import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np 
import pandas as pd 


app=Flask(__name__)

#Loading model
regmodel=pickle.load(open('house_pred.pkl','rb'))
scaler=pickle.load(open('scaling.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    new_data=scaler.transform((np.array(list(data.values())).reshape(1,-1)))
    op=regmodel.predict(new_data)
    return jsonify(op[0])


if __name__=="__main__":
    app.run(debug=True)
    
    
    
    
    
