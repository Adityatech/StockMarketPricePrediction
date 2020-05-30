from flask import Flask,request,render_template,make_response,url_for
import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import warnings
import json
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
from datetime import date
import pandas_datareader as pdr
import os

app = Flask(__name__)

#IMAGES_FOLDER = os.path.join('static/images', 'charts')
#app.config['UPLOAD_FOLDER'] = IMAGES_FOLDER

def makePrediction(name):
    name = name.lower()
    today = date.today()
    try:
        df = pdr.data.get_data_yahoo(name, start='2020-03-01', end='today')
    except:
        return 'bad request!', 400
    df.reset_index(level=0, inplace=True)
    x = df[['High', 'Low']]
    y = df['Close']
    scaler = StandardScaler()
    svr = SVR(kernel='poly')
    # initialising pipeling
    regr = make_pipeline(scaler, svr)
    model = regr.fit(x, y)
    z = len(df)
    l = df['Low'][z - 1]
    h = df['High'][z - 1]
    c = df['Close'][z - 1]
    PP = (l + h + c) / 3
    R1 = (2 * PP - l)
    S1 = (2 * PP - h)
    closing = model.predict([[R1, S1]])
    A = (R1 + S1 + closing[0]) / 3
    R2 = (2 * A - S1)
    S2 = (2 * A - R1)
    closing2 = model.predict([[R2, S2]])[0]
    slbuy = S1 - (S1 * 0.2) / 100
    slsell = R1 + (R1 * 0.2) / 100
    train = df.loc[:z]
    valid = df.loc[z:]
    new_index = [z - 1, z + 1]
    valid = valid.reindex(new_index)
    previous_close = df['Close'].iloc[z - 1]
    # Visualize the data
    valid['Predictions'] = [closing, closing2]
    plt.figure(figsize=(10, 8))
    plt.title('Model')
    plt.xlabel('days', fontsize=18)
    plt.ylabel('Close Price ', fontsize=18)
    plt.plot(train['Close'])
    plt.plot(valid[['Predictions']])
    plt.legend(['Train', 'Predictions'], loc='lower right')
    filename = "images/"+name+".png"
    plt.savefig("./static/"+filename)
    closing = round(closing[0],2)
    slbuy = round(slbuy,2)
    slsell = round(slsell,2)
    if(slsell<closing):
        slsell = slsell+(closing-slsell)+1
    if(slbuy>closing):
        slbuy = slbuy - (slbuy-closing)-1
    if(previous_close>closing):
        sell = slsell
    else:
        sell=slbuy
    return closing,filename,sell


@app.route("/")
def home():
    return render_template('index.html')

@app.route("/predict",methods=['GET', 'POST'])
def predict():
    print("predict called")
    name = request.form['name']
    closing2,filename,sell = makePrediction(name)
    return render_template('dashboard.html',filename = filename,closing = closing2,sell=sell)

if __name__ == '__main__':
    app.run()