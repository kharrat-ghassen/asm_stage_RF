#pip install flask
#pip install numpy
#pip install scikit-learn==1.3.2
#python.exe -m pip install --upgrade pip
from flask import Flask, render_template, request, send_from_directory
import pickle
import numpy as np
import math
app = Flask(__name__)
# Define the file path
directory_Mnt_HT='C:/Users/kharr/Downloads/test - RF/models/RF_sales_prediction_model_Mnt_HT.pkl'
directory_Mnt_TTC='C:/Users/kharr/Downloads/test - RF/models/RF_sales_prediction_model_Mnt_TTC.pkl'
directory_Marge_HT='C:/Users/kharr/Downloads/test - RF/models/RF_sales_prediction_model_Marge_HT.pkl'
# Open the file and load the model
model_Mnt_HT = pickle.load(open(directory_Mnt_HT, 'rb'))
model_Mnt_TTC = pickle.load(open(directory_Mnt_TTC, 'rb'))
model_Marge_HT = pickle.load(open(directory_Marge_HT, 'rb'))
@app.route("/")
def home():
    return render_template("home.html")
@app.route('/predict', methods=['POST', 'GET'])
def predict():
    day = int(request.form['day'])
    month = int(request.form['month'])
    year = int(request.form['year'])
    station = request.form['station']
    unique_station=['POINT DE VENTE SILIANA', 'POINT DE VENTE KEF', 'POINT DE VENTE SFAX', 'POINT DE VENTE GAFSA', 'POINT DE VENTE TUNIS', 'POINT DE VENTE SOUSSE', 'POINT DE VENTE NABEUL']
    station_encoded=unique_station.index(station)

    X = np.array([[day,month,year,station_encoded]])

    Y_pred_Mnt_HT = model_Mnt_HT.predict(X)
    Y_pred_Mnt_TTC = model_Mnt_TTC.predict(X)
    Y_pred_Marge_HT = model_Marge_HT.predict(X)

    Mnt_HT_pred=round(Y_pred_Mnt_HT[0],3)
    Mnt_TTC_pred=round(Y_pred_Mnt_TTC[0],3)
    Marge_HT_pred=round(Y_pred_Marge_HT[0],3)

    date=f'{day}/{month}/{year}'

    return render_template("predict.html" , date=date , station=station , Mnt_HT_pred=Mnt_HT_pred , Mnt_TTC_pred=Mnt_TTC_pred , Marge_HT_pred=Marge_HT_pred)

if __name__ == "__main__":
    app.run()#