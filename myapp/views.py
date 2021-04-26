from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.contrib import messages
import numpy as np
import pandas as pd
import datetime
import pickle as pickle
#from sklearn.externals import joblib 
import joblib as joblib

path1 = 'myapp/models/model.pkl'

columns = ['citygrp','type' ,'date' , 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10',
 'p11', 'p12', 'p13', 'p14', 'p15', 'p16', 'p17', 'p18', 'p19', 'p20', 'p21', 'p22', 'p23', 'p24', 'p25',
 'p26', 'p27', 'p28', 'p29', 'p30', 'p31', 'p32', 'p33', 'p34', 'p35', 'p36', 'p37']

def home(requests):
    if requests.method=='GET':
        return render(requests,'index.html')
    else:
        temp = []
        for i in range(len(columns)):
              temp.append(requests.POST[columns[i]])
        print(temp)

        model = joblib.load(path1) 
        enc = model['encoder']
        ridge_model = model['ridge']
        lgb_model= model['lgb']
        knn_model= model['knn']
        lasso_model= model['lasso']
        el_model= model['el']
        xgb_model = model['xgb']
        
        temp = pd.DataFrame([temp],columns=columns)
        a = pd.DataFrame(enc.transform(temp[['citygrp','type']]).toarray())
        temp1 = pd.to_datetime(temp["date"])
        launch_date = datetime.datetime(2015, 3, 23)
        # scale days open
        temp['Days Open'] =  (launch_date - temp1).dt.days/1000
        temp.drop(['citygrp','type','date'],inplace=True,axis=1)
        temp = a.join(temp)
        x = temp.iloc[0]
        ridge = ridge_model.predict([x])
        lgb = lgb_model.predict([x])
        knn = knn_model.predict([x])
        lasso = lasso_model.predict([x])
        el = el_model.predict([x])
        xgb = xgb_model.predict([x])
        
        params={
            'ridge' : np.expm1(ridge)[0],
            'lgb' : np.expm1(lgb)[0],
            'knn' : np.expm1(knn)[0],
            'el' : np.expm1(el)[0],
            'xgb' : np.expm1(xgb)[0],
            'lasso' : np.expm1(lasso)[0],
            }

        return render(requests, 'result.html', params)