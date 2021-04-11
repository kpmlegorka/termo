import streamlit as st
import pandas as pd
import numpy as np
import joblib



'''
# Подбор геометрических параметров микроструктуры
#### Чтобы прогнозировать теплоотдачу с помощью "нейросетей", используйте переключатели ниже _(замедляют быстродействие)_
'''

rndm = joblib.load('rndmF_model.pkl')
Xmodel=joblib.load('XGBR_model.pkl')
lm=joblib.load('GBR_model.pkl')

rndm2 = joblib.load('rndmF_model_2.pkl')
Xmodel2=joblib.load('XGBR_model_2.pkl')
lm2=joblib.load('GBR_model_2.pkl')


colum1, colum2, colum3= st.beta_columns(3)
with colum1:
    rndFors=st.checkbox("RandomForest", False)
with colum2:
    linReg=st.checkbox("GBRegressor", False)
with colum3:
    nerKa=st.checkbox("XGBoost", False)



genre = st.radio("Выберите вид структуры: 3D или 2D",('3D', '2D'))
if genre == '3D':
#Kq
    x1 = st.sidebar.slider('Kq', min_value=8, max_value=12000,  value=1190)

#angle/90
    x2 = st.sidebar.slider('angle/90', min_value=0.78, max_value=1.00,  value=0.89)

#h/lo
    x3 = st.sidebar.slider('h/lo', min_value=0.09, max_value=0.71,  value=0.23)

#D/lo
    x4 = st.sidebar.slider('Δ/lo', min_value=0.01, max_value=0.30,  value=0.08)

#d/lo
    x5 = st.sidebar.slider('δ/lo', min_value=0.04, max_value=0.40,  value=0.06)

#u/lo
    x6 = st.sidebar.slider('u/lo', min_value=0.01, max_value=0.30,  value=0.07)

#s/lo
    x7 = st.sidebar.slider('s/lo', min_value=0.01, max_value=0.79,  value=0.06)
#Pr
    x8 = st.sidebar.slider('Pr', min_value=1.7, max_value=6.8,  value=1.75)

    y=1.49*x1**(-0.15)*x2**(-1.720)*x3**(0.313)*x4**(0.069)*x5**(0.078)*x6**(-0.454)*x7**(-0.492)   
    data_slider = {'Kq': [x1], 'angle/90': [x2], 'h/lo': [x3], 'D/lo': [x4], 'd/lo': [x5], 'u/lo': [x6], 's/lo': [x7], 'Pr': [x8]}
    nm = pd.DataFrame(data=data_slider)
    
    col1, col2= st.beta_columns(2)
    with col1:
        st.header("3D структура")
        st.image('3d.jpg',  use_column_width=True)
    with col2:
        st.header("Значение интесификации теплоотдачи")  
        st.write('Kq=', x1,'; ','angle/90=', x2,'; ','h/lo=', x3,'; ','Δ/lo=', x4,'; ','δ/lo=', x5,'; ','u/lo=', x6,'; ','s/lo=', x7, '; ','Pr', x8)
        st.write('Полиноминальная регрессия: α/α0=',round(y, 2))
        if rndFors:
            #rndm=RandomForestRegressor(n_estimators=100, max_features ='sqrt')
            #rndm.fit(X_train, y_train)
            y_forest=rndm.predict(nm)
            st.write('RandomForest: α/α0=',round(y_forest[0], 2))
        if linReg:
            #lm = LinearRegression()
            #model = lm.fit(X_train, y_train)
            y_linReg = lm.predict(nm)
            st.write('GBRegressor: α/α0=',round(y_linReg[0], 2))
        if nerKa:
            #dataset=xdd.to_numpy()      
            #X_np = dataset[:,0:7]
            #y_np = dataset[:,7]
            #Xmodel = xgboost.XGBRegressor()
            #Xmodel.fit(X_np, y_np)
            y_nerKa = Xmodel.predict(nm)  #(xnm)
            st.write('XGBoost: α/α0=',round(y_nerKa[0], 2))
else:
#Kq
    x1 = st.sidebar.slider('Kq', min_value=13, max_value=13660,  value=203)

#угол/90
    x2 = st.sidebar.slider('angle/90', min_value=0.72, max_value=1.00,  value=1.00)

#h/lo
    x3 = st.sidebar.slider('h/lo', min_value=0.03, max_value=1.45,  value=1.45)

#D/lo
    x4 = st.sidebar.slider('Δ/lo', min_value=0.01, max_value=1.30,  value=1.29)

#d/lo
    x5 = st.sidebar.slider('δ/lo', min_value=0.01, max_value=1.00,  value=0.22)
#Pr
    x6 = st.sidebar.slider('Pr', min_value=1.7, max_value=6.8,  value=1.75)

    y=2.66*x1**(-0.09)*x2**(-0.091)*x3**(0.133)*x4**(0.035)*x5**(-0.149)
    
    data_slider = {'Kq': [x1], 'angle/90': [x2], 'h/lo': [x3], 'D/lo': [x4], 'd/lo': [x5], 'Pr': [x6]}
    nm = pd.DataFrame(data=data_slider)
    xnm=np.array([[x1, x2, x3, x4, x5]])
    col1, col2= st.beta_columns(2)
    with col1:
        st.header("2D структура")
        st.image('2d.jpg',  use_column_width=True)
    with col2:
        st.header("Значение теплоотдачи")
        st.write('Kq=', x1,'; ','angle/90=', x2,'; ','h/lo=', x3,'; ','Δ/lo=', x4,'; ','δ/lo=', x5,'; ','Pr=', x6)
        st.write('Формула: α/α0=',round(y, 2))
        if rndFors:
            #rndm=RandomForestRegressor(n_estimators=100, max_features ='sqrt')
            #rndm.fit(XU_train, yU_train)
            y_forest=rndm2.predict(nm)
            st.write('Лес: α/α0=',round(y_forest[0], 2))
        if linReg:
            #lm = LinearRegression()
            #model = lm.fit(XU_train, yU_train)
            y_linReg = lm2.predict(nm)
            st.write('ЛинРегрессия: α/α0=',round(y_linReg[0], 2))
        if nerKa:
            #dataset=xdd.to_numpy()      
            #X_np = dataset[:,0:7]
            #y_np = dataset[:,7]
            #Xmodel = xgboost.XGBRegressor()
            #Xmodel.fit(X_np, y_np)
            y_nerKa = Xmodel2.predict(xnm)
            st.write('Градиент: α/α0=',round(y_nerKa[0], 2))
