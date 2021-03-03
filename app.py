import streamlit as st
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
#from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures

import xgboost
from sklearn import model_selection    #cross_validation


'''
# Подбор геометрических параметров микроструктуры
#### Чтобы прогнозировать теплоотдачу с помощью "нейросетей", используйте переключатели ниже _(замедляют быстродействие)_
'''

@st.cache
def read_data():
    return pd.read_csv('datatermoRedact.csv')
df=read_data()

@st.cache
def read_dataU():
    return pd.read_csv('datatermoRedact2.csv')
Uf=read_dataU()

@st.cache
def read_data2():
    xd=pd.read_csv('only_XandY.csv')
    xd.drop(['Unnamed: 0'], axis='columns', inplace=True)
    return xd
xdd=read_data2()

XU_train = Uf[['Kq','angle/90', 'h/lo', 'D/lo', 'd/lo']]
yU_train = Uf['a/a_smooth_Bor']
#XU_train, XU_test, yU_train, yU_test = train_test_split(XU, yU, test_size=0.3, random_state=0)

X_train = df[['Kq','angle/90', 'h/lo', 'D/lo', 'd/lo', 'u/lo', 's/lo']]
y_train = df['a/a_smooth_Bor']
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

colum1, colum2, colum3= st.beta_columns(3)
with colum1:
    rndFors=st.checkbox("RandomForest", False)
with colum2:
    linReg=st.checkbox("LinearRegression", False)
with colum3:
    nerKa=st.checkbox("XGBoost", False)



genre = st.radio("Выберите вид структуры 3D или 2D",('3D', '2D'))
if genre == '3D':
#Kq
    x1 = st.sidebar.slider('Kq', min_value=8, max_value=12000,  value=1190)

#угол/90
    x2 = st.sidebar.slider('угол/90', min_value=0.78, max_value=1.00,  value=0.89)

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

    y=1.49*x1**(-0.15)*x2**(-1.720)*x3**(0.313)*x4**(0.069)*x5**(0.078)*x6**(-0.454)*x7**(-0.492)   
    data_slider = {'Kq': [x1], 'угол/90': [x2], 'h/lo': [x3], 'D/lo': [x4], 'd/lo': [x5], 'u/lo': [x6], 's/lo': [x7]}
    nm = pd.DataFrame(data=data_slider)
    xnm=np.array([[x1, x2, x3, x4, x5, x6, x7]])
    col1, col2= st.beta_columns(2)
    with col1:
        st.header("3D структура")
        st.image('3d.jpg',  use_column_width=True)
    with col2:
        st.header("Значение теплоотдачи")  
        st.write('Kq=', x1,'; ','угол/90=', x2,'; ','h/lo=', x3,'; ','Δ/lo=', x4,'; ','δ/lo=', x5,'; ','u/lo=', x6,'; ','s/lo=', x7)
        st.write('Формула: α/α0=',round(y, 2))
        if rndFors:
            rndm=RandomForestRegressor(n_estimators=100, max_features ='sqrt')
            rndm.fit(X_train, y_train)
            y_forest=rndm.predict(nm)
            st.write('Лес: α/α0=',round(y_forest[0], 2))
        if linReg:
            lm = LinearRegression()
            model = lm.fit(X_train, y_train)
            y_linReg = lm.predict(nm)
            st.write('ЛинРегрессия: α/α0=',round(y_linReg[0], 2))
        if nerKa:
            dataset=xdd.to_numpy()      
            X_np = dataset[:,0:7]
            y_np = dataset[:,7]
            Xmodel = xgboost.XGBRegressor()
            Xmodel.fit(X_np, y_np)
            y_nerKa = Xmodel.predict(xnm)
            st.write('Градиент: α/α0=',round(y_nerKa[0], 2))
else:
#Kq
    x1 = st.sidebar.slider('Kq', min_value=13, max_value=13660,  value=203)

#угол/90
    x2 = st.sidebar.slider('угол/90', min_value=0.72, max_value=1.00,  value=1.00)

#h/lo
    x3 = st.sidebar.slider('h/lo', min_value=0.03, max_value=1.45,  value=1.45)

#D/lo
    x4 = st.sidebar.slider('Δ/lo', min_value=0.01, max_value=1.30,  value=1.29)

#d/lo
    x5 = st.sidebar.slider('δ/lo', min_value=0.01, max_value=1.00,  value=0.22)


    y=2.66*x1**(-0.09)*x2**(-0.091)*x3**(0.133)*x4**(0.035)*x5**(-0.149)
    
    data_slider = {'Kq': [x1], 'угол/90': [x2], 'h/lo': [x3], 'D/lo': [x4], 'd/lo': [x5]}
    nm = pd.DataFrame(data=data_slider)
    xnm=np.array([[x1, x2, x3, x4, x5]])
    col1, col2= st.beta_columns(2)
    with col1:
        st.header("2D структура")
        st.image('2d.jpg',  use_column_width=True)
    with col2:
        st.header("Значение теплоотдачи")
        st.write('Kq=', x1,'; ','угол/90=', x2,'; ','h/lo=', x3,'; ','Δ/lo=', x4,'; ','δ/lo=', x5)
        st.write('Формула: α/α0=',round(y, 2))
        if rndFors:
            rndm=RandomForestRegressor(n_estimators=100, max_features ='sqrt')
            rndm.fit(XU_train, yU_train)
            y_forest=rndm.predict(nm)
            st.write('Лес: α/α0=',round(y_forest[0], 2))
        if linReg:
            lm = LinearRegression()
            model = lm.fit(XU_train, yU_train)
            y_linReg = lm.predict(nm)
            st.write('ЛинРегрессия: α/α0=',round(y_linReg[0], 2))
        if nerKa:
            dataset=xdd.to_numpy()      
            X_np = dataset[:,0:7]
            y_np = dataset[:,7]
            Xmodel = xgboost.XGBRegressor()
            Xmodel.fit(X_np, y_np)
            y_nerKa = Xmodel.predict(xnm)
            st.write('Градиент: α/α0=',round(y_nerKa[0], 2))
