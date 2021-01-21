import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures

@st.cache
def read_data():
    return pd.read_csv('datatermoRedact.csv')

df=read_data()

X = df[['Kq','angle/90', 'h/lo', 'D/lo', 'd/lo', 'u/lo', 's/lo', 'Pr']]
y = df['a/a_smooth_Bor']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

@st.cache
def rndFors():
    rndm=RandomForestRegressor(n_estimators=100, max_features ='sqrt')

    rndm.fit(X_train, y_train)
    rndm_y_pred = rndm.predict(X)
    firstY=df['a/a_smooth_Bor']
    firstY=firstY.values
    endYYY=rndm_y_pred/firstY
    endYYY=pd.DataFrame(data=endYYY)
    endYYY=endYYY.loc[:,0]
    return endYYY

YYY=rndFors()
plt.scatter(df['q'], YYY, edgecolor='b')
plt.ylim([0.1, 10])
plt.xlim([1000, 10000000])
plt.yscale('log')
plt.xscale('log')
plt.hlines(1, 0, 10000000, color = 'black', linewidth = 1)
plt.hlines(3, 0, 10000000, color = 'black', linewidth = 2)
plt.hlines(0.6, 0, 10000000, color = 'black', linewidth = 2)
plt.text(1500, 3.2, '+30%')
plt.text(1500, 0.5, '-30%')
#plt.show()
st.pyplot(plt)



genre = st.radio("Выберите вид структуры 3D или 2D",('3D', '2D'))
if genre == '3D':
#Kq
    x1 = st.sidebar.slider('Kq', min_value=2, max_value=12000)

#угол/90
    x2 = st.sidebar.slider('угол/90', min_value=0.78, max_value=1.00)

#h/lo
    x3 = st.sidebar.slider('h/lo', min_value=0.01, max_value=1.00)

#D/lo
    x4 = st.sidebar.slider('D/lo', min_value=0.01, max_value=1.00)

#d/lo
    x5 = st.sidebar.slider('d/lo', min_value=0.01, max_value=1.00)

#u/lo
    x6 = st.sidebar.slider('u/lo', min_value=0.01, max_value=1.00)

#s/lo
    x7 = st.sidebar.slider('s/lo', min_value=0.01, max_value=1.00)

    y=1.49*x1**(-0.15)*x2**(-1.720)*x3**(0.313)*x4**(0.069)*x5**(0.078)*x6**(-0.454)*x7**(-0.492)

    col1, col2= st.beta_columns(2)
    with col1:
        st.header("3D структура")
        st.image('3d.jpg',  use_column_width=True)
    with col2:
        st.header("Значение теплоотдачи")  
        st.write('Kq=', x1,'; ','угол/90=', x2,'; ','h/lo=', x3,'; ','D/lo=', x4,'; ','d/lo=', x5,'; ','u/lo=', x6,'; ','s/lo=', x7)
        st.write('α/α0=',y)
else:
#Kq
    x1 = st.sidebar.slider('Kq', min_value=2, max_value=12000)

#угол/90
    x2 = st.sidebar.slider('угол/90', min_value=0.01, max_value=1.00)

#h/lo
    x3 = st.sidebar.slider('h/lo', min_value=0.01, max_value=1.00)

#D/lo
    x4 = st.sidebar.slider('D/lo', min_value=0.01, max_value=1.00)

#d/lo
    x5 = st.sidebar.slider('d/lo', min_value=0.01, max_value=1.00)


    y=6*x1**(-0.2)*x2**(0.554)*x3**(0.19)*x4**(0.201)*x5**(-0.394)

    col1, col2= st.beta_columns(2)
    with col1:
        st.header("2D структура")
        st.image('2d.jpg',  use_column_width=True)
    with col2:
        st.header("Значение теплоотдачи")
        st.write('Kq=', x1,'; ','угол/90=', x2,'; ','h/lo=', x3,'; ','D/lo=', x4,'; ','d/lo=', x5)
        st.write('α/α0=',y)
