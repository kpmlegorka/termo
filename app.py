import streamlit as st
import pandas as pd


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