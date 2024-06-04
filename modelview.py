import streamlit as st
import numpy as np
import joblib

st.markdown('Iris Classifier')
sepal_length= st.number_input('Enter the sepal length(cm)')
sepal_width= st.number_input('Enter the sepal width(cm)')
petal_length= st.number_input('Enter the petal length(cm)')
petal_width= st.number_input('Enter the petal_width(cm)')

if st.button('Predict'):
    model= joblib.load('iris_model.pkl')
    X= np.array([sepal_length,sepal_width,petal_length,petal_width])
    if any(X<=0):
        st.markdown('inputs must be greater than 0')
    else:
        # st.markdown(' The prediction is %d')
        st.markdown(f'### Prediction is {model.predict([[sepal_length, sepal_width, petal_length, petal_width]])[0]}')
