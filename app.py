import pandas as pd
import streamlit as st
import joblib

model = joblib.load('Logistic_Heart.pkl')
scaler = joblib.load('scaker.pkl')
expected_columns = joblib.load('columns.pkl')

st.title("heart strpke prediction by Prince")
st.markdown('Provide the following details')

age = st.slider('Age', 18, 100, 40)
