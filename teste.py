import streamlit as st
import pandas as pd

st.title('Teste')

st.write("teste")

with st.container():
    dados = pd.read_csv('dados_biodiesel_OS.csv')
    st.write(dados.head())