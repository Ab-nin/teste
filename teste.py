import streamlit as st
import pandas as pd

st.title('Teste')

st.write("teste")

with st.container():
    dados = pd.read_excel('dados_CN.xlsx')
    st.write(dados.head())