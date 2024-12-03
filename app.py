import streamlit as st
import gdown
import pickle
from prophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import requests
import numpy as np

# Função para carregar o modelo Prophet
@st.cache_resource
def carregar_modelo():
    # URL do modelo Prophet salvo no Google Drive
    #https://drive.google.com/file/d/11eL3dI9aeUGjVKUSDGrDccLJ4tPQMHTD/view?usp=sharing
    url = 'https://drive.google.com/uc?id=11eL3dI9aeUGjVKUSDGrDccLJ4tPQMHTD'
    
    # Fazer o download do modelo
    gdown.download(url, 'prophet_model.pkl', quiet=False)
    
    # Carregar o modelo Prophet com pickle
    with open('prophet_model.pkl', 'rb') as f:
        modelo = pickle.load(f)
    
    return modelo

# Função principal do aplicativo
def main():
    # Configuração da página
    st.set_page_config(page_title='MVP para análise temporal de petróleo',
                       page_icon='🛢️')
    
    st.write('# MVP para análise de preço do petróleo Brent')

    # Carregar o modelo Prophet
    modelo = carregar_modelo()
    st.success("Modelo carregado com sucesso!")

    # Entradas do usuário para previsão
    st.sidebar.header("Configurações da Previsão")
    periods = st.sidebar.number_input("Quantos dias você quer prever?", min_value=1, value=30)

    # Botão para gerar previsão
    if st.button("Gerar Previsão"):
        # Criar DataFrame futuro
        future = modelo.make_future_dataframe(periods=periods)

        # Fazer a previsão
        forecast = modelo.predict(future)

        # Mostrar resultados
        st.subheader("Resultados da Previsão")
        st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])

        # Plotar o gráfico
        st.subheader("Gráfico da Previsão")
        st.line_chart(forecast[['ds', 'yhat']].set_index('ds'))

    # Seções adicionais como placeholders
    st.write("## Dashboard")
    st.write("## Insights")
    st.write("## Modelo")

# Rodar o aplicativo
if __name__ == "__main__":
    main()