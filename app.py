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

# Rodar o aplicativo
if __name__ == "__main__":
    main()