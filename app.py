import streamlit as st
import gdown
import pickle
from prophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from datetime import datetime
import numpy as np

# Função para carregar o modelo Prophet
@st.cache_resource
def carregar_modelo():
    url = 'https://drive.google.com/uc?id=11eL3dI9aeUGjVKUSDGrDccLJ4tPQMHTD'
    gdown.download(url, 'prophet_model.pkl', quiet=False)

    with open('prophet_model.pkl', 'rb') as f:
        modelo = pickle.load(f)
    return modelo

# Função para carregar os dados
@st.cache_data
def carregar_dados():
    url = 'https://raw.githubusercontent.com/marialuisamartins/tech_fase4/6ad3e07bc901fd984eedb3030510b2816aaf7383/ipeadata%5B03-11-2024-01-09%5D.xlsx'
    response = requests.get(url)
    with open('ipeadata.xlsx', 'wb') as file:
        file.write(response.content)
    ipeadata = pd.read_excel('ipeadata.xlsx', engine='openpyxl')
    ipeadata['data'] = pd.to_datetime(ipeadata['data'])
    return ipeadata

def main():
    # Configuração da página
    st.set_page_config(page_title='MVP para análise temporal de petróleo', page_icon='🛢️')

    st.write('# MVP para análise de preço do petróleo Brent')

    # Carregar dados e modelo
    modelo = carregar_modelo()
    dados = carregar_dados()
    st.success("Modelo e dados carregados com sucesso!")

    # Processar dados
    dados.set_index('data', inplace=True)
    dados_filtered = dados[dados.index >= '2021-01-01']

    # Exibição de estatísticas
    st.markdown("### Estatísticas do Período (2021-2024)")
    st.write(f"**Valor máximo**: ${dados_filtered['preco'].max():,.2f}")
    st.write(f"**Valor mínimo**: ${dados_filtered['preco'].min():,.2f}")
    st.write(f"**Média**: ${dados_filtered['preco'].mean():,.2f}")
    st.write(f"**Desvio Padrão**: ${dados_filtered['preco'].std():,.2f}")

    # Previsão
    futuro = modelo.make_future_dataframe(periods=2)
    forecast = modelo.predict(futuro)
    previsao_hoje = forecast.iloc[-2]['yhat']
    previsao_amanha = forecast.iloc[-1]['yhat']
    tendencia = "subindo" if previsao_amanha > previsao_hoje else "descendo"
    icone_tendencia = "🔼" if tendencia == "subindo" else "🔽"

    st.markdown("### Previsão")
    st.write(f"**Hoje**: ${previsao_hoje:,.2f}")
    st.write(f"**Amanhã**: ${previsao_amanha:,.2f} {icone_tendencia} ({tendencia})")

    # Gráficos
    st.markdown("### Gráfico de Preço Diário")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=dados_filtered, x=dados_filtered.index, y='preco', ax=ax, color='blue')
    ax.set_title('Evolução do Preço do Petróleo (2021-2024)')
    ax.set_xlabel('Data')
    ax.set_ylabel('Preço (USD)')
    st.pyplot(fig)

    # Gráfico de Média Móvel
    st.markdown("### Média Móvel de 30 dias")
    dados_filtered['media_movel_30'] = dados_filtered['preco'].rolling(30).mean()
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.plot(dados_filtered.index, dados_filtered['preco'], label='Preço Diário', color='blue', alpha=0.5)
    ax2.plot(dados_filtered.index, dados_filtered['media_movel_30'], label='Média Móvel de 30 dias', color='orange')
    ax2.set_title('Preço do Petróleo e Média Móvel (30 dias)')
    ax2.legend()
    st.pyplot(fig2)

# Executar o app
if __name__ == "__main__":
    main()
