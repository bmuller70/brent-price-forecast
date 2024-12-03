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
    url = 'https://drive.google.com/uc?id=11eL3dI9aeUGjVKUSDGrDccLJ4tPQMHTD'
    gdown.download(url, 'prophet_model.pkl', quiet=False)

    with open('prophet_model.pkl', 'rb') as f:
        modelo = pickle.load(f)
    
    return modelo

# Função para baixar e salvar o arquivo Excel
@st.cache_data
def dados_xls():
    url = 'https://raw.githubusercontent.com/marialuisamartins/tech_fase4/6ad3e07bc901fd984eedb3030510b2816aaf7383/ipeadata%5B03-11-2024-01-09%5D.xlsx'
    arquivo_local = 'ipeadata.xlsx'
    response = requests.get(url)
    with open(arquivo_local, 'wb') as file:
        file.write(response.content)

    return arquivo_local

# Função principal do aplicativo
def main():
    st.set_page_config(page_title='MVP para análise temporal de petróleo', page_icon='🛢️')
    st.write('# MVP para análise de preço do petróleo Brent')

    # Carregar modelo e dados
    arquivo_excel = dados_xls()
    modelo = carregar_modelo()
    st.success("Modelo carregado com sucesso!")
    
    # Ler o arquivo Excel
    ipeadata = pd.read_excel(arquivo_excel, engine='openpyxl')
    ipeadata['data'] = pd.to_datetime(ipeadata['data'], errors='coerce')
    ipeadata = ipeadata.dropna(subset=['data', 'preco'])
    ipeadata.set_index('data', inplace=True)
    ipeadata_filtered = ipeadata[ipeadata.index >= '2021-01-01']

    # Estilização HTML
    st.markdown("""
        <style>
            .header-bar {
                background-color: #2C3E50;
                padding: 10px;
                color: white;
                text-align: center;
                font-size: 24px;
                font-weight: bold;
            }
        </style>
        <div class="header-bar">Dashboard Interativo de Previsão e Análise do Preço do Petróleo</div>
    """, unsafe_allow_html=True)

    st.sidebar.image('https://upload.wikimedia.org/wikipedia/commons/5/54/Logo_FIAP.png', use_container_width=True)
    st.sidebar.header("Exploração do Preço do Petróleo")

    # Estatísticas
    st.markdown("### Estatísticas e Insights")
    st.write(f"**Valor máximo no período**: ${ipeadata_filtered['preco'].max():,.2f}")
    st.write(f"**Valor mínimo no período**: ${ipeadata_filtered['preco'].min():,.2f}")
    st.write(f"**Média do período**: ${ipeadata_filtered['preco'].mean():,.2f}")
    st.write(f"**Desvio padrão**: ${ipeadata_filtered['preco'].std():,.2f}")

    # Previsões
    df_forecast = pd.DataFrame({'ds': ipeadata_filtered.index, 'y': ipeadata_filtered['preco']})
    future = modelo.make_future_dataframe(periods=2)
    forecast = modelo.predict(future)
    previsao_hoje = forecast[forecast['ds'] == df_forecast.index[-1]]['yhat'].values[0]
    previsao_amanha = forecast[forecast['ds'] == future['ds'].iloc[-1]]['yhat'].values[0]
    tendencia = "subindo" if previsao_amanha > previsao_hoje else "descendo"
    icone_tendencia = "🔼" if tendencia == "subindo" else "🔽"

    st.markdown("### Previsão de Preço")
    st.write(f"**Previsão para hoje**: ${previsao_hoje:,.2f}")
    st.write(f"**Previsão para amanhã**: ${previsao_amanha:,.2f} {icone_tendencia} ({tendencia})")

    # Gráfico comparativo por ano
    ipeadata_filtered['year'] = ipeadata_filtered.index.year
    fig1, ax1 = plt.subplots(figsize=(14, 6))
    sns.lineplot(data=ipeadata_filtered, x=ipeadata_filtered.index, y='preco', hue='year', marker='o', palette='tab10')
    ax1.set_title("Comparação de Preços por Ano")
    st.pyplot(fig1)

    # Gráfico de picos mensais
    ipeadata_filtered['month'] = ipeadata_filtered.index.month
    monthly_max = ipeadata_filtered.groupby('month')['preco'].max()
    fig2, ax2 = plt.subplots(figsize=(14, 6))
    monthly_max.plot(kind='bar', color='orange', ax=ax2)
    ax2.set_title("Picos Mensais de Preço")
    st.pyplot(fig2)

    # Últimos 15 dias
    st.markdown("### Últimos 15 Dias")
    fig3, ax3 = plt.subplots(figsize=(14, 6))
    ax3.plot(ipeadata_filtered.tail(15).index, ipeadata_filtered.tail(15)['preco'], marker='o', linestyle='-', color='blue')
    ax3.set_title("Preços - Últimos 15 Dias")
    st.pyplot(fig3)

    # Análises adicionais
    analise_opcao = st.selectbox("Escolha uma análise:", ["Média Móvel de 7 dias", "Distribuição dos Preços"])
    if analise_opcao == "Média Móvel de 7 dias":
        ipeadata_filtered['media_movel_7'] = ipeadata_filtered['preco'].rolling(window=7).mean()
        fig4, ax4 = plt.subplots(figsize=(14, 6))
        ax4.plot(ipeadata_filtered.index, ipeadata_filtered['media_movel_7'], label="Média Móvel 7 dias", color='orange')
        st.pyplot(fig4)
    elif analise_opcao == "Distribuição dos Preços":
        fig5, ax5 = plt.subplots(figsize=(12, 6))
        sns.histplot(ipeadata_filtered['preco'], kde=True, bins=30, color='skyblue', ax=ax5)
        st.pyplot(fig5)

# Rodar o aplicativo
if __name__ == "__main__":
    main()
