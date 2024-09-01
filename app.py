# 1 - IMPORTAÇÃO DAS BILIOTECAS
# -------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import to_time_series_dataset
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import datetime

# 2 - IMPORTAÇÃO DOS DATASETS
# -------------------------------------------------

# dataset do histórico dos preços dos ativos
df_historico = pd.read_csv('./data/df_model.csv', index_col=['Unnamed: 0', 'Unnamed: 1'])

# dataset para normalização dos dados
df_pre_normal = pd.read_csv('./data/df_pre_normal.csv', index_col=['Unnamed: 0', 'Unnamed: 1'])

base = pd.read_csv('./data/h_consolidado.csv')

# 3 - CONSTRUÇÃO DE FUNÇÕES PARA CLUSTERIZAÇÃO DE DADOS
# -------------------------------------------------

# 3.1 função para normalização dos dados
def modelar_dados(ativos=['11'], data_inicial='2024-07-26'):

  df = df_pre_normal.loc[(ativos,),data_inicial:]
  df.dropna(inplace=True)
  colunas = df.columns
  # reconstruindo o histórico dos ativos normalizado
  df_normal = pd.DataFrame(index=df.index, columns=colunas)
  df_normal[data_inicial] = 1

  for i in range(len(colunas[1:])):
    df_normal[colunas[i + 1]] = df_normal[colunas[i]] * df[colunas[i + 1]]
   
  return df_normal # histórico dos ativos normalizados

# função para clusterizar os dados
@st.cache_resource
def clusterizar(n_cluster=2):

  km = TimeSeriesKMeans(n_clusters=n_cluster, n_init=2, verbose=True, metric='dtw')

  return km  

# função para construção de dataframes com histórico dos preços clusterizados
@st.cache_resource
def dados_acoes(df_normal):

  valores = []
  indices = []

  for indice in df_normal.index:
    valores.append(df_historico.loc[indice, df_normal.columns[0]:].values)
    indices.append(indice[1])  

  dataframe = pd.DataFrame({'valores': valores}, index=indices)
    
  return dataframe  

def welcome():
    return 'welcome all'
  
# 4 - RENDERIZAÇÃO DO APP
# -------------------------------------------------
  
# início função principal para a construção o app 
def main():    

    st.set_page_config(page_title = 'Renda Analisys', \
        page_icon = './images/telmarketing_icon.png',
        layout ='wide',
        initial_sidebar_state='expanded')
     
       
    st.title('Clustering de Ativos Financeiros')
    st.subheader('', divider='rainbow')
    
    st.subheader('Somatório das rendas')

    # recuperação das funções
    dados = modelar_dados()
    km = clusterizar()
    # preparando os dados para clusterização
    dados_model = to_time_series_dataset(dados)
    # clusterizando os dados
    y_pred = km.fit_predict(dados_model)
    # salvando os clusters nos dataframes
    dados['cluster'] = y_pred
    variacoes = dados_acoes(dados)

    st.sidebar.write("## Faça sua clusterização aqui")

    with st.sidebar.form(key='my_form'):

      tipo_ativo = st.multiselect("Tipo de Ativo (conf. sufixo numérico)", options=dados.index.levels[0], default=['11'])
      n_cluster = st.number_input('Quantidade de Clusters', min_value=2, max_value=9, step=1)
      data_inicial = st.date_input("Data Inicial do Histórico dos Ativos", 
                                  min_value=datetime.date(2021, 1, 4), 
                                  max_value=datetime.date(2024, 7, 31))
                               

      data_inicial = str(data_inicial)

    

      if st.form_submit_button("Predict"):
        # recuperação das funções  
        dados = modelar_dados(tipo_ativo, data_inicial)
        km = clusterizar(n_cluster)
        # preparando os dados para clusterização
        dados_model = to_time_series_dataset(dados)
        # clusterizando os dados
        y_pred = km.fit_predict(dados_model)
        # salvando os clusters nos dataframes
        dados['cluster'] = y_pred
        variacoes = dados_acoes(dados)

    # plotando gráfico de barras representando o tamanho dos clusters
    fig_1 = px.histogram(dados, x='cluster', color_discrete_sequence=['#838A08'])
    st.plotly_chart(fig_1, theme="streamlit", use_container_width=True)

    # gráfigo preços ativos

    with st.expander("See explanation"):

      ativo = st.multiselect("Tipo de Ativo (conf. sufixo numérico)", options=variacoes.index)

      ff = base[base.cod_negociacao.isin(ativo)]

      fig_2 = go.Figure(data=[go.Candlestick(x=ff['data_pregao'],
                      open=ff['preco_abertura'],
                      high=ff['preco_maximo'],
                      low=ff['preco_minimo'],
                      close=ff['preco_ultimo_negocio'])])

      fig_2.update_layout(xaxis_rangeslider_visible=False)
      fig_2.update_traces(increasing_line_color='#b58900', decreasing_line_color='#e83e8c')
      st.plotly_chart(fig_2, theme="streamlit", use_container_width=True)

    # preparativos para a plotagem dos dataframes
    lista_tabs = ['Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5', 'Cluster 6', 'Cluster 7', 'Cluster 8'  ]
    tabs = st.tabs(lista_tabs[:n_cluster])

    with st.container():
      for cluster in range(n_cluster):      

        with tabs[cluster]:

          col1, col2 = st.columns([3, 2])

          with col1:
                
            dados_var = dados.loc[dados['cluster'] == cluster]
            variacoes = dados_acoes(dados_var)
            st.subheader('Histórico dos Ativos')
            st.dataframe(variacoes, use_container_width=True, column_config={'valores': st.column_config.LineChartColumn('valores')}) 

          with col2:
          
            fig, ax = plt.subplots()
            fig.set_facecolor('#424242')
            ax.set_facecolor('#424242')
            ax.spines[['top', 'left', 'right', 'bottom']].set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])
            for xx in dados_model[y_pred == cluster]:

              plt.plot(xx.ravel(), "k-", alpha=.2)
              plt.plot(km.cluster_centers_[cluster].ravel(), color='#838A08')
            st.subheader('Variações dos Ativos')
            st.pyplot(fig)

if __name__=='__main__':
    main()