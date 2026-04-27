
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# --- Funções de Processamento de Dados (Replicadas do Notebook) ---

def load_and_prepare_data(filepath):
    """Carrega os dados e realiza filtros básicos."""
    df = pd.read_csv(filepath, encoding="Latin 1", delimiter=";")

    df = df[df['NM_MUNICIPIO'].str.upper() == 'BATURITÉ']
    df = df[df['QT_VOTOS_NOMINAIS_VALIDOS'] > 0]

    return df

def feature_engineering(df):
    """Cria novas variáveis (features) que auxiliam no entendimento estratégico."""
    termos_eleito = ['ELEITO', 'ELEITO POR QP', 'ELEITO POR MÉDIA', 'SUPLENTE']
    df['IS_ELEITO'] = df['DS_SIT_TOT_TURNO'].apply(lambda x: 1 if str(x).upper() in termos_eleito else 0)

    df['TOTAL_VOTOS_CARGO'] = df.groupby('DS_CARGO')['QT_VOTOS_NOMINAIS_VALIDOS'].transform('sum')
    df['PERCENTUAL_VOTOS_CANDIDATO'] = (df['QT_VOTOS_NOMINAIS_VALIDOS'] / df['TOTAL_VOTOS_CARGO']) * 100

    df['VOTOS_TOTAIS_PARTIDO'] = df.groupby(['SG_PARTIDO', 'DS_CARGO'])['QT_VOTOS_NOMINAIS_VALIDOS'].transform('sum')
    df['QTD_CANDIDATOS_PARTIDO'] = df.groupby(['SG_PARTIDO', 'DS_CARGO'])['SQ_CANDIDATO'].transform('count')
    df['PESO_CANDIDATO_NO_PARTIDO_PERCENT'] = (df['QT_VOTOS_NOMINAIS_VALIDOS'] / df['VOTOS_TOTAIS_PARTIDO']) * 100

    return df

def eda_visualizations(df):
    """Gera visualizações estatísticas focadas em estratégia de campanha."""

    st.header("Análise Exploratória Geral")

    df_vereador = df[df['DS_CARGO'].str.upper() == 'VEREADOR'].copy()
    df_prefeito = df[df['DS_CARGO'].str.upper() == 'PREFEITO'].copy()

    if df_vereador.empty:
        st.warning("Nenhum dado de vereador encontrado para Baturité.")
        return

    # VISUALIZAÇÃO 1: Top 15 Vereadores Mais Votados
    st.subheader("Top 15 Vereadores Mais Votados")
    top_vereadores = df_vereador.sort_values(by='QT_VOTOS_NOMINAIS_VALIDOS', ascending=False).head(15)
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    sns.barplot(data=top_vereadores, x='QT_VOTOS_NOMINAIS_VALIDOS', y='NM_URNA_CANDIDATO', hue='SG_PARTIDO', dodge=False, ax=ax1)
    ax1.set_title('Top 15 Vereadores Mais Votados em Baturité (2024)')
    ax1.set_xlabel('Quantidade de Votos Válidos')
    ax1.set_ylabel('Candidato')
    ax1.legend(title='Partido', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    st.pyplot(fig1)

    # VISUALIZAÇÃO 2: Força Partidária
    st.subheader("Força dos Partidos - Total de Votos para Vereador")
    forca_partido = df_vereador[['SG_PARTIDO', 'VOTOS_TOTAIS_PARTIDO']].drop_duplicates().sort_values(by='VOTOS_TOTAIS_PARTIDO', ascending=False)
    fig2 = px.bar(forca_partido, x='SG_PARTIDO', y='VOTOS_TOTAIS_PARTIDO',
                       title='Força dos Partidos - Total de Votos para Vereador',
                       labels={'SG_PARTIDO': 'Partido', 'VOTOS_TOTAIS_PARTIDO': 'Total de Votos'},
                       text_auto='.2s', color='VOTOS_TOTAIS_PARTIDO', color_continuous_scale='Viridis')
    st.plotly_chart(fig2)

    # VISUALIZAÇÃO 3: Relação entre Nº de Candidatos do Partido vs Votos
    st.subheader("Estratégia dos Partidos: Nº de Candidatos vs Votos Totais")
    chapas = df_vereador[['SG_PARTIDO', 'QTD_CANDIDATOS_PARTIDO', 'VOTOS_TOTAIS_PARTIDO']].drop_duplicates()
    fig3 = px.scatter(chapas, x='QTD_CANDIDATOS_PARTIDO', y='VOTOS_TOTAIS_PARTIDO', text='SG_PARTIDO',
                             size='VOTOS_TOTAIS_PARTIDO', color='SG_PARTIDO',
                             title='Estratégia dos Partidos: N de Candidatos vs Votos Totais',
                             labels={'QTD_CANDIDATOS_PARTIDO': 'Quantidade de Candidatos na Chapa',
                                     'VOTOS_TOTAIS_PARTIDO': 'Total de Votos da Legenda'})
    fig3.update_traces(textposition='top center')
    st.plotly_chart(fig3)

    # VISUALIZAÇÃO 4: Distribuição de Votos - Eleitos vs Não Eleitos
    st.subheader("Distribuição de Votos: Não Eleitos (0) vs Eleitos (1)")
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df_vereador, x='IS_ELEITO', y='QT_VOTOS_NOMINAIS_VALIDOS', palette='Set2', ax=ax4)
    ax4.set_title('Distribuição de Votos: Não Eleitos / Suplentes (0) vs Eleitos (1)')
    ax4.set_xlabel('Status Eleitoral')
    ax4.set_ylabel('Votos Nominais Válidos')
    ax4.set_xticks([0, 1], labels=['Não Eleitos / Suplentes', 'Eleitos'])
    plt.tight_layout()
    st.pyplot(fig4)

    eleitos = df_vereador[df_vereador['IS_ELEITO'] == 1]
    if not eleitos.empty:
        nota_corte = eleitos['QT_VOTOS_NOMINAIS_VALIDOS'].min()
        st.write(f"**[ESTRATÉGIA]** O vereador eleito com menos votos teve: {nota_corte} votos.")
        st.write(f"**[ESTRATÉGIA]** A média de votos dos eleitos foi: {eleitos['QT_VOTOS_NOMINAIS_VALIDOS'].mean():.0f} votos.")
    else:
        st.write("**[ESTRATÉGIA]** Não há candidatos eleitos registrados para calcular a nota de corte.")

def analyze_candidate_plotly(df, candidate_urna_name):
    """
    Analisa e visualiza estatísticas de um candidato específico usando Plotly.
    """
    candidate_df = df[df['NM_URNA_CANDIDATO'].str.upper() == candidate_urna_name.upper()]

    if candidate_df.empty:
        st.error(f"❌ Candidato '{candidate_urna_name}' não encontrado no dataset.")
        return

    candidate_data = candidate_df.iloc[0]

    st.subheader(f"Análise Detalhada: {candidate_data['NM_CANDIDATO']} ({candidate_data['NM_URNA_CANDIDATO']})")
    st.write(f"**Partido:** {candidate_data['SG_PARTIDO']} - {candidate_data['NM_PARTIDO']}")
    st.write(f"**Cargo:** {candidate_data['DS_CARGO']}")
    st.write(f"**Situação:** {candidate_data['DS_SIT_TOT_TURNO']} (Eleito: {'Sim' if candidate_data['IS_ELEITO'] == 1 else 'Não'})")
    st.write(f"**Votos Nominais Válidos:** {candidate_data['QT_VOTOS_NOMINAIS_VALIDOS']:,}")
    st.write(f"**Percentual de Votos no Cargo:** {candidate_data['PERCENTUAL_VOTOS_CANDIDATO']:.2f}%")
    st.write(f"**Peso do Candidato no Partido:** {candidate_data['PESO_CANDIDATO_NO_PARTIDO_PERCENT']:.2f}%")
    st.write(f"**Total de Votos do Partido para o Cargo:** {candidate_data['VOTOS_TOTAIS_PARTIDO']:,}")
    st.write(f"**Total de Votos Válidos para o Cargo (Município):** {candidate_data['TOTAL_VOTOS_CARGO']:,}")

    # Visualização: Comparação de Votos
    comparison_data = pd.DataFrame({
        'Métrica': ['Votos do Candidato', 'Votos Totais do Partido', 'Votos Totais do Cargo'],
        'Valor': [
            candidate_data['QT_VOTOS_NOMINAIS_VALIDOS'],
            candidate_data['VOTOS_TOTAIS_PARTIDO'],
            candidate_data['TOTAL_VOTOS_CARGO']
        ]
    })

    fig_comp = px.bar(comparison_data, x='Métrica', y='Valor',
                 title=f'Comparativo de Votos para {candidate_data['NM_URNA_CANDIDATO']}',
                 color='Métrica',
                 text_auto=True,
                 labels={'Métrica': '', 'Valor': 'Quantidade de Votos'})
    fig_comp.update_layout(yaxis_title='Quantidade de Votos')
    st.plotly_chart(fig_comp)

    # Visualização: Percentual do Candidato (Gauge)
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = candidate_data['PERCENTUAL_VOTOS_CANDIDATO'],
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f'Percentual de Votos de {candidate_data['NM_URNA_CANDIDATO']} no Cargo'},
        gauge = {
            'axis': {'range': [None, 100], 'tickvals': [0, 25, 50, 75, 100]},
            'bar': {'color': 'rgba(0, 128, 0, 0.7)'},
            'bgcolor': 'white',
            'borderwidth': 2,
            'bordercolor': 'gray'}
    ))
    fig_gauge.update_layout(font_size=12)
    st.plotly_chart(fig_gauge)

# --- Aplicação Streamlit Principal ---
def main():
    st.set_page_config(layout="wide")
    st.title("Análise Eleitoral de Baturité (2024)")

    filepath = "votacao_candidato_munzona_2024_CE.csv"

    # Carregar e preparar os dados
    df_raw = load_and_prepare_data(filepath)
    df_featured = feature_engineering(df_raw)

    # Sidebar para navegação
    st.sidebar.title("Navegação")
    page = st.sidebar.radio("Escolha a seção:", ["Visão Geral e EDA", "Análise por Candidato"])

    if page == "Visão Geral e EDA":
        eda_visualizations(df_featured)
    elif page == "Análise por Candidato":
        st.header("Análise Individual de Candidatos")
        unique_candidates = df_featured['NM_URNA_CANDIDATO'].unique()
        selected_candidate = st.selectbox(
            "Selecione um candidato para analisar:",
            options=unique_candidates
        )
        if selected_candidate:
            analyze_candidate_plotly(df_featured, selected_candidate)

if __name__ == "__main__":
    main()
