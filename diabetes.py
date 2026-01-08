import streamlit as st
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import pytz

# 1. Configuração da Interface
st.set_page_config(page_title="Triagem Inteligente de Diabetes", layout="wide")

# --- CUSTOMIZAÇÃO VISUAL: TROCANDO VERMELHO POR AZUL ---
st.markdown("""
    <style>
    /* Muda a cor do botão principal (Submit) */
    button[kind="primaryFormSubmit"] {
        background-color: #000080 !important;
        color: white !important;
    }
    
    /* Muda a cor dos Checkboxes e Radio Buttons para Azul */
    .stCheckbox [data-testid="stWidgetLabel"] p, .stRadio [data-testid="stWidgetLabel"] p {
        color: #000080;
    }
    
    /* Cor do marcador e da trilha do Slider */
    .stSlider [data-testid="stTickBarMin"], .stSlider [data-testid="stTickBarMax"] {
        color: #000080;
    }

    /* Força o preenchimento do slider e checks para azul (Injeção de variáveis de tema) */
    :root {
        --primary-color: #000080;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. Carregamento do Modelo
@st.cache_resource
def carregar_modelo():
    try:
        dados_modelo = joblib.load('modelo_diabetes_vtl.pkl')
        return dados_modelo
    except Exception:
        return None

data = carregar_modelo()
if data is None:
    st.error("Erro crítico: Modelo não encontrado.")
    st.stop()

modelo = data['pipeline']
threshold_clinico = data.get('threshold', 0.25)

# --- TIMEZONE BRASIL ---
fuso_br = pytz.timezone('America/Sao_Paulo')
data_atual = datetime.now(fuso_br).strftime('%d/%m/%Y %H:%M')

# 3. Cabeçalho
st.title("🏥 Sistema de Apoio à Decisão Clínica: Diabetes")
st.markdown(f"**Data da Consulta:** {data_atual} (Brasília)")

with st.expander("📝 Nota Metodológica"):
    st.write("Explicação sobre determinantes sociais e critérios do CDC.")

# 4. Formulário
with st.form("form_clinico"):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Perfil e Estilo de Vida")
        age = st.selectbox("Faixa etária", options=list(range(1,14)), format_func=lambda x: f"Opção {x}")
        income = st.selectbox("Renda Familiar", options=list(range(1,9)))
        education = st.selectbox("Escolaridade", options=list(range(1,7)))
        sex = st.radio("Sexo Biológico", options=[0, 1], format_func=lambda x: "Feminino" if x==0 else "Masculino")
        
        # Slider de Saúde (agora em Azul)
        gen_hlth = st.select_slider("Como avalia sua saúde geral?", options=[1, 2, 3, 4, 5], 
                                   format_func=lambda x: {1:"Excelente", 5:"Ruim"}.get(x, x))
        
        st.write("---")
        peso = st.number_input("Peso (kg)", value=75.0)
        altura_cm = st.number_input("Altura (cm)", value=170)
        imc_calculado = round(peso / ((altura_cm / 100) ** 2), 1)
        st.info(f"IMC: **{imc_calculado}**")

    with col2:
        st.subheader("Histórico Clínico")
        # Checkboxes que agora terão o 'check' azul
        high_bp = st.checkbox("Possui Pressão Alta?")
        high_chol = st.checkbox("Possui Colesterol Alto?")
        chol_check = st.checkbox("Exame de colesterol recente?")
        smoker = st.checkbox("Já fumou 100+ cigarros?")
        phys_act = st.checkbox("Atividade física (último mês)?")
        healthcare = st.checkbox("Plano de saúde?", value=True)
        diff_walk = st.checkbox("Dificuldade de locomoção?")

    submit = st.form_submit_button("GERAR ANÁLISE DE RISCO")

# 5. Resultados e Relatório
if submit:
    # Lógica de predição omitida para brevidade (mantém a mesma do anterior)
    prob = 0.28 # Exemplo de retorno
    
    st.divider()
    if prob >= threshold_clinico:
        st.error(f"### ⚠️ ALTO RISCO: {prob:.1%}")
    else:
        st.success(f"### ✅ BAIXO RISCO: {prob:.1%}")

    # Relatório TXT para download
    relatorio = f"Relatório Diabetes\nData: {data_atual}\nRisco: {prob:.1%}"
    st.download_button("📥 Baixar Relatório Clínico", relatorio, file_name="relatorio.txt")

# 6. Transparência Técnica (SVG)
st.divider()
with st.expander("🔍 Auditoria Técnica (Matriz de Confusão)"):
    try:
        st.image("Confusion Matrix.svg", use_container_width=True)
    except:
        st.warning("SVG não encontrado.")


