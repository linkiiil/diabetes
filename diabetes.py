import io
import os
import joblib
import pytz
import base64
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# ---------------------------
# Configuração da Página
# ---------------------------
st.set_page_config(page_title="Triagem Inteligente de Diabetes", layout="wide")

# ---------------------------
# Funções de Suporte
# ---------------------------
@st.cache_resource
def carregar_modelo(path="modelo_diabetes_vtl.pkl"):
    if not os.path.exists(path):
        return None
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Erro ao carregar o arquivo .pkl: {e}")
        return None

def display_svg(path: str, caption: str):
    """Renderiza SVG convertendo para Base64 para compatibilidade total."""
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                svg_data = f.read()
            b64 = base64.b64encode(svg_data).decode("utf-8")
            html = f"""
            <div style="display: flex; justify-content: center; background-color: white; padding: 20px; border-radius: 10px; border: 1px solid #eee;">
                <img src="data:image/svg+xml;base64,{b64}" style="width: 100%; max-width: 800px; height: auto;"/>
            </div>
            """
            st.write(f"**{caption}**")
            st.components.v1.html(html, height=550)
        except Exception as e:
            st.error(f"Erro ao processar {path}: {e}")
    else:
        st.warning(f"⚠️ Arquivo não encontrado: {path}")

# ---------------------------
# Carregamento do modelo
# ---------------------------
data = carregar_modelo()
if data is None:
    st.error("❌ ERRO CRÍTICO: 'modelo_diabetes_vtl.pkl' não encontrado.")
    st.stop()

modelo = data.get('pipeline')
threshold_clinico = data.get('threshold', 0.25)
recall_val = data.get('recall_pos')
pr_auc_val = data.get('avg_precision')
roc_auc_val = data.get('roc_auc')
f1_val = data.get('f1_pos')

# ---------------------------
# Cabeçalho
# ---------------------------
fuso_br = pytz.timezone('America/Sao_Paulo')
data_atual = datetime.now(fuso_br).strftime('%d/%m/%Y %H:%M')

st.title("🏥 Sistema de Apoio à Decisão Clínica: Diabetes")
st.markdown(f"Analista Responsável: Portal de Triagem Preventiva | Data: {data_atual} (Horário de Brasília)")

st.markdown(
    "Origem dos dados: Este projeto utiliza o dataset *Diabetes Health Indicators* do Centers for Disease Control and Prevention (CDC), "
    "uma base de dados robusta com mais de 250 mil registros que traduzem o perfil de saúde, estilo de vida e indicadores socioeconômicos da população."
)

with st.expander("📝 Nota Metodológica e Motivação Técnica"):
    st.markdown("""
    Justificativa das Variáveis:
    * 💰 **Socioeconômicos:** Renda e educação são determinantes sociais críticos.
    * 🚬 **Estilo de Vida:** Tabagismo e sedentarismo são marcadores de risco metabólico.
    * 🏃 **Atividade Física:** Identifica sedentarismo, um marcador crítico de risco metabólico.
    * 🏥 **Custo:** Avalia barreiras financeiras que impedem o diagnóstico precoce.

    **Estratificação Socioeconômica (FGV):**
    As classes econômicas foram mapeadas em níveis ordinais baseados em Salários Mínimos (SM) para adaptar o modelo ao contexto brasileiro.
    """)

# ---------------------------
# Formulário de entrada
# ---------------------------
with st.form("form_clinico"):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Perfil e Estilo de Vida")
        opcoes_age = {1:"18-24", 2:"25-29", 3:"30-34", 4:"35-39", 5:"40-44", 6:"45-49",
                      7:"50-54", 8:"55-59", 9:"60-64", 10:"65-69", 11:"70-74", 12:"75-79", 13:"80+"}
        age = st.selectbox("Faixa etária", options=list(opcoes_age.keys()), format_func=lambda x: opcoes_age[x])

        map_sm_fgv = {
            "Classe E (Até 1 SM)": 1, "Classe D (1 a 2 SM)": 2, "Classe D (2 a 4 SM)": 3,
            "Classe C (4 a 7 SM)": 4, "Classe C (7 a 15 SM)": 5, "Classe B (15 a 20 SM)": 6,
            "Classe A (20 a 30 SM)": 7, "Classe A (Acima de 30 SM)": 8
        }
        escolha_renda = st.selectbox("Classificação Econômica (FGV - Salários Mínimos)", options=list(map_sm_fgv.keys()))
        income = map_sm_fgv[escolha_renda]

        opcoes_edu = {1:"Fundamental incompleto", 2:"Fundamental", 3:"Médio incompleto", 4:"Médio completo", 5:"Técnico/Superior inc.", 6:"Graduado"}
        education = st.selectbox("Escolaridade", options=list(opcoes_edu.keys()), format_func=lambda x: opcoes_edu[x])

        sex = st.radio("Sexo Biológico", options=[0, 1], format_func=lambda x: "Feminino" if x==0 else "Masculino")

        opcoes_gen = {1:"Excelente", 2:"Muito Boa", 3:"Boa", 4:"Regular", 5:"Ruim"}
        gen_hlth = st.select_slider("Como avalia sua saúde geral?", options=list(opcoes_gen.keys()), format_func=lambda x: opcoes_gen[x])

        st.write("---")
        st.markdown("**Cálculo de IMC**")
        c1_imc, c2_imc = st.columns(2)
        peso = c1_imc.number_input("Peso (kg)", min_value=30.0, value=75.0)
        altura_cm = c2_imc.number_input("Altura (cm)", min_value=100, value=170)
        imc_calculado = round(peso / ((altura_cm / 100) ** 2), 1)
        st.info(f"IMC Calculado: **{imc_calculado}**")

    with col2:
        st.subheader("Histórico Clínico")
        high_bp = st.checkbox("Pressão Alta?")
        high_chol = st.checkbox("Colesterol Alto?")
        chol_check = st.checkbox("Exame de colesterol (últimos 5 anos)?")
        stroke = st.checkbox("Já teve AVC?")
        heart_dis = st.checkbox("Doença Cardíaca ou Infarto?")
        smoker = st.checkbox("Já fumou 100+ cigarros na vida?")
        phys_act = st.checkbox("Atividade física no último mês?")
        fruits = st.checkbox("Consome Frutas regularmente?")
        veggies = st.checkbox("Consome Vegetais regularmente?")
        hvy_alcohol = st.checkbox("Consumo excessivo de álcool?")
        healthcare = st.checkbox("Possui plano de saúde?")
        no_doc_cost = st.checkbox("Deixou de ir ao médico por custo?")
        diff_walk = st.checkbox("Dificuldade para caminhar/subir escadas?")
        submit = st.form_submit_button("GERAR ANÁLISE DE RISCO")

# ---------------------------
# Previsão
# ---------------------------
if submit:
    input_data = pd.DataFrame([{
        'HighBP': 1 if high_bp else 0, 'HighChol': 1 if high_chol else 0, 'CholCheck': 1 if chol_check else 0,
        'BMI': imc_calculado, 'Smoker': 1 if smoker else 0, 'Stroke': 1 if stroke else 0,
        'HeartDiseaseorAttack': 1 if heart_dis else 0, 'PhysActivity': 1 if phys_act else 0,
        'Fruits': 1 if fruits else 0, 'Veggies': 1 if veggies else 0, 'HvyAlcoholConsump': 1 if hvy_alcohol else 0,
        'AnyHealthcare': 1 if healthcare else 0, 'NoDocbcCost': 1 if no_doc_cost else 0, 'GenHlth': gen_hlth,
        'DiffWalk': 1 if diff_walk else 0, 'Sex': sex, 'Age': age, 'Education'

