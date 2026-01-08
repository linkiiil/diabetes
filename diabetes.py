Para evitar qualquer conflito visual e garantir que o seu projeto de MBA tenha uma estética profissional e funcional, vamos usar a estratégia mais segura: definir a cor primária via :root (que altera checks, sliders e seletores de forma nativa) e estilizar os botões sem sobrepor o conteúdo.

Aqui está o código completo, limpo e com o azul Navy aplicado corretamente:

Python

import streamlit as st
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import pytz

# 1. Configuração da Interface
st.set_page_config(page_title="Triagem Inteligente de Diabetes", layout="wide")

# --- CUSTOMIZAÇÃO VISUAL DEFINITIVA (NAVY BLUE) ---
st.markdown("""
    <style>
    /* Altera a cor primária do Streamlit (Checks, Sliders, Radio) */
    :root {
        --primary-color: #000080;
    }

    /* Estilização do Botão de Submissão */
    button[kind="primaryFormSubmit"] {
        background-color: #000080 !important;
        color: white !important;
        width: 100%;
        border-radius: 8px;
        height: 3em;
        font-weight: bold;
    }
    
    /* Hover do botão de submissão */
    button[kind="primaryFormSubmit"]:hover {
        background-color: #0000a0 !important;
        border: 1px solid #000080;
    }

    /* Estilização do Botão de Download */
    div.stDownloadButton > button {
        color: #000080 !important;
        border: 1px solid #000080 !important;
        background-color: transparent !important;
    }
    
    div.stDownloadButton > button:hover {
        background-color: #000080 !important;
        color: white !important;
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
    st.error("Erro crítico: Arquivo 'modelo_diabetes_vtl.pkl' não encontrado.")
    st.stop()

modelo = data['pipeline']
threshold_clinico = data.get('threshold', 0.25)

# --- AJUSTE DE TIMEZONE (BRASIL) ---
fuso_br = pytz.timezone('America/Sao_Paulo')
data_atual = datetime.now(fuso_br).strftime('%d/%m/%Y %H:%M')

# 3. Cabeçalho
st.title("🏥 Sistema de Apoio à Decisão Clínica: Diabetes")
st.markdown(f"**Data da Consulta:** {data_atual} (Horário de Brasília)")

with st.expander("📝 Nota Metodológica: Por que essas perguntas são necessárias?"):
    st.markdown("""
    Este sistema utiliza o padrão epidemiológico do **CDC**. Justificativas técnicas:
    * **💰 Socioeconômicos:** Renda e educação impactam o acesso a alimentos e exames.
    * **🚬 100 Cigarros:** Marco clínico para distinguir o **tabagismo estabelecido**.
    * **🏃 Atividade Física:** Identifica sedentarismo, marcador crítico de risco metabólico.
    """)

# 4. Formulário de Entrada
with st.form("form_clinico"):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Perfil e Estilo de Vida")
        
        opcoes_age = {1:"18-24", 2:"25-29", 3:"30-34", 4:"35-39", 5:"40-44", 6:"45-49", 
                      7:"50-54", 8:"55-59", 9:"60-64", 10:"65-69", 11:"70-74", 12:"75-79", 13:"80+"}
        age = st.selectbox("Faixa etária", options=list(opcoes_age.keys()), format_func=lambda x: opcoes_age[x])

        opcoes_inc = {1:"Até $10k", 2:"$10k-$15k", 3:"$15k-$20k", 4:"$20k-$25k", 5:"$25k-$35k", 6:"$35k-$50k", 7:"$50k-$75k", 8:"$75k+"}
        income = st.selectbox("Faixa de Renda Anual (USD)", options=list(opcoes_inc.keys()), format_func=lambda x: opcoes_inc[x])

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
        high_bp = st.checkbox("Possui Pressão Alta?")
        high_chol = st.checkbox("Possui Colesterol Alto?")
        chol_check = st.checkbox("Exame de colesterol (últimos 5 anos)?")
        stroke = st.checkbox("Já teve AVC?")
        heart_dis = st.checkbox("Doença Cardíaca ou Infarto?")
        smoker = st.checkbox("Já fumou 100+ cigarros na vida?")
        phys_act = st.checkbox("Atividade física no último mês?")
        fruits = st.checkbox("Consome Frutas regularmente?")
        veggies = st.checkbox("Consome Vegetais regularmente?")
        hvy_alcohol = st.checkbox("Consumo excessivo de álcool?")
        healthcare = st.checkbox("Possui plano de saúde?", value=True)
        no_doc_cost = st.checkbox("Deixou de ir ao médico por custo?")
        diff_walk = st.checkbox("Dificuldade para caminhar/escadas?")

    submit = st.form_submit_button("GERAR ANÁLISE DE RISCO")

# 5. Processamento e Relatório
if submit:
    input_data = pd.DataFrame([{
        'HighBP': 1 if high_bp else 0, 'HighChol': 1 if high_chol else 0, 'CholCheck': 1 if chol_check else 0,
        'BMI': imc_calculado, 'Smoker': 1 if smoker else 0, 'Stroke': 1 if stroke else 0,
        'HeartDiseaseorAttack': 1 if heart_dis else 0, 'PhysActivity': 1 if phys_act else 0,
        'Fruits': 1 if fruits else 0, 'Veggies': 1 if veggies else 0, 'HvyAlcoholConsump': 1 if hvy_alcohol else 0,
        'AnyHealthcare': 1 if healthcare else 0, 'NoDocbcCost': 1 if no_doc_cost else 0,
        'GenHlth': gen_hlth, 'DiffWalk': 1 if diff_walk else 0, 'Sex': sex, 'Age': age,
        'Education': education, 'Income': income
    }])

    input_data = input_data[modelo.feature_names_in_]
    prob = modelo.predict_proba(input_data)[0][1]
    
    st.divider()
    status_clinico = "ALTO RISCO" if prob >= threshold_clinico else "BAIXO RISCO"
    
    if prob >= threshold_clinico:
        st.error(f"### ⚠️ {status_clinico} IDENTIFICADO: {prob:.1%}")
        st.markdown(f"**Recomendação:** Procure um médico para exames confirmatórios (Glicemia/HbA1c).")
    else:
        st.success(f"### ✅ {status_clinico} IDENTIFICADO: {prob:.1%}")

    texto_relatorio = f"""
    ==================================================
    RELATÓRIO DE TRIAGEM PREVENTIVA - DIABETES (IA)
    ==================================================
    Data: {data_atual} (Brasília)
    Risco Estimado: {prob:.1%}
    Status: {status_clinico}
    --------------------------------------------------
    SÍNTESE DOS INDICADORES:
    - IMC: {imc_calculado}
    - Pressão Alta: {"Sim" if high_bp else "Não"}
    - Colesterol Alto: {"Sim" if high_chol else "Não"}
    --------------------------------------------------
    SUGESTÃO DE CONDUTA (PROFISSIONAL DE SAÚDE):
    Modelo com Sensibilidade de 94.8%. 
    Sugere-se avaliar Glicemia de Jejum e Hemoglobina Glicada.
    ==================================================
    """
    
    st.download_button(
        label="📥 Baixar Relatório Clínico (.txt)",
        data=texto_relatorio,
        file_name=f"relatorio_diabetes_{datetime.now(fuso_br).strftime('%Y%m%d_%H%M')}.txt",
        mime="text/plain"
    )

# 6. Transparência Técnica
st.divider()
with st.expander("🔍 Auditoria Técnica (Matriz de Confusão)"):
    col_text, col_img = st.columns([1, 1.5])
    with col_text:
        st.write(f"**Recall:** 94.86% | **Threshold:** {threshold_clinico}")
        st.markdown("Estratégia focada em minimizar Falsos Negativos.")
    with col_img:
        try:
            st.image("Confusion Matrix.svg", use_container_width=True)
        except:
            st.warning("SVG da Matriz não encontrado.")

st.caption("Aviso: Ferramenta de triagem estatística. Não substitui diagnóstico médico.




