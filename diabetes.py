import streamlit as st

import joblib

import pandas as pd

import numpy as np

from datetime import datetime

import pytz  # Necessário: pip install pytz


# 1. Configuração da Interface

st.set_page_config(page_title="Triagem Inteligente de Diabetes", layout="wide")



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



# 3. Cabeçalho e Nota Metodológica

st.title("🏥 Sistema de Apoio à Decisão Clínica: Diabetes")

st.markdown(f"**Analista Responsável:** Portal de Triagem Preventiva | **Data:** {data_atual} (Horário de Brasília)")



with st.expander("📝 Nota Metodológica: Por que essas perguntas são necessárias?"):

    st.markdown("""

    Este sistema utiliza o padrão epidemiológico do **CDC**. Algumas perguntas possuem justificativas técnicas:

    

    * **💰 Socioeconômicos:** Renda e educação impactam o acesso a alimentos de qualidade e exames.

    * **🚬 100 Cigarros:** Marco clínico para distinguir uso social de **tabagismo estabelecido**.

    * **🏃 Atividade Física:** Identifica sedentarismo, um marcador crítico de risco metabólico.

    * **🏥 Custo:** Avalia barreiras financeiras que impedem o diagnóstico precoce.

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

        c1, c2 = st.columns(2)

        peso = c1.number_input("Peso (kg)", min_value=30.0, value=75.0)

        altura_cm = c2.number_input("Altura (cm)", min_value=100, value=170)

        imc_calculado = round(peso / ((altura_cm / 100) ** 2), 1)

        st.info(f"IMC: **{imc_calculado}**")



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



    # Garantir ordem das colunas

    input_data = input_data[modelo.feature_names_in_]

    prob = modelo.predict_proba(input_data)[0][1]

    

    st.divider()

    

    if prob >= threshold_clinico:

        st.error(f"### ⚠️ ALTO RISCO IDENTIFICADO: {prob:.1%}")

        st.markdown(f"**Recomendação:** Procure um médico para exames confirmatórios (Glicemia/HbA1c).")

    else:

        st.success(f"### ✅ BAIXO RISCO IDENTIFICADO: {prob:.1%}")



    # --- GERADOR DE RELATÓRIO ---

    texto_relatorio = f"""

    ==================================================

    RELATÓRIO DE TRIAGEM PREVENTIVA - DIABETES (IA)

    ==================================================

    Data: {data_atual}

    Risco Estimado: {prob:.1%}

    Status: {"ALTO RISCO" if prob >= threshold_clinico else "BAIXO RISCO"}

    --------------------------------------------------

    SÍNTESE DOS INDICADORES:

    - IMC: {imc_calculado}

    - Pressão Alta: {"Sim" if high_bp else "Não"}

    - Colesterol Alto: {"Sim" if high_chol else "Não"}

    - Tabagismo (+100 cig): {"Sim" if smoker else "Não"}

    --------------------------------------------------

    SUGESTÃO DE CONDUTA (PROFISSIONAL DE SAÚDE):

    Modelo com Sensibilidade de 94.8%. 

    Sugere-se avaliar Glicemia de Jejum e Hemoglobina Glicada.

    ==================================================

    """

    

    st.download_button(

        label="📥 Baixar Relatório Clínico (.txt)",

        data=texto_relatorio,

        file_name=f"relatorio_diabetes_{datetime.now().strftime('%Y%m%d')}.txt",

        mime="text/plain"

    )

    st.caption("Para salvar em PDF: Abra o relatório baixado e use a opção 'Imprimir -> Salvar como PDF'.")



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

            st.warning("SVG da Matriz não encontrado no repositório.")


st.caption("Aviso: Ferramenta de triagem estatística. Não substitui diagnóstico médico.")

