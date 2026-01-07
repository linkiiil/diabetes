import streamlit as st
import joblib
import pandas as pd
import numpy as np

# 1. Configuração da Interface
st.set_page_config(page_title="Triagem Inteligente de Diabetes", layout="wide")

# 2. Carregamento do Modelo
@st.cache_resource
def carregar_modelo():
    try:
        dados_modelo = joblib.load('modelo_diabetes_vtl.pkl')
        return dados_modelo
    except FileNotFoundError:
        return None

data = carregar_modelo()
if data is None:
    st.error("Erro: Arquivo 'modelo_diabetes_vtl.pkl' não encontrado!")
    st.stop()

modelo = data['pipeline']
threshold_clinico = data.get('threshold', 0.25)

# 3. Interface do Usuário
st.title("🏥 Sistema de Apoio à Decisão Clínica: Diabetes")
st.markdown("IA para triagem populacional baseada nos dados do CDC.")

with st.form("form_clinico"):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Perfil e Estilo de Vida")
        
        # Faixa Etária
        opcoes_age = {1:"18-24", 2:"25-29", 3:"30-34", 4:"35-39", 5:"40-44", 6:"45-49", 
                      7:"50-54", 8:"55-59", 9:"60-64", 10:"65-69", 11:"70-74", 12:"75-79", 13:"80+"}
        age = st.selectbox("Qual sua faixa etária?", options=list(opcoes_age.keys()), format_func=lambda x: opcoes_age[x])

        # Renda em Dólares (Ajustado)
        opcoes_inc = {1:"Menos de $10.000 (USD)", 2:"$10.000 a $15.000 (USD)", 3:"$15.000 a $20.000 (USD)", 
                      4:"$20.000 a $25.000 (USD)", 5:"$25.000 a $35.000 (USD)", 6:"$35.000 a $50.000 (USD)", 
                      7:"$50.000 a $75.000 (USD)", 8:"$75.000 ou mais (USD)"}
        income = st.selectbox("Qual sua faixa de renda anual (em Dólares)?", options=list(opcoes_inc.keys()), format_func=lambda x: opcoes_inc[x])

        # Escolaridade
        opcoes_edu = {1:"Nunca estudou/Jardim", 2:"1ª a 8ª série (Fundamental)", 3:"9ª a 11ª (Médio inc.)", 
                      4:"12ª série/GED (Médio comp.)", 5:"Superior incompleto/Técnico", 6:"Graduado"}
        education = st.selectbox("Escolaridade", options=list(opcoes_edu.keys()), format_func=lambda x: opcoes_edu[x])

        sex = st.radio("Sexo Biológico", options=[0, 1], format_func=lambda x: "Feminino" if x==0 else "Masculino")
        
        # Saúde Geral
        opcoes_gen = {1:"Excelente", 2:"Muito Boa", 3:"Boa", 4:"Regular", 5:"Ruim"}
        gen_hlth = st.select_slider("Como avalia sua saúde geral?", options=list(opcoes_gen.keys()), format_func=lambda x: opcoes_gen[x])
        
        # --- CALCULADORA DE IMC INTEGRADA ---
        st.write("---")
        st.markdown("**Cálculo Automático de IMC**")
        c1, c2 = st.columns(2)
        peso = c1.number_input("Peso (kg)", min_value=30.0, max_value=250.0, value=70.0)
        altura_cm = c2.number_input("Altura (cm)", min_value=100, max_value=230, value=170)
        
        # Fórmula: peso / (altura_m * altura_m)
        imc_calculado = round(peso / ((altura_cm / 100) ** 2), 1)
        st.info(f"Seu IMC calculado é: **{imc_calculado}**")

    with col2:
        st.subheader("Histórico e Comportamento")
        
        high_bp = st.checkbox("Possui Pressão Alta?")
        high_chol = st.checkbox("Possui Colesterol Alto?")
        chol_check = st.checkbox("Fez exame de colesterol nos últimos 5 anos?")
        stroke = st.checkbox("Já teve um AVC?")
        heart_dis = st.checkbox("Possui doença cardíaca (CHD) ou infarto (MI)?")
        smoker = st.checkbox("Já fumou pelo menos 100 cigarros na vida?")
        phys_act = st.checkbox("Praticou atividade física no último mês?")
        fruits = st.checkbox("Consome Frutas 1+ vezes ao dia?")
        veggies = st.checkbox("Consome Vegetais 1+ vezes ao dia?")
        hvy_alcohol = st.checkbox("Consumo excessivo de álcool?")
        healthcare = st.checkbox("Possui plano de saúde?", value=True)
        no_doc_cost = st.checkbox("Deixou de ir ao médico por custo?")
        diff_walk = st.checkbox("Dificuldade séria para caminhar ou subir escadas?")

    submit = st.form_submit_button("ANALISAR RISCO")

# 4. Lógica de Predição
if submit:
    # Preparação do DataFrame respeitando as 19 colunas oficiais
    input_data = pd.DataFrame([{
        'HighBP': 1 if high_bp else 0,
        'HighChol': 1 if high_chol else 0,
        'CholCheck': 1 if chol_check else 0,
        'BMI': imc_calculado, # Usa o valor calculado automaticamente
        'Smoker': 1 if smoker else 0,
        'Stroke': 1 if stroke else 0,
        'HeartDiseaseorAttack': 1 if heart_dis else 0,
        'PhysActivity': 1 if phys_act else 0,
        'Fruits': 1 if fruits else 0,
        'Veggies': 1 if veggies else 0,
        'HvyAlcoholConsump': 1 if hvy_alcohol else 0,
        'AnyHealthcare': 1 if healthcare else 0,
        'NoDocbcCost': 1 if no_doc_cost else 0,
        'GenHlth': gen_hlth,
        'DiffWalk': 1 if diff_walk else 0,
        'Sex': sex,
        'Age': age,
        'Education': education,
        'Income': income
    }])

    # Garantir a ordem exata das colunas que o modelo espera
    colunas_modelo = ['HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke', 
                      'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies', 
                      'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth', 
                      'DiffWalk', 'Sex', 'Age', 'Education', 'Income']
    
    input_data = input_data[colunas_modelo]

    prob = modelo.predict_proba(input_data)[0][1]
    
    st.divider()
    st.subheader(f"Probabilidade Calculada: {prob:.1%}")
    
    if prob >= threshold_clinico:
        st.error(f"RESULTADO: ALTO RISCO (Acima do Threshold de {threshold_clinico*100:.0f}%)")
        st.info("Estratégia focada em alta sensibilidade (Recall).")
    else:
        st.success(f"RESULTADO: BAIXO RISCO (Abaixo do Threshold de {threshold_clinico*100:.0f}%)")