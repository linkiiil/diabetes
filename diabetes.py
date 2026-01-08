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
        # Tenta carregar o dicionário contendo o pipeline e o threshold
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

# --- NOVO: NOTA METODOLÓGICA (LOGO NO INÍCIO PARA ESCLARECER O USUÁRIO) ---
with st.expander("📝 Nota Metodológica: Por que fazemos perguntas sobre renda, educação e cigarros?"):
    st.markdown("""
    Este formulário segue os critérios internacionais de triagem do **CDC (Centers for Disease Control and Prevention)**. 
    Algumas perguntas possuem justificativas técnicas importantes:
    
    * **💰 Aspectos Socioeconômicos:** Renda e educação são **Determinantes Sociais de Saúde**. Eles influenciam diretamente o acesso a alimentos saudáveis, academias e frequência de exames preventivos.
    * **🚬 Os 100 Cigarros:** É o marco clínico global para distinguir o uso social do **tabagismo estabelecido**, onde os danos metabólicos e a resistência à insulina aumentam.
    * **🏃 Atividade Física:** O foco é identificar o **sedentarismo**. Não praticar nenhuma atividade no último mês é um indicador de alerta para o metabolismo da glicose.
    * **🏥 Custo e Saúde:** Avaliamos barreiras financeiras que impedem o diagnóstico precoce, mesmo para quem possui plano de saúde.
    """)

with st.form("form_clinico"):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Perfil e Estilo de Vida")
        
        opcoes_age = {1:"18-24", 2:"25-29", 3:"30-34", 4:"35-39", 5:"40-44", 6:"45-49", 
                      7:"50-54", 8:"55-59", 9:"60-64", 10:"65-69", 11:"70-74", 12:"75-79", 13:"80+"}
        age = st.selectbox("Qual sua faixa etária?", options=list(opcoes_age.keys()), format_func=lambda x: opcoes_age[x])

        opcoes_inc = {1:"Menos de $10.000 (USD)", 2:"$10.000 a $15.000 (USD)", 3:"$15.000 a $20.000 (USD)", 
                      4:"$20.000 a $25.000 (USD)", 5:"$25.000 a $35.000 (USD)", 6:"$35.000 a $50.000 (USD)", 
                      7:"$50.000 a $75.000 (USD)", 8:"$75.000 ou mais (USD)"}
        income = st.selectbox("Faixa de Renda Familiar Anual", options=list(opcoes_inc.keys()), format_func=lambda x: opcoes_inc[x])

        opcoes_edu = {1:"Nunca estudou/Jardim", 2:"1ª a 8ª série (Fundamental)", 3:"9ª a 11ª (Médio inc.)", 
                      4:"12ª série/GED (Médio comp.)", 5:"Superior incompleto/Técnico", 6:"Graduado"}
        education = st.selectbox("Nível de Escolaridade", options=list(opcoes_edu.keys()), format_func=lambda x: opcoes_edu[x])

        sex = st.radio("Sexo Biológico", options=[0, 1], format_func=lambda x: "Feminino" if x==0 else "Masculino")
        
        opcoes_gen = {1:"Excelente", 2:"Muito Boa", 3:"Boa", 4:"Regular", 5:"Ruim"}
        gen_hlth = st.select_slider("Como avalia sua saúde geral?", options=list(opcoes_gen.keys()), format_func=lambda x: opcoes_gen[x])
        
        st.write("---")
        st.markdown("**Cálculo de IMC**")
        c1, c2 = st.columns(2)
        peso = c1.number_input("Peso (kg)", min_value=30.0, max_value=250.0, value=70.0)
        altura_cm = c2.number_input("Altura (cm)", min_value=100, max_value=230, value=170)
        imc_calculado = round(peso / ((altura_cm / 100) ** 2), 1)
        st.info(f"IMC: **{imc_calculado}**")

    with col2:
        st.subheader("Histórico e Comportamento")
        
        high_bp = st.checkbox("Possui Pressão Alta?")
        high_chol = st.checkbox("Possui Colesterol Alto?")
        chol_check = st.checkbox("Fez exame de colesterol nos últimos 5 anos?")
        stroke = st.checkbox("Já teve um AVC?")
        heart_dis = st.checkbox("Possui doença cardíaca (CHD) ou infarto (MI)?")
        smoker = st.checkbox("Já fumou pelo menos 100 cigarros na vida?")
        phys_act = st.checkbox("Praticou atividade física no último mês?")
        fruits = st.checkbox("Consome Frutas regularmente?")
        veggies = st.checkbox("Consome Vegetais regularmente?")
        hvy_alcohol = st.checkbox("Consumo excessivo de álcool?")
        healthcare = st.checkbox("Possui plano de saúde?", value=True)
        no_doc_cost = st.checkbox("Deixou de ir ao médico por custo?")
        diff_walk = st.checkbox("Dificuldade séria para caminhar ou subir escadas?")

    submit = st.form_submit_button("ANALISAR MEU RISCO")

# 4. Lógica de Predição e Resultados
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

    # Ordenação rigorosa para o modelo
    colunas_modelo = ['HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke', 
                      'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies', 
                      'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth', 
                      'DiffWalk', 'Sex', 'Age', 'Education', 'Income']
    input_data = input_data[colunas_modelo]

    prob = modelo.predict_proba(input_data)[0][1]
    
    st.divider()
    st.subheader(f"Resultado da Análise")
    
    # Exibição Visual do Risco com Alertas Coloridos
    if prob >= threshold_clinico:
        st.error(f"### ⚠️ ALTO RISCO IDENTIFICADO ({prob:.1%})")
        st.markdown(f"""
        **Recomendação Médica:** Seus indicadores de saúde e perfil socioeconômico sugerem uma alta probabilidade estatística 
        para o diabetes. Recomendamos procurar um profissional de saúde para realizar exames laboratoriais como 
        **Glicemia de Jejum** e **Hemoglobina Glicada (HbA1c)**.
        
        *Este modelo é calibrado para alta sensibilidade (Recall de 94.8%), priorizando a detecção preventiva.*
        """)
    else:
        st.success(f"### ✅ BAIXO RISCO IDENTIFICADO ({prob:.1%})")
        st.markdown(f"""
        Seus indicadores sugerem, no momento, um baixo risco para diabetes. Continue mantendo hábitos saudáveis e 
        realizando seus exames de rotina regularmente.
        """)
    
    # Botão para baixar relatório para levar ao médico
    relatorio = f"RELATÓRIO DE TRIAGEM IA\nProbabilidade: {prob:.1%}\nThreshold: {threshold_clinico}\nIMC: {imc_calculado}"
    st.download_button("📥 Baixar Relatório para Consulta", relatorio, "meu_risco_diabetes.txt")

# --- SEÇÃO DE TRANSPARÊNCIA TÉCNICA (ARQUIVO SVG) ---
st.divider()
with st.expander("🔍 Detalhes Técnicos e Matriz de Confusão (Auditoria)"):
    st.write("### Estratégia de Decisão Clínica")
    
    col_text, col_img = st.columns([1, 1.5])
    
    with col_text:
        st.markdown(f"""
        **Métricas de Validação (Dataset CDC):**
        * **Recall (Sensibilidade):** 94.86%
        * **Diabetes Identificadas:** 6.658 (True Positives)
        * **Falsos Negativos:** 361 (Casos não detectados)
        
        **Estratégia do Threshold (0.25):**
        O modelo não busca apenas acerto (Acurácia), mas sim a **segurança do paciente**. 
        Definimos um ponto de corte conservador para garantir que a vasta maioria dos diabéticos 
        seja encaminhada para triagem clínica.
        """)
    
    with col_img:
        try:
            st.image("Confusion Matrix.svg", use_container_width=True, caption="Performance Vetorial (SVG)")
        except:
            st.warning("⚠️ Arquivo 'Confusion Matrix.svg' não encontrado no GitHub.")

# --- RODAPÉ ÉTICO ---
st.caption("Aviso: Esta é uma ferramenta de triagem baseada em estatística. Não substitui diagnóstico médico.")
