import streamlit as st
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import pytz

# 1. Configuração da Interface
st.set_page_config(page_title="Triagem Inteligente de Diabetes", layout="wide")

# --- CUSTOMIZAÇÃO VISUAL (NAVY BLUE) ---
st.markdown("""
    <style>
    :root { --primary-color: #000080; }
    button[kind="primaryFormSubmit"] {
        background-color: #000080 !important;
        color: white !important;
        width: 100%;
        border-radius: 8px;
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

# 3. Cabeçalho e Nota Metodológica
st.title("🏥 Sistema de Apoio à Decisão Clínica: Diabetes")
st.markdown(f"**Analista Responsável:** Portal de Triagem Preventiva | **Data:** {data_atual} (Horário de Brasília)")

with st.expander("📝 Nota Metodológica e Motivação Técnica"):
    st.markdown("""
    **Justificativa das Variáveis:**
    * **💰 Socioeconômicos:** Renda e educação são determinantes sociais críticos. 
    * **🚬 Estilo de Vida:** Tabagismo e sedentarismo são marcadores de risco metabólico.
    * **🏃 Atividade Física:** Identifica sedentarismo, um marcador crítico de risco metabólico.
    * **🏥 Custo:** Avalia barreiras financeiras que impedem o diagnóstico precoce.

    **Mudança Metodológica (Critério FGV):**
    **Estratificação Socioeconômica:**
    O modelo original utiliza faixas em dólares (USD). Para o contexto brasileiro, adaptamos a entrada para **Salários Mínimos (SM)** seguindo a classificação da **FGV**:
    * **Critério:** As Classes A, B, C, D e E foram mapeadas nos 8 níveis ordinais do modelo.
    * **Justificativa:** O Salário Mínimo atua como um *proxy* para o poder de compra e acesso a determinantes de saúde (alimentação, medicamentos e exames), mantendo a integridade estatística da predição original mesmo com variações inflacionárias.
    """)

# 4. Formulário de Entrada
with st.form("form_clinico"):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Perfil e Estilo de Vida")
        
        opcoes_age = {1:"18-24", 2:"25-29", 3:"30-34", 4:"35-39", 5:"40-44", 6:"45-49", 
                      7:"50-54", 8:"55-59", 9:"60-64", 10:"65-69", 11:"70-74", 12:"75-79", 13:"80+"}
        age = st.selectbox("Faixa etária", options=list(opcoes_age.keys()), format_func=lambda x: opcoes_age[x])

        # --- MUDANÇA: CLASSES FGV (Corrigido dentro do escopo da col1) ---
        map_sm_fgv = {
            "Classe E (Até 1 SM)": 1,
            "Classe D (1 a 2 SM)": 2,
            "Classe D (2 a 4 SM)": 3,
            "Classe C (4 a 7 SM)": 4,
            "Classe C (7 a 15 SM)": 5,
            "Classe B (15 a 20 SM)": 6,
            "Classe A (20 a 30 SM)": 7,
            "Classe A (Acima de 30 SM)": 8
        }

        escolha_renda = st.selectbox("Classificação Econômica (FGV - Salários Mínimos)", 
                                    options=list(map_sm_fgv.keys()))
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

    # Reordenar conforme modelo
    input_data = input_data[modelo.feature_names_in_]
    prob = modelo.predict_proba(input_data)[0][1]
    
    st.divider()
    status_risco = "ALTO RISCO" if prob >= threshold_clinico else "BAIXO RISCO"
    
    if prob >= threshold_clinico:
        st.error(f"### ⚠️ {status_risco} IDENTIFICADO: {prob:.1%}")
        st.markdown("**Conduta sugerida:** Encaminhamento para Glicemia de Jejum e HbA1c.")
    else:
        st.success(f"### ✅ {status_risco} IDENTIFICADO: {prob:.1%}")

    # Relatório Clínico
    texto_relatorio = f"""
    ==================================================
    RELATÓRIO DE TRIAGEM PREVENTIVA - DIABETES (IA)
    ==================================================
    Data/Hora: {data_atual}
    Risco: {prob:.1%} ({status_risco})
    --------------------------------------------------
    SÍNTESE DOS DADOS:
    - IMC: {imc_calculado}
    - Renda: {escolha_renda}
    - Saúde Geral: {opcoes_gen[gen_hlth]}
    --------------------------------------------------
    NOTA: Baseado em modelo preditivo CDC/BRFSS.
    ==================================================
    """
    
    st.download_button(label="📥 Baixar Relatório Clínico", data=texto_relatorio, 
                       file_name=f"triagem_{datetime.now().strftime('%d%m%Y')}.txt")

# 6. Rodapé Técnico
st.divider()
with st.expander("🔍 Auditoria Técnica (Matriz de Confusão)"):
    col_a, col_b = st.columns([1, 1.5])
    col_a.write(f"Recall: 94.56% | Threshold: {threshold_clinico}")
    try:
        col_b.image("Confusion Matrix.svg", use_container_width=True)
    except:
        col_b.warning("SVG não encontrado no repositório.")

st.caption("Aviso: Ferramenta estatística de suporte. Não substitui o diagnóstico médico.")






