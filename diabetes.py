import streamlit as st
import joblib
import pandas as pd

# ============================================================
# CONFIGURAÇÃO GERAL
# ============================================================
st.set_page_config(
    page_title="Triagem Inteligente de Diabetes",
    layout="wide"
)

MODELO_PATH = "modelo_diabetes_vtl.pkl"
CONF_MATRIX_PATH = "Confusion Matrix.svg"
THRESHOLD_DEFAULT = 0.25

# ============================================================
# DEFINIÇÕES DE DOMÍNIO (CDC)
# ============================================================
AGE_MAP = {
    1: "18–24", 2: "25–29", 3: "30–34", 4: "35–39", 5: "40–44",
    6: "45–49", 7: "50–54", 8: "55–59", 9: "60–64", 10: "65–69",
    11: "70–74", 12: "75–79", 13: "80+"
}

INCOME_MAP = {
    1: "< $10.000", 2: "$10.000–15.000", 3: "$15.000–20.000",
    4: "$20.000–25.000", 5: "$25.000–35.000",
    6: "$35.000–50.000", 7: "$50.000–75.000", 8: "$75.000+"
}

EDUCATION_MAP = {
    1: "Nunca estudou / Jardim",
    2: "Fundamental incompleto",
    3: "Médio incompleto",
    4: "Médio completo / GED",
    5: "Superior incompleto / Técnico",
    6: "Superior completo"
}

GENHLTH_MAP = {
    1: "Excelente", 2: "Muito boa", 3: "Boa", 4: "Regular", 5: "Ruim"
}

FEATURE_ORDER = [
    'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
    'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
    'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
    'DiffWalk', 'Sex', 'Age', 'Education', 'Income'
]

# ============================================================
# CARREGAMENTO DO MODELO
# ============================================================
@st.cache_resource
def carregar_modelo():
    dados = joblib.load(MODELO_PATH)
    return dados["pipeline"], dados.get("threshold", THRESHOLD_DEFAULT)

try:
    modelo, threshold_clinico = carregar_modelo()
except Exception:
    st.error("Erro ao carregar o modelo de triagem.")
    st.stop()

# ============================================================
# CABEÇALHO
# ============================================================
st.title("Sistema de Apoio à Decisão Clínica — Diabetes")
st.markdown(
    "Ferramenta de **triagem populacional baseada em IA**, treinada com dados do CDC."
)

# ============================================================
# NOTA METODOLÓGICA
# ============================================================
with st.expander("Nota metodológica e justificativa clínica"):
    st.markdown("""
    Este instrumento segue critérios epidemiológicos internacionais do **CDC**.

    - **Determinantes Sociais de Saúde** (renda e escolaridade) impactam adesão, prevenção e diagnóstico precoce.
    - **100 cigarros** é o marco clínico global para tabagismo estabelecido.
    - **Atividade física** é avaliada como proxy de sedentarismo metabólico.
    - **Barreiras financeiras** são preditores independentes de pior desfecho clínico.
    """)

# ============================================================
# FORMULÁRIO CLÍNICO
# ============================================================
with st.form("form_triangem"):

    col_left, col_right = st.columns(2)

    # ---------------- PERFIL ----------------
    with col_left:
        st.subheader("Perfil e Estilo de Vida")

        age = st.selectbox("Faixa etária", AGE_MAP, format_func=AGE_MAP.get)
        income = st.selectbox("Renda familiar anual", INCOME_MAP, format_func=INCOME_MAP.get)
        education = st.selectbox("Escolaridade", EDUCATION_MAP, format_func=EDUCATION_MAP.get)
        sex = st.radio("Sexo biológico", {0: "Feminino", 1: "Masculino"}, format_func=lambda x: {0: "Feminino", 1: "Masculino"}[x])
        gen_hlth = st.select_slider("Autoavaliação de saúde", GENHLTH_MAP, format_func=GENHLTH_MAP.get)

        st.divider()
        st.markdown("**Índice de Massa Corporal (IMC)**")
        peso = st.number_input("Peso (kg)", 30.0, 250.0, 70.0)
        altura = st.number_input("Altura (cm)", 100, 230, 170)
        bmi = round(peso / ((altura / 100) ** 2), 1)
        st.info(f"IMC calculado: **{bmi}**")

    # ---------------- HISTÓRICO ----------------
    with col_right:
        st.subheader("Histórico Clínico e Comportamental")

        high_bp = st.checkbox("Hipertensão arterial")
        high_chol = st.checkbox("Colesterol elevado")
        chol_check = st.checkbox("Exame de colesterol nos últimos 5 anos")
        stroke = st.checkbox("Histórico de AVC")
        heart = st.checkbox("Doença cardíaca ou infarto")
        smoker = st.checkbox("Já fumou ≥ 100 cigarros")
        phys = st.checkbox("Atividade física no último mês")
        fruits = st.checkbox("Consumo regular de frutas")
        veggies = st.checkbox("Consumo regular de vegetais")
        alcohol = st.checkbox("Consumo excessivo de álcool")
        healthcare = st.checkbox("Possui plano de saúde", value=True)
        no_doc = st.checkbox("Evitou médico por custo")
        diff_walk = st.checkbox("Dificuldade significativa de locomoção")

    submit = st.form_submit_button("Analisar risco")

# ============================================================
# INFERÊNCIA
# ============================================================
if submit:
    input_df = pd.DataFrame([{
        'HighBP': int(high_bp),
        'HighChol': int(high_chol),
        'CholCheck': int(chol_check),
        'BMI': bmi,
        'Smoker': int(smoker),
        'Stroke': int(stroke),
        'HeartDiseaseorAttack': int(heart),
        'PhysActivity': int(phys),
        'Fruits': int(fruits),
        'Veggies': int(veggies),
        'HvyAlcoholConsump': int(alcohol),
        'AnyHealthcare': int(healthcare),
        'NoDocbcCost': int(no_doc),
        'GenHlth': gen_hlth,
        'DiffWalk': int(diff_walk),
        'Sex': sex,
        'Age': age,
        'Education': education,
        'Income': income
    }][FEATURE_ORDER])

    prob = modelo.predict_proba(input_df)[0, 1]

    st.divider()
    st.subheader("Resultado da Triagem")

    if prob >= threshold_clinico:
        st.error(f"ALTO RISCO ESTIMADO — {prob:.1%}")
        st.markdown("""
        **Encaminhamento recomendado:**  
        Realizar **glicemia de jejum** e **HbA1c** com profissional de saúde.

        Modelo calibrado para **alta sensibilidade (Recall ≈ 95%)**, priorizando segurança clínica.
        """)
    else:
        st.success(f"BAIXO RISCO ESTIMADO — {prob:.1%}")
        st.markdown("Manter hábitos saudáveis e acompanhamento preventivo regular.")

    st.download_button(
        "Baixar relatório",
        f"Probabilidade estimada: {prob:.1%}\nIMC: {bmi}\nThreshold: {threshold_clinico}",
        "triagem_diabetes.txt"
    )

# ============================================================
# TRANSPARÊNCIA
# ============================================================
st.divider()
with st.expander("Auditoria técnica e validação"):
    st.markdown("""
    **Dataset:** CDC  
    **Recall:** 94.86%  
    **Falsos negativos:** 361  
    **Threshold clínico:** 0.25 (estratégia conservadora)
    """)

    try:
        st.image(CONF_MATRIX_PATH, use_container_width=True)
    except:
        st.warning("Matriz de confusão não encontrada.")

st.caption("Ferramenta de triagem estatística. Não substitui diagnóstico médico.")
