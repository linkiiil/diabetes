import io
import os
import joblib
import pytz
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional

# Tenta importar bibliotecas para conversão de SVG para PNG (melhora a renderização)
try:
    import cairosvg
    from PIL import Image
    CAIROSVG_AVAILABLE = True
except Exception:
    CAIROSVG_AVAILABLE = False

# ---------------------------
# Configuração da Página
# ---------------------------
st.set_page_config(page_title="Triagem Inteligente de Diabetes", layout="wide")

# Estilo CSS para melhorar a estética
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

# ---------------------------
# Funções de Suporte
# ---------------------------
@st.cache_resource
def carregar_modelo(path="modelo_diabetes_vtl.pkl"):
    try:
        return joblib.load(path)
    except Exception:
        return None

def display_svg_high_quality(path: str, scale: int = 2, caption: Optional[str] = None, max_height: int = 640):
    if not os.path.exists(path):
        st.warning(f"Arquivo não encontrado: {os.path.basename(path)}")
        return

    if CAIROSVG_AVAILABLE:
        try:
            with open(path, "rb") as f:
                svg_bytes = f.read()
            png_bytes = cairosvg.svg2png(bytestring=svg_bytes, scale=scale)
            img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
            st.image(img, use_column_width=True, caption=caption)
            return
        except Exception:
            pass

    # Fallback: embed SVG direto via HTML
    try:
        with open(path, "r", encoding="utf-8") as f:
            svg_text = f.read()
        html = f"<div style='max-height:{max_height}px; overflow:auto;'>{svg_text}</div>"
        st.components.v1.html(html, height=max_height + 40, scrolling=True)
        if caption:
            st.caption(caption)
    except Exception:
        st.warning(f"Não foi possível renderizar: {os.path.basename(path)}")

# ---------------------------
# Carregamento do modelo
# ---------------------------
data = carregar_modelo()
if data is None:
    st.error("Erro crítico: Arquivo 'modelo_diabetes_vtl.pkl' não encontrado.")
    st.stop()

modelo = data.get('pipeline') or data.get('model') or data.get('estimator')
threshold_clinico = data.get('threshold', 0.25)

# ---------------------------
# Cabeçalho e Timezone
# ---------------------------
fuso_br = pytz.timezone('America/Sao_Paulo')
data_atual = datetime.now(fuso_br).strftime('%d/%m/%Y %H:%M')

st.title("🏥 Sistema de Apoio à Decisão Clínica: Diabetes")
st.markdown(f"**Analista Responsável:** Portal de Triagem Preventiva | **Data:** {data_atual}")

with st.expander("📝 Nota Metodológica e Motivação Técnica"):
    st.markdown("""
    **Origem dos dados:** Dataset CDC (250 mil+ registros).
    **Justificativa:** O modelo integra fatores socioeconômicos (renda/educação) e estilo de vida para prever o risco metabólico.
    **Estratificação FGV:** Mapeamento de renda adaptado ao contexto brasileiro (Salários Mínimos).
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
        escolha_renda = st.selectbox("Renda (FGV)", options=list(map_sm_fgv.keys()))
        income = map_sm_fgv[escolha_renda]
        
        opcoes_edu = {1:"Fundamental incompleto", 2:"Fundamental", 3:"Médio incompleto", 4:"Médio completo", 5:"Técnico/Superior inc.", 6:"Graduado"}
        education = st.selectbox("Escolaridade", options=list(opcoes_edu.keys()), format_func=lambda x: opcoes_edu[x])
        
        sex = st.radio("Sexo Biológico", options=[0, 1], format_func=lambda x: "Feminino" if x==0 else "Masculino")
        
        opcoes_gen = {1:"Excelente", 2:"Muito Boa", 3:"Boa", 4:"Regular", 5:"Ruim"}
        gen_hlth = st.select_slider("Saúde geral", options=list(opcoes_gen.keys()), format_func=lambda x: opcoes_gen[x])
        
        st.write("---")
        peso = st.number_input("Peso (kg)", min_value=30.0, value=75.0)
        altura_cm = st.number_input("Altura (cm)", min_value=100, value=170)
        imc_calculado = round(peso / ((altura_cm / 100) ** 2), 1)
        st.info(f"IMC: {imc_calculado}")

    with col2:
        st.subheader("Histórico Clínico")
        high_bp = st.checkbox("Pressão Alta?")
        high_chol = st.checkbox("Colesterol Alto?")
        chol_check = st.checkbox("Exame de colesterol (5 anos)?")
        stroke = st.checkbox("Já teve AVC?")
        heart_dis = st.checkbox("Doença Cardíaca?")
        smoker = st.checkbox("Fumante (100+ cigarros)?")
        phys_act = st.checkbox("Atividade física recente?")
        fruits = st.checkbox("Consome Frutas?")
        veggies = st.checkbox("Consome Vegetais?")
        hvy_alcohol = st.checkbox("Álcool em excesso?")
        healthcare = st.checkbox("Plano de saúde?")
        no_doc_cost = st.checkbox("Barreira de custo médico?")
        diff_walk = st.checkbox("Dificuldade de locomoção?")
        
        submit = st.form_submit_button("GERAR ANÁLISE DE RISCO")

# ---------------------------
# Lógica de Previsão
# ---------------------------
if submit:
    input_df = pd.DataFrame([{
        'HighBP': 1 if high_bp else 0, 'HighChol': 1 if high_chol else 0, 'CholCheck': 1 if chol_check else 0,
        'BMI': imc_calculado, 'Smoker': 1 if smoker else 0, 'Stroke': 1 if stroke else 0,
        'HeartDiseaseorAttack': 1 if heart_dis else 0, 'PhysActivity': 1 if phys_act else 0,
        'Fruits': 1 if fruits else 0, 'Veggies': 1 if veggies else 0, 'HvyAlcoholConsump': 1 if hvy_alcohol else 0,
        'AnyHealthcare': 1 if healthcare else 0, 'NoDocbcCost': 1 if no_doc_cost else 0, 'GenHlth': gen_hlth,
        'DiffWalk': 1 if diff_walk else 0, 'Sex': sex, 'Age': age, 'Education': education, 'Income': income
    }])

    try:
        input_df = input_df[modelo.feature_names_in_]
    except:
        pass

    prob = modelo.predict_proba(input_df)[0][1]
    status = "ALTO RISCO" if prob >= threshold_clinico else "BAIXO RISCO"
    
    st.divider()
    if prob >= threshold_clinico:
        st.error(f"### ⚠️ {status}: {prob:.1%}")
        st.warning("Conduta: Encaminhamento prioritário para exames de Glicemia/HbA1c.")
    else:
        st.success(f"### ✅ {status}: {prob:.1%}")
    
    # Botão de download do relatório
    relatorio = f"RELATÓRIO DE TRIAGEM\nData: {data_atual}\nResultado: {status} ({prob:.1%})\nIMC: {imc_calculado}"
    st.download_button("📥 Baixar Relatório", relatorio, file_name="resultado_triagem.txt")

# ---------------------------
# Auditoria Técnica (Abas)
# ---------------------------
st.divider()
st.subheader("📊 Auditoria de Performance do Modelo")
tab_pr, tab_sep, tab_brier, tab_conf, tab_metrics = st.tabs([
    "Curvas Recall-Precision", "Separação de Classes", "Brier Score", "Matriz de Confusão", "Métricas"
])

with tab_pr:
    display_svg_high_quality("Curvas Recall-Precision.svg", caption="Trade-off Precisão/Sensibilidade")

with tab_sep:
    display_svg_high_quality("Separação de Classes.svg", caption="Distribuição de Probabilidades")

with tab_brier:
    display_svg_high_quality("Brier Score.svg", caption="Calibração do Modelo")

with tab_conf:
    display_svg_high_quality("Matriz de Confusão.svg", caption="Erros e Acertos na Validação")

with tab_metrics:
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Recall (Sensibilidade)", f"{data.get('recall', 0):.2%}")
    with col_b:
        st.metric("PR AUC (Average Precision)", f"{data.get('pr_auc', 0):.3f}")

st.caption("Aviso: Esta é uma ferramenta estatística de suporte e não substitui o diagnóstico médico.")
