import io
import os
import joblib
import pytz
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional

# Optional libs for SVG -> PNG conversion (improves raster rendering quality)
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

# ---------------------------
# Carregamento do modelo / artefato
# ---------------------------
@st.cache_resource
def carregar_modelo(path="modelo_diabetes_vtl.pkl"):
    try:
        return joblib.load(path)
    except Exception:
        return None

data = carregar_modelo()
if data is None:
    st.error("Erro crítico: Arquivo 'modelo_diabetes_vtl.pkl' não encontrado ou inválido.")
    st.stop()

modelo = data.get('pipeline') or data.get('model') or data.get('estimator')
threshold_clinico = data.get('threshold', 0.25)

# ---------------------------
# Timezone e cabeçalho
# ---------------------------
fuso_br = pytz.timezone('America/Sao_Paulo')
data_atual = datetime.now(fuso_br).strftime('%d/%m/%Y %H:%M')

st.title("🏥 Sistema de Apoio à Decisão Clínica: Diabetes")
st.markdown(f"**Analista Responsável:** Portal de Triagem Preventiva | **Data:** {data_atual} (Horário de Brasília)")

# Nota adicional solicitada pelo usuário
st.markdown(
    "**Origem dos dados:** Este projeto utiliza o dataset *Diabetes Health Indicators* do Centers for Disease Control and Prevention (CDC), uma base de dados robusta com mais de 250 mil registros que traduzem o perfil de saúde, estilo de vida e indicadores socioeconômicos da população."
)

with st.expander("📝 Nota Metodológica e Motivação Técnica"):
    st.markdown("""
    **Justificativa das Variáveis:**
    * **💰 Socioeconômicos:** Renda e educação são determinantes sociais críticos. 
    * **🚬 Estilo de Vida:** Tabagismo e sedentarismo são marcadores de risco metabólico.
    * **🏃 Atividade Física:** Identifica sedentarismo, um marcador crítico de risco metabólico.
    * **🏥 Custo:** Avalia barreiras financeiras que impedem o diagnóstico precoce.

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

# ---------------------------
# Previsão e relatório
# ---------------------------
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

    # Reordenar conforme modelo (se disponível)
    try:
        input_data = input_data[modelo.feature_names_in_]
    except Exception:
        # se não for possível reordenar, assume-se que as colunas já batem
        pass

    prob = modelo.predict_proba(input_data)[0][1]

    st.divider()
    status_risco = "ALTO RISCO" if prob >= threshold_clinico else "BAIXO RISCO"

    if prob >= threshold_clinico:
        st.error(f"### ⚠️ {status_risco} IDENTIFICADO: {prob:.1%}")
        st.markdown("**Conduta sugerida:** Encaminhamento para Glicemia de Jejum e HbA1c.")
    else:
        st.success(f"### ✅ {status_risco} IDENTIFICADO: {prob:.1%}")

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

# ---------------------------
# Util: exibir SVG com melhor qualidade quando possível
# ---------------------------
def display_svg_high_quality(path: str, scale: int = 2, caption: Optional[str] = None):
    """
    Tenta converter SVG para PNG em alta resolução usando cairosvg (se disponível).
    Se cairosvg não estiver instalado, embute o SVG diretamente via components.html.
    scale: multiplicador de resolução (1,2,3...). Para SVG vetorial, a conversão melhora raster output.
    """
    if not os.path.exists(path):
        st.warning(f"Arquivo não encontrado: {os.path.basename(path)}")
        return

    if CAIROSVG_AVAILABLE:
        try:
            with open(path, "rb") as f:
                svg_bytes = f.read()
            # converter para PNG em memória com escala
            png_bytes = cairosvg.svg2png(bytestring=svg_bytes, scale=scale)
            img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
            st.image(img, use_column_width=True, caption=caption)
            return
        except Exception:
            # fallback para embed SVG
            pass

    # fallback: embed SVG diretamente (mantém qualidade vetorial em navegadores que suportam)
    try:
        svg_text = open(path, "r", encoding="utf-8").read()
        # ajustar largura responsiva
        html = f"""
        <div style="width:100%; display:flex; justify-content:center;">
            {svg_text}
        </div>
        """
        st.components.v1.html(html, height=400)
        if caption:
            st.caption(caption)
    except Exception:
        st.warning(f"Não foi possível renderizar o arquivo: {os.path.basename(path)}")

# ---------------------------
# Auditoria Técnica: exibir apenas Recall e Average Precision
# ---------------------------
st.divider()
with st.expander("🔍 Auditoria Técnica (Gráficos e Métricas)"):
    st.write("Abaixo estão os artefatos de avaliação do modelo. Se algum SVG não estiver disponível, uma mensagem será exibida.")

    # Layout 2x2 para gráficos
    row1_col1, row1_col2 = st.columns(2)
    row2_col1, row2_col2 = st.columns(2)

    # Curva Precisão-Recall
    with row1_col1:
        display_svg_high_quality("Curva Precisão-Recall.svg", scale=3, caption="Curva Precisão-Recall")

    # Separação de Classes
    with row1_col2:
        display_svg_high_quality("Separação de Classes.svg", scale=3, caption="Separação de Classes")

    # Brier Score (exibe gráfico; valor já presente no gráfico)
    with row2_col1:
        display_svg_high_quality("Brier Score.svg", scale=3, caption="Brier Score (gráfico)")

    # Matriz de Confusão
    with row2_col2:
        # tenta nomes alternativos para compatibilidade
        if os.path.exists("Matriz de Confusão.svg"):
            display_svg_high_quality("Matriz de Confusão.svg", scale=3, caption="Matriz de Confusão")
        elif os.path.exists("Confusion Matrix.svg"):
            display_svg_high_quality("Confusion Matrix.svg", scale=3, caption="Matriz de Confusão")
        else:
            st.warning("Arquivo 'Matriz de Confusão.svg' não encontrado no repositório.")

    st.markdown("---")
    # Exibir apenas Recall e Average Precision (PR AUC)
    recall_val = None
    pr_auc_val = None
    if isinstance(data, dict):
        recall_val = data.get('recall', None)
        pr_auc_val = data.get('pr_auc', None)

    col_a, col_b = st.columns(2)
    if recall_val is None:
        col_a.metric("Recall (validação)", "N/A")
    else:
        col_a.metric("Recall (validação)", f"{recall_val:.2%}")

    if pr_auc_val is None:
        col_b.metric("Average Precision (PR AUC)", "N/A")
    else:
        col_b.metric("Average Precision (PR AUC)", f"{pr_auc_val:.3f}")

st.caption("Aviso: Ferramenta estatística de suporte. Não substitui o diagnóstico médico.")
