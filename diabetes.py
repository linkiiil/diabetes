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
        # Carrega o dicionário que contém o pipeline treinado e o threshold
        dados_modelo = joblib.load('modelo_diabetes_vtl.pkl')
        return dados_modelo
    except FileNotFoundError:
        return None

data = carregar_modelo()
if data is None:
    st.error("Erro crítico: Arquivo 'modelo_diabetes_vtl.pkl' não encontrado no diretório raiz.")
    st.stop()

# Extração dos componentes do modelo
modelo = data['pipeline']
threshold_clinico = data.get('threshold', 0.25)

# 3. Cabeçalho e Nota Metodológica
st.title("🏥 Sistema de Apoio à Decisão Clínica: Diabetes")
st.markdown("Utilizando Inteligência Artificial para triagem populacional baseada nos critérios do CDC.")

with st.expander("📝 Nota Metodológica: Por que essas perguntas são necessárias?"):
    st.markdown("""
    Este sistema utiliza o padrão epidemiológico do **CDC (Centers for Disease Control and Prevention)**. Algumas perguntas possuem justificativas técnicas importantes para o cálculo de risco:
    
    * **💰 Aspectos Socioeconômicos:** Escolaridade e renda são "Determinantes Sociais de Saúde". Eles impactam o acesso a alimentos de qualidade e a frequência de cuidados preventivos.
    * **🚬 Os 100 Cigarros:** Marco clínico global para distinguir o uso social do **tabagismo estabelecido**, onde o risco de resistência à insulina aumenta consideravelmente.
    * **🏃 Atividade Física:** O foco é identificar o sedentarismo. Não praticar exercícios no último mês é um marcador crítico de risco metabólico.
    * **🏥 Custo e Saúde:** Identifica barreiras financeiras que impedem o diagnóstico precoce, fator comum em doenças crônicas subnotificadas.
    """)

# 4. Formulário de Entrada de Dados
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
        income = st.selectbox("Sua faixa de renda familiar anual", options=list(opcoes_inc.keys()), format_func=lambda x: opcoes_inc[x])

        opcoes_edu = {1:"Nunca estudou/Jardim", 2:"1ª a 8ª série (Fundamental)", 3:"9ª a 11ª (Médio inc.)", 
                      4:"12ª série/GED (Médio comp.)", 5:"Superior incompleto/Técnico", 6:"Graduado"}
        education = st.selectbox("Nível de Escolaridade", options=list(opcoes_edu.keys()), format_func=lambda x: opcoes_edu[x])

        sex = st.radio("Sexo Biológico", options=[0, 1], format_func=lambda x: "Feminino" if x==0 else "Masculino")
        
        opcoes_gen = {1:"Excelente", 2:"Muito Boa", 3:"Boa", 4:"Regular", 5:"Ruim"}
        gen_hlth = st.select_slider("Como avalia sua saúde geral?", options=list(opcoes_gen.keys()), format_func=lambda x: opcoes_gen[x])
        
        st.write("---")
        st.markdown("**Cálculo de IMC (Índice de Massa Corporal)**")
        c1, c2 = st.columns(2)
        peso = c1.number_input("Peso (kg)", min_value=30.0, max_value=250.0, value=75.0)
        altura_cm = c2.number_input("Altura (cm)", min_value=100, max_value=230, value=170)
        imc_calculado = round(peso / ((altura_cm / 100) ** 2), 1)
        st.info(f"Seu IMC calculado é: **{imc_calculado}**")

    with col2:
        st.subheader("Histórico Clínico e Comportamento")
        
        high_bp = st.checkbox("Possui Pressão Alta?")
        high_chol = st.checkbox("Possui Colesterol Alto?")
        chol_check = st.checkbox("Fez exame de colesterol nos últimos 5 anos?")
        stroke = st.checkbox("Já teve um AVC?")
        heart_dis = st.checkbox("Possui histórico de doença cardíaca ou infarto?")
        smoker = st.checkbox("Já fumou pelo menos 100 cigarros ao longo da vida?")
        phys_act = st.checkbox("Praticou atividade física no último mês?")
        fruits = st.checkbox("Consome Frutas regularmente (1+ vez ao dia)?")
        veggies = st.checkbox("Consome Vegetais regularmente (1+ vez ao dia)?")
        hvy_alcohol = st.checkbox("Possui consumo excessivo de álcool?")
        healthcare = st.checkbox("Possui acesso a algum plano/seguro de saúde?", value=True)
        no_doc_cost = st.checkbox("Já deixou de ir ao médico por causa do custo?")
        diff_walk = st.checkbox("Dificuldade séria para caminhar ou subir escadas?")

    submit = st.form_submit_button("GERAR ANÁLISE DE RISCO")

# 5. Processamento e Exibição de Resultados
if submit:
    # Construção do DataFrame respeitando a ordem das 19 colunas do treinamento
    input_data = pd.DataFrame([{
        'HighBP': 1 if high_bp else 0,
        'HighChol': 1 if high_chol else 0,
        'CholCheck': 1 if chol_check else 0,
        'BMI': imc_calculado,
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

    # Reordenamento explícito das colunas
    colunas_modelo = ['HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke', 
                      'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies', 
                      'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth', 
                      'DiffWalk', 'Sex', 'Age', 'Education', 'Income']
    input_data = input_data[colunas_modelo]

    # Predição de Probabilidade
    prob = modelo.predict_proba(input_data)[0][1]
    
    st.divider()
    
    # Exibição do Resultado com Alertas Coloridos
    if prob >= threshold_clinico:
        st.error(f"### ⚠️ ALTO RISCO IDENTIFICADO: {prob:.1%}")
        st.markdown(f"""
        **Análise Clínica:** Seu perfil apresenta indicadores que, estatisticamente, estão associados a uma maior probabilidade de diabetes. 
        O modelo utilizou um ponto de corte conservador (Threshold {threshold_clinico*100:.0f}%) para garantir a máxima segurança na triagem.
        
        **👉 Próximo Passo:** Recomendamos que você procure um clínico geral para realizar exames laboratoriais de confirmação (Glicemia de Jejum e HbA1c).
        """)
    else:
        st.success(f"### ✅ BAIXO RISCO IDENTIFICADO: {prob:.1%}")
        st.markdown("""
        Seus indicadores atuais sugerem um baixo risco para diabetes segundo o modelo preditivo. 
        Continue mantendo hábitos saudáveis e realizando seus exames de rotina regularmente.
        """)

    # --- GERAÇÃO DO RELATÓRIO PARA DOWNLOAD ---
    texto_relatorio = f"""
    ==================================================
    RELATÓRIO DE TRIAGEM PREVENTIVA - DIABETES (IA)
    ==================================================
    
    RESULTADO DA ANÁLISE:
    - Probabilidade Calculada: {prob:.1%}
    - Classificação: {"ALTO RISCO" if prob >= threshold_clinico else "BAIXO RISCO"}
    - Threshold Utilizado: {threshold_clinico}
    
    SÍNTESE DOS INDICADORES:
    - IMC: {imc_calculado}
    - Pressão Alta: {"Sim" if high_bp else "Não"}
    - Colesterol Alto: {"Sim" if high_chol else "Não"}
    - Dificuldade Locomoção: {"Sim" if diff_walk else "Não"}
    - Histórico Cardíaco: {"Sim" if heart_dis else "Não"}
    
    OBSERVAÇÃO PARA O PROFISSIONAL DE SAÚDE:
    Este relatório foi gerado por um modelo de Machine Learning treinado com 
    dados do CDC/BRFSS. O algoritmo foi calibrado para Alta Sensibilidade (Recall: 94.8%) 
    objetivando a triagem populacional precoce.
    
    SUGESTÃO DE CONDUTA:
    Avaliar necessidade de exames confirmatórios:
    [ ] Glicemia de Jejum
    [ ] Hemoglobina Glicada (HbA1c)
    
    Data: {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M')}
    ==================================================
    """
    
    st.download_button(
        label="📥 Baixar Relatório Completo para Consulta Médica",
        data=texto_relatorio,
        file_name="relatorio_diabetes.txt",
        mime="text/plain"
    )

# 6. Transparência Técnica (Matriz de Confusão)
st.divider()
with st.expander("🔍 Detalhes Técnicos e Matriz de Confusão (Auditoria)"):
    st.write("### Performance do Modelo no Conjunto de Teste")
    
    col_text, col_img = st.columns([1, 1.5])
    
    with col_text:
        st.markdown(f"""
        **Métricas de Validação:**
        * **Recall (Sensibilidade):** 94.86%
        * **Casos Identificados:** 6.658 (True Positives)
        * **Falsos Negativos:** 361 (Casos perdidos)
        
        **Estratégia:**
        Optamos por um Threshold de **{threshold_clinico}**. Na saúde pública, o custo de um 'falso alarme' 
        (falso positivo) é menor do que o risco de não detectar um paciente doente (falso negativo).
        """)
    
    with col_img:
        try:
            # Busca o arquivo SVG que deve estar na mesma pasta do código
            st.image("Confusion Matrix.svg", use_container_width=True, caption="Matriz de Confusão Vetorial (SVG)")
        except:
            st.warning("⚠️ Arquivo 'Confusion Matrix.svg' não encontrado no repositório.")

# Rodapé Ético
st.caption("Aviso: Esta ferramenta é um simulador estatístico para fins educacionais e de triagem. Não substitui o diagnóstico médico profissional.")
