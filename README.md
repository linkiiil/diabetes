# Diabetes

🏥 Sistema de Triagem Inteligente: Diabetes Risk Predictor

Este projeto apresenta uma solução de Machine Learning para a identificação precoce de risco de diabetes, utilizando a base de dados histórica do CDC (Centers for Disease Control and Prevention). O objetivo central é fornecer uma ferramenta de suporte à decisão clínica com foco em Alta Sensibilidade (Recall) para viabilizar triagens populacionais preventivas.

📋 Contexto e Objetivos

O diabetes é uma patologia crônica de elevado impacto socioeconômico. Este modelo foi desenvolvido para converter indicadores de saúde, comportamento e dados socioeconômicos em probabilidades de risco, permitindo intervenções médicas antes do agravamento do quadro clínico.

🛠️ Metodologia e Tecnologias

Algoritmo Principal: LightGBM (LGBM) otimizado através de busca hiperparamétrica (RandomizedSearchCV).

Pré-processamento: Limpeza de dados duplicados, tratamento de desbalanceamento de classe e mitigação de Data Leakage através da exclusão das variáveis MentHlth e PhysHlth.

Interpretabilidade: Implementação de valores SHAP (SHapley Additive exPlanations) para explicar a contribuição de cada variável no risco calculado.

🇧🇷 Adaptação Socioeconômica (Critério FGV)
Um diferencial técnico deste projeto é a transposição metodológica das variáveis de renda para o contexto brasileiro.

O modelo original utiliza faixas em dólares (USD). Para evitar distorções cambiais e inflacionárias, aplicamos o conceito de Salários Mínimos (SM) baseado na classificação de classes econômicas da FGV:

  Motivação: O Salário Mínimo atua como um proxy fiel para o poder de compra e acesso a determinantes de saúde (alimentação e medicina preventiva) no Brasil;

  Mapeamento: As 5 classes oficiais da FGV (A a E) foram distribuídas nos 8 níveis ordinais do modelo original, garantindo que a "vulnerabilidade financeira" lida   pela IA corresponda à realidade nacional.

Classe FGV	   Faixa (em SM)	  Código IA	   Nível de Acesso Estimado
-----------------------------------------------------------------------------------
Classe E	     Até 1 SM	        1	           Extrema vulnerabilidade social
Classe D	     1 a 4 SM	        2 - 3	       Baixa renda com barreiras de acesso
Classe C	     4 a 15 SM	      4 - 5	       Classe média (maior variação de risco)
Classe B	     15 a 20 SM	      6	           Alta renda e acesso à saúde privada
Classe A	     Acima de 20 SM	  7 - 8	       Topo da pirâmide e prevenção plena
-----------------------------------------------------------------------------------
📈 Resultados Técnicos

O modelo foi estrategicamente calibrado para priorizar a captura de casos positivos (Diabetes), resultando em:

Recall (Sensibilidade): 94,86% (identificando 6.658 casos reais no conjunto de teste).

ROC-AUC: 0,8165.

Threshold Clínico: 0,25 — ponto de corte otimizado para maximizar a triagem preventiva.

📊 Matriz de Confusão e Estratégia Clínica

Priorizamos a Sensibilidade para assegurar que o paciente receba orientação precoce.

A imagem detalha a performance utilizando o Threshold de 0,25. Embora gere 22.420 falsos positivos, a estratégia garante que apenas 361 casos reais (falsos negativos) não sejam detectados, priorizando a segurança clínica.

📂 Estrutura do Repositório

diabetes.py: Aplicação interativa desenvolvida em Streamlit.

modelo_diabetes_vtl.pkl: Binário do modelo LGBM, incluindo transformadores e parâmetros de calibração.

requirements.txt: Lista de dependências técnicas (Pandas, Scikit-Learn, XGBoost, LightGBM).

matriz_confusao.svg: Gráfico vetorial de performance para documentação.

Projeto - Diabetes.pdf: Relatório técnico completo contendo EDA, SHAP e validação de hipóteses.

README.md: Guia de apresentação e documentação do projeto.

💻 Como Executar Localmente

Clone o repositório:

Bash

git clone https://github.com/linkiiil/diabetes.git

Instale as dependências:

Bash

pip install -r requirements.txt

Execute a aplicação:

Bash

streamlit run diabetes.py
