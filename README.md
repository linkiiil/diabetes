# 🩺 Diabetes

### 🏥 Sistema de Triagem Inteligente: Diabetes Risk Predictor

Este projeto apresenta uma solução de Machine Learning para a identificação precoce de risco de diabetes, utilizando a base de dados histórica do CDC (Centers for Disease Control and Prevention). O objetivo central é fornecer uma ferramenta de suporte à decisão clínica com foco em Alta Sensibilidade (Recall) para viabilizar triagens populacionais preventivas.

### 📋 Contexto e Objetivos

O diabetes é uma patologia crônica de elevado impacto socioeconômico. Este modelo foi desenvolvido para converter indicadores de saúde, comportamento e dados socioeconômicos em probabilidades de risco, permitindo intervenções médicas antes do agravamento do quadro clínico.

### 🛠️ Metodologia e Tecnologias

Algoritmo Principal: LightGBM (LGBM) otimizado através de busca hiperparamétrica (RandomizedSearchCV).

Tratamento de Desbalanceamento: Modelagem da realidade do fenômeno sem o uso de técnicas de reamostragem sintética (ex: SMOTE). O desbalanceamento foi tratado diretamente na calibração probabilística do algoritmo e no ajuste do limite de decisão.

Mitigação de Data Leakage: Remoção cirúrgica de variáveis de consequência (sintomas tardios) como MentHlth e PhysHlth, garantindo que o modelo identifique causas e fatores de risco em pacientes assintomáticos.

Interpretabilidade (XAI): Implementação de valores SHAP (SHapley Additive exPlanations) para explicar o impacto individual de cada variável no risco calculado, eliminando o efeito "caixa preta".

Threshold Clínico: Deslocamento do ponto de corte padrão de 0.50 para 0.25, otimizado especificamente para a realidade de saúde pública, onde o custo de um falso negativo supera o de um falso positivo.

🇧🇷 Adaptação Socioeconômica (Critério FGV)
Um diferencial técnico deste projeto é a transposição metodológica das variáveis de renda do dataset original (em USD) para o contexto brasileiro. Para evitar distorções cambiais e inflacionárias, aplicamos o conceito de Salários Mínimos (SM) baseado na classificação de classes econômicas da FGV:

Motivação: O Salário Mínimo atua como um proxy fiel para o poder de compra e acesso a determinantes de saúde (alimentação e medicina preventiva) no Brasil.

Mapeamento: As 5 classes oficiais da FGV (A a E) foram distribuídas nos 8 níveis ordinais do modelo original, garantindo que a "vulnerabilidade financeira" lida pela IA corresponda à realidade nacional.

| Classe FGV | Faixa de Renda (SM) | Código IA | Nível de Acesso Estimado |
| :--- | :--- | :---: | :--- |
| **Classe E** | Até 1 SM | 1 | Extrema vulnerabilidade social |
| **Classe D** | 1 a 4 SM | 2 - 3 | Baixa renda com barreiras de acesso |
| **Classe C** | 4 a 15 SM | 4 - 5 | Classe média (maior variação de risco) |
| **Classe B** | 15 a 20 SM | 6 | Alta renda e acesso à saúde privada |
| **Classe A** | Acima de 20 SM | 7 - 8 | Topo da pirâmide e prevenção plena |

### 📈 Resultados Técnicos e Estratégia Clínica

O modelo foi estrategicamente calibrado para priorizar a captura de casos positivos:

Recall (Sensibilidade): 94.89% (identificando 6.660 casos reais no conjunto de teste).

ROC-AUC: 0.8166.

Estratégia: Priorizamos a Sensibilidade para assegurar orientação precoce. Essa configuração funciona como uma "rede de triagem fina": gera-se um volume maior de falsos positivos (que são descartados posteriormente com exames de sangue baratos), mas garante-se que o número de casos reais perdidos (falsos negativos) seja reduzido ao mínimo (apenas 359 casos não detectados).

### 🔍 Auditoria Técnica e Transparência

A aplicação conta com uma aba dedicada à Auditoria Técnica, permitindo que gestores de saúde e cientistas de dados validem a confiabilidade de cada predição através de:

Brier Score: Validação da calibração (avalia a precisão da probabilidade prevista em relação à frequência observada).

Separação de Classes: Visualização da densidade de probabilidade para pacientes saudáveis e em risco.

Curva Recall-Precision: Demonstração visual do trade-off assumido ao estabelecer o threshold de 0.25.

### ⚠️ Nota sobre o IMC (Body Mass Index)

O modelo utiliza o IMC derivado de dados autorreferidos. Durante a análise exploratória (EDA), observou-se que 92,1% dos dados concentram-se entre os IMCs 20 e 40. Valores extremos (ex: IMC > 60) representam o ruído estatístico comum em surveys (informações irreais preenchidas pelos usuários). Optou-se por mantê-los para preservar a distribuição original dos dados, com o modelo tratando a base de forma robusta, porém exige-se cautela na avaliação clínica de casos limítrofes individuais.

### 📂 Estrutura do Repositório

├── Diabetes_Predictive_Radar.pdf   
Apresentação em slides resumindo o contexto clínico, a estratégia adotada e os resultados alcançados.

├── Projeto - Diabetes.ipynb   
Notebook completo contendo a Análise Exploratória (EDA), desenvolvimento do modelo, análise SHAP e validação das hipóteses.

├── diabetes.py   
Aplicação interativa para triagem desenvolvida em Streamlit.

├── modelo_diabetes_vtl.pkl   
Pipeline serializado do LGBM, incluindo transformadores e parâmetros de calibração.

├── requirements.txt   
Lista de dependências (Pandas, Scikit-Learn, LightGBM, Streamlit, SHAP).

├── Gráficos Vetoriais (.svg)  
Curvas Recall-Precision, Brier Score, Separação de Classes e Matriz de Confusão exportadas em alta qualidade.

### 💻 Como Executar Localmente

Clone o repositório:

Bash
git clone https://github.com/linkiiil/diabetes.git
Instale as dependências:

Bash
pip install -r requirements.txt
Execute a aplicação:

Bash
streamlit run diabetes.py
