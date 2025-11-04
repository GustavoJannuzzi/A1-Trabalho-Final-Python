# Aplicação Web de Análise de Dados e Machine Learning

**Avaliação Final - Python**  
---

## Descrição do Projeto

Esta é uma aplicação web interativa desenvolvida em Python usando Streamlit que permite aos usuários:

- **Upload de arquivos CSV** com dados estruturados
- **Análise exploratória** de dados com estatísticas e visualizações
- **Visualizações interativas** (histogramas, box plots, scatter plots, matriz de correlação)
- **Machine Learning** com múltiplos algoritmos de regressão e classificação
- **Predições personalizadas** baseadas em modelos treinados
- **Treinamento dinâmico** de modelos com novos dados

---

## Funcionalidades Principais

### 1. Upload e Flexibilidade dos Dados
- Suporte para arquivos CSV com dados estruturados
- Detecção automática de tipos de dados (numéricos, categóricos)
- Análise de valores nulos e duplicados
- Tratamento automático de dados categóricos

### 2. Análise Exploratória de Dados
- **Visão Geral:** Métricas básicas (linhas, colunas, memória, valores nulos)
- **Estatísticas Descritivas:** Média, mediana, desvio padrão, quartis
- **Informações de Colunas:** Tipos de dados, valores únicos, porcentagem de nulos

### 3. Visualizações Avançadas

#### Gráficos Disponíveis:
- **Histogramas:** Distribuição de variáveis numéricas
- **Box Plots:** Detecção de outliers
- **Scatter Plots:** Relação entre variáveis com linha de tendência
- **Matriz de Correlação:** Heatmap de correlações entre features
- **Gráficos de Barras e Pizza:** Distribuição de variáveis categóricas
- **Mapa de Valores Nulos:** Visualização de dados faltantes

### 4. Machine Learning

#### Algoritmos de Regressão:
- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)

#### Algoritmos de Classificação:
- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier
- Support Vector Machine (SVC)
- K-Nearest Neighbors (KNN)

#### Métricas de Avaliação:

**Para Regressão:**
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² Score
- MAPE (Mean Absolute Percentage Error)

**Para Classificação:**
- Acurácia
- Precisão
- Recall
- F1-Score

### 5. Sistema de Predições
- Interface interativa para inserir novos dados
- Predições com qualquer modelo treinado
- Visualização de métricas do modelo usado

---

## Tecnologias 

- **Python 3.x**
- **Streamlit:** Framework para aplicações web interativas
- **Pandas:** Manipulação e análise de dados
- **NumPy:** Computação numérica
- **Matplotlib & Seaborn:** Visualização de dados
- **Scikit-learn:** Machine Learning e avaliação de modelos

---

## Instalação

### Passo a Passo

1. **Clone ou baixe o projeto:**
```bash
git clone github.com/GustavoJannuzzi/A1-Trabalho-Final-Python
cd A1-Trabalho-Final-Python
```

2. **Instale as dependências:**
```bash
pip install -r requirements.txt
```

4. **Execute a aplicação:**
```bash
streamlit run app.py
```

---

## Estrutura do Projeto

```
projeto/
│
├── app.py                  # Arquivo principal do Streamlit
├── data_processor.py       # Processamento e limpeza de dados
├── visualizations.py       # Criação de gráficos e visualizações
├── ml_models.py           # Treinamento e avaliação de modelos ML
├── requirements.txt       # Dependências do projeto
└── README.md             
```

### Descrição dos Módulos

#### `app.py`
Arquivo principal que contém:
- Interface do usuário com Streamlit
- Sistema de abas (Visão Geral, Análise Visual, ML, Predições)
- Gerenciamento de estado da sessão
- Coordenação entre todos os módulos

#### `data_processor.py`
Responsável por:
- Limpeza de dados (valores nulos, duplicados)
- Codificação de variáveis categóricas
- Tratamento de outliers
- Preparação de dados para ML

#### `visualizations.py`
Contém funções para:
- Criação de histogramas e box plots
- Gráficos de dispersão com linha de tendência
- Matriz de correlação
- Visualização de categorias
- Comparação de modelos

#### `ml_models.py`
Implementa:
- Treinamento de múltiplos modelos
- Avaliação com métricas apropriadas
- Sistema de predições
- Validação cruzada
- Re-treinamento dinâmico

---

## Como Usar

### 1. Upload de Dados

1. Clique no botão "Browse files" na barra lateral
2. Selecione um arquivo CSV
3. O sistema carregará e mostrará informações básicas

### 2. Análise Exploratória

- Navegue pela aba **"Visão Geral dos Dados"**
- Veja métricas, amostra dos dados e estatísticas
- Identifique colunas numéricas e categóricas

### 3. Visualização

- Acesse a aba **"Análise Visual"**
- Escolha o tipo de gráfico desejado
- Selecione as colunas para visualizar
- Explore padrões e relações nos dados

### 4. Machine Learning

- Vá para a aba **"Machine Learning"**
- Selecione as features (atributos preditivos)
- Escolha a variável alvo (target)
- O sistema detecta automaticamente se é regressão ou classificação
- Selecione os algoritmos que deseja treinar
- Configure parâmetros de treinamento
- Clique em **"Executar Análise"**
- Compare os resultados dos modelos

### 5. Fazer Predições

- Acesse a aba **"Fazer Predições"**
- Insira os valores para cada feature
- Selecione o modelo a usar
- Clique em **"Fazer Predição"**
- Veja o resultado e as métricas do modelo

---

## Exemplos de Datasets Compatíveis

### Imóveis
```csv
bairro,area_m2,quartos,banheiros,ano_construcao,preco
Centro,120,3,2,2015,350000
Jardim,85,2,1,2018,280000
```

### Vendas
```csv
marketing_spend,region,season,sales
5000,North,Summer,45000
3000,South,Winter,32000
```

### Classificação de Flores
```csv
sepal_length,sepal_width,petal_length,petal_width,species
5.1,3.5,1.4,0.2,setosa
7.0,3.2,4.7,1.4,versicolor
```
---

## Configurações Avançadas

### Parâmetros de Treinamento

- **Tamanho do conjunto de teste:** 10% - 50% (padrão: 20%)
- **Semente aleatória:** Para reprodutibilidade dos resultados
- **Tratamento de valores nulos:** Opcional, remove linhas com dados faltantes

### Personalização de Modelos

Os parâmetros dos modelos podem ser ajustados editando o arquivo `ml_models.py`:

```python
# Exemplo: Alterar número de árvores no Random Forest
RandomForestRegressor(n_estimators=200, random_state=42)
```

---

## Requisitos dos Dados


1. **Formato:** Arquivo CSV com cabeçalho
2. **Codificação:** UTF-8 (recomendado)
3. **Separador:** Vírgula (`,`)
4. **Colunas:** Nomes únicos e descritivos
5. **Dados Numéricos:** Para features e target em regressão
6. **Valores Nulos:** Minimizar para melhor performance

---

## Métricas de Performance

### Regressão

- **R² Score:** Quanto o modelo explica a variância (0 a 1, maior é melhor)
- **RMSE:** Erro médio absoluto (menor é melhor)
- **MAE:** Erro absoluto médio (menor é melhor)

### Classificação

- **Acurácia:** Proporção de predições corretas
- **Precisão:** Qualidade das predições positivas
- **Recall:** Capacidade de encontrar todos os casos positivos
- **F1-Score:** Média harmônica entre precisão e recall

---

## 🚀 Melhorias Futuras

- [ ] Suporte para mais formatos (Excel, JSON)
- [ ] Gráficos 3D e mapas interativos
- [ ] Otimização automática de hiperparâmetros
- [ ] Export de modelos treinados
- [ ] Relatórios em PDF
- [ ] Detecção automática de outliers
- [ ] Feature engineering automático
- [ ] Comparação com baseline models

---

