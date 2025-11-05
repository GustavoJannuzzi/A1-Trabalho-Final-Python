# Aplicação Web de Análise de Dados e Machine Learning

**Avaliação Final - Python**

---

## Descrição do Projeto

Esta é uma aplicação web interativa desenvolvida em Python que combina um frontend em Streamlit com um backend em Flask API. A arquitetura separa a interface do usuário do processamento pesado, permitindo escalabilidade e melhor organização do código.

### Funcionalidades:

- **Upload de arquivos CSV** com dados estruturados
- **Análise exploratória** de dados com estatísticas e visualizações
- **Visualizações interativas** (histogramas, box plots, scatter plots, matriz de correlação)
- **Machine Learning** com múltiplos algoritmos de regressão e classificação
- **Predições personalizadas** baseadas em modelos treinados
- **Treinamento dinâmico** de modelos com novos dados

---

## Arquitetura

### Backend (Flask API)
Responsável pelo processamento pesado:
- Processamento e limpeza de dados
- Treinamento de modelos de Machine Learning
- Avaliação de métricas
- Predições

### Frontend (Streamlit)
Responsável pela interface do usuário:
- Upload de arquivos
- Visualizações interativas
- Apresentação de resultados
- Interface de configuração

---

## Estrutura do Projeto

```
projeto/
│
├── backend/
│   ├── api.py                    # API Flask principal
│   ├── data_processor.py         # Processamento de dados
│   ├── ml_models.py              # Modelos de Machine Learning
│   └── requirements.txt          # Dependências do backend
│
├── frontend/
│   ├── app.py                    # Aplicação Streamlit
│   ├── visualizations.py         # Visualizações de dados
│   └── requirements.txt          # Dependências do frontend
│
├── requirements.txt              # Dependências completas
└── README.md                     # Documentação
```

---

## Instalação

### Passo a Passo

1. **Clone o repositório:**
```bash
git clone https://github.com/GustavoJannuzzi/A1-Trabalho-Final-Python.git
cd A1-Trabalho-Final-Python
```

2. **Instale as dependências:**

**Opção 1 - Instalar tudo:**
```bash
pip install -r requirements.txt
```

**Opção 2 - Instalar separadamente:**
```bash
# Backend
pip install -r backend/requirements.txt

# Frontend
pip install -r frontend/requirements.txt
```

---

## Como Executar

### 1. Iniciar o Backend (Flask API)

Em um terminal, execute:

```bash
python backend/api.py
```

A API estará disponível em: `http://localhost:5000`

### 2. Iniciar o Frontend (Streamlit)

Em outro terminal, execute:

```bash
streamlit run frontend/app.py
```

O Streamlit abrirá automaticamente em: `http://localhost:8501`

**IMPORTANTE:** O backend DEVE estar rodando antes de iniciar o frontend!

---

## Tecnologias

### Backend
- **Flask:** Framework web para API REST
- **Pandas:** Manipulação e análise de dados
- **NumPy:** Computação numérica
- **Scikit-learn:** Machine Learning e avaliação de modelos

### Frontend
- **Streamlit:** Framework para aplicações web interativas
- **Matplotlib & Seaborn:** Visualização de dados
- **Requests:** Comunicação HTTP com a API

---

## API Endpoints

### Backend Flask API

| Endpoint | Método | Descrição |
|----------|--------|-----------|
| `/api/health` | GET | Verificar status da API |
| `/api/upload` | POST | Upload de arquivo CSV |
| `/api/data/overview` | GET | Visão geral dos dados |
| `/api/data/columns` | GET | Informações das colunas |
| `/api/train` | POST | Treinar modelos ML |
| `/api/predict` | POST | Fazer predições |
| `/api/models` | GET | Listar modelos treinados |
| `/api/reset` | POST | Limpar dados armazenados |

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

## Como Usar

### 1. Upload de Dados

1. Certifique-se de que o backend está rodando
2. Abra o frontend Streamlit
3. Clique no botão "Browse files" na barra lateral
4. Selecione um arquivo CSV
5. O sistema carregará e mostrará informações básicas

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

Os parâmetros dos modelos podem ser ajustados editando o arquivo `backend/ml_models.py`:

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

## Troubleshooting

### Erro: "Não foi possível conectar à API Flask"

**Solução:** Certifique-se de que o backend está rodando:
```bash
python backend/api.py
```

### Erro: "ModuleNotFoundError"

**Solução:** Instale as dependências:
```bash
pip install -r requirements.txt
```

### Erro: "Port already in use"

**Solução:** Altere a porta no arquivo `backend/api.py`:
```python
app.run(debug=True, host='0.0.0.0', port=5001)  # Mudar de 5000 para 5001
```

E no arquivo `frontend/app.py`:
```python
API_URL = "http://localhost:5001"  # Atualizar a porta
```

---

## Melhorias Futuras

- [ ] Suporte para mais formatos (Excel, JSON)
- [ ] Cache de modelos treinados com Redis
- [ ] Autenticação de usuários
- [ ] Deploy com Docker
- [ ] Otimização automática de hiperparâmetros
- [ ] Export de modelos treinados
- [ ] Relatórios em PDF
- [ ] API de versionamento de modelos

---

## Contribuindo

Contribuições são bem-vindas! Por favor:

1. Faça um fork do projeto
2. Crie uma branch para sua feature (`git checkout -b feature/NovaFuncionalidade`)
3. Commit suas mudanças (`git commit -m 'Adiciona nova funcionalidade'`)
4. Push para a branch (`git push origin feature/NovaFuncionalidade`)
5. Abra um Pull Request

---

## Licença

Este projeto foi desenvolvido para fins educacionais como parte da avaliação final da disciplina de Python.

---

## Autores

Desenvolvido por alunos da disciplina de Tópicos Especiais em Python

---

## Contato

Para dúvidas ou sugestões, abra uma issue no GitHub.
