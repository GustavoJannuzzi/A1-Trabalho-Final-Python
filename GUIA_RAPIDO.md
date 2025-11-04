# 🚀 GUIA RÁPIDO DE INÍCIO

## ⚡ Como Executar a Aplicação

### 1️⃣ Instalar Dependências
```bash
pip install -r requirements.txt
```

### 2️⃣ Executar Aplicação
```bash
streamlit run app.py
```

### 3️⃣ Acessar no Navegador
```
http://localhost:8501
```

---

## 📝 Como Usar (Passo a Passo)

### PASSO 1: Upload do Arquivo
1. Clique em "Browse files" na barra lateral
2. Selecione o arquivo `exemplo_imoveis.csv` (incluído no projeto)
3. Aguarde o carregamento

### PASSO 2: Explorar os Dados
- Vá para a aba **"Visão Geral dos Dados"**
- Veja estatísticas, tipos de colunas e valores nulos
- Ajuste o número de linhas para ver mais/menos dados

### PASSO 3: Visualizar Gráficos
- Acesse a aba **"Análise Visual"**
- Experimente diferentes tipos de gráficos:
  - Histogramas para ver distribuições
  - Box Plots para detectar outliers
  - Scatter Plot para ver relações (ex: area_m2 vs preco)
  - Matriz de Correlação para ver todas as correlações

### PASSO 4: Treinar Modelos
- Vá para a aba **"Machine Learning"**
- **Selecione Features:** area_m2, quartos, banheiros, ano_construcao
- **Selecione Target:** preco
- **Escolha Algoritmos:** Selecione 3-5 algoritmos
- Clique em **"Executar Análise"**
- Compare os resultados (R² Score indica a qualidade)

### PASSO 5: Fazer Predições
- Acesse a aba **"Fazer Predições"**
- Insira valores exemplo:
  - area_m2: 100
  - quartos: 3
  - banheiros: 2
  - ano_construcao: 2020
- Selecione o melhor modelo
- Clique em **"Fazer Predição"**
- Veja o preço estimado!

---

## 🎯 Dicas para Avaliação

### ✅ Demonstre estas Funcionalidades:

1. **Upload e Flexibilidade**
   - Mostre que funciona com diferentes datasets
   - Destaque a detecção automática de tipos

2. **Análise Visual**
   - Mostre múltiplos tipos de gráficos
   - Explique insights encontrados

3. **Machine Learning**
   - Mostre detecção automática (Regressão/Classificação)
   - Compare múltiplos algoritmos
   - Explique as métricas

4. **Código Limpo**
   - Mencione a organização modular
   - Destaque comentários e documentação

---

## 📊 Exemplo de Apresentação

### Roteiro Sugerido:

1. **Introdução (2 min)**
   - Apresentar o projeto e objetivos
   - Mostrar estrutura dos arquivos

2. **Upload e Análise (3 min)**
   - Fazer upload do CSV
   - Mostrar visão geral
   - Destacar tratamento de dados

3. **Visualizações (3 min)**
   - Criar 3-4 gráficos diferentes
   - Explicar insights

4. **Machine Learning (4 min)**
   - Configurar e treinar modelos
   - Comparar resultados
   - Explicar métricas

5. **Predições (2 min)**
   - Fazer uma predição exemplo
   - Mostrar como seria usado na prática

6. **Código (3 min)**
   - Mostrar organização modular
   - Destacar boas práticas
   - Mencionar documentação

---

## 💡 Pontos Fortes do Projeto

- ✅ Interface profissional e intuitiva
- ✅ Código modular e bem documentado
- ✅ Múltiplos algoritmos de ML
- ✅ Tratamento automático de dados
- ✅ Visualizações variadas
- ✅ Sistema de predições funcional
- ✅ README completo
- ✅ Detecção automática de regressão/classificação

---

## 🐛 Solução Rápida de Problemas

**Erro ao instalar dependências?**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Porta 8501 ocupada?**
```bash
streamlit run app.py --server.port 8502
```

**Erro ao carregar CSV?**
- Verifique a codificação (UTF-8)
- Confirme que tem cabeçalho
- Use o arquivo exemplo incluído

---

## 📌 Checklist de Avaliação

Antes de apresentar, verifique:

- [ ] Todos os arquivos estão presentes
- [ ] Requirements.txt instalado
- [ ] Aplicação executa sem erros
- [ ] Upload de CSV funciona
- [ ] Gráficos são gerados
- [ ] Modelos treinam corretamente
- [ ] Predições funcionam
- [ ] README está completo
- [ ] Código está comentado

---

## 🎓 Critérios Atendidos

### ✅ Qualidade Técnica
- Bibliotecas corretas (pandas, sklearn, matplotlib)
- Código limpo e comentado
- Estrutura modular

### ✅ Análise de Dados
- Manipulação com pandas
- Visualizações informativas
- Múltiplos tipos de gráficos

### ✅ Machine Learning
- Múltiplos modelos
- Configuração de parâmetros
- Métricas apropriadas
- Treinamento dinâmico

### ✅ Documentação
- README detalhado
- Comentários no código
- Instruções claras

---

**BOA SORTE NA AVALIAÇÃO! 🎯**
