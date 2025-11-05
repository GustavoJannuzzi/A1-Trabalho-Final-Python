# Guia de Início Rápido

## Inicialização Rápida

### Opção 1: Usar o script de inicialização (Windows)

Basta executar o arquivo `start.bat`:

```bash
start.bat
```

Isso abrirá dois terminais:
- Terminal 1: Backend Flask (http://localhost:5000)
- Terminal 2: Frontend Streamlit (http://localhost:8501)

### Opção 2: Inicialização manual

**Terminal 1 - Backend:**
```bash
python backend/api.py
```

**Terminal 2 - Frontend:**
```bash
streamlit run frontend/app.py
```

---

## Estrutura de Pastas

```
projeto/
│
├── backend/              # Backend Flask API
│   ├── api.py           # Servidor Flask
│   ├── data_processor.py # Processamento de dados
│   ├── ml_models.py     # Modelos ML
│   └── requirements.txt
│
├── frontend/            # Frontend Streamlit
│   ├── app.py          # Interface Streamlit
│   ├── visualizations.py # Visualizações
│   └── requirements.txt
│
└── start.bat           # Script de inicialização
```

---

## Fluxo de Trabalho

1. **Iniciar Backend** (SEMPRE PRIMEIRO!)
   - Roda em `http://localhost:5000`
   - Processa dados e treina modelos

2. **Iniciar Frontend**
   - Roda em `http://localhost:8501`
   - Interface do usuário

3. **Usar a aplicação:**
   - Upload CSV
   - Visualizar dados
   - Treinar modelos
   - Fazer predições

---

## Comunicação Entre Frontend e Backend

```
[Streamlit Frontend] <--HTTP--> [Flask Backend]
        |                              |
    Interface                    Processamento
    Visualizações               Machine Learning
                                Predições
```

### Endpoints Principais:

- `POST /api/upload` - Upload de CSV
- `POST /api/train` - Treinar modelos
- `POST /api/predict` - Fazer predições
- `GET /api/data/overview` - Visão geral dos dados

---

## Verificação de Instalação

Execute este comando para verificar se todas as dependências estão instaladas:

```bash
pip list | findstr "Flask streamlit pandas scikit-learn"
```

Se algo estiver faltando:

```bash
pip install -r requirements.txt
```

---

## Testando a API

Você pode testar a API diretamente:

```bash
curl http://localhost:5000/api/health
```

Resposta esperada:
```json
{
  "status": "ok",
  "message": "Flask API is running"
}
```

---

## Dicas Importantes

1. **SEMPRE inicie o backend antes do frontend**
2. **Não feche o terminal do backend** enquanto estiver usando o frontend
3. **Use Ctrl+C** para parar os servidores
4. **Porta 5000** deve estar livre para o Flask
5. **Porta 8501** deve estar livre para o Streamlit

---

## Troubleshooting Rápido

### Problema: "Connection Error" no Streamlit

**Solução:** Verifique se o backend está rodando:
```bash
curl http://localhost:5000/api/health
```

### Problema: "Port already in use"

**Solução:** Mate o processo na porta:
```bash
# Windows
netstat -ano | findstr :5000
taskkill /PID <numero_do_pid> /F
```

### Problema: "Module not found"

**Solução:** Reinstale as dependências:
```bash
pip install -r requirements.txt
```

---

## Primeiro Uso - Passo a Passo

1. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

2. Inicie o backend:
   ```bash
   python backend/api.py
   ```
   Aguarde a mensagem: "Running on http://0.0.0.0:5000"

3. Inicie o frontend (em outro terminal):
   ```bash
   streamlit run frontend/app.py
   ```
   O navegador abrirá automaticamente

4. Faça upload de um arquivo CSV

5. Explore as funcionalidades!

---

## Arquivos de Teste

Você pode criar um CSV de teste simples:

```csv
idade,salario,anos_experiencia,cargo
25,3000,2,Junior
30,5000,5,Pleno
35,8000,10,Senior
40,12000,15,Senior
28,4000,3,Pleno
```

Salve como `dados_teste.csv` e faça upload na aplicação.

---

## URLs Importantes

- **Frontend Streamlit:** http://localhost:8501
- **Backend Flask:** http://localhost:5000
- **API Health Check:** http://localhost:5000/api/health

---

## Comandos Úteis

### Verificar se as portas estão em uso:
```bash
# Windows
netstat -ano | findstr :5000
netstat -ano | findstr :8501
```

### Limpar cache do Streamlit:
```bash
streamlit cache clear
```

### Ver logs do Flask:
Os logs aparecem no terminal onde você iniciou `backend/api.py`

---

## Arquitetura Simplificada

```
Usuario
  |
  v
[Streamlit Frontend] -----> [Flask Backend] -----> [Machine Learning]
     |                           |                        |
  Interface                  Endpoints               Scikit-learn
  Visualizações             REST API                  Pandas
                                                     NumPy
```

---

Pronto! Agora você está pronto para usar a aplicação.

Para mais informações detalhadas, consulte o arquivo README.md
