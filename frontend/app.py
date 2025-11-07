"""
Aplicação Web de Análise de Dados e Machine Learning
Frontend Streamlit - Avaliação Final Python
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from visualizations import DataVisualizer

API_URL = "http://localhost:5000"

# Configuração da página
st.set_page_config(
    page_title="Análise de Dados & ML",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)


def check_api_connection():
    """Verifica se a API Flask está rodando"""
    try:
        response = requests.get(f"{API_URL}/api/health", timeout=2)
        return response.status_code == 200
    except:
        return False


def initialize_session_state():
    """Inicializa variáveis de sessão do Streamlit"""
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'data_uploaded' not in st.session_state:
        st.session_state.data_uploaded = False
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    if 'models_results' not in st.session_state:
        st.session_state.models_results = None
    if 'processed_data_info' not in st.session_state:
        st.session_state.processed_data_info = None


def main():
    """Função principal da aplicação"""
    initialize_session_state()

    # Cabeçalho principal
    st.markdown('<h1 class="main-header">Análise de Dados & Machine Learning</h1>',
                unsafe_allow_html=True)
    st.markdown("---")

    if not check_api_connection():
        st.error("ERRO: Não foi possível conectar à API Flask. Certifique-se de que o backend está rodando em http://localhost:5000")
        st.info("Para iniciar o backend, execute: python backend/api.py")
        return

    with st.sidebar:
        st.title("Configurações")

        # Upload do arquivo
        st.markdown("### Upload de Dados")
        uploaded_file = st.file_uploader(
            "Carregue seu arquivo CSV",
            type=['csv'],
            help="Selecione um arquivo CSV com dados estruturados"
        )

        if uploaded_file is not None:
            try:
                # Enviar arquivo para API
                files = {'file': uploaded_file}
                response = requests.post(f"{API_URL}/api/upload", files=files)

                if response.status_code == 200:
                    data_info = response.json()

                    # Carregar dados localmente para visualizações
                    uploaded_file.seek(0)
                    st.session_state.data = pd.read_csv(uploaded_file)
                    st.session_state.data_uploaded = True

                    st.success("Arquivo carregado com sucesso!")
                    st.info(f"{data_info['rows']} linhas e {data_info['columns']} colunas")
                else:
                    error_data = response.json()
                    st.error(f"Erro ao carregar arquivo: {error_data.get('error', 'Erro desconhecido')}")
                    return

            except Exception as e:
                st.error(f"Erro ao carregar arquivo: {str(e)}")
                return

        st.markdown("---")

        # Informações da aplicação
        with st.expander("Sobre a Aplicação"):
            st.markdown("""
            **Funcionalidades:**
            - Upload de arquivos CSV
            - Análise exploratória de dados
            - Visualizações interativas
            - Machine Learning com múltiplos algoritmos
            - Predições personalizadas

            **Arquitetura:**
            - Frontend: Streamlit
            - Backend: Flask API
            - Processamento: Python/Scikit-learn
            """)

    if not st.session_state.data_uploaded:
        st.info("Por favor, faça upload de um arquivo CSV para começar a análise")

        st.markdown("### Formato Esperado dos Dados")
        st.markdown("""
        Seu arquivo CSV deve conter:
        - **Colunas nomeadas** na primeira linha
        - **Dados estruturados** em formato tabular
        - **Valores numéricos** para atributos preditivos
        - **Coluna alvo** (target) para predição

        **Exemplos de datasets compatíveis:**
        - Preços de imóveis (área, quartos, banheiros → preço)
        - Dados de vendas (marketing, região, sazonalidade → vendas)
        - Dados de saúde (idade, peso, pressão → diagnóstico)
        """)

    else:
        tab1, tab2, tab3, tab4 = st.tabs([
            "Visão Geral dos Dados",
            "Análise Visual",
            "Machine Learning",
            "Fazer Predições"
        ])

        with tab1:
            show_data_overview()

        with tab2:
            show_data_visualization()

        with tab3:
            show_machine_learning()

        with tab4:
            show_predictions()


def show_data_overview():
    """Exibe visão geral dos dados carregados"""
    st.markdown('<h2 class="sub-header">Visão Geral dos Dados</h2>', unsafe_allow_html=True)

    try:
        response = requests.get(f"{API_URL}/api/data/overview")

        if response.status_code != 200:
            st.error("Erro ao buscar dados da API")
            return

        data = response.json()
        df = st.session_state.data

        # Métricas principais
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Linhas", data['shape']['rows'])
        with col2:
            st.metric("Colunas", data['shape']['columns'])
        with col3:
            st.metric("Valores Nulos", df.isnull().sum().sum())
        with col4:
            st.metric("Memória", f"{df.memory_usage(deep=True).sum() / 1024:.2f} KB")

        st.markdown("---")

        # Mostrar amostra dos dados
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("### Amostra dos Dados")
        with col2:
            n_rows = st.slider("Número de linhas", 5, 50, 10)

        st.dataframe(df.head(n_rows), use_container_width=True)

        st.markdown("### Informações das Colunas")
        col_info = pd.DataFrame({
            'Tipo': df.dtypes,
            'Não-Nulos': df.notnull().sum(),
            'Nulos': df.isnull().sum(),
            '% Nulos': (df.isnull().sum() / len(df) * 100).round(2),
            'Únicos': df.nunique()
        })
        st.dataframe(col_info, use_container_width=True)

        st.markdown("### Estatísticas Descritivas")
        st.dataframe(df.describe(), use_container_width=True)

    except Exception as e:
        st.error(f"Erro ao carregar visão geral: {str(e)}")


def show_data_visualization():
    """Exibe visualizações dos dados"""
    st.markdown('<h2 class="sub-header">Análise Visual</h2>', unsafe_allow_html=True)

    df = st.session_state.data
    visualizer = DataVisualizer(df)

    # Seleção de colunas para visualização
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    if len(numeric_cols) == 0:
        st.warning("Não há colunas numéricas para visualizar")
        return

    # Tipo de gráfico
    viz_type = st.selectbox(
        "Selecione o tipo de visualização",
        ["Histogramas", "Box Plots", "Scatter Plot", "Matriz de Correlação",
         "Distribuição de Categorias", "Heatmap de Valores Nulos"]
    )

    if viz_type == "Histogramas":
        st.markdown("### Distribuição das Variáveis Numéricas")
        selected_cols = st.multiselect("Selecione as colunas", numeric_cols, default=numeric_cols[:3])
        if selected_cols:
            fig = visualizer.plot_histograms(selected_cols)
            st.pyplot(fig)

    elif viz_type == "Box Plots":
        st.markdown("### Box Plots - Detecção de Outliers")
        selected_cols = st.multiselect("Selecione as colunas", numeric_cols, default=numeric_cols[:3])
        if selected_cols:
            fig = visualizer.plot_boxplots(selected_cols)
            st.pyplot(fig)

    elif viz_type == "Scatter Plot":
        st.markdown("### Gráfico de Dispersão")
        col1, col2 = st.columns(2)
        with col1:
            x_col = st.selectbox("Eixo X", numeric_cols)
        with col2:
            y_col = st.selectbox("Eixo Y", numeric_cols, index=min(1, len(numeric_cols)-1))

        if categorical_cols:
            hue_col = st.selectbox("Colorir por (opcional)", [None] + categorical_cols)
        else:
            hue_col = None

        fig = visualizer.plot_scatter(x_col, y_col, hue_col)
        st.pyplot(fig)

    elif viz_type == "Matriz de Correlação":
        st.markdown("### Matriz de Correlação")
        fig = visualizer.plot_correlation_matrix(numeric_cols)
        st.pyplot(fig)

    elif viz_type == "Distribuição de Categorias":
        if categorical_cols:
            st.markdown("### Distribuição de Variáveis Categóricas")
            selected_col = st.selectbox("Selecione a coluna categórica", categorical_cols)
            fig = visualizer.plot_categorical_distribution(selected_col)
            st.pyplot(fig)
        else:
            st.warning("Não há colunas categóricas no dataset")

    elif viz_type == "Heatmap de Valores Nulos":
        st.markdown("### Visualização de Valores Nulos")
        fig = visualizer.plot_missing_values()
        st.pyplot(fig)


def show_machine_learning():
    """Exibe interface de Machine Learning"""
    st.markdown('<h2 class="sub-header">Machine Learning</h2>', unsafe_allow_html=True)

    df = st.session_state.data

    # Seleção de features e target
    st.markdown("### Configuração do Modelo")

    all_columns = df.columns.tolist()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Atributos Preditivos (Features)")
        features = st.multiselect(
            "Selecione as features para o modelo",
            all_columns,
            default=numeric_cols[:min(3, len(numeric_cols))],
            help="Selecione as variáveis que serão usadas para fazer predições"
        )

    with col2:
        st.markdown("#### Variável Alvo (Target)")
        target = st.selectbox(
            "Selecione a variável que deseja prever",
            all_columns,
            help="Esta é a variável que o modelo aprenderá a prever"
        )

    if not features or not target:
        st.warning("Por favor, selecione as features e o target")
        return

    if target in features:
        st.error("A variável alvo não pode estar entre os atributos preditivos!")
        return

    # Verificar se é regressão ou classificação
    is_regression = pd.api.types.is_numeric_dtype(df[target])

    if is_regression:
        task_type = "Regressão"
    else:
        task_type = "Classificação"

    st.info(f"Tipo de problema detectado: **{task_type}**")

    st.markdown("---")

    # Seleção de algoritmos
    st.markdown("### Seleção de Algoritmos")

    if is_regression:
        available_models = {
            "Linear Regression": "Regressão Linear - Modelo simples e interpretável",
            "Random Forest": "Random Forest - Ensemble de árvores de decisão",
            "Gradient Boosting": "Gradient Boosting - Algoritmo de boosting poderoso",
            "Support Vector Machine": "SVM - Máquina de Vetores de Suporte",
            "K-Nearest Neighbors": "KNN - K Vizinhos Mais Próximos"
        }
    else:
        available_models = {
            "Logistic Regression": "Regressão Logística - Classificador linear",
            "Random Forest": "Random Forest - Ensemble de árvores de decisão",
            "Gradient Boosting": "Gradient Boosting - Algoritmo de boosting poderoso",
            "Support Vector Machine": "SVM - Máquina de Vetores de Suporte",
            "K-Nearest Neighbors": "KNN - K Vizinhos Mais Próximos"
        }

    selected_models = st.multiselect(
        "Selecione os algoritmos que deseja treinar",
        list(available_models.keys()),
        default=list(available_models.keys())[:3],
        format_func=lambda x: f"{x} - {available_models[x]}"
    )

    if not selected_models:
        st.warning("Selecione pelo menos um algoritmo")
        return

    st.markdown("### Configurações de Treinamento")

    col1, col2, col3 = st.columns(3)

    with col1:
        test_size = st.slider(
            "Tamanho do conjunto de teste (%)",
            10, 50, 20,
            help="Porcentagem dos dados que será usada para testar o modelo"
        )

    with col2:
        random_state = st.number_input(
            "Semente aleatória",
            0, 1000, 42,
            help="Para reprodutibilidade dos resultados"
        )

    with col3:
        handle_missing = st.checkbox(
            "Tratar valores nulos",
            value=True,
            help="Remove linhas com valores nulos automaticamente"
        )

    st.markdown("---")

    if st.button("Executar Análise e Treinar Modelos", type="primary", use_container_width=True):
        with st.spinner("Processando dados e treinando modelos..."):
            try:
                payload = {
                    'features': features,
                    'target': target,
                    'models': selected_models,
                    'test_size': test_size / 100,
                    'random_state': int(random_state),
                    'handle_missing': handle_missing
                }

                response = requests.post(f"{API_URL}/api/train", json=payload)

                if response.status_code == 200:
                    result = response.json()

                    st.session_state.models_results = result['results']
                    st.session_state.model_trained = True
                    st.session_state.processed_data_info = {
                        'features': features,
                        'target': target,
                        'is_regression': result['task_type'] == 'regression'
                    }

                    st.success("Modelos treinados com sucesso!")
                    st.info(f"Amostras treinadas: {result['samples_trained']} | Features: {result['features_count']}")

                else:
                    error_data = response.json()
                    st.error(f"Erro durante o treinamento: {error_data.get('error', 'Erro desconhecido')}")
                    return

            except Exception as e:
                st.error(f"Erro ao treinar modelos: {str(e)}")
                return

    # Mostrar resultados se já treinados
    if st.session_state.model_trained and st.session_state.models_results:
        st.markdown("---")
        st.markdown("### Resultados dos Modelos")

        results_df = pd.DataFrame(st.session_state.models_results).T

        # Ordenar por métrica principal
        if st.session_state.processed_data_info['is_regression']:
            results_df = results_df.sort_values('R² Score', ascending=False)
            metric = 'R² Score'
        else:
            results_df = results_df.sort_values('Acurácia', ascending=False)
            metric = 'Acurácia'

        st.dataframe(results_df.style.highlight_max(axis=0), use_container_width=True)

        # Gráfico comparativo
        st.markdown("### Comparação Visual dos Modelos")
        fig = DataVisualizer.plot_model_comparison(results_df, metric)
        st.pyplot(fig)

        # Melhor modelo
        best_model = results_df.index[0]
        st.success(f"Melhor modelo: **{best_model}** com {metric} = {results_df.loc[best_model, metric]:.4f}")


def show_predictions():
    """Interface para fazer predições com o modelo treinado"""
    st.markdown('<h2 class="sub-header">Fazer Predições</h2>', unsafe_allow_html=True)

    if not st.session_state.model_trained:
        st.warning("Por favor, treine os modelos primeiro na aba 'Machine Learning'")
        return

    info = st.session_state.processed_data_info
    features = info['features']
    target = info['target']
    is_regression = info['is_regression']

    st.markdown("### Insira os Valores para Predição")

    # Criar inputs para cada feature
    input_data = {}

    # Organizar inputs em colunas
    n_cols = min(3, len(features))
    cols = st.columns(n_cols)

    for idx, feature in enumerate(features):
        col_idx = idx % n_cols
        with cols[col_idx]:
            # Verificar tipo de dados
            if pd.api.types.is_numeric_dtype(st.session_state.data[feature]):
                min_val = float(st.session_state.data[feature].min())
                max_val = float(st.session_state.data[feature].max())
                mean_val = float(st.session_state.data[feature].mean())

                input_data[feature] = st.number_input(
                    f"{feature}",
                    min_value=min_val,
                    max_value=max_val,
                    value=mean_val,
                    help=f"Valor entre {min_val:.2f} e {max_val:.2f}"
                )
            else:
                unique_vals = st.session_state.data[feature].unique().tolist()
                input_data[feature] = st.selectbox(
                    f"{feature}",
                    unique_vals
                )

    st.markdown("---")

    # Selecionar modelo para predição
    available_models = list(st.session_state.models_results.keys())
    selected_model = st.selectbox(
        "Selecione o modelo para fazer a predição",
        available_models,
        help="Escolha qual modelo treinado usar para a predição"
    )

    # Fazer predição
    if st.button("Fazer Predição", type="primary", use_container_width=True):
        try:
            payload = {
                'model_name': selected_model,
                'input_data': input_data
            }

            response = requests.post(f"{API_URL}/api/predict", json=payload)

            if response.status_code == 200:
                result = response.json()
                prediction = result['prediction']
                model_metrics = result['model_metrics']

                st.markdown("---")
                st.markdown("### Resultado da Predição")

                col1, col2 = st.columns([2, 1])

                with col1:
                    st.markdown("#### Valores de Entrada:")
                    for feature, value in input_data.items():
                        st.write(f"**{feature}:** {value}")

                with col2:
                    st.markdown(f"#### Predição:")
                    st.markdown(f"### **{target}**")
                    if is_regression:
                        st.metric("Valor Predito", f"{prediction:.2f}")
                    else:
                        st.metric("Classe Predita", prediction)

                st.markdown("---")
                st.markdown(f"### Métricas do Modelo **{selected_model}**")

                cols = st.columns(len(model_metrics))
                for idx, (metric, value) in enumerate(model_metrics.items()):
                    with cols[idx]:
                        st.metric(metric, f"{value:.4f}")
            else:
                error_data = response.json()
                st.error(f"Erro ao fazer predição: {error_data.get('error', 'Erro desconhecido')}")

        except Exception as e:
            st.error(f"Erro ao fazer predição: {str(e)}")


if __name__ == "__main__":
    main()
