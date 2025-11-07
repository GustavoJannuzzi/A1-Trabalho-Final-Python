"""
API Flask para Processamento de Dados e Machine Learning
Backend responsável por todo processamento pesado
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import io
import base64
import traceback
from data_processor import DataProcessor
from ml_models import MLModelTrainer

app = Flask(__name__)
CORS(app)

# Armazenamento temporário em memória
storage = {
    'data': None,
    'processed_data': None,
    'models_results': None,
    'trainer': None,
    'processor': None
}

@app.route('/api/health', methods=['GET'])
def health_check():
    """Verifica se a API está funcionando"""
    return jsonify({
        'status': 'ok',
        'message': 'Flask API is running'
    }), 200


@app.route('/api/upload', methods=['POST'])
def upload_data():
    """
    Endpoint para upload de arquivo CSV
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'Nenhum arquivo enviado'}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'Nome de arquivo vazio'}), 400

        if not file.filename.endswith('.csv'):
            return jsonify({'error': 'Apenas arquivos CSV são suportados'}), 400

        # Ler CSV
        df = pd.read_csv(file)

        # Armazenar em memória
        storage['data'] = df
        storage['processor'] = DataProcessor(df)

        # Retornar informações básicas
        info = {
            'success': True,
            'rows': int(df.shape[0]),
            'columns': int(df.shape[1]),
            'column_names': df.columns.tolist(),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'null_values': int(df.isnull().sum().sum()),
            'memory_kb': float(df.memory_usage(deep=True).sum() / 1024)
        }

        return jsonify(info), 200

    except Exception as e:
        return jsonify({
            'error': f'Erro ao processar arquivo: {str(e)}',
            'traceback': traceback.format_exc()
        }), 500


@app.route('/api/data/overview', methods=['GET'])
def get_data_overview():
    """
    Retorna visão geral dos dados carregados
    """
    try:
        if storage['data'] is None:
            return jsonify({'error': 'Nenhum dado carregado'}), 400

        df = storage['data']

        # Estatísticas descritivas
        desc_stats = df.describe().to_dict()

        # Informações das colunas
        col_info = {
            'types': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'non_null': df.notnull().sum().to_dict(),
            'null_count': df.isnull().sum().to_dict(),
            'null_percentage': (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
            'unique_values': df.nunique().to_dict()
        }

        # Primeiras linhas
        sample_data = df.head(50).to_dict(orient='records')

        return jsonify({
            'success': True,
            'statistics': desc_stats,
            'column_info': col_info,
            'sample_data': sample_data,
            'shape': {'rows': int(df.shape[0]), 'columns': int(df.shape[1])}
        }), 200

    except Exception as e:
        return jsonify({
            'error': f'Erro ao gerar visão geral: {str(e)}',
            'traceback': traceback.format_exc()
        }), 500


@app.route('/api/data/columns', methods=['GET'])
def get_columns():
    """
    Retorna informações sobre as colunas
    """
    try:
        if storage['data'] is None:
            return jsonify({'error': 'Nenhum dado carregado'}), 400

        df = storage['data']

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

        return jsonify({
            'success': True,
            'all_columns': df.columns.tolist(),
            'numeric_columns': numeric_cols,
            'categorical_columns': categorical_cols
        }), 200

    except Exception as e:
        return jsonify({
            'error': f'Erro ao obter colunas: {str(e)}'
        }), 500


@app.route('/api/train', methods=['POST'])
def train_models():
    """
    Treina modelos de Machine Learning
    """
    try:
        if storage['data'] is None:
            return jsonify({'error': 'Nenhum dado carregado'}), 400

        data = request.get_json()

        features = data.get('features')
        target = data.get('target')
        selected_models = data.get('models')
        test_size = data.get('test_size', 0.2)
        random_state = data.get('random_state', 42)
        handle_missing = data.get('handle_missing', True)

        if not features or not target:
            return jsonify({'error': 'Features e target são obrigatórios'}), 400

        if target in features:
            return jsonify({'error': 'Target não pode estar nas features'}), 400

        # Processar dados
        processor = storage['processor']
        X, y = processor.prepare_data(features, target, handle_missing)

        if X is None or y is None or len(X) == 0:
            return jsonify({'error': 'Erro ao processar dados. Verifique valores nulos.'}), 400

        # Detectar tipo de problema
        is_regression = pd.api.types.is_numeric_dtype(storage['data'][target])

        # Treinar modelos
        trainer = MLModelTrainer(is_regression)
        results = trainer.train_models(
            X, y,
            selected_models=selected_models,
            test_size=test_size,
            random_state=random_state
        )

        # Armazenar resultados
        storage['models_results'] = results
        storage['trainer'] = trainer
        storage['processed_data'] = {
            'X': X,
            'y': y,
            'features': features,
            'target': target,
            'is_regression': is_regression
        }

        return jsonify({
            'success': True,
            'results': results,
            'task_type': 'regression' if is_regression else 'classification',
            'samples_trained': int(len(X)),
            'features_count': int(X.shape[1])
        }), 200

    except Exception as e:
        return jsonify({
            'error': f'Erro ao treinar modelos: {str(e)}',
            'traceback': traceback.format_exc()
        }), 500


@app.route('/api/predict', methods=['POST'])
def make_prediction():
    """
    Faz predição com modelo treinado
    """
    try:
        if storage['trainer'] is None:
            return jsonify({'error': 'Nenhum modelo treinado'}), 400

        data = request.get_json()

        model_name = data.get('model_name')
        input_data = data.get('input_data')

        if not model_name or not input_data:
            return jsonify({'error': 'model_name e input_data são obrigatórios'}), 400

        # Criar DataFrame com input
        input_df = pd.DataFrame([input_data])

        # Fazer predição
        trainer = storage['trainer']
        prediction = trainer.predict(model_name, input_df)

        # Obter métricas do modelo
        model_metrics = storage['models_results'].get(model_name, {})

        return jsonify({
            'success': True,
            'prediction': float(prediction[0]) if storage['processed_data']['is_regression'] else str(prediction[0]),
            'model_metrics': model_metrics,
            'input_data': input_data
        }), 200

    except Exception as e:
        return jsonify({
            'error': f'Erro ao fazer predição: {str(e)}',
            'traceback': traceback.format_exc()
        }), 500


@app.route('/api/models', methods=['GET'])
def get_models():
    """
    Retorna informações sobre modelos treinados
    """
    try:
        if storage['models_results'] is None:
            return jsonify({
                'success': True,
                'models_trained': False,
                'models': []
            }), 200

        return jsonify({
            'success': True,
            'models_trained': True,
            'models': list(storage['models_results'].keys()),
            'results': storage['models_results'],
            'processed_data_info': {
                'features': storage['processed_data']['features'],
                'target': storage['processed_data']['target'],
                'is_regression': storage['processed_data']['is_regression']
            }
        }), 200

    except Exception as e:
        return jsonify({
            'error': f'Erro ao obter modelos: {str(e)}'
        }), 500


@app.route('/api/reset', methods=['POST'])
def reset_storage():
    """
    Limpa todos os dados armazenados
    """
    try:
        storage['data'] = None
        storage['processed_data'] = None
        storage['models_results'] = None
        storage['trainer'] = None
        storage['processor'] = None

        return jsonify({
            'success': True,
            'message': 'Dados limpos com sucesso'
        }), 200

    except Exception as e:
        return jsonify({
            'error': f'Erro ao limpar dados: {str(e)}'
        }), 500


if __name__ == '__main__':
    print("Iniciando Flask API...")
    print("API disponível em: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
