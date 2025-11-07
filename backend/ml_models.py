"""
Módulo de Machine Learning
Responsável por treinar, avaliar e fazer predições com modelos de ML
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

# Modelos de Regressão
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

# Modelos de Classificação
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

class MLModelTrainer:
    """
    Classe para treinar e avaliar modelos de Machine Learning
    """
    
    def __init__(self, is_regression=True):
        """
        Inicializa o treinador de modelos
        """
        self.is_regression = is_regression
        self.models = {}
        self.trained_models = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # Inicializar modelos disponíveis
        self._initialize_models()
    
    def _initialize_models(self):
        """
        Inicializa os modelos disponíveis
        """
        if self.is_regression:
            self.models = {
                'Linear Regression': LinearRegression(),
                'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
                'Support Vector Machine': SVR(kernel='rbf'),
                'K-Nearest Neighbors': KNeighborsRegressor(n_neighbors=5)
            }
        else:
            self.models = {
                'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
                'Support Vector Machine': SVC(kernel='rbf', random_state=42),
                'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5)
            }
    
    def train_models(self, X, y, selected_models=None, test_size=0.2, random_state=42):
        """
        Treina múltiplos modelos e avalia seu desempenho
        """
        # Validar tamanho dos dados
        if len(X) == 0 or len(y) == 0:
            raise ValueError("Dados vazios! Não é possível treinar modelos.")
        
        # Calcular tamanho mínimo necessário
        min_samples = max(5, int(1 / test_size))
        if len(X) < min_samples:
            raise ValueError(f"Dataset muito pequeno! São necessárias pelo menos {min_samples} amostras, mas há apenas {len(X)}.")
        
        # Ajustar test_size se necessário
        min_test_samples = 2
        if int(len(X) * test_size) < min_test_samples:
            test_size = min_test_samples / len(X)
            print(f"Ajustando test_size para {test_size:.2f} para garantir pelo menos {min_test_samples} amostras de teste")
        
        # Dividir dados em treino e teste
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Se não especificados, usar todos os modelos
        if selected_models is None:
            selected_models = list(self.models.keys())
        
        results = {}
        
        # Treinar cada modelo selecionado
        for model_name in selected_models:
            if model_name in self.models:
                print(f"Treinando {model_name}...")
                
                # Obter modelo
                model = self.models[model_name]
                
                # Treinar
                model.fit(self.X_train, self.y_train)
                
                # Salvar modelo treinado
                self.trained_models[model_name] = model
                
                # Fazer predições
                y_pred = model.predict(self.X_test)
                
                # Avaliar
                if self.is_regression:
                    results[model_name] = self._evaluate_regression(y_pred)
                else:
                    results[model_name] = self._evaluate_classification(y_pred)
        
        return results
    
    def _evaluate_regression(self, y_pred):
        """
        Avalia modelo de regressão
        """
        mse = mean_squared_error(self.y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        
        # Calcular MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((self.y_test - y_pred) / self.y_test)) * 100
        
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R² Score': r2,
            'MAPE (%)': mape
        }
    
    def _evaluate_classification(self, y_pred):
        """
        Avalia modelo de classificação
        """
        accuracy = accuracy_score(self.y_test, y_pred)
        
        # Para classificação binária ou multiclasse
        avg_method = 'binary' if len(np.unique(self.y_test)) == 2 else 'weighted'
        
        precision = precision_score(self.y_test, y_pred, average=avg_method, zero_division=0)
        recall = recall_score(self.y_test, y_pred, average=avg_method, zero_division=0)
        f1 = f1_score(self.y_test, y_pred, average=avg_method, zero_division=0)
        
        return {
            'Acurácia': accuracy,
            'Precisão': precision,
            'Recall': recall,
            'F1-Score': f1
        }
    
    def predict(self, model_name, X_new):
        """
        Faz predições com um modelo treinado
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Modelo {model_name} não foi treinado")
        
        model = self.trained_models[model_name]
        
        # Garantir que X_new tenha as mesmas colunas que X_train
        if isinstance(X_new, pd.DataFrame):
            # Reordenar colunas para corresponder ao treinamento
            X_new = X_new[self.X_train.columns]
        
        return model.predict(X_new)
    
    def cross_validate(self, model_name, X, y, cv=5):
        """
        Realiza validação cruzada
        """
        if model_name not in self.models:
            raise ValueError(f"Modelo {model_name} não encontrado")
        
        model = self.models[model_name]
        
        # Escolher métrica apropriada
        if self.is_regression:
            scoring = 'r2'
        else:
            scoring = 'accuracy'
        
        # Realizar validação cruzada
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        
        return {
            'scores': scores,
            'mean': scores.mean(),
            'std': scores.std(),
            'min': scores.min(),
            'max': scores.max()
        }
    
    def get_feature_importance(self, model_name):
        """
        Obtém importância das features (para modelos baseados em árvore)
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Modelo {model_name} não foi treinado")
        
        model = self.trained_models[model_name]
        
        # Verificar se o modelo tem feature_importances_
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.X_train.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return importance_df
        else:
            return None
    
    def get_model_parameters(self, model_name):
        """
        Obtém parâmetros de um modelo treinado
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Modelo {model_name} não foi treinado")
        
        model = self.trained_models[model_name]
        return model.get_params()
    
    def get_confusion_matrix(self, model_name):
        """
        Gera matriz de confusão para modelos de classificação
        """
        if self.is_regression:
            raise ValueError("Matriz de confusão só está disponível para classificação")
        
        if model_name not in self.trained_models:
            raise ValueError(f"Modelo {model_name} não foi treinado")
        
        model = self.trained_models[model_name]
        y_pred = model.predict(self.X_test)
        
        return confusion_matrix(self.y_test, y_pred)
    
    def get_classification_report(self, model_name):
        """
        Gera relatório de classificação detalhado
        """
        if self.is_regression:
            raise ValueError("Relatório de classificação só está disponível para classificação")
        
        if model_name not in self.trained_models:
            raise ValueError(f"Modelo {model_name} não foi treinado")
        
        model = self.trained_models[model_name]
        y_pred = model.predict(self.X_test)
        
        return classification_report(self.y_test, y_pred)
    
    def retrain_model(self, model_name, X_new, y_new):
        """
        Re-treina um modelo com novos dados
        """
        if model_name not in self.models:
            raise ValueError(f"Modelo {model_name} não encontrado")
        
        # Re-inicializar modelo
        self._initialize_models()
        model = self.models[model_name]
        
        # Dividir novos dados
        X_train, X_test, y_train, y_test = train_test_split(
            X_new, y_new, test_size=0.2, random_state=42
        )
        
        # Treinar
        model.fit(X_train, y_train)
        
        # Salvar
        self.trained_models[model_name] = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        # Avaliar
        y_pred = model.predict(X_test)
        
        if self.is_regression:
            return self._evaluate_regression(y_pred)
        else:
            return self._evaluate_classification(y_pred)
    
    def save_model(self, model_name, filepath):
        """
        Salva um modelo treinado em disco
        """
        import joblib
        
        if model_name not in self.trained_models:
            raise ValueError(f"Modelo {model_name} não foi treinado")
        
        model = self.trained_models[model_name]
        joblib.dump(model, filepath)
    
    def load_model(self, model_name, filepath):
        """
        Carrega um modelo salvo
        """
        import joblib
        
        model = joblib.load(filepath)
        self.trained_models[model_name] = model