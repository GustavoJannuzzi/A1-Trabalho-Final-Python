"""
Módulo de Processamento de Dados
Responsável por limpar, transformar e preparar dados para análise e modelagem
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

class DataProcessor:
    """
    Classe para processar e preparar dados para análise e machine learning
    """
    
    def __init__(self, dataframe):
        """
        Inicializa o processador de dados
        """
        self.df = dataframe.copy()
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def remove_missing_values(self, threshold=0.5):
        """
        Remove colunas e linhas com muitos valores nulos
        """
        # Remove colunas com mais de threshold% de valores nulos
        null_columns = self.df.columns[self.df.isnull().mean() > threshold]
        self.df = self.df.drop(columns=null_columns)
        
        # Remove linhas com valores nulos
        self.df = self.df.dropna()
        
        return self.df
    
    def encode_categorical_features(self, columns):
        """
        Codifica variáveis categóricas usando Label Encoding
        """
        for col in columns:
            if col in self.df.columns:
                # Verifica se a coluna é categórica ou do tipo objeto
                if self.df[col].dtype == 'object' or pd.api.types.is_categorical_dtype(self.df[col]):
                    le = LabelEncoder()
                    self.df[col] = le.fit_transform(self.df[col].astype(str))
                    self.label_encoders[col] = le
        
        return self.df
    
    def handle_outliers(self, columns, method='iqr', threshold=1.5):
        """
        Trata outliers usando o método IQR ou Z-score
        """
        for col in columns:
            if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):
                if method == 'iqr':
                    Q1 = self.df[col].quantile(0.25)
                    Q3 = self.df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR
                    
                    # Substituir outliers pelos limites
                    self.df[col] = self.df[col].clip(lower_bound, upper_bound)
                    
                elif method == 'zscore':
                    mean = self.df[col].mean()
                    std = self.df[col].std()
                    
                    z_scores = np.abs((self.df[col] - mean) / std)
                    self.df = self.df[z_scores < threshold]
        
        return self.df
    
    def normalize_features(self, columns):
        """
        Normaliza features numéricas usando StandardScaler
        """
        numeric_cols = [col for col in columns if col in self.df.columns 
                       and pd.api.types.is_numeric_dtype(self.df[col])]
        
        if numeric_cols:
            self.df[numeric_cols] = self.scaler.fit_transform(self.df[numeric_cols])
        
        return self.df
    
    def prepare_data(self, feature_columns, target_column, handle_missing=True):
        """
        Prepara os dados completos para machine learning
        """
        try:
            # Criar cópia do dataframe
            df_processed = self.df.copy()
            
            # Verificar se as colunas existem
            missing_cols = set(feature_columns + [target_column]) - set(df_processed.columns)
            if missing_cols:
                raise ValueError(f"Colunas não encontradas: {missing_cols}")
            
            # Selecionar apenas as colunas necessárias
            all_cols = feature_columns + [target_column]
            df_processed = df_processed[all_cols].copy()
            
            # Remover valores nulos se solicitado
            if handle_missing:
                df_processed = df_processed.dropna()
            
            # Verificar se ainda temos dados
            if len(df_processed) == 0:
                raise ValueError("Não há dados suficientes após remover valores nulos")
            
            # Codificar variáveis categóricas nas features
            categorical_features = [col for col in feature_columns 
                                   if df_processed[col].dtype == 'object']
            
            for col in categorical_features:
                le = LabelEncoder()
                # Garantir que não há valores nulos antes de codificar
                df_processed[col] = df_processed[col].fillna('Unknown')
                df_processed[col] = le.fit_transform(df_processed[col].astype(str))
                self.label_encoders[col] = le
            
            # Codificar target se for categórico
            if df_processed[target_column].dtype == 'object':
                le = LabelEncoder()
                df_processed[target_column] = df_processed[target_column].fillna('Unknown')
                df_processed[target_column] = le.fit_transform(df_processed[target_column].astype(str))
                self.label_encoders[target_column] = le
            
            # Separar features e target
            X = df_processed[feature_columns].copy()
            y = df_processed[target_column].copy()
            
            # Verificar se há valores infinitos ou NaN
            X = X.replace([np.inf, -np.inf], np.nan)
            
            # Remover linhas com NaN (se houver)
            if X.isnull().any().any():
                mask = ~X.isnull().any(axis=1)
                X = X[mask]
                y = y[mask]
            
            # Resetar índices
            X = X.reset_index(drop=True)
            y = y.reset_index(drop=True)
            
            print(f"Dados finais: X shape={X.shape}, y shape={y.shape}")
            
            # Verificação final
            if len(X) == 0 or len(y) == 0:
                raise ValueError("Dados vazios após processamento completo")
            
            return X, y
            
        except Exception as e:
            print(f"Erro ao preparar dados: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def get_data_summary(self):
        """
        Retorna um resumo estatístico dos dados
        """
        summary = {
            'n_rows': len(self.df),
            'n_columns': len(self.df.columns),
            'missing_values': self.df.isnull().sum().sum(),
            'missing_percentage': (self.df.isnull().sum().sum() / (len(self.df) * len(self.df.columns))) * 100,
            'numeric_columns': len(self.df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': len(self.df.select_dtypes(include=['object']).columns),
            'memory_usage': self.df.memory_usage(deep=True).sum() / 1024,
            'duplicated_rows': self.df.duplicated().sum()
        }
        
        return summary
    
    def detect_and_handle_duplicates(self):
        """
        Detecta e remove linhas duplicadas
        """
        n_duplicates = self.df.duplicated().sum()
        self.df = self.df.drop_duplicates()
        
        return n_duplicates
    
    def fill_missing_values(self, strategy='mean', columns=None):
        """
        Preenche valores nulos usando diferentes estratégias
        """
        if columns is None:
            columns = self.df.columns
        
        for col in columns:
            if col in self.df.columns:
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    if strategy == 'mean':
                        self.df[col].fillna(self.df[col].mean(), inplace=True)
                    elif strategy == 'median':
                        self.df[col].fillna(self.df[col].median(), inplace=True)
                    elif strategy == 'constant':
                        self.df[col].fillna(0, inplace=True)
                else:
                    # Para colunas categóricas, usar moda
                    self.df[col].fillna(self.df[col].mode()[0] if not self.df[col].mode().empty else 'Unknown', 
                                       inplace=True)
        
        return self.df
    
    def get_column_types(self):
        """
        Identifica tipos de colunas no dataset
        """
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = self.df.select_dtypes(include=['datetime64']).columns.tolist()
        
        return {
            'numeric': numeric_cols,
            'categorical': categorical_cols,
            'datetime': datetime_cols,
            'all': self.df.columns.tolist()
        }