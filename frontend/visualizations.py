"""
Módulo de Visualização de Dados
Responsável por criar gráficos e visualizações interativas
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

class DataVisualizer:
    """
    Classe para criar visualizações de dados
    """
    
    def __init__(self, dataframe):
        """
        Inicializa o visualizador
        
        Args:
            dataframe (pd.DataFrame): DataFrame com os dados
        """
        self.df = dataframe
        
        # Configurar estilo dos gráficos
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)
        plt.rcParams['font.size'] = 10
        
    def plot_histograms(self, columns, bins=30):
        """
        Cria histogramas para colunas numéricas
        """
        n_cols = len(columns)
        n_rows = (n_cols + 2) // 3
        
        fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
        
        for idx, col in enumerate(columns):
            if col in self.df.columns:
                ax = axes[idx]
                
                # Plotar histograma
                self.df[col].hist(bins=bins, ax=ax, color='skyblue', edgecolor='black', alpha=0.7)
                
                # Adicionar linha de média
                mean_val = self.df[col].mean()
                ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Média: {mean_val:.2f}')
                
                # Configurações do gráfico
                ax.set_title(f'Distribuição de {col}', fontsize=12, fontweight='bold')
                ax.set_xlabel(col)
                ax.set_ylabel('Frequência')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        # Remover eixos extras
        for idx in range(len(columns), len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        return fig
    
    def plot_boxplots(self, columns):
        """
        Cria box plots para detecção de outliers
        """
        n_cols = len(columns)
        n_rows = (n_cols + 2) // 3
        
        fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
        
        for idx, col in enumerate(columns):
            if col in self.df.columns:
                ax = axes[idx]
                
                # Criar box plot
                box_data = self.df[col].dropna()
                bp = ax.boxplot(box_data, vert=True, patch_artist=True)
                
                # Colorir o box plot
                for patch in bp['boxes']:
                    patch.set_facecolor('lightblue')
                
                # Configurações
                ax.set_title(f'Box Plot - {col}', fontsize=12, fontweight='bold')
                ax.set_ylabel(col)
                ax.grid(True, alpha=0.3)
                
                # Adicionar estatísticas
                q1 = box_data.quantile(0.25)
                q3 = box_data.quantile(0.75)
                median = box_data.median()
                
                ax.text(1.1, q1, f'Q1: {q1:.2f}', fontsize=8)
                ax.text(1.1, median, f'Mediana: {median:.2f}', fontsize=8, fontweight='bold')
                ax.text(1.1, q3, f'Q3: {q3:.2f}', fontsize=8)
        
        # Remover eixos extras
        for idx in range(len(columns), len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        return fig
    
    def plot_scatter(self, x_col, y_col, hue_col=None):
        """
        Cria gráfico de dispersão
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        if hue_col and hue_col in self.df.columns:
            # Scatter plot com cores
            for category in self.df[hue_col].unique():
                mask = self.df[hue_col] == category
                ax.scatter(
                    self.df.loc[mask, x_col],
                    self.df.loc[mask, y_col],
                    label=category,
                    alpha=0.6,
                    s=50
                )
            ax.legend(title=hue_col)
        else:
            # Scatter plot simples
            ax.scatter(
                self.df[x_col],
                self.df[y_col],
                alpha=0.6,
                s=50,
                color='steelblue'
            )
        
        # Linha de tendência
        z = np.polyfit(self.df[x_col].dropna(), self.df[y_col].dropna(), 1)
        p = np.poly1d(z)
        ax.plot(
            self.df[x_col].sort_values(),
            p(self.df[x_col].sort_values()),
            "r--",
            alpha=0.8,
            linewidth=2,
            label='Tendência'
        )
        
        # Configurações
        ax.set_xlabel(x_col, fontsize=12)
        ax.set_ylabel(y_col, fontsize=12)
        ax.set_title(f'Relação entre {x_col} e {y_col}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_correlation_matrix(self, columns):
        """
        Cria matriz de correlação
        
        Args:
            columns (list): Lista de colunas numéricas
            
        Returns:
            matplotlib.figure.Figure: Figura com a matriz de correlação
        """
        # Selecionar apenas colunas numéricas
        numeric_data = self.df[columns].select_dtypes(include=[np.number])
        
        # Calcular correlação
        corr_matrix = numeric_data.corr()
        
        # Criar figura
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Criar heatmap
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={"shrink": 0.8},
            ax=ax
        )
        
        # Configurações
        ax.set_title('Matriz de Correlação', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        
        return fig
    
    def plot_categorical_distribution(self, column, top_n=10):
        """
        Cria gráfico de barras para variável categórica
        """
        # Contar valores
        value_counts = self.df[column].value_counts().head(top_n)
        
        # Criar figura
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Gráfico de barras
        colors = plt.cm.Set3(range(len(value_counts)))
        bars = ax1.bar(range(len(value_counts)), value_counts.values, color=colors, edgecolor='black')
        ax1.set_xticks(range(len(value_counts)))
        ax1.set_xticklabels(value_counts.index, rotation=45, ha='right')
        ax1.set_ylabel('Frequência')
        ax1.set_title(f'Distribuição de {column}', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Adicionar valores nas barras
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=9)
        
        # Gráfico de pizza
        ax2.pie(
            value_counts.values,
            labels=value_counts.index,
            autopct='%1.1f%%',
            colors=colors,
            startangle=90
        )
        ax2.set_title(f'Proporção de {column}', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def plot_missing_values(self):
        """
        Visualiza valores nulos no dataset
        
        Returns:
            matplotlib.figure.Figure: Figura com visualização de nulos
        """
        # Calcular porcentagem de valores nulos
        missing_data = self.df.isnull().sum()
        missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
        
        if len(missing_data) == 0:
            # Se não houver valores nulos
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'Não há valores nulos no dataset!',
                   horizontalalignment='center',
                   verticalalignment='center',
                   fontsize=20,
                   color='green',
                   fontweight='bold')
            ax.axis('off')
            return fig
        
        missing_percent = (missing_data / len(self.df)) * 100
        
        # Criar figura
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Gráfico de barras
        colors = ['red' if x > 50 else 'orange' if x > 20 else 'yellow' for x in missing_percent]
        bars = ax1.barh(range(len(missing_data)), missing_percent.values, color=colors, edgecolor='black')
        ax1.set_yticks(range(len(missing_data)))
        ax1.set_yticklabels(missing_data.index)
        ax1.set_xlabel('Porcentagem de Valores Nulos (%)')
        ax1.set_title('Valores Nulos por Coluna', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Adicionar valores
        for i, (bar, val) in enumerate(zip(bars, missing_percent.values)):
            ax1.text(val + 1, bar.get_y() + bar.get_height()/2,
                    f'{val:.1f}%',
                    va='center', fontsize=9)
        
        # Heatmap de valores nulos
        sns.heatmap(
            self.df.isnull(),
            yticklabels=False,
            cbar=True,
            cmap='RdYlGn_r',
            ax=ax2
        )
        ax2.set_title('Mapa de Valores Nulos', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_model_comparison(results_df, metric):
        """
        Cria gráfico comparativo de modelos
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Criar gráfico de barras
        colors = plt.cm.viridis(np.linspace(0, 1, len(results_df)))
        bars = ax.bar(results_df.index, results_df[metric], color=colors, edgecolor='black', linewidth=1.5)
        
        # Adicionar valores nas barras
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Configurações
        ax.set_ylabel(metric, fontsize=12)
        ax.set_title(f'Comparação de Modelos - {metric}', fontsize=14, fontweight='bold')
        ax.set_xticklabels(results_df.index, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Adicionar linha média
        mean_val = results_df[metric].mean()
        ax.axhline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Média: {mean_val:.4f}')
        ax.legend()
        
        plt.tight_layout()
        return fig
