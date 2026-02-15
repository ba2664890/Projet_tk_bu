"""Analyse multivariée avancée."""

from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import f_oneway


class MultivariateAnalysis:
    """Analyses multivariées avancées"""

    @staticmethod
    def anova(df: pd.DataFrame, numeric_col: str, group_col: str) -> Dict:
        """ANOVA à un facteur"""
        data = df[[group_col, numeric_col]].dropna()
        groups = data.groupby(group_col)[numeric_col].apply(list).to_dict()
        group_values = [g for g in groups.values() if len(g) > 1]

        if len(group_values) < 2:
            return {'error': 'Au moins 2 groupes nécessaires'}

        f_stat, p_value = f_oneway(*group_values)

        group_stats = {}
        for name, values in groups.items():
            group_stats[str(name)] = {
                'n': len(values),
                'mean': np.mean(values),
                'std': np.std(values)
            }

        grand_mean = data[numeric_col].mean()
        ss_between = sum(len(v) * (np.mean(v) - grand_mean)**2 for v in group_values)
        ss_total = sum((data[numeric_col] - grand_mean)**2)
        eta_squared = ss_between / ss_total if ss_total > 0 else 0

        return {
            'f_statistic': f_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'eta_squared': eta_squared,
            'group_stats': group_stats
        }

    @staticmethod
    def linear_regression(df: pd.DataFrame, y_col: str, x_cols: List[str]) -> Dict:
        """Régression linéaire multiple"""
        try:
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import r2_score, mean_squared_error
        except ImportError:
            return {'error': "Scikit-learn non installé (pip install scikit-learn)"}

        if not y_col or y_col not in df.columns:
            return {'error': "Variable cible invalide"}

        valid_x_cols = [c for c in x_cols if c and c in df.columns and c != y_col]
        if not valid_x_cols:
            return {'error': "Aucune variable explicative valide"}

        data = df[[y_col] + valid_x_cols].dropna()
        if len(data) < len(valid_x_cols) + 1:
            return {'error': "Pas assez d'observations"}

        X, y = data[valid_x_cols], data[y_col]
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)

        return {
            'r2': r2_score(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'coefficients': dict(zip(valid_x_cols, model.coef_.tolist())),
            'intercept': model.intercept_,
            'n_observations': len(data),
            'features': valid_x_cols,
            'y_true': y.tolist(),
            'y_pred': y_pred.tolist(),
            'residuals': (y - y_pred).tolist()
        }

    @staticmethod
    def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
        """Matrice de corrélation"""
        return df.select_dtypes(include=[np.number]).corr()

    @staticmethod
    def pca_analysis(df: pd.DataFrame, n_components: int = 5) -> Dict:
        """Analyse en Composantes Principales"""
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        numeric_df = df.select_dtypes(include=[np.number]).dropna()
        if len(numeric_df.columns) < 2:
            return {'error': 'Au moins 2 variables numériques nécessaires'}

        scaler = StandardScaler()
        scaled = scaler.fit_transform(numeric_df)

        pca = PCA(n_components=min(n_components, len(numeric_df.columns)))
        scores = pca.fit_transform(scaled)

        return {
            'eigenvalues': pca.explained_variance_.tolist(),
            'variance_ratio': (pca.explained_variance_ratio_ * 100).tolist(),
            'cumulative_variance': (np.cumsum(pca.explained_variance_ratio_) * 100).tolist(),
            'feature_names': numeric_df.columns.tolist(),
            'components': pca.components_.tolist(),
            'scores': scores[:, :2].tolist() if scores.shape[1] >= 2 else []
        }
