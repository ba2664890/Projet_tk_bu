"""Chargement et optimisation des sources de données."""

import os
from typing import Tuple

import numpy as np
import pandas as pd


class DataSourceManager:
    """Gestionnaire pour différentes sources de données"""

    @staticmethod
    def load_file(file_path: str, **kwargs) -> Tuple[pd.DataFrame, str]:
        """Charge un fichier selon son extension"""
        ext = os.path.splitext(file_path)[1].lower()

        try:
            if ext == '.csv':
                csv_kwargs = dict(kwargs)
                csv_kwargs.setdefault("low_memory", False)
                try:
                    df = pd.read_csv(file_path, **csv_kwargs)
                except Exception as csv_err:
                    # Pandas: low_memory n'est pas supporté avec engine='python'
                    if "low_memory" in str(csv_err) and "python" in str(csv_err):
                        csv_kwargs.pop("low_memory", None)
                        df = pd.read_csv(file_path, **csv_kwargs)
                    else:
                        raise
            elif ext in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path, **kwargs)
            elif ext == '.json':
                df = pd.read_json(file_path, **kwargs)
            elif ext == '.parquet':
                df = pd.read_parquet(file_path, **kwargs)
            elif ext == '.dta':
                df = pd.read_stata(file_path, **kwargs)
            elif ext == '.sas7bdat':
                df = pd.read_sas(file_path, **kwargs)
            elif ext in ['.sav', '.por']:
                df = pd.read_spss(file_path, **kwargs)
            elif ext == '.feather':
                df = pd.read_feather(file_path, **kwargs)
            elif ext in ['.pkl', '.pickle']:
                df = pd.read_pickle(file_path, **kwargs)
            else:
                raise ValueError(f"Format non supporté: {ext}")

            df = DataSourceManager.optimize_memory(df)
            return df, f"✓ Fichier {ext.upper()} chargé"

        except Exception as e:
            raise ValueError(f"Erreur: {str(e)}")

    @staticmethod
    def optimize_memory(df: pd.DataFrame) -> pd.DataFrame:
        """Optimise l'utilisation mémoire"""
        for col in df.columns:
            col_type = df[col].dtype
            if col_type == 'object':
                if df[col].nunique() / len(df) < 0.5:
                    df[col] = df[col].astype('category')
            elif col_type in ['int64', 'int32']:
                df[col] = pd.to_numeric(df[col], downcast='integer')
            elif col_type in ['float64', 'float32']:
                df[col] = pd.to_numeric(df[col], downcast='float')
        return df

    @staticmethod
    def load_from_api(url: str, headers: dict = None) -> Tuple[pd.DataFrame, str]:
        """Charge depuis une API REST"""
        try:
            import requests
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()

            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict) and 'data' in data:
                df = pd.DataFrame(data['data'])
            elif isinstance(data, dict) and 'results' in data:
                df = pd.DataFrame(data['results'])
            else:
                df = pd.DataFrame([data])

            return DataSourceManager.optimize_memory(df), "✓ Données API chargées"
        except ImportError:
            raise ImportError("Installez 'requests': pip install requests")
        except Exception as e:
            raise ValueError(f"Erreur API: {str(e)}")
