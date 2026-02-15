"""Application principale EDA Desk Modern."""

import json
import os
import re
import textwrap
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Dict, List, Optional

import customtkinter as ctk
import matplotlib
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from scipy import stats
from scipy.stats import kendalltau, pearsonr, spearmanr

from .ai_report import HuggingFaceReportAssistant
from .analysis import MultivariateAnalysis
from .data_sources import DataSourceManager
from .theme import Colors
from .widgets import ModernButton, ModernCard

_SEABORN_AVAILABLE = True
try:
    import seaborn as sns
except Exception:
    _SEABORN_AVAILABLE = False

    class _SeabornFallback:
        """Subset API for this app when seaborn is unavailable."""

        @staticmethod
        def set_theme(style: str = "whitegrid") -> None:
            del style  # Only one fallback style is supported.
            for mpl_style in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid", "ggplot"):
                try:
                    matplotlib.style.use(mpl_style)
                    return
                except Exception:
                    continue

        @staticmethod
        def boxplot(*, data: pd.DataFrame, x: str, y: str, ax, color=None, **kwargs) -> None:
            del kwargs
            plot_data = data[[x, y]].dropna().copy()
            if plot_data.empty:
                return
            plot_data[x] = plot_data[x].astype(str)
            groups = list(dict.fromkeys(plot_data[x].tolist()))
            series = [plot_data.loc[plot_data[x] == g, y].values for g in groups]
            bp = ax.boxplot(series, labels=groups, patch_artist=True)
            box_color = color or "#4C72B0"
            for box in bp.get("boxes", []):
                box.set_facecolor(box_color)
                box.set_alpha(0.75)

        @staticmethod
        def stripplot(*, data: pd.DataFrame, x: str, y: str, ax, color="white", alpha=0.35, size=2, **kwargs) -> None:
            del kwargs
            plot_data = data[[x, y]].dropna().copy()
            if plot_data.empty:
                return
            plot_data[x] = plot_data[x].astype(str)
            groups = list(dict.fromkeys(plot_data[x].tolist()))
            positions = {group: idx + 1 for idx, group in enumerate(groups)}
            rng = np.random.default_rng(42)
            jitter = rng.uniform(-0.15, 0.15, len(plot_data))
            x_values = plot_data[x].map(positions).to_numpy(dtype=float) + jitter
            ax.scatter(
                x_values,
                plot_data[y].to_numpy(dtype=float),
                color=color,
                alpha=alpha,
                s=max(1, size) * 8,
                edgecolors="none",
            )

        @staticmethod
        def heatmap(
            data: pd.DataFrame,
            annot=True,
            fmt=".2f",
            cmap="RdBu_r",
            center=None,
            square=False,
            ax=None,
            **kwargs,
        ) -> None:
            del kwargs
            if ax is None:
                ax = plt.gca()

            matrix = data.to_numpy(dtype=float)
            aspect = "equal" if square else "auto"
            if center is None:
                image = ax.imshow(matrix, cmap=cmap, aspect=aspect)
            else:
                max_delta = np.nanmax(np.abs(matrix - center))
                if not np.isfinite(max_delta) or max_delta == 0:
                    max_delta = 1.0
                image = ax.imshow(
                    matrix,
                    cmap=cmap,
                    vmin=center - max_delta,
                    vmax=center + max_delta,
                    aspect=aspect,
                )
            ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

            ax.set_xticks(np.arange(data.shape[1]))
            ax.set_xticklabels([str(col) for col in data.columns], rotation=45, ha="right")
            ax.set_yticks(np.arange(data.shape[0]))
            ax.set_yticklabels([str(idx) for idx in data.index])

            if annot:
                for i in range(data.shape[0]):
                    for j in range(data.shape[1]):
                        value = data.iat[i, j]
                        if pd.notna(value):
                            ax.text(j, i, f"{value:{fmt}}", ha="center", va="center", fontsize=8)

            ax.set_xlim(-0.5, data.shape[1] - 0.5)
            ax.set_ylim(data.shape[0] - 0.5, -0.5)

    sns = _SeabornFallback()

# Configure backend before importing pyplot. Prefer TkAgg for embedded Tk plots.
try:
    matplotlib.use("TkAgg", force=True)
    import matplotlib.pyplot as plt
except Exception:
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

# Configuration UI
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")
ctk.DrawEngine.preferred_drawing_method = "polygon_shapes"
sns.set_theme(style="whitegrid")


class EDADeskPro(ctk.CTk):
    """Application principale EDA-Desk Pro"""
    
    def __init__(self):
        super().__init__()
        
        # Configuration fenêtre
        self.title("EDA-Desk Pro - Exploratory Data Analysis Tool")
        self.geometry("1500x950")
        self.minsize(1300, 800)
        
        # Variables de données
        self.df = None
        self.raw_df = None
        self.file_path = None
        self.numeric_cols = []
        self.categorical_cols = []
        self.boolean_cols = []
        self.active_filters: List[str] = []
        
        # Config IA (fichiers externes)
        # Les fichiers de config restent au niveau projet (même dossier que eda_desk_modern.py).
        self.project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.hf_config_path = os.path.join(self.project_dir, "hf_report_config.json")
        self.hf_prompt_path = os.path.join(self.project_dir, "hf_report_prompt.txt")
        self.ai_context_path: Optional[str] = None
        self.enable_ai_report = ctk.BooleanVar(value=True)
        self.report_brief: Dict[str, str] = {
            "theme": "",
            "resume_contexte": "",
            "objectif_general": "",
            "objectifs_specifiques": "",
            "questions_cle": "",
            "public_cible": "",
            "periode_etendue": "",
            "hypotheses": "",
            "limitations_connues": "",
            "format_souhaite": "Institutionnel",
            "niveau_detail": "Tres detaille",
        }
        
        HuggingFaceReportAssistant.ensure_default_files(self.hf_config_path, self.hf_prompt_path)
        
        # Lecteur vidéo page de garde
        self.default_cover_video_filename = "20611243-hd_1920_1080_24fps.mp4"
        self.default_cover_video_path = os.path.join(self.project_dir, self.default_cover_video_filename)
        self.cover_video_path: Optional[str] = (
            self.default_cover_video_path if os.path.exists(self.default_cover_video_path) else None
        )
        self.cover_video_capture = None
        self.cover_video_backend: Optional[str] = None
        self.cover_video_playing = False
        self.cover_video_job = None
        self.gst_player = None
        self.gst_video_sink = None
        self.gst_bus = None
        self.gst_poll_job = None
        self._is_closing = False
        
        # Configuration grid
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # Création de l'interface
        self.create_sidebar()
        self.create_main_area()

        self.protocol("WM_DELETE_WINDOW", self.on_close)
        
    def create_sidebar(self):
        """Crée la barre latérale"""
        self.sidebar = ctk.CTkFrame(
            self,
            width=300,
            corner_radius=0,
            fg_color=Colors.BG_SIDEBAR,
            border_width=1,
            border_color=Colors.BORDER
        )
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_propagate(False)
        
        # Logo et titre
        logo_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        logo_frame.pack(fill="x", padx=20, pady=20)
        
        ctk.CTkLabel(
            logo_frame, text="📊 EDA-Desk Pro",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color=Colors.PRIMARY
        ).pack(anchor="w")
        
        ctk.CTkLabel(
            logo_frame, text="Exploratory Data Analysis",
            font=ctk.CTkFont(size=12),
            text_color=Colors.TEXT_SECONDARY
        ).pack(anchor="w")
        
        # Séparateur
        ctk.CTkFrame(self.sidebar, height=2, fg_color=Colors.BG_INPUT).pack(fill="x", padx=20, pady=10)
        
        # === SECTION CHARGEMENT ===
        ctk.CTkLabel(
            self.sidebar, text="📁 SOURCE DE DONNÉES",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=Colors.TEXT_MUTED
        ).pack(anchor="w", padx=20, pady=(15, 5))
        
        # Menu source
        self.source_var = ctk.StringVar(value="CSV")
        sources = ["CSV", "Excel", "JSON", "Parquet", "Stata", "SAS", "SPSS"]
        
        self.source_menu = ctk.CTkOptionMenu(
            self.sidebar, variable=self.source_var,
            values=sources, corner_radius=10,
            fg_color=Colors.BG_INPUT, button_color=Colors.PRIMARY,
            button_hover_color=Colors.PRIMARY_HOVER
        )
        self.source_menu.pack(fill="x", padx=20, pady=5)
        
        # Boutons de chargement
        ModernButton(self.sidebar, "📂 Ouvrir Fichier", self.load_data).pack(fill="x", padx=20, pady=5)
        
        btn_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        btn_frame.pack(fill="x", padx=20, pady=5)
        
        ModernButton(btn_frame, "🌐 API", self.load_from_api, width=110).pack(side="left", padx=(0, 5))
        ModernButton(btn_frame, "🗄️ SQL", self.load_from_sql, width=110).pack(side="right")
        
        # === SECTION THÈME ===
        ctk.CTkLabel(
            self.sidebar, text="🎨 APPARENCE",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=Colors.TEXT_MUTED
        ).pack(anchor="w", padx=20, pady=(20, 5))
        
        self.theme_var = ctk.StringVar(value="Light")
        theme_switch = ctk.CTkOptionMenu(
            self.sidebar, variable=self.theme_var,
            values=["Dark", "Light", "System"],
            command=self.change_theme,
            corner_radius=10,
            fg_color=Colors.BG_INPUT,
            button_color=Colors.PRIMARY,
            button_hover_color=Colors.PRIMARY_HOVER
        )
        theme_switch.pack(fill="x", padx=20, pady=5)
        
        # === SECTION INFOS ===
        self.info_card = ModernCard(self.sidebar, "📋 Informations")
        self.info_card.pack(fill="x", padx=20, pady=15)
        
        self.lbl_filename = ctk.CTkLabel(
            self.info_card.content_frame, text="Aucun fichier chargé",
            font=ctk.CTkFont(size=12), text_color=Colors.TEXT_SECONDARY
        )
        self.lbl_filename.pack(anchor="w")
        
        self.lbl_dimensions = ctk.CTkLabel(
            self.info_card.content_frame, text="",
            font=ctk.CTkFont(size=11), text_color=Colors.TEXT_MUTED
        )
        self.lbl_dimensions.pack(anchor="w", pady=(5, 0))
        
        self.lbl_memory = ctk.CTkLabel(
            self.info_card.content_frame, text="",
            font=ctk.CTkFont(size=11), text_color=Colors.INFO
        )
        self.lbl_memory.pack(anchor="w")
        
        # === SECTION TYPES ===
        self.types_frame = ctk.CTkFrame(self.info_card.content_frame, fg_color="transparent")
        self.types_frame.pack(fill="x", pady=(10, 0))
        
        self.lbl_numeric = ctk.CTkLabel(self.types_frame, text="🔢 0", font=ctk.CTkFont(size=11))
        self.lbl_numeric.pack(side="left")
        
        self.lbl_categorical = ctk.CTkLabel(self.types_frame, text="📝 0", font=ctk.CTkFont(size=11))
        self.lbl_categorical.pack(side="left", padx=10)
        
        self.lbl_bool = ctk.CTkLabel(self.types_frame, text="✅ 0", font=ctk.CTkFont(size=11))
        self.lbl_bool.pack(side="left")

        self.lbl_filter_state = ctk.CTkLabel(
            self.info_card.content_frame, text="🧪 Filtres actifs: 0",
            font=ctk.CTkFont(size=11), text_color=Colors.SECONDARY
        )
        self.lbl_filter_state.pack(anchor="w", pady=(8, 0))
        
        # === SECTION RAPPORT IA ===
        ctk.CTkFrame(self.sidebar, height=2, fg_color=Colors.BG_INPUT).pack(fill="x", padx=20, pady=(10, 8))
        ctk.CTkLabel(
            self.sidebar, text="🤖 RAPPORT IA (HUGGING FACE)",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=Colors.TEXT_MUTED
        ).pack(anchor="w", padx=20, pady=(4, 2))
        
        self.ai_switch = ctk.CTkSwitch(
            self.sidebar, text="Activer interprétation IA",
            variable=self.enable_ai_report,
            progress_color=Colors.PRIMARY,
            button_color=Colors.PRIMARY,
            button_hover_color=Colors.PRIMARY_HOVER
        )
        self.ai_switch.pack(anchor="w", padx=20, pady=4)
        
        self.lbl_ai_context = ctk.CTkLabel(
            self.sidebar, text="Contexte IA: aucun fichier",
            font=ctk.CTkFont(size=10), text_color=Colors.TEXT_SECONDARY
        )
        self.lbl_ai_context.pack(anchor="w", padx=20, pady=(0, 4))
        
        ai_btns = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        ai_btns.pack(fill="x", padx=20, pady=(0, 8))
        ModernButton(ai_btns, "⚙️ Config IA", self.open_ai_config, style="secondary", width=120).pack(side="left")
        ModernButton(ai_btns, "📎 Fichier contexte", self.select_ai_context_file, style="secondary", width=140).pack(side="right")

        # === SECTION BRIEF RAPPORT ===
        ctk.CTkFrame(self.sidebar, height=2, fg_color=Colors.BG_INPUT).pack(fill="x", padx=20, pady=(4, 8))
        ctk.CTkLabel(
            self.sidebar, text="🧭 BRIEF RAPPORT",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=Colors.TEXT_MUTED
        ).pack(anchor="w", padx=20, pady=(4, 2))

        self.lbl_report_brief = ctk.CTkLabel(
            self.sidebar, text="Brief: non renseigne",
            font=ctk.CTkFont(size=10), text_color=Colors.TEXT_SECONDARY
        )
        self.lbl_report_brief.pack(anchor="w", padx=20, pady=(0, 4))

        ModernButton(
            self.sidebar, "📝 Editer brief rapport", self.open_report_brief_editor, style="secondary"
        ).pack(fill="x", padx=20, pady=(0, 8))

        self._update_report_brief_label()
        
        # === Boutons action ===
        ctk.CTkFrame(self.sidebar, height=2, fg_color=Colors.BG_INPUT).pack(fill="x", padx=20, pady=15)
        
        ModernButton(self.sidebar, "📄 Générer Rapport", self.generate_report, style="success").pack(fill="x", padx=20, pady=5)
        ModernButton(self.sidebar, "🔄 Réinitialiser", self.reset_app, style="secondary").pack(fill="x", padx=20, pady=5)
        
    def create_main_area(self):
        """Crée la zone principale (hors page d'accueil)."""
        self.main_frame = ctk.CTkFrame(self, corner_radius=0, fg_color=Colors.BG_DARK)
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        self.main_frame.grid_columnconfigure(0, weight=3, uniform="workspace")
        self.main_frame.grid_columnconfigure(1, weight=2, uniform="workspace")
        self.main_frame.grid_rowconfigure(0, weight=0)
        self.main_frame.grid_rowconfigure(1, weight=1)
        self.main_frame.grid_rowconfigure(2, weight=1)

        # === BARRE SUPÉRIEURE WORKSPACE ===
        self.workspace_topbar = ctk.CTkFrame(self.main_frame, corner_radius=12, fg_color=Colors.BG_CARD)
        self.workspace_topbar.grid(row=0, column=0, columnspan=2, sticky="ew", padx=5, pady=5)

        top_left = ctk.CTkFrame(self.workspace_topbar, fg_color="transparent")
        top_left.pack(side="left", padx=12, pady=8)
        ModernButton(
            top_left,
            "📂 Ouvrir CSV",
            self._open_csv_from_topbar,
            style="primary",
            width=130,
        ).pack(side="left")

        top_info = ctk.CTkFrame(self.workspace_topbar, fg_color="transparent")
        top_info.pack(side="right", padx=12, pady=8)
        self.lbl_topbar_file = ctk.CTkLabel(
            top_info,
            text="Aucun fichier chargé",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=Colors.TEXT_PRIMARY,
        )
        self.lbl_topbar_file.pack(anchor="e")
        self.lbl_topbar_dims = ctk.CTkLabel(
            top_info,
            text="Dimensions: -",
            font=ctk.CTkFont(size=11),
            text_color=Colors.TEXT_SECONDARY,
        )
        self.lbl_topbar_dims.pack(anchor="e")
        self.lbl_topbar_filters = ctk.CTkLabel(
            top_info,
            text="Filtres actifs: 0",
            font=ctk.CTkFont(size=10),
            text_color=Colors.TEXT_MUTED,
        )
        self.lbl_topbar_filters.pack(anchor="e")

        # === ZONE GAUCHE : APERÇU TABLE ===
        self.preview_frame = ctk.CTkFrame(self.main_frame, corner_radius=15, fg_color=Colors.BG_CARD)
        self.preview_frame.grid(row=1, column=0, sticky="nsew", padx=(5, 3), pady=5)
        self.preview_frame.grid_columnconfigure(0, weight=1)
        self.preview_frame.grid_rowconfigure(1, weight=1)
        self.create_preview_area()

        # === ZONE DROITE : CONTRÔLES D'ANALYSE ===
        self.tabs_panel = ctk.CTkFrame(self.main_frame, corner_radius=15, fg_color=Colors.BG_CARD)
        self.tabs_panel.grid(row=1, column=1, sticky="nsew", padx=(3, 5), pady=5)

        tabs_header = ctk.CTkFrame(self.tabs_panel, fg_color="transparent")
        tabs_header.pack(fill="x", padx=10, pady=(10, 5))

        ctk.CTkLabel(
            tabs_header, text="Navigation:",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=Colors.TEXT_SECONDARY
        ).pack(side="left", padx=(0, 8))

        self.nav_tab_buttons: Dict[str, ModernButton] = {}
        nav_specs = [
            ("Diagnostic", "🩺 Diagnostic"),
            ("Statistiques", "📈 Statistiques"),
            ("Multivarie", "🧠 Multivarié"),
            ("Visualisation", "🎨 Visualisation"),
            ("Filtres", "🧪 Filtres"),
        ]
        for tab_name, label in nav_specs:
            btn = ModernButton(
                tabs_header,
                label,
                command=lambda t=tab_name: self.open_workspace_tab(t),
                style="secondary",
                width=138,
            )
            btn.pack(side="left", padx=3)
            self.nav_tab_buttons[tab_name] = btn

        ModernButton(
            tabs_header,
            "🏠 Accueil",
            self.show_home_page,
            style="primary",
            width=118
        ).pack(side="right")

        self.tabs_content = ctk.CTkFrame(self.tabs_panel, fg_color="transparent")
        self.tabs_content.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        # Créer les conteneurs d'onglets
        self.tab_diagnostic = ctk.CTkFrame(self.tabs_content, fg_color="transparent")
        self.tab_stats = ctk.CTkFrame(self.tabs_content, fg_color="transparent")
        self.tab_multivar = ctk.CTkFrame(self.tabs_content, fg_color="transparent")
        self.tab_viz = ctk.CTkFrame(self.tabs_content, fg_color="transparent")
        self.tab_filters = ctk.CTkFrame(self.tabs_content, fg_color="transparent")

        # Configuration des onglets
        self.setup_diagnostic_tab()
        self.setup_stats_tab()
        self.setup_multivar_tab()
        self.setup_viz_tab()
        self.setup_filters_tab()
        self.switch_tab("Diagnostic")

        # === ZONE INFÉRIEURE : RÉSULTATS ===
        self.create_results_area()
        self._refresh_preview_table()

        # === PAGE D'ACCUEIL (séparée) ===
        self.create_home_page()
        self.show_home_page()
        
    def switch_tab(self, tab_name: str):
        """Affiche l'onglet sélectionné"""
        tabs = {
            "Diagnostic": self.tab_diagnostic,
            "Statistiques": self.tab_stats,
            "Multivarie": self.tab_multivar,
            "Visualisation": self.tab_viz,
            "Filtres": self.tab_filters,
        }

        if tab_name not in tabs:
            tab_name = "Diagnostic"

        for name, frame in tabs.items():
            if name == tab_name:
                frame.pack(fill="both", expand=True)
            else:
                frame.pack_forget()

        for name, btn in self.nav_tab_buttons.items():
            if name == tab_name:
                btn.configure(
                    fg_color=Colors.PRIMARY,
                    hover_color=Colors.PRIMARY_HOVER,
                    text_color="#FFFFFF",
                )
            else:
                btn.configure(
                    fg_color="#E2ECF8",
                    hover_color="#CFDFF2",
                    text_color=Colors.TEXT_PRIMARY,
                )

    def open_workspace_tab(self, tab_name: str):
        """Ouvre une page métier depuis l'accueil ou la barre de navigation."""
        self.show_workspace_page()
        self.switch_tab(tab_name)

    def create_home_page(self):
        """Crée une vraie page d'accueil séparée avec vidéo de fond et accès aux pages."""
        self.home_frame = ctk.CTkFrame(
            self.main_frame,
            corner_radius=16,
            fg_color=Colors.BG_DARK,
            border_width=1,
            border_color=Colors.BORDER
        )

        self.cover_video_label = ctk.CTkLabel(
            self.home_frame,
            text=self._cover_video_placeholder_text(),
            text_color=Colors.TEXT_MUTED,
            font=ctk.CTkFont(size=16, weight="bold"),
            justify="center",
            fg_color=Colors.BG_DARK,
        )
        self.cover_video_label.place(relx=0, rely=0, relwidth=1, relheight=1)

        self.home_overlay = ctk.CTkFrame(
            self.home_frame,
            corner_radius=18,
            fg_color=Colors.BG_CARD,
            border_width=1,
            border_color=Colors.BORDER,
        )
        self.home_overlay.place(relx=0.5, rely=0.52, anchor="center", relwidth=0.82, relheight=0.78)

        ctk.CTkLabel(
            self.home_overlay,
            text="EDA-Desk Pro",
            font=ctk.CTkFont(size=36, weight="bold"),
            text_color=Colors.TEXT_PRIMARY
        ).pack(pady=(28, 4))

        ctk.CTkLabel(
            self.home_overlay,
            text="Une page d'accueil claire, avec navigation par boutons comme un site web",
            font=ctk.CTkFont(size=15),
            text_color=Colors.TEXT_SECONDARY
        ).pack()

        ctk.CTkLabel(
            self.home_overlay,
            text="Choisissez une section pour démarrer",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=Colors.INFO
        ).pack(pady=(20, 8))

        nav_grid = ctk.CTkFrame(self.home_overlay, fg_color="transparent")
        nav_grid.pack(pady=(0, 14))
        for idx in range(3):
            nav_grid.grid_columnconfigure(idx, weight=1)

        home_actions = [
            ("🩺 Diagnostic", lambda: self.open_workspace_tab("Diagnostic"), "primary"),
            ("📈 Statistiques", lambda: self.open_workspace_tab("Statistiques"), "secondary"),
            ("🧠 Multivarié", lambda: self.open_workspace_tab("Multivarie"), "secondary"),
            ("🎨 Visualisation", lambda: self.open_workspace_tab("Visualisation"), "secondary"),
            ("🧪 Filtres", lambda: self.open_workspace_tab("Filtres"), "secondary"),
            ("📄 Générer rapport", self.generate_report, "success"),
        ]
        for idx, (label, cmd, style) in enumerate(home_actions):
            row = idx // 3
            col = idx % 3
            ModernButton(nav_grid, label, cmd, style=style, width=210).grid(row=row, column=col, padx=8, pady=8)

        self.lbl_cover_video_name = ctk.CTkLabel(
            self.home_overlay,
            text=(
                f"Vidéo de fond fixe: {os.path.basename(self.cover_video_path)}"
                if self.cover_video_path
                else f"Vidéo introuvable: {self.default_cover_video_filename}"
            ),
            font=ctk.CTkFont(size=11),
            text_color=Colors.TEXT_MUTED
        )
        self.lbl_cover_video_name.pack(pady=(8, 4))

        self.lbl_cover_video_status = ctk.CTkLabel(
            self.home_overlay,
            text="Statut vidéo: prêt",
            font=ctk.CTkFont(size=11),
            text_color=Colors.INFO
        )
        self.lbl_cover_video_status.pack(pady=(8, 0))
        if not self.cover_video_path:
            self._set_cover_video_status("Statut vidéo: fichier introuvable", "#FCA5A5")

    def show_home_page(self):
        """Affiche la landing page (sans mélange avec les pages métier)."""
        if hasattr(self, "workspace_topbar"):
            self.workspace_topbar.grid_remove()
        if hasattr(self, "preview_frame"):
            self.preview_frame.grid_remove()
        if hasattr(self, "tabs_panel"):
            self.tabs_panel.grid_remove()
        if hasattr(self, "results_frame"):
            self.results_frame.grid_remove()
        self.home_frame.grid(row=0, column=0, columnspan=2, rowspan=3, sticky="nsew", padx=5, pady=5)
        if self.cover_video_path and not self.cover_video_playing:
            self.toggle_cover_video()

    def _cover_video_placeholder_text(self) -> str:
        """Texte d'attente pour la zone vidéo d'accueil."""
        if self.cover_video_path:
            return f"Vidéo de fond: {os.path.basename(self.cover_video_path)}"
        return f"Vidéo introuvable: {self.default_cover_video_filename}"

    def _set_cover_video_status(self, text: str, color: str = "#93C5FD"):
        """Met à jour le statut vidéo affiché sur la landing page."""
        if hasattr(self, "lbl_cover_video_status"):
            self._safe_configure(self.lbl_cover_video_status, text=text, text_color=color)

    def _set_video_toggle_label(self, playing: bool):
        """Met à jour le texte du bouton lecture/pause si le bouton existe."""
        if hasattr(self, "btn_video_toggle"):
            self._safe_configure(
                self.btn_video_toggle,
                text="⏸️ Pause" if playing else "▶️ Lecture"
            )

    def _safe_configure(self, widget, **kwargs) -> bool:
        """Configure un widget en ignorant les erreurs Tk de fermeture."""
        try:
            widget.configure(**kwargs)
            return True
        except (tk.TclError, RuntimeError):
            return False

    def show_workspace_page(self):
        """Affiche l'espace de travail (pages métier)."""
        if hasattr(self, "home_frame"):
            self.home_frame.grid_remove()
        if hasattr(self, "workspace_topbar"):
            self.workspace_topbar.grid()
        if hasattr(self, "preview_frame"):
            self.preview_frame.grid()
        self.tabs_panel.grid()
        self.results_frame.grid()
        self._refresh_preview_table()
        if self.cover_video_playing:
            self.stop_cover_video()

    def _open_csv_from_topbar(self):
        """Raccourci barre supérieure: ouvre directement un CSV."""
        if hasattr(self, "source_var"):
            self.source_var.set("CSV")
        self.load_data()

    def create_preview_area(self):
        """Crée la zone d'aperçu tabulaire (50 premières lignes)."""
        header = ctk.CTkFrame(self.preview_frame, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=12, pady=(10, 6))
        header.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            header,
            text="📄 Aperçu des Données",
            font=ctk.CTkFont(size=15, weight="bold"),
            text_color=Colors.PRIMARY,
        ).grid(row=0, column=0, sticky="w")

        self.lbl_preview_meta = ctk.CTkLabel(
            header,
            text="Aucune donnée",
            font=ctk.CTkFont(size=11),
            text_color=Colors.TEXT_SECONDARY,
        )
        self.lbl_preview_meta.grid(row=0, column=1, sticky="e")

        table_wrap = ctk.CTkFrame(
            self.preview_frame,
            corner_radius=10,
            fg_color=Colors.BG_INPUT,
            border_width=1,
            border_color=Colors.BORDER,
        )
        table_wrap.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 12))
        table_wrap.grid_rowconfigure(0, weight=1)
        table_wrap.grid_columnconfigure(0, weight=1)

        try:
            style = ttk.Style()
            style.configure("Preview.Treeview", rowheight=24)
            style.configure("Preview.Treeview.Heading", font=("TkDefaultFont", 10, "bold"))
        except Exception:
            pass

        self.preview_table = ttk.Treeview(
            table_wrap,
            columns=(),
            show="headings",
            style="Preview.Treeview",
        )
        self.preview_table.grid(row=0, column=0, sticky="nsew")

        y_scroll = ttk.Scrollbar(table_wrap, orient="vertical", command=self.preview_table.yview)
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll = ttk.Scrollbar(table_wrap, orient="horizontal", command=self.preview_table.xview)
        x_scroll.grid(row=1, column=0, sticky="ew")
        self.preview_table.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)

    def _refresh_preview_table(self, max_rows: int = 50):
        """Met à jour l'aperçu tabulaire."""
        if not hasattr(self, "preview_table"):
            return

        tree = self.preview_table
        try:
            tree.delete(*tree.get_children())
        except tk.TclError:
            return

        if self.df is None:
            tree.configure(columns=("etat",), show="headings")
            tree.heading("etat", text="Aperçu")
            tree.column("etat", width=320, anchor="w", stretch=True)
            tree.insert("", "end", values=("Aucune donnée chargée.",))
            if hasattr(self, "lbl_preview_meta"):
                self.lbl_preview_meta.configure(text="Aucune donnée")
            return

        df_view = self.df.head(max_rows).copy()
        columns = [str(col) for col in df_view.columns]
        if not columns:
            columns = ["(vide)"]
            tree.configure(columns=columns, show="headings")
            tree.heading("(vide)", text="Aperçu")
            tree.column("(vide)", width=320, anchor="w", stretch=True)
            tree.insert("", "end", values=("Dataset sans colonnes.",))
            if hasattr(self, "lbl_preview_meta"):
                self.lbl_preview_meta.configure(text="0 ligne")
            return

        tree.configure(columns=columns, show="headings")
        for col in columns:
            tree.heading(col, text=col)
            col_width = min(260, max(110, len(col) * 10))
            tree.column(col, width=col_width, minwidth=80, anchor="w", stretch=True)

        for _, row in df_view.iterrows():
            values: List[str] = []
            for val in row.tolist():
                try:
                    is_missing = bool(pd.isna(val))
                except Exception:
                    is_missing = False
                if is_missing:
                    values.append("")
                else:
                    txt = str(val)
                    values.append(txt if len(txt) <= 180 else f"{txt[:177]}...")
            tree.insert("", "end", values=values)

        if hasattr(self, "lbl_preview_meta"):
            self.lbl_preview_meta.configure(
                text=f"{len(df_view):,} lignes affichées / {len(self.df):,}"
            )

    def setup_cover_tab(self):
        """Ancienne page onglet 'Accueil' (désactivée, remplacée par la landing page dédiée)."""
        return

    def setup_filters_tab(self):
        """Configure l'onglet de filtres avancés."""
        filter_card = ModernCard(self.tab_filters, "🧪 Filtres avancés")
        filter_card.pack(fill="x", padx=10, pady=8)

        ctk.CTkLabel(
            filter_card.content_frame,
            text="Combinez recherche texte, plage numérique et modalités catégorielles.",
            text_color=Colors.TEXT_SECONDARY,
            font=ctk.CTkFont(size=12)
        ).pack(anchor="w", pady=(0, 10))

        self.filter_search_var = ctk.StringVar()
        row_search = ctk.CTkFrame(filter_card.content_frame, fg_color="transparent")
        row_search.pack(fill="x", pady=4)
        ctk.CTkLabel(row_search, text="Recherche texte globale:", width=190, anchor="w").pack(side="left")
        ctk.CTkEntry(row_search, textvariable=self.filter_search_var, width=480, corner_radius=10).pack(side="left", padx=6)

        self.filter_num_col_var = ctk.StringVar(value="")
        self.filter_min_var = ctk.StringVar(value="")
        self.filter_max_var = ctk.StringVar(value="")
        row_num = ctk.CTkFrame(filter_card.content_frame, fg_color="transparent")
        row_num.pack(fill="x", pady=4)
        ctk.CTkLabel(row_num, text="Filtre numérique:", width=190, anchor="w").pack(side="left")
        self.combo_filter_num = ctk.CTkOptionMenu(row_num, variable=self.filter_num_col_var, values=[""], width=160)
        self.combo_filter_num.pack(side="left", padx=(6, 4))
        ctk.CTkEntry(row_num, textvariable=self.filter_min_var, width=120, placeholder_text="Min").pack(side="left", padx=2)
        ctk.CTkEntry(row_num, textvariable=self.filter_max_var, width=120, placeholder_text="Max").pack(side="left", padx=2)

        self.filter_cat_col_var = ctk.StringVar(value="")
        self.filter_cat_values_var = ctk.StringVar(value="")
        row_cat = ctk.CTkFrame(filter_card.content_frame, fg_color="transparent")
        row_cat.pack(fill="x", pady=4)
        ctk.CTkLabel(row_cat, text="Filtre catégoriel:", width=190, anchor="w").pack(side="left")
        self.combo_filter_cat = ctk.CTkOptionMenu(row_cat, variable=self.filter_cat_col_var, values=[""], width=160)
        self.combo_filter_cat.pack(side="left", padx=(6, 4))
        ctk.CTkEntry(
            row_cat, textvariable=self.filter_cat_values_var, width=360, corner_radius=10,
            placeholder_text="Valeurs séparées par virgule (ex: A, B, C)"
        ).pack(side="left", padx=2)

        self.filter_exclude_na = ctk.BooleanVar(value=False)
        ctk.CTkSwitch(
            filter_card.content_frame,
            text="Exclure les lignes contenant des NA après filtrage",
            variable=self.filter_exclude_na,
            progress_color=Colors.PRIMARY,
            button_color=Colors.PRIMARY,
            button_hover_color=Colors.PRIMARY_HOVER
        ).pack(anchor="w", pady=(8, 5))

        action_row = ctk.CTkFrame(filter_card.content_frame, fg_color="transparent")
        action_row.pack(fill="x", pady=(4, 0))
        ModernButton(action_row, "✅ Appliquer filtres", self.apply_filters, style="success", width=170).pack(side="left")
        ModernButton(action_row, "🧼 Effacer filtres", self.clear_filters, style="secondary", width=150).pack(side="left", padx=8)
        ModernButton(action_row, "🔎 Aperçu", self.preview_filtered_data, style="primary", width=120).pack(side="left")
        
    def setup_diagnostic_tab(self):
        """Configure l'onglet Diagnostic"""
        # Boutons
        btn_frame = ctk.CTkFrame(self.tab_diagnostic, fg_color="transparent")
        btn_frame.pack(fill="x", padx=10, pady=10)
        
        buttons = [
            ("📊 Valeurs Manquantes", self.analyze_missing, "primary"),
            ("⚠️ Variables Problèmes", self.detect_problems, "warning"),
            ("🎯 Outliers IQR", self.detect_outliers, "danger"),
            ("🔄 Doublons", self.detect_duplicates, "secondary"),
            ("📋 Diagnostic Complet", self.full_diagnostic, "success")
        ]
        
        for text, cmd, style in buttons:
            ModernButton(btn_frame, text, cmd, style).pack(side="left", padx=3)
            
    def setup_stats_tab(self):
        """Configure l'onglet Statistiques"""
        # Sélection variable
        select_frame = ctk.CTkFrame(self.tab_stats, fg_color="transparent")
        select_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(select_frame, text="Variable:", font=ctk.CTkFont(size=13)).pack(side="left", padx=5)
        
        self.var_stats = ctk.StringVar()
        self.combo_stats = ctk.CTkOptionMenu(
            select_frame, variable=self.var_stats,
            values=[""], corner_radius=10, width=250
        )
        self.combo_stats.pack(side="left", padx=5)
        
        # Boutons
        btn_frame = ctk.CTkFrame(self.tab_stats, fg_color="transparent")
        btn_frame.pack(fill="x", padx=10, pady=5)
        
        buttons = [
            ("📊 Stats Numériques", self.numeric_stats, "primary"),
            ("📋 Stats Catégorielles", self.categorical_stats, "primary"),
            ("📑 Résumé Complet", self.full_summary, "success"),
            ("📊 Toutes Variables", self.all_vars_stats, "secondary")
        ]
        
        for text, cmd, style in buttons:
            ModernButton(btn_frame, text, cmd, style).pack(side="left", padx=3)
            
    def setup_multivar_tab(self):
        """Configure l'onglet Multivarié"""
        # Sélection des variables
        select_frame = ctk.CTkFrame(self.tab_multivar, fg_color="transparent")
        select_frame.pack(fill="x", padx=10, pady=10)
        
        # Variable numérique 1
        ctk.CTkLabel(select_frame, text="Numérique 1:", font=ctk.CTkFont(size=11)).pack(side="left", padx=5)
        self.var_num1 = ctk.StringVar()
        self.combo_num1 = ctk.CTkOptionMenu(
            select_frame, variable=self.var_num1, values=[""], corner_radius=8, width=120
        )
        self.combo_num1.pack(side="left", padx=3)
        
        # Variable numérique 2
        ctk.CTkLabel(select_frame, text="Numérique 2:", font=ctk.CTkFont(size=11)).pack(side="left", padx=5)
        self.var_num2 = ctk.StringVar()
        self.combo_num2 = ctk.CTkOptionMenu(
            select_frame, variable=self.var_num2, values=[""], corner_radius=8, width=120
        )
        self.combo_num2.pack(side="left", padx=3)
        
        # Variable catégorielle
        ctk.CTkLabel(select_frame, text="Catégorielle:", font=ctk.CTkFont(size=11)).pack(side="left", padx=5)
        self.var_cat = ctk.StringVar()
        self.combo_cat = ctk.CTkOptionMenu(
            select_frame, variable=self.var_cat, values=[""], corner_radius=8, width=120
        )
        self.combo_cat.pack(side="left", padx=3)
        
        # Variable groupe
        ctk.CTkLabel(select_frame, text="Groupe:", font=ctk.CTkFont(size=11)).pack(side="left", padx=5)
        self.var_group = ctk.StringVar()
        self.combo_group = ctk.CTkOptionMenu(
            select_frame, variable=self.var_group, values=[""], corner_radius=8, width=120
        )
        self.combo_group.pack(side="left", padx=3)
        
        # Boutons
        btn_frame = ctk.CTkFrame(self.tab_multivar, fg_color="transparent")
        btn_frame.pack(fill="x", padx=10, pady=5)
        
        buttons = [
            ("📊 ANOVA", self.run_anova, "primary"),
            ("🔢 Test t", self.run_ttest, "primary"),
            ("📈 Corrélations", self.run_correlations, "primary"),
            ("🎯 Chi²", self.run_chi2, "warning"),
            ("📉 Régression", self.run_regression, "success"),
            ("🌀 ACP", self.run_pca, "secondary")
        ]
        
        for text, cmd, style in buttons:
            ModernButton(btn_frame, text, cmd, style).pack(side="left", padx=3)
            
    def setup_viz_tab(self):
        """Configure l'onglet Visualisation"""
        # Sélection
        select_frame = ctk.CTkFrame(self.tab_viz, fg_color="transparent")
        select_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(select_frame, text="Variable 1:", font=ctk.CTkFont(size=11)).pack(side="left", padx=5)
        self.var_viz = ctk.StringVar()
        self.combo_viz1 = ctk.CTkOptionMenu(
            select_frame, variable=self.var_viz, values=[""], corner_radius=8, width=150
        )
        self.combo_viz1.pack(side="left", padx=3)
        
        ctk.CTkLabel(select_frame, text="Variable 2:", font=ctk.CTkFont(size=11)).pack(side="left", padx=5)
        self.var_viz2 = ctk.StringVar()
        self.combo_viz2 = ctk.CTkOptionMenu(
            select_frame, variable=self.var_viz2, values=[""], corner_radius=8, width=150
        )
        self.combo_viz2.pack(side="left", padx=3)
        
        # Boutons
        btn_frame = ctk.CTkFrame(self.tab_viz, fg_color="transparent")
        btn_frame.pack(fill="x", padx=10, pady=5)
        
        buttons = [
            ("📉 Histogramme", self.plot_histogram, "primary"),
            ("📦 Boxplot", self.plot_boxplot, "primary"),
            ("⚪ Scatter", self.plot_scatter, "primary"),
            ("🔥 Heatmap", self.plot_heatmap, "danger"),
            ("📊 Bar", self.plot_bar, "secondary"),
            ("🥧 Pie", self.plot_pie, "warning"),
            ("📈 Pairplot", self.plot_pairplot, "success"),
            ("📉 Distribution", self.plot_distribution, "secondary")
        ]
        
        for text, cmd, style in buttons:
            ModernButton(btn_frame, text, cmd, style).pack(side="left", padx=3)
            
    def create_results_area(self):
        """Crée la zone de résultats"""
        # Frame principale des résultats
        self.results_frame = ctk.CTkFrame(self.main_frame, corner_radius=15, fg_color=Colors.BG_CARD)
        self.results_frame.grid(row=2, column=0, columnspan=2, sticky="nsew", padx=5, pady=(3, 5))
        self.results_frame.grid_columnconfigure(0, weight=1)
        self.results_frame.grid_rowconfigure(1, weight=1)
        
        # Header
        header = ctk.CTkFrame(self.results_frame, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=15, pady=10)
        
        ctk.CTkLabel(
            header, text="📝 Résultats",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=Colors.PRIMARY
        ).pack(side="left")
        
        # Bouton copier
        ModernButton(header, "📋 Copier", self.copy_results, style="secondary", width=80).pack(side="right")
        
        # Zone de texte avec scroll
        self.txt_results = ctk.CTkTextbox(
            self.results_frame, corner_radius=10,
            fg_color=Colors.BG_INPUT,
            text_color=Colors.TEXT_PRIMARY,
            font=ctk.CTkFont(family="Consolas", size=12),
            wrap="word"
        )
        self.txt_results.grid(row=1, column=0, sticky="nsew", padx=15, pady=(0, 15))
        
        # Message d'accueil
        self.show_welcome()
        
    def show_welcome(self):
        """Affiche le message d'accueil"""
        welcome = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                     EDA-Desk Pro Modern - Light Experience                   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║   SOURCES: CSV (paramètres avancés), Excel, JSON, Parquet, API, SQL        ║
║   FILTRES: texte global, plage numérique, catégories multi-valeurs          ║
║   RAPPORTS: PDF + interprétation IA Hugging Face (optionnelle)             ║
║                                                                              ║
║  Chargez un fichier pour commencer.                                           ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
        self.txt_results.delete("1.0", "end")
        self.txt_results.insert("1.0", welcome)

    def on_close(self):
        """Nettoyage à la fermeture."""
        self._is_closing = True
        try:
            self.stop_cover_video()
        except tk.TclError:
            pass
        finally:
            try:
                self.destroy()
            except tk.TclError:
                pass

    def _update_dataset_info(self):
        """Synchronise les cartes d'information dataset + filtres."""
        if self.df is None:
            self.lbl_filename.configure(text="Aucun fichier chargé")
            self.lbl_dimensions.configure(text="")
            self.lbl_memory.configure(text="")
            self.lbl_filter_state.configure(text="🧪 Filtres actifs: 0")
            if hasattr(self, "lbl_topbar_file"):
                self.lbl_topbar_file.configure(text="Aucun fichier chargé")
            if hasattr(self, "lbl_topbar_dims"):
                self.lbl_topbar_dims.configure(text="Dimensions: -")
            if hasattr(self, "lbl_topbar_filters"):
                self.lbl_topbar_filters.configure(text="Filtres actifs: 0")
            return

        display_name = os.path.basename(self.file_path) if self.file_path else "Dataset"
        self.lbl_filename.configure(text=f"📄 {display_name}")
        self.lbl_dimensions.configure(text=f"📊 {self.df.shape[0]:,} × {self.df.shape[1]}")
        mem_mb = self.df.memory_usage(deep=True).sum() / 1024**2
        self.lbl_memory.configure(text=f"💾 {mem_mb:.1f} MB")
        self.lbl_filter_state.configure(text=f"🧪 Filtres actifs: {len(self.active_filters)}")
        if hasattr(self, "lbl_topbar_file"):
            self.lbl_topbar_file.configure(text=f"📄 {display_name}")
        if hasattr(self, "lbl_topbar_dims"):
            self.lbl_topbar_dims.configure(text=f"Dimensions: {self.df.shape[0]:,} × {self.df.shape[1]}")
        if hasattr(self, "lbl_topbar_filters"):
            self.lbl_topbar_filters.configure(text=f"Filtres actifs: {len(self.active_filters)}")

    def _build_dataset_summary_text(self) -> str:
        """Construit un résumé riche du dataset pour le reporting."""
        if self.df is None:
            return "Aucune donnée chargée"

        df = self.df
        total_cells = max(1, df.shape[0] * df.shape[1])
        missing_total = int(df.isnull().sum().sum())
        duplicates = int(df.duplicated().sum())
        miss_pct = (missing_total / total_cells) * 100
        num_cols = len(self.numeric_cols)
        cat_cols = len(self.categorical_cols)
        bool_cols = len(self.boolean_cols)

        top_missing = (df.isnull().mean() * 100).sort_values(ascending=False).head(8)
        top_missing_txt = ", ".join(
            f"{c}:{v:.1f}%" for c, v in top_missing.items() if v > 0
        ) or "Aucune valeur manquante significative"

        top_cats = []
        for col in (self.categorical_cols + self.boolean_cols)[:5]:
            s = df[col].dropna()
            if len(s) == 0:
                continue
            vc = s.value_counts().head(1)
            if not vc.empty:
                top_cats.append(f"{col} -> {vc.index[0]} ({vc.iloc[0]/len(s)*100:.1f}%)")

        num_summary = []
        for col in self.numeric_cols[:8]:
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(s) == 0:
                continue
            num_summary.append(
                f"{col}: mean={s.mean():.3f}, median={s.median():.3f}, std={s.std():.3f}"
            )

        return textwrap.dedent(
            f"""
            Dimensions: {df.shape[0]:,} lignes x {df.shape[1]} colonnes
            Types: numériques={num_cols}, catégorielles={cat_cols}, booléennes={bool_cols}
            Valeurs manquantes: {missing_total:,} ({miss_pct:.2f}%)
            Doublons lignes: {duplicates:,}
            Filtres actifs: {', '.join(self.active_filters) if self.active_filters else 'Aucun'}
            Colonnes les plus incomplètes: {top_missing_txt}
            Résumé numérique: {' | '.join(num_summary) if num_summary else 'N/A'}
            Résumé catégoriel: {' | '.join(top_cats) if top_cats else 'N/A'}
            """
        ).strip()

    def _report_brief_completion(self) -> Dict[str, Any]:
        keys = [
            "theme",
            "resume_contexte",
            "objectif_general",
            "objectifs_specifiques",
            "questions_cle",
            "public_cible",
            "periode_etendue",
            "hypotheses",
            "limitations_connues",
            "format_souhaite",
            "niveau_detail",
        ]
        filled = [k for k in keys if str(self.report_brief.get(k, "")).strip()]
        return {"filled": len(filled), "total": len(keys), "ratio": len(filled) / max(1, len(keys))}

    def _update_report_brief_label(self):
        info = self._report_brief_completion()
        theme = str(self.report_brief.get("theme", "")).strip() or "non renseigne"
        public = str(self.report_brief.get("public_cible", "")).strip() or "non renseigne"
        self.lbl_report_brief.configure(
            text=f"Brief: {info['filled']}/{info['total']} | Theme: {theme[:26]} | Public: {public[:20]}"
        )

    def _build_report_brief_text(self) -> str:
        labels = {
            "theme": "Theme",
            "resume_contexte": "Resume du contexte",
            "objectif_general": "Objectif general",
            "objectifs_specifiques": "Objectifs specifiques",
            "questions_cle": "Questions cles",
            "public_cible": "Public cible",
            "periode_etendue": "Periode etendue",
            "hypotheses": "Hypotheses",
            "limitations_connues": "Limitations connues",
            "format_souhaite": "Format souhaite",
            "niveau_detail": "Niveau de detail",
        }
        lines = []
        for k, label in labels.items():
            v = str(self.report_brief.get(k, "")).strip()
            lines.append(f"{label}: {v if v else 'Non renseigne'}")
        return "\n".join(lines)

    def open_report_brief_editor(self):
        """Edite les informations de cadrage pour un rapport precis et coherent."""
        win = ctk.CTkToplevel(self)
        win.title("Brief du rapport")
        win.geometry("860x760")
        win.transient(self)

        root = ctk.CTkScrollableFrame(win, fg_color="transparent")
        root.pack(fill="both", expand=True, padx=16, pady=14)

        ctk.CTkLabel(
            root,
            text="Renseignez le brief pour relier toutes les analyses entre elles",
            font=ctk.CTkFont(size=15, weight="bold"),
            text_color=Colors.TEXT_PRIMARY
        ).pack(anchor="w", pady=(0, 8))

        ctk.CTkLabel(
            root,
            text="Plus le brief est précis (thème, objectifs, public, questions), plus le rapport final est ciblé.",
            font=ctk.CTkFont(size=11),
            text_color=Colors.TEXT_SECONDARY
        ).pack(anchor="w", pady=(0, 12))

        def add_entry(label: str, key: str, placeholder: str = "", width: int = 780):
            ctk.CTkLabel(root, text=label, font=ctk.CTkFont(size=12, weight="bold")).pack(anchor="w", pady=(6, 2))
            var = ctk.StringVar(value=str(self.report_brief.get(key, "")))
            ent = ctk.CTkEntry(root, width=width, textvariable=var, placeholder_text=placeholder, corner_radius=10)
            ent.pack(anchor="w")
            return var

        def add_text(label: str, key: str, placeholder: str = "", height: int = 90):
            ctk.CTkLabel(root, text=label, font=ctk.CTkFont(size=12, weight="bold")).pack(anchor="w", pady=(8, 2))
            txt = ctk.CTkTextbox(root, width=790, height=height, corner_radius=10)
            txt.pack(anchor="w")
            txt.insert("1.0", str(self.report_brief.get(key, "")))
            if not str(self.report_brief.get(key, "")) and placeholder:
                txt.insert("1.0", placeholder)
            return txt

        theme_var = add_entry("Theme du rapport", "theme", "Ex: Etat des ODP regionaux et performance de service")
        public_var = add_entry("Public cible", "public_cible", "Ex: Direction generale, ANSD, partenaires techniques")
        period_var = add_entry("Periode / etendue", "periode_etendue", "Ex: 2020-2025, couverture nationale et regionale")
        format_var = add_entry("Format souhaite", "format_souhaite", "Ex: Institutionnel / Decisionnel / Technique")
        detail_var = add_entry("Niveau de detail", "niveau_detail", "Ex: Tres detaille")

        context_txt = add_text(
            "Resume du contexte",
            "resume_contexte",
            "Decrire le contexte metier, institutionnel et les enjeux principaux.",
            height=100,
        )
        obj_gen_txt = add_text(
            "Objectif general",
            "objectif_general",
            "Ex: Evaluer la qualite et la coherence des indicateurs ODP par region.",
            height=80,
        )
        obj_spec_txt = add_text(
            "Objectifs specifiques",
            "objectifs_specifiques",
            "Ex: 1) comparer les regions, 2) identifier les facteurs explicatifs, 3) proposer un plan d'action.",
            height=95,
        )
        q_txt = add_text(
            "Questions cles a traiter",
            "questions_cle",
            "Ex: Quelles regions sont en retard? Quelles variables expliquent les ecarts?",
            height=95,
        )
        hyp_txt = add_text(
            "Hypotheses / attentes",
            "hypotheses",
            "Ex: Les ecarts regionaux sont lies au niveau d'equipement et a la couverture de service.",
            height=80,
        )
        lim_txt = add_text(
            "Limitations connues",
            "limitations_connues",
            "Ex: manquants sur certaines regions, periodisation incomplete, biais de declaration.",
            height=80,
        )

        def tval(widget):
            return widget.get("1.0", "end").strip()

        def save_brief():
            self.report_brief.update({
                "theme": theme_var.get().strip(),
                "public_cible": public_var.get().strip(),
                "periode_etendue": period_var.get().strip(),
                "format_souhaite": format_var.get().strip(),
                "niveau_detail": detail_var.get().strip(),
                "resume_contexte": tval(context_txt),
                "objectif_general": tval(obj_gen_txt),
                "objectifs_specifiques": tval(obj_spec_txt),
                "questions_cle": tval(q_txt),
                "hypotheses": tval(hyp_txt),
                "limitations_connues": tval(lim_txt),
            })
            self._update_report_brief_label()
            self.show_message("✅ Brief rapport mis a jour", "success")
            win.destroy()

        btns = ctk.CTkFrame(root, fg_color="transparent")
        btns.pack(fill="x", pady=(12, 4))
        ModernButton(btns, "💾 Enregistrer brief", save_brief, style="success", width=180).pack(side="left")
        ModernButton(btns, "❌ Annuler", win.destroy, style="secondary", width=120).pack(side="left", padx=8)

    def open_ai_config(self):
        """Édite la configuration Hugging Face du rapport IA."""
        try:
            cfg = HuggingFaceReportAssistant.load_config(self.hf_config_path)
        except Exception:
            cfg = dict(HuggingFaceReportAssistant.DEFAULT_CONFIG)

        win = ctk.CTkToplevel(self)
        win.title("Configuration IA Hugging Face")
        win.geometry("760x560")
        win.transient(self)

        form = ctk.CTkFrame(win, fg_color="transparent")
        form.pack(fill="both", expand=True, padx=20, pady=15)

        def add_row(label: str, default: str):
            row = ctk.CTkFrame(form, fg_color="transparent")
            row.pack(fill="x", pady=5)
            ctk.CTkLabel(row, text=label, width=210, anchor="w").pack(side="left")
            ent = ctk.CTkEntry(row, width=500, corner_radius=10)
            ent.pack(side="left")
            ent.insert(0, str(default))
            return ent

        profile_row = ctk.CTkFrame(form, fg_color="transparent")
        profile_row.pack(fill="x", pady=5)
        ctk.CTkLabel(profile_row, text="Profil modèle:", width=210, anchor="w").pack(side="left")
        profile_var = ctk.StringVar(value=str(cfg.get("model_profile", "report_generation")))
        profile_menu = ctk.CTkOptionMenu(
            profile_row,
            variable=profile_var,
            values=HuggingFaceReportAssistant.get_profile_names(),
            width=500,
        )
        profile_menu.pack(side="left")

        apply_profile_row = ctk.CTkFrame(form, fg_color="transparent")
        apply_profile_row.pack(fill="x", pady=(2, 6))
        ctk.CTkLabel(apply_profile_row, text="", width=210).pack(side="left")
        apply_profile_var = ctk.BooleanVar(value=bool(cfg.get("apply_profile_defaults", True)))
        ctk.CTkSwitch(
            apply_profile_row,
            text="Appliquer les valeurs par défaut du profil",
            variable=apply_profile_var,
            progress_color=Colors.PRIMARY,
            button_color=Colors.PRIMARY,
            button_hover_color=Colors.PRIMARY_HOVER,
        ).pack(side="left")

        ent_chat_api = add_row("Chat API URL:", cfg.get("chat_api_url", "https://router.huggingface.co/v1/chat/completions"))
        ent_chat_model = add_row("Chat model principal:", cfg.get("chat_model", HuggingFaceReportAssistant.DEFAULT_CONFIG.get("chat_model", "")))
        ent_chat_fallbacks = add_row(
            "Fallback chat models:",
            ", ".join(cfg.get("fallback_chat_models", HuggingFaceReportAssistant.DEFAULT_CONFIG.get("fallback_chat_models", [])))
        )
        ent_url = add_row("API URL modèle:", cfg.get("api_url", ""))
        ent_env = add_row("Variable token:", cfg.get("token_env_var", "HF_API_TOKEN"))
        ent_token = add_row("Token direct (option):", cfg.get("api_token", ""))
        ent_max = add_row("max_new_tokens:", cfg.get("max_new_tokens", 900))
        ent_temp = add_row("temperature:", cfg.get("temperature", 0.2))
        ent_top_p = add_row("top_p:", cfg.get("top_p", 0.9))
        ent_timeout = add_row("timeout_sec:", cfg.get("timeout_sec", 120))

        ctk.CTkLabel(
            form,
            text=f"Fichier prompt: {self.hf_prompt_path}",
            text_color=Colors.TEXT_MUTED,
            font=ctk.CTkFont(size=11)
        ).pack(anchor="w", pady=(10, 0))
        ctk.CTkLabel(
            form,
            text="Définissez votre token dans l'environnement: export HF_API_TOKEN='...'",
            text_color=Colors.TEXT_MUTED,
            font=ctk.CTkFont(size=11)
        ).pack(anchor="w", pady=(3, 2))
        ctk.CTkLabel(
            form,
            text="Conseil: utiliser des modèles avec suffixe provider (ex: :hf-inference).",
            text_color=Colors.TEXT_MUTED,
            font=ctk.CTkFont(size=11)
        ).pack(anchor="w", pady=(0, 10))

        def save_cfg():
            try:
                fallback_chat_models = [
                    m.strip() for m in ent_chat_fallbacks.get().split(",") if m.strip()
                ]
                payload = dict(cfg)
                payload.update({
                    "model_profile": profile_var.get().strip() or "report_generation",
                    "apply_profile_defaults": bool(apply_profile_var.get()),
                    "use_chat_completions": True,
                    "chat_api_url": ent_chat_api.get().strip() or "https://router.huggingface.co/v1/chat/completions",
                    "chat_model": ent_chat_model.get().strip() or HuggingFaceReportAssistant.DEFAULT_CONFIG.get("chat_model", ""),
                    "fallback_chat_models": fallback_chat_models,
                    "api_url": ent_url.get().strip(),
                    "token_env_var": ent_env.get().strip() or "HF_API_TOKEN",
                    "api_token": ent_token.get().strip(),
                    "max_new_tokens": int(ent_max.get().strip()),
                    "temperature": float(ent_temp.get().strip()),
                    "top_p": float(ent_top_p.get().strip()),
                    "timeout_sec": int(ent_timeout.get().strip()),
                    "wait_for_model": True,
                })
                with open(self.hf_config_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2, ensure_ascii=False)
                win.destroy()
                self.show_message(f"✅ Configuration IA sauvegardée: {self.hf_config_path}", "success")
            except Exception as e:
                self.show_message(f"❌ Config IA invalide: {e}", "error")

        btn_row = ctk.CTkFrame(form, fg_color="transparent")
        btn_row.pack(fill="x", pady=(8, 0))
        ModernButton(btn_row, "💾 Enregistrer", save_cfg, style="success", width=140).pack(side="left")
        ModernButton(btn_row, "❌ Annuler", win.destroy, style="secondary", width=120).pack(side="left", padx=8)

    def select_ai_context_file(self):
        """Sélectionne un fichier de contexte métier pour le rapport IA."""
        path = filedialog.askopenfilename(
            title="Sélectionner un fichier de contexte IA",
            filetypes=[
                ("Fichiers texte", "*.txt *.md *.log *.rst"),
                ("JSON", "*.json"),
                ("CSV", "*.csv"),
                ("Tous", "*.*"),
            ]
        )
        if not path:
            return
        self.ai_context_path = path
        self.lbl_ai_context.configure(text=f"Contexte IA: {os.path.basename(path)}")
        self.show_message(f"✅ Fichier de contexte IA sélectionné: {path}", "success")

    def toggle_cover_video(self):
        """Démarre / met en pause la lecture vidéo."""
        if not self.cover_video_path:
            self._set_cover_video_status("Statut vidéo: fichier introuvable", "#FCA5A5")
            self.show_message(
                f"⚠️ Vidéo d'accueil introuvable: {self.default_cover_video_filename}",
                "warning"
            )
            return

        # Gestion pause/reprise quand un backend est déjà actif.
        if self.cover_video_backend == "gst" and self.gst_player is not None:
            try:
                import gi
                gi.require_version("Gst", "1.0")
                from gi.repository import Gst
                if self.cover_video_playing:
                    self.gst_player.set_state(Gst.State.PAUSED)
                    self.cover_video_playing = False
                    self._set_video_toggle_label(False)
                    self._set_cover_video_status("Statut vidéo: en pause (GStreamer)", "#FBBF24")
                else:
                    self.gst_player.set_state(Gst.State.PLAYING)
                    self.cover_video_playing = True
                    self._set_video_toggle_label(True)
                    self._set_cover_video_status("Statut vidéo: lecture active (GStreamer)", "#86EFAC")
                    self._poll_gst_bus()
            except Exception as e:
                self._set_cover_video_status("Statut vidéo: erreur reprise GStreamer", "#FCA5A5")
                self.show_message(f"⚠️ Reprise vidéo impossible: {e}", "warning")
            return

        if self.cover_video_playing:
            self.cover_video_playing = False
            self._set_video_toggle_label(False)
            if self.cover_video_backend == "cv2":
                self._set_cover_video_status("Statut vidéo: en pause (OpenCV)", "#FBBF24")
            return

        # Nettoyage visuel avant lancement.
        self._safe_configure(self.cover_video_label, text="", image=None)
        self.cover_video_label.image = None

        # Backend 1: OpenCV (si installé)
        if self._start_cv2_video():
            self._set_cover_video_status("Statut vidéo: lecture active (OpenCV)", "#86EFAC")
            return

        # Backend 2: GStreamer (fallback robuste sur Linux)
        if self._start_gst_video():
            self._set_cover_video_status("Statut vidéo: lecture active (GStreamer)", "#86EFAC")
            return

        self._set_cover_video_status(
            "Statut vidéo: impossible de lire MP4 (codec/dépendances manquants)",
            "#FCA5A5",
        )
        self.show_message(
            "❌ Lecture vidéo indisponible: installez 'opencv-python' ou un décodeur H264 GStreamer.",
            "error",
        )

    def stop_cover_video(self):
        """Arrête et libère la vidéo."""
        self.cover_video_playing = False
        if self.cover_video_job:
            try:
                self.after_cancel(self.cover_video_job)
            except Exception:
                pass
            self.cover_video_job = None

        if self.gst_poll_job:
            try:
                self.after_cancel(self.gst_poll_job)
            except Exception:
                pass
            self.gst_poll_job = None

        if self.cover_video_capture is not None:
            try:
                self.cover_video_capture.release()
            except Exception:
                pass
            self.cover_video_capture = None

        if self.gst_player is not None:
            try:
                import gi
                gi.require_version("Gst", "1.0")
                from gi.repository import Gst
                self.gst_player.set_state(Gst.State.NULL)
            except Exception:
                pass
            self.gst_player = None
            self.gst_video_sink = None
            self.gst_bus = None

        self.cover_video_backend = None
        self._set_video_toggle_label(False)
        self._set_cover_video_status("Statut vidéo: arrêtée", "#93C5FD")
        if hasattr(self, "cover_video_label") and not self._is_closing:
            self._safe_configure(
                self.cover_video_label,
                image=None,
                text=self._cover_video_placeholder_text()
            )
            self.cover_video_label.image = None

    def _start_cv2_video(self) -> bool:
        """Démarre la lecture via OpenCV si disponible."""
        try:
            import cv2
            from PIL import Image, ImageTk
        except Exception:
            return False

        if self.cover_video_capture is None:
            self.cover_video_capture = cv2.VideoCapture(self.cover_video_path)
            if not self.cover_video_capture or not self.cover_video_capture.isOpened():
                self.cover_video_capture = None
                return False

        self.cover_video_backend = "cv2"
        self.cover_video_playing = True
        self._set_video_toggle_label(True)

        def update_frame():
            if (
                self._is_closing
                or
                not self.cover_video_playing
                or self.cover_video_capture is None
                or self.cover_video_backend != "cv2"
            ):
                return
            ok, frame = self.cover_video_capture.read()
            if not ok:
                self.cover_video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self.cover_video_capture.read()
                if not ok:
                    self.stop_cover_video()
                    return

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            target_w = max(960, int(self.cover_video_label.winfo_width()))
            target_h = max(560, int(self.cover_video_label.winfo_height()))
            frame = cv2.resize(frame, (target_w, target_h))
            tk_img = ImageTk.PhotoImage(Image.fromarray(frame))
            if not self._safe_configure(self.cover_video_label, image=tk_img, text=""):
                self.cover_video_playing = False
                self.cover_video_job = None
                return
            self.cover_video_label.image = tk_img
            try:
                self.cover_video_job = self.after(40, update_frame)
            except tk.TclError:
                self.cover_video_job = None

        update_frame()
        return True

    def _start_gst_video(self) -> bool:
        """Démarre la lecture via GStreamer (fallback sans OpenCV)."""
        try:
            import gi
            gi.require_version("Gst", "1.0")
            gi.require_version("GstVideo", "1.0")
            from gi.repository import Gst, GstVideo

            Gst.init(None)
            self.stop_cover_video()

            # Sans décodeur H264, un MP4 H264 ne peut pas être lu.
            h264_decoders = [
                "avdec_h264",
                "openh264dec",
                "vaapih264dec",
                "v4l2h264dec",
                "nvh264dec",
            ]
            if not any(Gst.ElementFactory.find(name) is not None for name in h264_decoders):
                self._set_cover_video_status("Statut vidéo: codec H264 absent (GStreamer)", "#FCA5A5")
                return False

            self.gst_player = Gst.ElementFactory.make("playbin", "cover_player")
            if self.gst_player is None:
                return False

            sink = None
            for sink_name in ("glimagesink", "ximagesink", "autovideosink"):
                sink = Gst.ElementFactory.make(sink_name, None)
                if sink is not None:
                    break
            if sink is not None:
                self.gst_video_sink = sink
                self.gst_player.set_property("video-sink", sink)

            self.gst_player.set_property("uri", Path(self.cover_video_path).resolve().as_uri())
            self.gst_player.set_property("volume", 0.0)
            self.gst_bus = self.gst_player.get_bus()
            self._safe_configure(self.cover_video_label, text="", image=None)
            self.cover_video_label.image = None

            self.update_idletasks()
            xid = int(self.cover_video_label.winfo_id())
            if self.gst_video_sink is not None:
                try:
                    GstVideo.VideoOverlay.set_window_handle(self.gst_video_sink, xid)
                except Exception:
                    pass

            self.gst_player.set_state(Gst.State.PLAYING)
            self.cover_video_backend = "gst"
            self.cover_video_playing = True
            self._set_video_toggle_label(True)
            self._poll_gst_bus()
            return True
        except Exception:
            return False

    def _poll_gst_bus(self):
        """Boucle de surveillance GStreamer (EOS/erreurs)."""
        if (
            self._is_closing
            or not self.cover_video_playing
            or self.cover_video_backend != "gst"
            or self.gst_player is None
        ):
            return
        try:
            import gi
            gi.require_version("Gst", "1.0")
            from gi.repository import Gst

            if self.gst_bus is None:
                self.gst_bus = self.gst_player.get_bus()

            msg_mask = (
                Gst.MessageType.ERROR
                | Gst.MessageType.EOS
                | Gst.MessageType.WARNING
            )
            while True:
                msg = self.gst_bus.timed_pop_filtered(0, msg_mask)
                if msg is None:
                    break
                if msg.type == Gst.MessageType.EOS:
                    self.gst_player.seek_simple(
                        Gst.Format.TIME,
                        Gst.SeekFlags.FLUSH | Gst.SeekFlags.KEY_UNIT,
                        0,
                    )
                    self.gst_player.set_state(Gst.State.PLAYING)
                elif msg.type == Gst.MessageType.ERROR:
                    err, _ = msg.parse_error()
                    self.stop_cover_video()
                    self._set_cover_video_status("Statut vidéo: erreur GStreamer", "#FCA5A5")
                    self.show_message(f"❌ Erreur lecture vidéo (GStreamer): {err}", "error")
                    return

            if not self._is_closing:
                try:
                    self.gst_poll_job = self.after(120, self._poll_gst_bus)
                except tk.TclError:
                    self.gst_poll_job = None
        except Exception:
            # Si la surveillance échoue, on retente doucement.
            if not self._is_closing:
                try:
                    self.gst_poll_job = self.after(250, self._poll_gst_bus)
                except tk.TclError:
                    self.gst_poll_job = None

    def _ask_csv_options(self) -> Optional[Dict[str, Any]]:
        """Dialogue de paramètres CSV avancés."""
        dialog = ctk.CTkToplevel(self)
        dialog.title("Options CSV avancées")
        dialog.geometry("640x440")
        dialog.transient(self)

        result = {"value": None}
        body = ctk.CTkFrame(dialog, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=20, pady=15)

        ctk.CTkLabel(
            body,
            text="Personnalisez l'import CSV (séparateur, encodage, décimal, NA, header).",
            font=ctk.CTkFont(size=12),
            text_color=Colors.TEXT_SECONDARY
        ).pack(anchor="w", pady=(0, 12))

        sep_var = ctk.StringVar(value="Auto")
        enc_var = ctk.StringVar(value="utf-8")
        decimal_var = ctk.StringVar(value="Point (.)")
        quote_var = ctk.StringVar(value='"')
        na_var = ctk.StringVar(value="")
        header_var = ctk.StringVar(value="Oui (ligne 1)")

        def row_option(label: str, widget_factory):
            row = ctk.CTkFrame(body, fg_color="transparent")
            row.pack(fill="x", pady=6)
            ctk.CTkLabel(row, text=label, width=210, anchor="w").pack(side="left")
            widget = widget_factory(row)
            widget.pack(side="left")

        row_option("Séparateur:", lambda parent: ctk.CTkOptionMenu(
            parent, variable=sep_var,
            values=["Auto", "Virgule (,)", "Point-virgule (;)", "Tabulation (\\t)", "Pipe (|)"],
            width=260
        ))
        row_option("Encodage:", lambda parent: ctk.CTkOptionMenu(
            parent, variable=enc_var,
            values=["utf-8", "utf-8-sig", "latin-1", "cp1252"],
            width=260
        ))
        row_option("Décimal:", lambda parent: ctk.CTkOptionMenu(
            parent, variable=decimal_var,
            values=["Point (.)", "Virgule (,)"],
            width=260
        ))
        row_option("Quote char:", lambda parent: ctk.CTkEntry(parent, textvariable=quote_var, width=120))
        row_option("Valeurs NA custom:", lambda parent: ctk.CTkEntry(
            parent, textvariable=na_var, width=360, placeholder_text="Ex: NA,null,missing"
        ))
        row_option("Header:", lambda parent: ctk.CTkOptionMenu(
            parent, variable=header_var,
            values=["Oui (ligne 1)", "Non (pas d'en-tête)"],
            width=260
        ))

        def confirm():
            sep_map = {
                "Auto": None,
                "Virgule (,)": ",",
                "Point-virgule (;)": ";",
                "Tabulation (\\t)": "\t",
                "Pipe (|)": "|",
            }
            decimal_map = {"Point (.)": ".", "Virgule (,)": ","}
            opts: Dict[str, Any] = {
                "encoding": enc_var.get().strip(),
                "decimal": decimal_map.get(decimal_var.get(), "."),
                "quotechar": (quote_var.get() or '"')[0],
                "header": 0 if header_var.get().startswith("Oui") else None,
            }
            sep_value = sep_map.get(sep_var.get())
            if sep_value is not None:
                opts["sep"] = sep_value
            else:
                opts["sep"] = None
                opts["engine"] = "python"

            na_tokens = [v.strip() for v in na_var.get().split(",") if v.strip()]
            if na_tokens:
                opts["na_values"] = na_tokens

            result["value"] = opts
            dialog.destroy()

        btns = ctk.CTkFrame(body, fg_color="transparent")
        btns.pack(fill="x", pady=(15, 0))
        ModernButton(btns, "✅ Importer", confirm, style="success", width=140).pack(side="left")
        ModernButton(btns, "❌ Annuler", dialog.destroy, style="secondary", width=120).pack(side="left", padx=8)

        self.wait_window(dialog)
        return result["value"]

    def _ask_file_options(self, source_type: str) -> Optional[Dict[str, Any]]:
        """Retourne les options de lecture selon le type de source."""
        if source_type == "CSV":
            return self._ask_csv_options()
        return {}

    def apply_filters(self):
        """Applique les filtres avancés sur la table source."""
        if self.raw_df is None:
            return self.show_error()

        df = self.raw_df.copy()
        applied: List[str] = []

        text_q = self.filter_search_var.get().strip()
        if text_q:
            str_cols = list(df.select_dtypes(include=["object", "string", "category"]).columns)
            if str_cols:
                mask = np.zeros(len(df), dtype=bool)
                for col in str_cols:
                    mask |= df[col].astype(str).str.contains(text_q, case=False, na=False)
                df = df[mask]
                applied.append(f"texte='{text_q}'")

        num_col = self.filter_num_col_var.get()
        min_raw = self.filter_min_var.get().strip()
        max_raw = self.filter_max_var.get().strip()
        if num_col in df.columns and (min_raw or max_raw):
            try:
                series = pd.to_numeric(df[num_col], errors="coerce")
                if min_raw:
                    df = df[series >= float(min_raw)]
                    applied.append(f"{num_col}>={min_raw}")
                if max_raw:
                    series = pd.to_numeric(df[num_col], errors="coerce")
                    df = df[series <= float(max_raw)]
                    applied.append(f"{num_col}<={max_raw}")
            except ValueError:
                self.show_message("❌ Min/Max numérique invalide", "error")
                return

        cat_col = self.filter_cat_col_var.get()
        cat_values = [v.strip() for v in self.filter_cat_values_var.get().split(",") if v.strip()]
        if cat_col in df.columns and cat_values:
            norm = {v.lower() for v in cat_values}
            df = df[df[cat_col].astype(str).str.lower().isin(norm)]
            applied.append(f"{cat_col} in {cat_values}")

        if self.filter_exclude_na.get():
            before = len(df)
            df = df.dropna()
            removed = before - len(df)
            applied.append(f"drop_na({removed} lignes)")

        self.df = df.reset_index(drop=True)
        self.active_filters = applied
        self.detect_types()
        self.update_combos()
        self._update_dataset_info()
        self._refresh_preview_table()
        self.show_message(
            f"✅ Filtres appliqués ({len(applied)}).\nLignes: {self.df.shape[0]:,} / {self.raw_df.shape[0]:,}",
            "success"
        )

    def clear_filters(self):
        """Efface les filtres et restaure le dataset original."""
        if self.raw_df is None:
            return self.show_error()

        self.df = self.raw_df.copy()
        self.active_filters = []
        self.filter_search_var.set("")
        self.filter_min_var.set("")
        self.filter_max_var.set("")
        self.filter_cat_values_var.set("")
        self.filter_exclude_na.set(False)
        self.detect_types()
        self.update_combos()
        self._update_dataset_info()
        self._refresh_preview_table()
        self.show_message(f"✅ Filtres effacés. Dataset restauré ({len(self.df):,} lignes).", "success")

    def preview_filtered_data(self):
        """Aperçu rapide de la table filtrée."""
        if self.df is None:
            return self.show_error()
        self.clear_results()
        self.print_header("🔎 APERÇU TABLE FILTRÉE")
        self.txt_results.insert("end", f"\nLignes: {self.df.shape[0]:,} | Colonnes: {self.df.shape[1]}\n")
        self.txt_results.insert("end", f"Filtres: {', '.join(self.active_filters) if self.active_filters else 'Aucun'}\n")
        self._insert_dataframe_table("Aperçu (head 20)", self.df.head(20), max_rows=20)
        
    # ========================================================================
    # GESTION DES DONNÉES
    # ========================================================================
    
    def load_data(self):
        """Charge un fichier"""
        source_type = self.source_var.get()
        
        filetypes = {
            'CSV': [('Fichiers CSV', '*.csv')],
            'Excel': [('Fichiers Excel', '*.xlsx *.xls')],
            'JSON': [('Fichiers JSON', '*.json')],
            'Parquet': [('Fichiers Parquet', '*.parquet')],
            'Stata': [('Fichiers Stata', '*.dta')],
            'SAS': [('Fichiers SAS', '*.sas7bdat')],
            'SPSS': [('Fichiers SPSS', '*.sav *.por')]
        }
        
        file_path = filedialog.askopenfilename(
            title=f"Sélectionner un fichier {source_type}",
            filetypes=filetypes.get(source_type, [('Tous', '*.*')])
        )
        
        if not file_path:
            return
            
        try:
            load_options = self._ask_file_options(source_type)
            if load_options is None:
                return

            self.df, message = DataSourceManager.load_file(file_path, **load_options)
            self.raw_df = self.df.copy()
            self.active_filters = []
            self.file_path = file_path
            
            # Détection types
            self.detect_types()
            self.update_combos()
            self._update_dataset_info()
            self._refresh_preview_table()
            
            self.show_message(f"✅ {message}\n{self.df.shape[0]:,} observations, {self.df.shape[1]} variables", "success")
            self.open_workspace_tab("Diagnostic")
            
        except Exception as e:
            self.show_message(f"❌ Erreur: {str(e)}", "error")
            
    def load_from_api(self):
        """Charge depuis API"""
        dialog = ctk.CTkToplevel(self)
        dialog.title("Charger depuis API")
        dialog.geometry("500x150")
        dialog.transient(self)
        
        ctk.CTkLabel(dialog, text="URL de l'API:", font=ctk.CTkFont(size=12)).pack(pady=10)
        url_entry = ctk.CTkEntry(dialog, width=450, corner_radius=10)
        url_entry.pack(pady=5)
        url_entry.insert(0, "https://jsonplaceholder.typicode.com/users")
        
        def load():
            try:
                self.df, message = DataSourceManager.load_from_api(url_entry.get())
                self.raw_df = self.df.copy()
                self.active_filters = []
                self.file_path = url_entry.get()
                self.detect_types()
                self.update_combos()
                self._update_dataset_info()
                self._refresh_preview_table()
                
                dialog.destroy()
                self.show_message(f"✅ {message}", "success")
                self.open_workspace_tab("Diagnostic")
            except Exception as e:
                self.show_message(f"❌ Erreur: {str(e)}", "error")
                
        ModernButton(dialog, "Charger", load).pack(pady=15)
        
    def load_from_sql(self):
        """Charge depuis SQL"""
        dialog = ctk.CTkToplevel(self)
        dialog.title("Charger depuis SQL")
        dialog.geometry("500x200")
        dialog.transient(self)
        
        ctk.CTkLabel(dialog, text="Connection String:", font=ctk.CTkFont(size=12)).pack(pady=5)
        conn_entry = ctk.CTkEntry(dialog, width=450, corner_radius=10)
        conn_entry.pack(pady=5)
        
        ctk.CTkLabel(dialog, text="Requête SQL:", font=ctk.CTkFont(size=12)).pack(pady=5)
        query_entry = ctk.CTkEntry(dialog, width=450, corner_radius=10)
        query_entry.pack(pady=5)
        query_entry.insert(0, "SELECT * FROM table_name LIMIT 1000")
        
        def load():
            try:
                import sqlalchemy
                engine = sqlalchemy.create_engine(conn_entry.get())
                self.df = pd.read_sql(query_entry.get(), engine)
                self.df = DataSourceManager.optimize_memory(self.df)
                self.raw_df = self.df.copy()
                self.active_filters = []
                self.file_path = "SQL Query"
                self.detect_types()
                self.update_combos()
                self._update_dataset_info()
                self._refresh_preview_table()
                
                dialog.destroy()
                self.show_message("✅ Données SQL chargées", "success")
                self.open_workspace_tab("Diagnostic")
            except Exception as e:
                self.show_message(f"❌ Erreur: {str(e)}", "error")
                
        ModernButton(dialog, "Charger", load).pack(pady=10)
        
    def detect_types(self):
        """Détecte les types de variables"""
        self.numeric_cols = []
        self.categorical_cols = []
        self.boolean_cols = []
        
        for col in self.df.columns:
            dtype = self.df[col].dtype
            n_unique = self.df[col].nunique()
            
            if dtype == 'bool' or (n_unique == 2 and set(self.df[col].dropna().unique()) <= {0, 1, True, False}):
                self.boolean_cols.append(col)
            elif pd.api.types.is_numeric_dtype(dtype):
                self.numeric_cols.append(col)
            else:
                self.categorical_cols.append(col)
        
        self.lbl_numeric.configure(text=f"🔢 {len(self.numeric_cols)}")
        self.lbl_categorical.configure(text=f"📝 {len(self.categorical_cols)}")
        self.lbl_bool.configure(text=f"✅ {len(self.boolean_cols)}")
        
    def update_combos(self):
        """Met à jour les listes déroulantes"""
        all_cols = list(self.df.columns)
        cat_cols = self.categorical_cols + self.boolean_cols

        def set_combo(combo, values, variable, default_index=0):
            safe_values = [str(v) for v in values] if values else [""]
            combo.configure(values=safe_values)
            if values:
                variable.set(safe_values[min(default_index, len(safe_values) - 1)])
            else:
                variable.set("")
        
        # Stats
        set_combo(self.combo_stats, all_cols, self.var_stats)
        
        # Multivarié
        set_combo(self.combo_num1, self.numeric_cols, self.var_num1, 0)
        set_combo(self.combo_num2, self.numeric_cols, self.var_num2, 1 if len(self.numeric_cols) > 1 else 0)
        
        set_combo(self.combo_cat, cat_cols, self.var_cat, 0)
        set_combo(self.combo_group, cat_cols, self.var_group, 1 if len(cat_cols) > 1 else 0)
        
        # Viz
        set_combo(self.combo_viz1, all_cols, self.var_viz, 0)
        set_combo(self.combo_viz2, all_cols, self.var_viz2, 1 if len(all_cols) > 1 else 0)

        # Filtres
        if hasattr(self, "combo_filter_num"):
            set_combo(self.combo_filter_num, self.numeric_cols, self.filter_num_col_var, 0)
        if hasattr(self, "combo_filter_cat"):
            set_combo(self.combo_filter_cat, cat_cols, self.filter_cat_col_var, 0)
        
    def reset_app(self):
        """Réinitialise l'application"""
        self.df = None
        self.raw_df = None
        self.file_path = None
        self.numeric_cols = []
        self.categorical_cols = []
        self.boolean_cols = []
        self.active_filters = []
        
        self._update_dataset_info()
        self._refresh_preview_table()
        self.lbl_numeric.configure(text="🔢 0")
        self.lbl_categorical.configure(text="📝 0")
        self.lbl_bool.configure(text="✅ 0")
        
        for combo in [self.combo_stats, self.combo_num1, self.combo_num2, self.combo_cat,
                      self.combo_group, self.combo_viz1, self.combo_viz2]:
            combo.configure(values=[""])
        
        for var in [self.var_stats, self.var_num1, self.var_num2, self.var_cat,
                    self.var_group, self.var_viz, self.var_viz2]:
            var.set("")

        if hasattr(self, "filter_search_var"):
            self.filter_search_var.set("")
            self.filter_min_var.set("")
            self.filter_max_var.set("")
            self.filter_cat_values_var.set("")
            self.filter_exclude_na.set(False)
        
        self.show_welcome()
        
    def change_theme(self, value):
        """Change le thème"""
        ctk.set_appearance_mode(value)
        
    def copy_results(self):
        """Copie les résultats"""
        self.clipboard_clear()
        self.clipboard_append(self.txt_results.get("1.0", "end"))
        self.show_message("📋 Résultats copiés!", "success")
        
    # ========================================================================
    # DIAGNOSTIC
    # ========================================================================
    
    def analyze_missing(self):
        """Analyse des valeurs manquantes"""
        if self.df is None:
            return self.show_error()
            
        self.clear_results()
        self.print_header("📊 ANALYSE DES VALEURS MANQUANTES")
        
        missing = self.df.isnull().sum()
        pct = (missing / len(self.df)) * 100
        
        self.txt_results.insert("end", f"\n{'Variable':<30} {'Manquantes':>12} {'%':>10} {'Statut':<12}\n")
        self.txt_results.insert("end", "─" * 70 + "\n")
        
        critical = []
        for col in self.df.columns:
            n_miss = missing[col]
            p = pct[col]
            
            if p > 30:
                status = "🔴 CRITIQUE"
                critical.append(col)
            elif p > 10:
                status = "🟡 ATTENTION"
                critical.append(col)
            elif p > 0:
                status = "🟢 OK"
            else:
                status = "✅ Complet"
                
            self.txt_results.insert("end", f"{col:<30} {n_miss:>12,} {p:>9.2f}% {status}\n")
        
        total_cells = self.df.shape[0] * self.df.shape[1]
        total_missing = missing.sum()
        
        self.txt_results.insert("end", f"\n{'═' * 70}\n")
        self.txt_results.insert("end", f"Total: {total_missing:,} / {total_cells:,} ({total_missing/total_cells*100:.2f}%)\n")
        
    def detect_problems(self):
        """Détecte les variables problématiques"""
        if self.df is None:
            return self.show_error()
            
        self.clear_results()
        self.print_header("⚠️ VARIABLES PROBLÉMATIQUES")
        
        # Manquantes > 30%
        self.txt_results.insert("end", "\n🔴 MANQUANTES > 30%:\n")
        for col in self.df.columns:
            pct = self.df[col].isnull().sum() / len(self.df) * 100
            if pct > 30:
                self.txt_results.insert("end", f"   • {col}: {pct:.1f}%\n")
        
        # Quasi constantes
        self.txt_results.insert("end", "\n🟠 QUASI CONSTANTES > 95%:\n")
        for col in self.df.columns:
            top_freq = self.df[col].value_counts(normalize=True, dropna=False).iloc[0]
            if top_freq > 0.95:
                self.txt_results.insert("end", f"   • {col}: {top_freq*100:.1f}% même valeur\n")
        
        # Haute cardinalité
        self.txt_results.insert("end", "\n🟡 HAUTE CARDINALITÉ > 50:\n")
        for col in self.categorical_cols:
            n = self.df[col].nunique()
            if n > 50:
                self.txt_results.insert("end", f"   • {col}: {n} valeurs uniques\n")
                
    def detect_outliers(self):
        """Détecte les outliers"""
        if self.df is None:
            return self.show_error()
        if not self.numeric_cols:
            self.clear_results()
            self.txt_results.insert("end", "❌ Aucune variable numérique")
            return
            
        self.clear_results()
        self.print_header("🎯 OUTLIERS (IQR)")
        
        total = 0
        for col in self.numeric_cols:
            data = self.df[col].dropna()
            Q1, Q3 = data.quantile([0.25, 0.75])
            IQR = Q3 - Q1
            lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
            
            outliers = data[(data < lower) | (data > upper)]
            n_out = len(outliers)
            
            if n_out > 0:
                total += n_out
                self.txt_results.insert("end", f"\n📌 {col}: {n_out:,} outliers ({n_out/len(data)*100:.1f}%)\n")
                self.txt_results.insert("end", f"   Bornes: [{lower:.2f}, {upper:.2f}]\n")
        
        self.txt_results.insert("end", f"\n📊 Total: {total:,} outliers\n")
        
    def detect_duplicates(self):
        """Analyse des doublons"""
        if self.df is None:
            return self.show_error()
            
        self.clear_results()
        self.print_header("🔄 DOUBLONS")
        
        n_dup = self.df.duplicated().sum()
        self.txt_results.insert("end", f"\n📊 RÉSULTATS:\n")
        self.txt_results.insert("end", f"   • Lignes: {len(self.df):,}\n")
        self.txt_results.insert("end", f"   • Doublons: {n_dup:,} ({n_dup/len(self.df)*100:.2f}%)\n")
        self.txt_results.insert("end", f"   • Uniques: {len(self.df.drop_duplicates()):,}\n")
        
    def full_diagnostic(self):
        """Diagnostic complet"""
        if self.df is None:
            return self.show_error()
            
        self.clear_results()
        self.print_header("📋 DIAGNOSTIC COMPLET")
        
        total_cells = self.df.shape[0] * self.df.shape[1]
        total_missing = self.df.isnull().sum().sum()
        
        self.txt_results.insert("end", f"\n📊 GÉNÉRAL:\n")
        self.txt_results.insert("end", f"   • Dimensions: {self.df.shape[0]:,} × {self.df.shape[1]}\n")
        self.txt_results.insert("end", f"   • Mémoire: {self.df.memory_usage(deep=True).sum()/1024**2:.2f} MB\n")
        
        self.txt_results.insert("end", f"\n🔍 QUALITÉ:\n")
        self.txt_results.insert("end", f"   • Manquantes: {total_missing:,} ({total_missing/total_cells*100:.2f}%)\n")
        self.txt_results.insert("end", f"   • Doublons: {self.df.duplicated().sum():,}\n")
        self.txt_results.insert("end", f"   • Lignes complètes: {len(self.df.dropna()):,}\n")
        
    # ========================================================================
    # STATISTIQUES
    # ========================================================================
    
    def numeric_stats(self):
        """Statistiques numériques"""
        if self.df is None:
            return self.show_error()
            
        var = self.var_stats.get()
        if var not in self.numeric_cols:
            var = self.numeric_cols[0] if self.numeric_cols else None
            
        if not var:
            return
            
        data = self.df[var].dropna()
        
        self.clear_results()
        self.print_header(f"📊 STATISTIQUES: {var}")
        
        self.txt_results.insert("end", f"\n📍 TENDANCE CENTRALE:\n")
        self.txt_results.insert("end", f"   • Moyenne:  {data.mean():.4f}\n")
        self.txt_results.insert("end", f"   • Médiane:  {data.median():.4f}\n")
        
        self.txt_results.insert("end", f"\n📏 DISPERSION:\n")
        self.txt_results.insert("end", f"   • Min:      {data.min():.4f}\n")
        self.txt_results.insert("end", f"   • Max:      {data.max():.4f}\n")
        self.txt_results.insert("end", f"   • Std:      {data.std():.4f}\n")
        self.txt_results.insert("end", f"   • CV:       {data.std()/data.mean()*100:.2f}%\n")
        
        self.txt_results.insert("end", f"\n📐 FORME:\n")
        skew = stats.skew(data)
        kurt = stats.kurtosis(data)
        self.txt_results.insert("end", f"   • Skewness: {skew:.4f}\n")
        self.txt_results.insert("end", f"   • Kurtosis: {kurt:.4f}\n")
        
        if len(data) >= 8:
            stat, p = stats.shapiro(data.sample(min(5000, len(data))))
            normal = "✅ Normale" if p > 0.05 else "❌ Non normale"
            self.txt_results.insert("end", f"\n🧪 NORMALITÉ (Shapiro-Wilk):\n")
            self.txt_results.insert("end", f"   • p-value: {p:.4f}\n")
            self.txt_results.insert("end", f"   • Conclusion: {normal}\n")
            
    def categorical_stats(self):
        """Statistiques catégorielles"""
        if self.df is None:
            return self.show_error()
            
        var = self.var_stats.get()
        if var not in (self.categorical_cols + self.boolean_cols):
            var = self.categorical_cols[0] if self.categorical_cols else None
            
        if not var:
            return
            
        data = self.df[var].dropna()
        counts = data.value_counts()
        
        self.clear_results()
        self.print_header(f"📋 CATÉGORIELLE: {var}")
        
        self.txt_results.insert("end", f"\n📊 INFOS:\n")
        self.txt_results.insert("end", f"   • Modalités: {len(counts)}\n")
        self.txt_results.insert("end", f"   • Dominante: {counts.index[0]} ({counts.iloc[0]/len(data)*100:.1f}%)\n")
        
        self.txt_results.insert("end", f"\n📋 DISTRIBUTION:\n")
        for val, count in counts.head(15).items():
            pct = count/len(data)*100
            self.txt_results.insert("end", f"   • {str(val)[:25]}: {count:,} ({pct:.1f}%)\n")
            
    def full_summary(self):
        """Résumé complet"""
        if self.df is None:
            return self.show_error()
            
        self.clear_results()
        self.print_header("📑 RÉSUMÉ COMPLET")
        
        self.txt_results.insert("end", f"\n{'Variable':<20} {'Type':<12} {'Manquantes':>10} {'Uniques':>10}\n")
        self.txt_results.insert("end", "─" * 55 + "\n")
        
        for col in self.df.columns:
            dtype = 'Num' if col in self.numeric_cols else ('Cat' if col in self.categorical_cols else 'Bool')
            miss = self.df[col].isnull().sum()
            uniq = self.df[col].nunique()
            self.txt_results.insert("end", f"{col[:20]:<20} {dtype:<12} {miss:>10,} {uniq:>10,}\n")
            
    def all_vars_stats(self):
        """Toutes les variables"""
        if self.df is None:
            return self.show_error()
        if not self.numeric_cols:
            return
            
        self.clear_results()
        self.print_header("📊 TOUTES VARIABLES NUMÉRIQUES")
        
        desc = self.df[self.numeric_cols].describe().T
        
        self.txt_results.insert("end", f"\n{'Variable':<15} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}\n")
        self.txt_results.insert("end", "─" * 60 + "\n")
        
        for col in self.numeric_cols:
            if col in desc.index:
                row = desc.loc[col]
                self.txt_results.insert("end", f"{col[:15]:<15} {row['mean']:>10.2f} {row['std']:>10.2f} {row['min']:>10.2f} {row['max']:>10.2f}\n")
                
    # ========================================================================
    # MULTIVARIÉ
    # ========================================================================
    
    def run_anova(self):
        """ANOVA"""
        if self.df is None:
            return self.show_error()
            
        cat_cols = self.categorical_cols + self.boolean_cols
        num_col = self.var_num1.get()
        group_col = self.var_group.get()
        
        if num_col not in self.numeric_cols:
            num_col = self.numeric_cols[0] if self.numeric_cols else ""
        if group_col not in cat_cols:
            group_col = cat_cols[0] if cat_cols else ""
        
        if not num_col or not group_col:
            self.show_message("❌ Sélectionnez les variables", "error")
            return
            
        result = MultivariateAnalysis.anova(self.df, num_col, group_col)
        
        self.clear_results()
        self.print_header(f"📊 ANOVA: {num_col} par {group_col}")
        
        if 'error' in result:
            self.txt_results.insert("end", f"\n❌ {result['error']}\n")
            return
            
        sig_text = "Significative" if result['significant'] else "Non significative"
        self.txt_results.insert("end", f"\n🔬 RÉSULTATS:\n")
        self.txt_results.insert("end", f"   • F-statistique: {result['f_statistic']:.4f}\n")
        self.txt_results.insert("end", f"   • p-value: {result['p_value']:.4e}\n")
        self.txt_results.insert("end", f"   • η²: {result['eta_squared']:.4f}\n")
        self.txt_results.insert("end", f"   • Conclusion: {sig_text}\n")
        
        data = self.df[[group_col, num_col]].dropna().copy()
        data[group_col] = data[group_col].astype(str)
        group_stats_df = data.groupby(group_col)[num_col].agg(
            n='count', moyenne='mean', ecart_type='std', mediane='median', min='min', max='max'
        ).sort_values("moyenne", ascending=False)
        self._insert_dataframe_table("Statistiques par groupe", group_stats_df.round(4))
        self._show_table_window(group_stats_df.round(4), f"ANOVA - Tableau ({num_col} ~ {group_col})")
        
        top_groups = data[group_col].value_counts().head(12).index
        plot_data = data[data[group_col].isin(top_groups)].copy()
        
        fig, ax = plt.subplots(figsize=(11, 6))
        sns.boxplot(data=plot_data, x=group_col, y=num_col, ax=ax, color=Colors.PRIMARY)
        sample_n = min(2000, len(plot_data))
        if sample_n > 0:
            sns.stripplot(
                data=plot_data.sample(sample_n, random_state=42),
                x=group_col, y=num_col, ax=ax,
                color="white", alpha=0.35, size=2
            )
        ax.set_title(f"ANOVA - Distribution de {num_col} par {group_col}")
        ax.set_xlabel(group_col)
        ax.set_ylabel(num_col)
        ax.tick_params(axis='x', rotation=40)
        plt.tight_layout()
        self._show_plot(fig, f"ANOVA - {num_col} ~ {group_col}")
        
    def run_ttest(self):
        """Test t"""
        if self.df is None:
            return self.show_error()
            
        cat_cols = self.categorical_cols + self.boolean_cols
        num_col = self.var_num1.get()
        group_col = self.var_group.get()
        
        if num_col not in self.numeric_cols:
            num_col = self.numeric_cols[0] if self.numeric_cols else ""
        if group_col not in cat_cols:
            group_col = cat_cols[0] if cat_cols else ""
        
        if not num_col or not group_col:
            self.show_message("❌ Sélectionnez les variables", "error")
            return
            
        from scipy.stats import ttest_ind, levene
        
        groups = self.df[[group_col, num_col]].dropna().groupby(group_col)[num_col].apply(list).to_dict()
        if len(groups) != 2:
            self.show_message("❌ Exactement 2 groupes nécessaires", "error")
            return
            
        g1, g2 = list(groups.values())
        
        self.clear_results()
        self.print_header(f"🔢 TEST t: {num_col}")
        
        # Test de Levene
        levene_stat, levene_p = levene(g1, g2)
        equal_var = levene_p > 0.05
        
        t_stat, p_value = ttest_ind(g1, g2, equal_var=equal_var)
        
        sig_text = "Significative" if p_value < 0.05 else "Non significative"
        mean_diff = np.mean(g1) - np.mean(g2)
        pooled_std = np.sqrt(((np.std(g1, ddof=1) ** 2) + (np.std(g2, ddof=1) ** 2)) / 2)
        cohens_d = (mean_diff / pooled_std) if pooled_std > 0 else 0
        
        self.txt_results.insert("end", f"\n🔬 RÉSULTATS:\n")
        self.txt_results.insert("end", f"   • t-statistique: {t_stat:.4f}\n")
        self.txt_results.insert("end", f"   • p-value: {p_value:.4e}\n")
        self.txt_results.insert("end", f"   • Égalité var: {'Oui' if equal_var else 'Non'}\n")
        self.txt_results.insert("end", f"   • Différence moyennes: {mean_diff:.4f}\n")
        self.txt_results.insert("end", f"   • Taille d'effet (Cohen d): {cohens_d:.4f}\n")
        self.txt_results.insert("end", f"   • Conclusion: {sig_text}\n")
        
        names = [str(k) for k in groups.keys()]
        stats_table = pd.DataFrame({
            "groupe": names,
            "n": [len(g1), len(g2)],
            "moyenne": [np.mean(g1), np.mean(g2)],
            "ecart_type": [np.std(g1, ddof=1), np.std(g2, ddof=1)],
            "mediane": [np.median(g1), np.median(g2)],
            "min": [np.min(g1), np.min(g2)],
            "max": [np.max(g1), np.max(g2)],
        }).set_index("groupe")
        self._insert_dataframe_table("Tableau comparatif des groupes", stats_table.round(4))
        self._show_table_window(stats_table.round(4), f"Test t - Tableau ({num_col} ~ {group_col})")
        
        data = self.df[[group_col, num_col]].dropna().copy()
        allowed = list(groups.keys())
        data = data[data[group_col].isin(allowed)]
        data[group_col] = data[group_col].astype(str)
        
        fig, ax = plt.subplots(figsize=(9.5, 6))
        sns.boxplot(data=data, x=group_col, y=num_col, ax=ax, color=Colors.INFO)
        sample_n = min(1500, len(data))
        if sample_n > 0:
            sns.stripplot(
                data=data.sample(sample_n, random_state=42),
                x=group_col, y=num_col, ax=ax,
                color="white", alpha=0.35, size=2
            )
        ax.set_title(f"Test t - {num_col} par {group_col}")
        ax.set_xlabel(group_col)
        ax.set_ylabel(num_col)
        plt.tight_layout()
        self._show_plot(fig, f"Test t - {num_col} ~ {group_col}")
        
    def run_correlations(self):
        """Corrélations"""
        if self.df is None:
            return self.show_error()
        if len(self.numeric_cols) < 2:
            self.show_message("❌ Au moins 2 variables numériques nécessaires", "error")
            return
            
        corr = MultivariateAnalysis.correlation_matrix(self.df)
        if corr.empty:
            self.show_message("❌ Matrice de corrélation indisponible", "error")
            return
        
        self.clear_results()
        self.print_header("📈 MATRICE DE CORRÉLATION")
        
        # Header
        header = f"{'':>12}"
        for col in corr.columns[:8]:
            header += f"{col[:8]:>10}"
        self.txt_results.insert("end", header + "\n")
        
        for idx, row in corr.iterrows():
            line = f"{idx[:12]:>12}"
            for val in row[:8]:
                line += f"{val:>10.3f}"
            self.txt_results.insert("end", line + "\n")
        
        # Fortes corrélations
        self.txt_results.insert("end", f"\n🔴 CORRÉLATIONS FORTES (|r| > 0.7):\n")
        for i, col1 in enumerate(corr.columns):
            for j, col2 in enumerate(corr.columns):
                if i < j:
                    r = corr.loc[col1, col2]
                    if abs(r) > 0.7:
                        self.txt_results.insert("end", f"   • {col1} vs {col2}: r = {r:.3f}\n")
        
        corr_full = corr.round(4)
        self._insert_dataframe_table("Matrice complète (aperçu)", corr_full, max_rows=20)
        self._show_table_window(corr_full, "Corrélations - Matrice complète")
        
        corr_plot = corr_full.iloc[:20, :20]
        fig, ax = plt.subplots(figsize=(11, 9))
        sns.heatmap(corr_plot, annot=True, fmt=".2f", cmap="RdBu_r", center=0, square=True, ax=ax)
        ax.set_title("Heatmap de corrélation (max 20 variables)")
        plt.tight_layout()
        self._show_plot(fig, "Corrélations - Heatmap")
                        
    def run_chi2(self):
        """Chi-carré"""
        if self.df is None:
            return self.show_error()
            
        cat_cols = self.categorical_cols + self.boolean_cols
        col1 = self.var_cat.get()
        col2 = self.var_group.get()
        
        if col1 not in cat_cols:
            col1 = cat_cols[0] if cat_cols else ""
        if col2 not in cat_cols:
            col2 = cat_cols[1] if len(cat_cols) > 1 else (cat_cols[0] if cat_cols else "")
        
        if not col1 or not col2:
            self.show_message("❌ Sélectionnez les variables catégorielles", "error")
            return
        if col1 == col2:
            self.show_message("❌ Choisissez 2 variables différentes", "error")
            return
            
        from scipy.stats import chi2_contingency
        
        contingency = pd.crosstab(self.df[col1], self.df[col2])
        chi2, p, dof, expected = chi2_contingency(contingency)
        
        n = contingency.sum().sum()
        min_dim = min(contingency.shape[0] - 1, contingency.shape[1] - 1)
        cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0
        
        self.clear_results()
        self.print_header(f"🎯 CHI²: {col1} × {col2}")
        
        sig_text = "Significative" if p < 0.05 else "Non significative"
        self.txt_results.insert("end", f"\n🔬 RÉSULTATS:\n")
        self.txt_results.insert("end", f"   • Chi²: {chi2:.4f}\n")
        self.txt_results.insert("end", f"   • p-value: {p:.4e}\n")
        self.txt_results.insert("end", f"   • ddl: {dof}\n")
        self.txt_results.insert("end", f"   • V de Cramer: {cramers_v:.4f}\n")
        self.txt_results.insert("end", f"   • Conclusion: {sig_text}\n")
        
        observed = contingency.copy()
        expected_df = pd.DataFrame(expected, index=contingency.index, columns=contingency.columns)
        
        observed_with_total = observed.copy()
        observed_with_total["Total"] = observed.sum(axis=1)
        total_row = observed.sum(axis=0)
        total_row["Total"] = observed.values.sum()
        observed_with_total.loc["Total"] = total_row
        
        self._insert_dataframe_table("Tableau de contingence observé", observed_with_total, max_rows=25)
        self._show_table_window(observed_with_total, f"Chi² - Observé ({col1} x {col2})")
        self._show_table_window(expected_df.round(2), f"Chi² - Effectifs attendus ({col1} x {col2})")
        
        contrib = ((observed - expected_df) ** 2 / expected_df).stack().sort_values(ascending=False).head(10)
        contrib_df = contrib.reset_index()
        contrib_df.columns = [col1, col2, "contribution_chi2"]
        self._insert_dataframe_table("Top contributions au Chi²", contrib_df.round(4), max_rows=10)
        
        row_pct = observed.div(observed.sum(axis=1), axis=0) * 100
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
        sns.heatmap(observed, annot=True, fmt=".0f", cmap="Blues", ax=axes[0])
        axes[0].set_title("Effectifs observés")
        axes[0].set_xlabel(col2)
        axes[0].set_ylabel(col1)
        sns.heatmap(row_pct, annot=True, fmt=".1f", cmap="YlOrBr", ax=axes[1])
        axes[1].set_title("Pourcentages par ligne (%)")
        axes[1].set_xlabel(col2)
        axes[1].set_ylabel(col1)
        plt.tight_layout()
        self._show_plot(fig, f"Chi² - {col1} x {col2}")
        
    def run_regression(self):
        """Régression"""
        if self.df is None:
            return self.show_error()
        if len(self.numeric_cols) < 2:
            self.show_message("❌ Au moins 2 variables numériques nécessaires", "error")
            return
            
        y_col = self.var_num1.get()
        if y_col not in self.numeric_cols:
            y_col = self.numeric_cols[0]
        
        x_cols = [c for c in self.numeric_cols if c != y_col][:5]
        
        if not x_cols:
            self.show_message("❌ Variables explicatives indisponibles", "error")
            return
            
        result = MultivariateAnalysis.linear_regression(self.df, y_col, x_cols)
        
        self.clear_results()
        self.print_header(f"📉 RÉGRESSION: {y_col}")
        
        if 'error' in result:
            self.txt_results.insert("end", f"\n❌ {result['error']}\n")
            return
            
        self.txt_results.insert("end", f"\n🔬 PERFORMANCE:\n")
        self.txt_results.insert("end", f"   • n obs: {result.get('n_observations', 0)}\n")
        self.txt_results.insert("end", f"   • R²: {result['r2']:.4f}\n")
        self.txt_results.insert("end", f"   • RMSE: {result['rmse']:.4f}\n")
        
        self.txt_results.insert("end", f"\n📊 COEFFICIENTS:\n")
        self.txt_results.insert("end", f"   • Intercept: {result['intercept']:.4f}\n")
        for var, coef in result['coefficients'].items():
            self.txt_results.insert("end", f"   • {var}: {coef:.4f}\n")
        
        coef_df = pd.DataFrame(
            [{"variable": "Intercept", "coefficient": result["intercept"]}] +
            [{"variable": var, "coefficient": coef} for var, coef in result["coefficients"].items()]
        ).set_index("variable")
        self._insert_dataframe_table("Tableau des coefficients", coef_df.round(6), max_rows=20)
        self._show_table_window(coef_df.round(6), f"Régression - Coefficients ({y_col})")
        
        y_true = np.array(result.get("y_true", []), dtype=float)
        y_pred = np.array(result.get("y_pred", []), dtype=float)
        residuals = np.array(result.get("residuals", []), dtype=float)
        
        if len(y_true) > 0 and len(y_pred) == len(y_true):
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            axes[0].scatter(y_true, y_pred, alpha=0.55, color=Colors.PRIMARY, edgecolors="white", linewidth=0.4)
            min_v = min(float(np.min(y_true)), float(np.min(y_pred)))
            max_v = max(float(np.max(y_true)), float(np.max(y_pred)))
            axes[0].plot([min_v, max_v], [min_v, max_v], linestyle="--", color="red", linewidth=1.6)
            axes[0].set_title("Réel vs Prédit")
            axes[0].set_xlabel("Valeurs réelles")
            axes[0].set_ylabel("Valeurs prédites")
            
            axes[1].scatter(y_pred, residuals, alpha=0.55, color=Colors.WARNING, edgecolors="white", linewidth=0.4)
            axes[1].axhline(0, linestyle="--", color="red", linewidth=1.5)
            axes[1].set_title("Résidus vs Prédictions")
            axes[1].set_xlabel("Valeurs prédites")
            axes[1].set_ylabel("Résidus")
            
            plt.tight_layout()
            self._show_plot(fig, f"Régression - Diagnostic ({y_col})")
            
    def run_pca(self):
        """ACP"""
        if self.df is None:
            return self.show_error()
            
        result = MultivariateAnalysis.pca_analysis(self.df)
        
        self.clear_results()
        self.print_header("🌀 ACP")
        
        if 'error' in result:
            self.txt_results.insert("end", f"\n❌ {result['error']}\n")
            return
            
        self.txt_results.insert("end", f"\n📊 VARIANCE EXPLIQUÉE:\n")
        self.txt_results.insert("end", f"{'CP':<8} {'Variance %':>12} {'Cumul %':>12}\n")
        self.txt_results.insert("end", "─" * 35 + "\n")
        
        for i, (var, cum) in enumerate(zip(result['variance_ratio'], result['cumulative_variance'])):
            self.txt_results.insert("end", f"CP{i+1:<6} {var:>11.2f}% {cum:>11.2f}%\n")
        
        pca_table = pd.DataFrame({
            "CP": [f"CP{i+1}" for i in range(len(result["variance_ratio"]))],
            "valeur_propre": result["eigenvalues"],
            "variance_%": result["variance_ratio"],
            "cumul_%": result["cumulative_variance"]
        }).set_index("CP")
        self._insert_dataframe_table("Tableau ACP (valeurs propres & variance)", pca_table.round(4))
        self._show_table_window(pca_table.round(4), "ACP - Variance expliquée")
        
        components = result.get("components", [])
        feature_names = result.get("feature_names", [])
        if components and feature_names:
            n_comp_display = min(3, len(components))
            loadings_df = pd.DataFrame(
                np.array(components[:n_comp_display]).T,
                index=feature_names,
                columns=[f"CP{i+1}" for i in range(n_comp_display)]
            )
            self._insert_dataframe_table("Loadings (composantes)", loadings_df.round(4), max_rows=20)
            self._show_table_window(loadings_df.round(4), "ACP - Loadings")
        
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))
        x = np.arange(1, len(result["variance_ratio"]) + 1)
        axes[0].bar(x, result["variance_ratio"], color=Colors.PRIMARY, alpha=0.8)
        axes[0].set_title("Scree plot")
        axes[0].set_xlabel("Composante principale")
        axes[0].set_ylabel("Variance expliquée (%)")
        axes[0].set_xticks(x)
        
        axes[1].plot(x, result["cumulative_variance"], marker="o", color=Colors.SUCCESS, linewidth=2)
        axes[1].axhline(80, linestyle="--", color=Colors.WARNING, linewidth=1.2)
        axes[1].set_title("Variance cumulée")
        axes[1].set_xlabel("Composante principale")
        axes[1].set_ylabel("Variance cumulée (%)")
        axes[1].set_xticks(x)
        axes[1].set_ylim(0, 105)
        plt.tight_layout()
        self._show_plot(fig, "ACP - Scree & variance cumulée")
        
        scores = result.get("scores", [])
        if scores:
            scores_arr = np.array(scores, dtype=float)
            if scores_arr.shape[1] >= 2:
                fig2, ax2 = plt.subplots(figsize=(8, 6))
                sample_n = min(3000, len(scores_arr))
                sampled = scores_arr[:sample_n]
                ax2.scatter(sampled[:, 0], sampled[:, 1], alpha=0.5, s=18, color=Colors.INFO, edgecolors="white", linewidth=0.3)
                ax2.axhline(0, color="gray", linewidth=1)
                ax2.axvline(0, color="gray", linewidth=1)
                ax2.set_title("Projection des observations (CP1 vs CP2)")
                ax2.set_xlabel("CP1")
                ax2.set_ylabel("CP2")
                plt.tight_layout()
                self._show_plot(fig2, "ACP - Projection CP1/CP2")
    
    def _insert_dataframe_table(self, title: str, df: pd.DataFrame, max_rows: int = 20):
        """Insère un tableau formaté dans la zone résultats."""
        self.txt_results.insert("end", f"\n📋 {title}:\n")
        if df is None or df.empty:
            self.txt_results.insert("end", "   (Aucune donnée)\n")
            return
        
        df_view = df.copy()
        truncated = len(df_view) > max_rows
        if truncated:
            df_view = df_view.head(max_rows)
        
        with pd.option_context("display.max_columns", None, "display.width", 220, "display.float_format", "{:,.4f}".format):
            self.txt_results.insert("end", df_view.to_string() + "\n")
        
        if truncated:
            self.txt_results.insert("end", f"... {len(df) - max_rows} lignes supplémentaires non affichées\n")
    
    def _show_table_window(self, df: pd.DataFrame, title: str, max_rows: int = 300):
        """Affiche un tableau détaillé dans une fenêtre dédiée."""
        if df is None or df.empty:
            return
        
        win = ctk.CTkToplevel(self)
        win.title(title)
        win.geometry("1050x700")
        
        txt = ctk.CTkTextbox(
            win, corner_radius=10,
            fg_color=Colors.BG_INPUT,
            text_color=Colors.TEXT_PRIMARY,
            font=ctk.CTkFont(family="Consolas", size=11),
            wrap="none"
        )
        txt.pack(fill="both", expand=True, padx=10, pady=10)
        
        df_view = df.copy()
        truncated = len(df_view) > max_rows
        if truncated:
            df_view = df_view.head(max_rows)
        
        with pd.option_context("display.max_columns", None, "display.width", 260, "display.float_format", "{:,.4f}".format):
            txt.insert("1.0", df_view.to_string())
        
        if truncated:
            txt.insert("end", f"\n\n... {len(df) - max_rows} lignes supplémentaires non affichées")
        
        txt.configure(state="disabled")
            
    # ========================================================================
    # VISUALISATION
    # ========================================================================
    
    def _show_plot(self, fig, title):
        """Affiche un graphique"""
        win = ctk.CTkToplevel(self)
        win.title(title)
        win.geometry("900x700")
        
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        
        toolbar = NavigationToolbar2Tk(canvas, win)
        toolbar.update()
        
    def plot_histogram(self):
        """Histogramme"""
        if self.df is None:
            return
            
        var = self.var_viz.get()
        if not var or var not in self.numeric_cols:
            var = self.numeric_cols[0] if self.numeric_cols else None
        if not var:
            return
            
        fig, ax = plt.subplots(figsize=(10, 6))
        data = self.df[var].dropna()
        
        ax.hist(data, bins=30, edgecolor='white', alpha=0.7, color=Colors.PRIMARY)
        ax.axvline(data.mean(), color='red', linestyle='--', label=f'Moyenne: {data.mean():.2f}')
        ax.axvline(data.median(), color='green', linestyle='--', label=f'Médiane: {data.median():.2f}')
        
        ax.set_xlabel(var)
        ax.set_ylabel('Fréquence')
        ax.set_title(f'Distribution de {var}')
        ax.legend()
        
        self._show_plot(fig, f"Histogramme - {var}")
        
    def plot_boxplot(self):
        """Boxplot"""
        if self.df is None:
            return
            
        var = self.var_viz.get()
        if not var or var not in self.numeric_cols:
            var = self.numeric_cols[0] if self.numeric_cols else None
        if not var:
            return
            
        fig, ax = plt.subplots(figsize=(10, 6))
        data = self.df[var].dropna()
        
        bp = ax.boxplot(data, vert=True, patch_artist=True)
        bp['boxes'][0].set_facecolor(Colors.PRIMARY)
        bp['boxes'][0].set_alpha(0.7)
        
        ax.set_ylabel(var)
        ax.set_title(f'Boxplot de {var}')
        
        self._show_plot(fig, f"Boxplot - {var}")
        
    def plot_scatter(self):
        """Scatter"""
        if self.df is None:
            return
            
        var1 = self.var_viz.get()
        var2 = self.var_viz2.get()
        if not var1 or not var2:
            return
            
        fig, ax = plt.subplots(figsize=(10, 6))
        data = self.df[[var1, var2]].dropna()
        
        ax.scatter(data[var1], data[var2], alpha=0.5, color=Colors.PRIMARY, edgecolors='white')
        
        # Régression
        z = np.polyfit(data[var1], data[var2], 1)
        p = np.poly1d(z)
        x_line = np.linspace(data[var1].min(), data[var1].max(), 100)
        ax.plot(x_line, p(x_line), color='red', linewidth=2, label='Tendance')
        
        corr = data[var1].corr(data[var2])
        
        ax.set_xlabel(var1)
        ax.set_ylabel(var2)
        ax.set_title(f'{var1} vs {var2} (r = {corr:.3f})')
        ax.legend()
        
        self._show_plot(fig, f"Scatter - {var1} vs {var2}")
        
    def plot_heatmap(self):
        """Heatmap"""
        if self.df is None or len(self.numeric_cols) < 2:
            return
            
        fig, ax = plt.subplots(figsize=(12, 10))
        corr = self.df[self.numeric_cols].corr()
        
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0, square=True, ax=ax)
        ax.set_title('Matrice de Corrélation')
        
        self._show_plot(fig, "Heatmap")
        
    def plot_bar(self):
        """Bar chart"""
        if self.df is None:
            return
            
        var = self.var_viz.get()
        if not var:
            return
            
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if var in self.categorical_cols:
            data = self.df[var].value_counts().head(15)
            ax.bar(range(len(data)), data.values, color=Colors.PRIMARY, alpha=0.7)
            ax.set_xticks(range(len(data)))
            ax.set_xticklabels(data.index, rotation=45, ha='right')
        
        ax.set_xlabel(var)
        ax.set_ylabel('Effectif')
        ax.set_title(f'Distribution de {var}')
        
        self._show_plot(fig, f"Bar - {var}")
        
    def plot_pie(self):
        """Pie chart"""
        if self.df is None:
            return
            
        var = self.var_viz.get()
        if not var or var not in (self.categorical_cols + self.boolean_cols):
            var = self.categorical_cols[0] if self.categorical_cols else None
        if not var:
            return
            
        fig, ax = plt.subplots(figsize=(10, 8))
        data = self.df[var].value_counts().head(10)
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(data)))
        ax.pie(data.values, labels=data.index, autopct='%1.1f%%', colors=colors)
        ax.set_title(f'Répartition de {var}')
        
        self._show_plot(fig, f"Pie - {var}")
        
    def plot_pairplot(self):
        """Pairplot"""
        if self.df is None or len(self.numeric_cols) < 2:
            return
            
        vars_plot = self.numeric_cols[:5]
        
        fig = plt.figure(figsize=(12, 12))
        n = len(vars_plot)
        
        for i, var1 in enumerate(vars_plot):
            for j, var2 in enumerate(vars_plot):
                ax = fig.add_subplot(n, n, i*n + j + 1)
                
                if i == j:
                    ax.hist(self.df[var1].dropna(), bins=20, color=Colors.PRIMARY, alpha=0.7)
                else:
                    data = self.df[[var1, var2]].dropna()
                    ax.scatter(data[var2], data[var1], alpha=0.3, s=10, color=Colors.PRIMARY)
                
                if j == 0:
                    ax.set_ylabel(var1[:10], fontsize=8)
                if i == n-1:
                    ax.set_xlabel(var2[:10], fontsize=8)
                    
        fig.suptitle('Pairplot')
        plt.tight_layout()
        
        self._show_plot(fig, "Pairplot")
        
    def plot_distribution(self):
        """Distribution KDE"""
        if self.df is None:
            return
            
        var = self.var_viz.get()
        if not var or var not in self.numeric_cols:
            var = self.numeric_cols[0] if self.numeric_cols else None
        if not var:
            return
            
        fig, ax = plt.subplots(figsize=(10, 6))
        data = self.df[var].dropna()
        
        ax.hist(data, bins=30, density=True, alpha=0.5, color=Colors.PRIMARY)
        data.plot.kde(ax=ax, color='red', linewidth=2)
        
        ax.set_xlabel(var)
        ax.set_ylabel('Densité')
        ax.set_title(f'Distribution de {var}')
        
        self._show_plot(fig, f"Distribution - {var}")
        
    # ========================================================================
    # RAPPORT
    # ========================================================================
    
    def generate_report(self):
        """Génère un rapport PDF complet (style institutionnel/ANSD)."""
        if self.df is None:
            return self.show_error()
            
        try:
            import math
            import shutil
            import tempfile
            from itertools import combinations

            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.enums import TA_CENTER
            from reportlab.platypus import (
                Image as RLImage,
                PageBreak,
                Paragraph,
                SimpleDocTemplate,
                Spacer,
                Table,
                TableStyle,
            )
            from reportlab.lib import colors
            from reportlab.lib.units import cm
            from scipy.stats import chi2_contingency
            
            output_path = filedialog.asksaveasfilename(
                title="Sauvegarder", defaultextension=".pdf",
                filetypes=[("PDF", "*.pdf")]
            )
            
            if not output_path:
                return

            self.show_message("📝 Génération du rapport ANSD complet en cours...", "info")

            df = self.df
            dataset_name = os.path.basename(self.file_path) if self.file_path else "Dataset"
            generated_at = datetime.now().strftime("%d/%m/%Y %H:%M")
            tmp_dir = tempfile.mkdtemp(prefix="eda_desk_report_")

            # Prépare les données d'analyse IA
            ai_text = ""
            ai_error = ""
            summary_text = self._build_dataset_summary_text()
            brief_text = self._build_report_brief_text()
            brief_info = self._report_brief_completion()
            latest_results = self.txt_results.get("1.0", "end").strip()

            context_payload = {
                "dataset_name": dataset_name,
                "generated_at": generated_at,
                "summary": summary_text,
                "brief_rapport": self.report_brief,
                "brief_completion": brief_info,
                "brief_text": brief_text,
                "latest_results": latest_results[:15000],
                "active_filters": self.active_filters,
            }
            context_json_path = os.path.splitext(output_path)[0] + "_hf_context.json"
            try:
                with open(context_json_path, "w", encoding="utf-8") as f:
                    json.dump(context_payload, f, indent=2, ensure_ascii=False)
            except Exception:
                context_json_path = ""

            if self.enable_ai_report.get():
                try:
                    self.show_message("🤖 Génération de l'interprétation IA en cours...", "info")
                    ai_text = HuggingFaceReportAssistant.generate_analysis(
                        config_path=self.hf_config_path,
                        prompt_path=self.hf_prompt_path,
                        dataset_summary=summary_text,
                        latest_results=latest_results,
                        context_file=self.ai_context_path
                    )
                except Exception as e:
                    ai_error = str(e)

            try:
                doc = SimpleDocTemplate(
                    output_path,
                    pagesize=A4,
                    rightMargin=1.4 * cm,
                    leftMargin=1.4 * cm,
                    topMargin=1.5 * cm,
                    bottomMargin=1.5 * cm,
                )

                styles = getSampleStyleSheet()
                style_cover_title = ParagraphStyle(
                    "cover_title", parent=styles["Title"],
                    fontSize=28, leading=32, textColor=colors.HexColor("#0EA5E9"),
                    alignment=TA_CENTER, spaceAfter=10
                )
                style_cover_subtitle = ParagraphStyle(
                    "cover_subtitle", parent=styles["Normal"],
                    fontSize=12, leading=17, textColor=colors.HexColor("#334155"),
                    alignment=TA_CENTER, spaceAfter=8
                )
                style_h1 = ParagraphStyle(
                    "h1", parent=styles["Heading1"],
                    fontSize=18, leading=22, textColor=colors.HexColor("#0F172A"), spaceAfter=8
                )
                style_h2 = ParagraphStyle(
                    "h2", parent=styles["Heading2"],
                    fontSize=14.5, leading=18, textColor=colors.HexColor("#0F172A"), spaceAfter=6
                )
                style_body = ParagraphStyle(
                    "body", parent=styles["Normal"],
                    fontSize=10.2, leading=14.5, textColor=colors.HexColor("#111827")
                )
                style_note = ParagraphStyle(
                    "note", parent=styles["Normal"],
                    fontSize=9, leading=12, textColor=colors.HexColor("#334155")
                )

                def safe_html(txt: Any) -> str:
                    return str(txt).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

                def para(txt: Any, style=style_body) -> Paragraph:
                    return Paragraph(safe_html(txt).replace("\n", "<br/>"), style)

                def fmt_val(v: Any) -> str:
                    if pd.isna(v):
                        return ""
                    if isinstance(v, (np.integer, int)):
                        return f"{int(v):,}"
                    if isinstance(v, (np.floating, float)):
                        return f"{float(v):,.4f}"
                    s = str(v)
                    return s if len(s) <= 90 else s[:87] + "..."

                def add_df_table(
                    elements_out: List[Any],
                    title: str,
                    table_df: pd.DataFrame,
                    col_widths: Optional[List[float]] = None,
                    max_rows_per_block: int = 26,
                    interpretation: Optional[str] = None
                ):
                    elements_out.append(Paragraph(title, style_h2))
                    if interpretation:
                        elements_out.append(para(interpretation, style_note))
                    if table_df is None or table_df.empty:
                        elements_out.append(para("Aucune donnée disponible."))
                        elements_out.append(Spacer(1, 0.25 * cm))
                        return

                    header = [str(c) for c in table_df.columns]
                    rows = [[fmt_val(v) for v in row] for row in table_df.itertuples(index=False, name=None)]

                    ncols = max(1, len(header))
                    if not col_widths:
                        col_widths = [16.8 * cm / ncols] * ncols

                    for start in range(0, len(rows), max_rows_per_block):
                        chunk = rows[start:start + max_rows_per_block]
                        table = Table([header] + chunk, colWidths=col_widths, repeatRows=1)
                        table.setStyle(TableStyle([
                            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DBEAFE")),
                            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#1E3A8A")),
                            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#BFDBFE")),
                            ("FONTSIZE", (0, 0), (-1, -1), 8.3),
                            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
                            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ]))
                        elements_out.append(table)
                        elements_out.append(Spacer(1, 0.18 * cm))

                plot_count = 0

                def add_figure(
                    elements_out: List[Any],
                    fig,
                    title: str,
                    interpretation: str,
                    width_cm: float = 16.6
                ):
                    nonlocal plot_count
                    plot_count += 1
                    img_path = os.path.join(tmp_dir, f"plot_{plot_count:03d}.png")
                    fig.savefig(img_path, dpi=180, bbox_inches="tight")
                    plt.close(fig)

                    img = RLImage(img_path)
                    target_w = width_cm * cm
                    ratio = target_w / img.drawWidth if img.drawWidth else 1
                    img.drawWidth = target_w
                    img.drawHeight = img.drawHeight * ratio

                    elements_out.append(Paragraph(title, style_h2))
                    elements_out.append(img)
                    elements_out.append(para(interpretation, style_note))
                    elements_out.append(Spacer(1, 0.22 * cm))

                def add_latex_formula(
                    elements_out: List[Any],
                    latex_expr: str,
                    interpretation: str,
                    width_cm: float = 16.0
                ):
                    """Rend une formule LaTeX en image pour insertion PDF."""
                    nonlocal plot_count
                    plot_count += 1
                    img_path = os.path.join(tmp_dir, f"formula_{plot_count:03d}.png")
                    try:
                        fig = plt.figure(figsize=(11, 1.1))
                        fig.patch.set_facecolor("white")
                        ax = fig.add_axes([0, 0, 1, 1])
                        ax.axis("off")
                        ax.text(0.01, 0.55, f"${latex_expr}$", fontsize=16, color="#0F172A")
                        fig.savefig(img_path, dpi=220, bbox_inches="tight")
                        plt.close(fig)

                        img = RLImage(img_path)
                        target_w = width_cm * cm
                        ratio = target_w / img.drawWidth if img.drawWidth else 1
                        img.drawWidth = target_w
                        img.drawHeight = img.drawHeight * ratio
                        elements_out.append(img)
                        elements_out.append(para(interpretation, style_note))
                        elements_out.append(Spacer(1, 0.12 * cm))
                    except Exception:
                        try:
                            plt.close("all")
                        except Exception:
                            pass
                        # Fallback texte pour éviter de perdre l'information mathématique.
                        elements_out.append(para(f"Formule: {latex_expr}", style_note))
                        elements_out.append(para(interpretation, style_note))
                        elements_out.append(Spacer(1, 0.08 * cm))

                def add_text_with_latex(
                    elements_out: List[Any],
                    text: str,
                    default_style=style_body
                ):
                    """Ajoute un texte en détectant des blocs LaTeX ($$...$$ ou ligne `$...$`)."""
                    if not text:
                        return
                    raw = str(text).replace("\r\n", "\n")
                    chunks = re.split(r"(\$\$.*?\$\$)", raw, flags=re.S)
                    for chunk in chunks:
                        if not chunk:
                            continue
                        if chunk.startswith("$$") and chunk.endswith("$$"):
                            expr = chunk[2:-2].strip()
                            if expr:
                                add_latex_formula(
                                    elements_out,
                                    expr,
                                    "Formule extraite automatiquement du texte d'analyse."
                                )
                            continue

                        for line in chunk.split("\n"):
                            stripped = line.strip()
                            if not stripped:
                                continue
                            full_inline = re.fullmatch(r"\$(.+?)\$", stripped)
                            if full_inline:
                                add_latex_formula(
                                    elements_out,
                                    full_inline.group(1).strip(),
                                    "Formule extraite automatiquement du texte d'analyse."
                                )
                            else:
                                # Supprime les délimiteurs inline pour éviter l'affichage brut `$...$`.
                                clean_line = re.sub(r"\$(.+?)\$", r"\1", line)
                                if clean_line.strip():
                                    elements_out.append(para(clean_line, default_style))

                missing_total = int(df.isnull().sum().sum())
                total_cells = max(1, df.shape[0] * df.shape[1])
                missing_pct = (missing_total / total_cells) * 100
                duplicates = int(df.duplicated().sum())
                duplicates_pct = (duplicates / max(1, len(df))) * 100
                mem_mb = df.memory_usage(deep=True).sum() / 1024**2

                # Dictionnaire complet des variables
                dict_rows = []
                for col in df.columns:
                    s = df[col]
                    missing = int(s.isna().sum())
                    missing_col_pct = (missing / max(1, len(df))) * 100
                    uniq = int(s.nunique(dropna=True))
                    dtype = str(s.dtype)
                    sample_vals = [str(v) for v in s.dropna().astype(str).head(3).tolist()]
                    dict_rows.append({
                        "variable": col,
                        "type": dtype,
                        "manquants": missing,
                        "%_manquants": round(missing_col_pct, 2),
                        "uniques": uniq,
                        "exemples": " | ".join(sample_vals) if sample_vals else "",
                    })
                dict_df = pd.DataFrame(dict_rows)

                # Résumé numérique + outliers
                num_summary_rows = []
                outlier_rows = []
                for col in self.numeric_cols:
                    s = pd.to_numeric(df[col], errors="coerce").dropna()
                    if len(s) == 0:
                        continue
                    q1, q3 = s.quantile(0.25), s.quantile(0.75)
                    iqr = q3 - q1
                    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                    n_out = int(((s < low) | (s > high)).sum())
                    out_pct = (n_out / max(1, len(s))) * 100
                    mean_v = float(s.mean())
                    std_v = float(s.std())
                    cv = (std_v / mean_v * 100) if mean_v != 0 else np.nan

                    num_summary_rows.append({
                        "variable": col,
                        "n": len(s),
                        "moyenne": mean_v,
                        "mediane": float(s.median()),
                        "std": std_v,
                        "cv_%": cv,
                        "min": float(s.min()),
                        "max": float(s.max()),
                        "skewness": float(stats.skew(s)),
                        "kurtosis": float(stats.kurtosis(s)),
                        "outliers_%": out_pct,
                    })
                    outlier_rows.append({
                        "variable": col,
                        "q1": float(q1),
                        "q3": float(q3),
                        "borne_inf": float(low),
                        "borne_sup": float(high),
                        "nb_outliers": n_out,
                        "outliers_%": out_pct,
                    })
                num_summary_df = pd.DataFrame(num_summary_rows).sort_values("outliers_%", ascending=False) if num_summary_rows else pd.DataFrame()
                outlier_df = pd.DataFrame(outlier_rows).sort_values("outliers_%", ascending=False) if outlier_rows else pd.DataFrame()

                # Résumé catégoriel
                cat_rows = []
                all_cat_cols = self.categorical_cols + self.boolean_cols
                for col in all_cat_cols:
                    s = df[col].dropna().astype(str)
                    if len(s) == 0:
                        cat_rows.append({
                            "variable": col, "n": 0, "modalites": 0,
                            "modalite_dominante": "", "dominance_%": np.nan, "cardinalite_%": np.nan
                        })
                        continue
                    vc = s.value_counts()
                    dom = str(vc.index[0])
                    dom_pct = (vc.iloc[0] / len(s)) * 100
                    cat_rows.append({
                        "variable": col,
                        "n": len(s),
                        "modalites": int(s.nunique()),
                        "modalite_dominante": dom,
                        "dominance_%": dom_pct,
                        "cardinalite_%": (s.nunique() / max(1, len(s))) * 100,
                    })
                cat_summary_df = pd.DataFrame(cat_rows).sort_values("dominance_%", ascending=False) if cat_rows else pd.DataFrame()

                # Corrélations
                corr_pairs_df = pd.DataFrame()
                strongest_pair = None
                if len(self.numeric_cols) >= 2:
                    corr = df[self.numeric_cols].corr()
                    pairs = []
                    for i, c1 in enumerate(corr.columns):
                        for j, c2 in enumerate(corr.columns):
                            if i < j:
                                r = corr.loc[c1, c2]
                                pairs.append({"var1": c1, "var2": c2, "r": float(r), "|r|": abs(float(r))})
                    corr_pairs_df = pd.DataFrame(pairs).sort_values("|r|", ascending=False)
                    if not corr_pairs_df.empty:
                        strongest_pair = corr_pairs_df.iloc[0]

                # Chi2 (catégoriel) sur la meilleure paire en V de Cramer
                chi2_result = None
                if len(all_cat_cols) >= 2:
                    best_v = -1.0
                    for c1, c2 in combinations(all_cat_cols[:8], 2):
                        s1 = df[c1].astype(str).fillna("NA")
                        s2 = df[c2].astype(str).fillna("NA")
                        top1 = s1.value_counts().head(12).index
                        top2 = s2.value_counts().head(12).index
                        s1r = s1.where(s1.isin(top1), "Autres")
                        s2r = s2.where(s2.isin(top2), "Autres")
                        tab = pd.crosstab(s1r, s2r)
                        if tab.shape[0] < 2 or tab.shape[1] < 2:
                            continue
                        chi2, pval, _, _ = chi2_contingency(tab)
                        n = tab.values.sum()
                        min_dim = min(tab.shape[0] - 1, tab.shape[1] - 1)
                        v = math.sqrt(chi2 / (n * min_dim)) if min_dim > 0 and n > 0 else 0
                        if v > best_v:
                            best_v = v
                            chi2_result = {
                                "var1": c1,
                                "var2": c2,
                                "chi2": float(chi2),
                                "pvalue": float(pval),
                                "cramers_v": float(v),
                                "table": tab,
                            }

                # Inference statistique sur variables numeriques (IC, normalite)
                numeric_infer_rows = []
                for col in self.numeric_cols[:25]:
                    s = pd.to_numeric(df[col], errors="coerce").dropna()
                    n = len(s)
                    if n < 3:
                        continue
                    mean_v = float(s.mean())
                    std_v = float(s.std(ddof=1))
                    se = std_v / math.sqrt(n) if n > 0 else np.nan
                    t_crit = float(stats.t.ppf(0.975, df=max(n - 1, 1)))
                    ci_low = mean_v - t_crit * se if pd.notna(se) else np.nan
                    ci_high = mean_v + t_crit * se if pd.notna(se) else np.nan

                    shapiro_p = np.nan
                    if n >= 8:
                        sample = s.sample(min(5000, n), random_state=42)
                        try:
                            shapiro_p = float(stats.shapiro(sample).pvalue)
                        except Exception:
                            shapiro_p = np.nan

                    jb_p = np.nan
                    try:
                        sample_jb = s.sample(min(5000, n), random_state=42)
                        jb_p = float(stats.jarque_bera(sample_jb).pvalue)
                    except Exception:
                        jb_p = np.nan

                    numeric_infer_rows.append({
                        "variable": col,
                        "n": n,
                        "moyenne": mean_v,
                        "ic95_bas": ci_low,
                        "ic95_haut": ci_high,
                        "p_shapiro": shapiro_p,
                        "p_jarque_bera": jb_p,
                        "normalite_5%": "Oui" if pd.notna(shapiro_p) and shapiro_p > 0.05 else "Non",
                    })
                numeric_infer_df = pd.DataFrame(numeric_infer_rows)

                # Regression multiple interpretable (si possible)
                reg_target = ""
                reg_result = None
                reg_coef_df = pd.DataFrame()
                if len(self.numeric_cols) >= 3 and not num_summary_df.empty:
                    reg_target = str(num_summary_df.sort_values("std", ascending=False).iloc[0]["variable"])
                    reg_features = [c for c in self.numeric_cols if c != reg_target][:5]
                    reg_result = MultivariateAnalysis.linear_regression(df, reg_target, reg_features)
                    if reg_result and "error" not in reg_result:
                        reg_coef_df = pd.DataFrame(
                            [{"variable": "Intercept", "coefficient": reg_result["intercept"]}]
                            + [{"variable": k, "coefficient": v} for k, v in reg_result["coefficients"].items()]
                        )

                # ANOVA automatique: meilleure variable categorielle selon p-value
                auto_anova = None
                if all_cat_cols and self.numeric_cols:
                    y_anova = self.numeric_cols[0]
                    if reg_target:
                        y_anova = reg_target
                    best_p = None
                    for g in all_cat_cols[:15]:
                        nmod = int(df[g].nunique(dropna=True))
                        if nmod < 2 or nmod > 15:
                            continue
                        res = MultivariateAnalysis.anova(df, y_anova, g)
                        if "error" in res:
                            continue
                        p = float(res.get("p_value", 1.0))
                        if best_p is None or p < best_p:
                            best_p = p
                            auto_anova = {"target": y_anova, "group": g, "result": res}

                # Commentaires pousses variable par variable (integres aux sections)
                numeric_commentaries: List[str] = []
                if not num_summary_df.empty:
                    top_num = num_summary_df.sort_values(["outliers_%", "cv_%"], ascending=False).head(8)
                    for _, r in top_num.iterrows():
                        skew_txt = "asymetrie a droite" if r["skewness"] > 0.5 else ("asymetrie a gauche" if r["skewness"] < -0.5 else "distribution plutot symetrique")
                        numeric_commentaries.append(
                            f"Variable numerique {r['variable']}: moyenne={r['moyenne']:.3f}, mediane={r['mediane']:.3f}, "
                            f"dispersion (CV={r['cv_%']:.2f}%), {skew_txt}, outliers={r['outliers_%']:.2f}%. "
                            "Interpretation: verifier la stabilite des estimations et envisager une transformation si la variabilite est forte."
                        )
                categorical_commentaries: List[str] = []
                if not cat_summary_df.empty:
                    top_cat = cat_summary_df.sort_values("dominance_%", ascending=False).head(8)
                    for _, r in top_cat.iterrows():
                        categorical_commentaries.append(
                            f"Variable categorielle {r['variable']}: {int(r['modalites'])} modalites, "
                            f"dominante={r['modalite_dominante']} ({r['dominance_%']:.2f}%). "
                            "Interpretation: un taux de dominance eleve peut reduire la capacite discriminante des modeles."
                        )

                elements: List[Any] = []

                # Page de garde ANSD-style
                cover = Table(
                    [
                        ["RAPPORT STATISTIQUE COMPLET - EDA DESK MODERN"],
                        [f"Jeu de données: {dataset_name}"],
                        [f"Date d'édition: {generated_at}"],
                    ],
                    colWidths=[17.2 * cm]
                )
                cover.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0EA5E9")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#E0F2FE")),
                    ("TEXTCOLOR", (0, 1), (-1, -1), colors.HexColor("#0F172A")),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                    ("FONTSIZE", (0, 0), (-1, 0), 15),
                    ("FONTSIZE", (0, 1), (-1, -1), 11),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                    ("TOPPADDING", (0, 0), (-1, -1), 10),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#93C5FD")),
                ]))

                toc_text = (
                    "Table des sections:\n"
                    "1. Cadrage utilisateur du rapport\n"
                    "2. Introduction générale\n"
                    "3. Résumé exécutif\n"
                    "4. Cadre méthodologique\n"
                    "5. Cadre théorique et formules mathématiques\n"
                    "6. Qualité des données\n"
                    "7. Dictionnaire complet des variables\n"
                    "8. Analyse descriptive numérique\n"
                    "9. Analyse descriptive catégorielle\n"
                    "10. Analyses bivariées et multivariées\n"
                    "11. Graphiques et interprétations\n"
                    "12. Recommandations opérationnelles\n"
                    "13. Conclusion générale\n"
                    "14. Annexes techniques"
                )
                elements.extend([
                    Spacer(1, 1.2 * cm),
                    Paragraph("Rapport ANSD - Structure complète", style_cover_title),
                    Paragraph("Qualité, statistiques, graphiques et interprétations opérationnelles", style_cover_subtitle),
                    Spacer(1, 0.4 * cm),
                    cover,
                    Spacer(1, 0.35 * cm),
                    para(toc_text, style_body),
                    PageBreak(),
                ])

                # Section 1 - Cadrage utilisateur
                elements.append(Paragraph("1. Cadrage utilisateur du rapport", style_h1))
                elements.append(para(
                    "Cette section formalise les attentes utilisateur afin de lier toutes les analyses "
                    "entre variables, résultats et recommandations."
                ))
                brief_table_df = pd.DataFrame([
                    {"Champ": "Theme", "Valeur": self.report_brief.get("theme", "") or "Non renseigne"},
                    {"Champ": "Resume du contexte", "Valeur": self.report_brief.get("resume_contexte", "") or "Non renseigne"},
                    {"Champ": "Objectif general", "Valeur": self.report_brief.get("objectif_general", "") or "Non renseigne"},
                    {"Champ": "Objectifs specifiques", "Valeur": self.report_brief.get("objectifs_specifiques", "") or "Non renseigne"},
                    {"Champ": "Questions cles", "Valeur": self.report_brief.get("questions_cle", "") or "Non renseigne"},
                    {"Champ": "Public cible", "Valeur": self.report_brief.get("public_cible", "") or "Non renseigne"},
                    {"Champ": "Periode / etendue", "Valeur": self.report_brief.get("periode_etendue", "") or "Non renseigne"},
                    {"Champ": "Hypotheses", "Valeur": self.report_brief.get("hypotheses", "") or "Non renseigne"},
                    {"Champ": "Limitations connues", "Valeur": self.report_brief.get("limitations_connues", "") or "Non renseigne"},
                    {"Champ": "Format souhaite", "Valeur": self.report_brief.get("format_souhaite", "") or "Non renseigne"},
                    {"Champ": "Niveau de detail", "Valeur": self.report_brief.get("niveau_detail", "") or "Non renseigne"},
                ])
                add_df_table(
                    elements,
                    "Tableau 0 - Brief utilisateur",
                    brief_table_df,
                    col_widths=[5.2 * cm, 11.2 * cm],
                    max_rows_per_block=30,
                    interpretation=(
                        f"Taux de completude du brief: {brief_info['filled']}/{brief_info['total']} "
                        f"({brief_info['ratio']*100:.1f}%). Un brief plus detaille rend le rapport plus precis."
                    ),
                )
                elements.append(PageBreak())

                # Section 2 - Introduction générale
                theme_txt = self.report_brief.get("theme", "").strip()
                context_txt = self.report_brief.get("resume_contexte", "").strip()
                obj_txt = self.report_brief.get("objectif_general", "").strip()
                qk_txt = self.report_brief.get("questions_cle", "").strip()
                public_txt = self.report_brief.get("public_cible", "").strip()
                period_txt = self.report_brief.get("periode_etendue", "").strip()
                detail_txt = self.report_brief.get("niveau_detail", "").strip()

                intro_parts = [
                    f"Ce rapport présente une analyse statistique complète du jeu de données {dataset_name}, "
                    f"qui contient {df.shape[0]:,} observations et {df.shape[1]} variables.",
                    "L'objectif est de produire une lecture cohérente et reliée des tableaux et graphiques, "
                    "afin de transformer les constats descriptifs en décisions opérationnelles.",
                    f"La structure des données inclut {len(self.numeric_cols)} variable(s) numérique(s), "
                    f"{len(self.categorical_cols)} variable(s) catégorielle(s) et {len(self.boolean_cols)} variable(s) booléenne(s).",
                ]
                if theme_txt:
                    intro_parts.append(f"Le thème directeur est: {theme_txt}.")
                if context_txt:
                    intro_parts.append(f"Contexte de l'étude: {context_txt}.")
                if obj_txt:
                    intro_parts.append(f"Objectif général: {obj_txt}.")
                if qk_txt:
                    intro_parts.append(f"Questions d'analyse prioritaires: {qk_txt}.")
                if public_txt:
                    intro_parts.append(f"Public cible du rapport: {public_txt}.")
                if period_txt:
                    intro_parts.append(f"Période et étendue considérées: {period_txt}.")
                if detail_txt:
                    intro_parts.append(f"Niveau de détail demandé: {detail_txt}.")
                intro_parts.append(
                    "La démarche combine contrôle de qualité, statistiques descriptives, inférence, analyse des associations "
                    "et recommandations hiérarchisées, dans un format de restitution institutionnel."
                )
                elements.append(Paragraph("2. Introduction générale", style_h1))
                elements.append(para(" ".join(intro_parts)))
                elements.append(Spacer(1, 0.2 * cm))

                # Section 3 - Résumé exécutif
                quality_flag = "bonne" if missing_pct < 5 else ("moyenne" if missing_pct < 20 else "fragile")
                strong_corr_txt = "Aucune corrélation calculable"
                if strongest_pair is not None:
                    strong_corr_txt = f"{strongest_pair['var1']} - {strongest_pair['var2']} (r={strongest_pair['r']:.3f})"

                exec_summary = (
                    f"Le dataset contient {df.shape[0]:,} observations et {df.shape[1]} variables. "
                    f"La complétude globale est de {100 - missing_pct:.2f}% et la qualité est jugée {quality_flag}. "
                    f"Le taux de doublons est de {duplicates_pct:.2f}%. "
                    f"Les analyses ont couvert les dimensions univariées, bivariées et multivariées. "
                    f"La relation linéaire la plus forte identifiée est: {strong_corr_txt}. "
                    f"Ce rapport propose également des recommandations opérationnelles priorisées."
                    + (f" Theme central: {theme_txt}." if theme_txt else "")
                    + (f" Objectif principal: {obj_txt}." if obj_txt else "")
                    + (f" Public cible: {public_txt}." if public_txt else "")
                )
                elements.append(Paragraph("3. Résumé exécutif", style_h1))
                elements.append(para(exec_summary))
                elements.append(Spacer(1, 0.2 * cm))

                kpi_df = pd.DataFrame([
                    {"Indicateur": "Lignes", "Valeur": f"{df.shape[0]:,}"},
                    {"Indicateur": "Colonnes", "Valeur": f"{df.shape[1]}"},
                    {"Indicateur": "Mémoire", "Valeur": f"{mem_mb:.2f} MB"},
                    {"Indicateur": "Valeurs manquantes", "Valeur": f"{missing_total:,} ({missing_pct:.2f}%)"},
                    {"Indicateur": "Doublons", "Valeur": f"{duplicates:,} ({duplicates_pct:.2f}%)"},
                    {"Indicateur": "Variables numériques", "Valeur": str(len(self.numeric_cols))},
                    {"Indicateur": "Variables catégorielles", "Valeur": str(len(self.categorical_cols))},
                    {"Indicateur": "Variables booléennes", "Valeur": str(len(self.boolean_cols))},
                ])
                add_df_table(
                    elements,
                    "Tableau 1 - Indicateurs synthétiques",
                    kpi_df,
                    col_widths=[6.2 * cm, 11 * cm],
                    max_rows_per_block=20,
                    interpretation="Lecture ANSD: ce tableau fournit les indicateurs de niveau macro pour juger de la robustesse initiale du fichier."
                )

                # Section 4 - Méthodologie
                elements.append(Paragraph("4. Cadre méthodologique", style_h1))
                methodology_text = (
                    "Le rapport est construit automatiquement selon un protocole EDA institutionnel: "
                    "(i) audit de qualité, (ii) description des distributions, "
                    "(iii) analyse des relations entre variables, "
                    "(iv) production de visualisations interprétées, "
                    "(v) formulation de recommandations. "
                    f"Filtres actifs au moment du calcul: {', '.join(self.active_filters) if self.active_filters else 'aucun'}."
                )
                qk = self.report_brief.get("questions_cle", "").strip()
                if qk:
                    methodology_text += f" Questions cles cibles: {qk}."
                objs = self.report_brief.get("objectifs_specifiques", "").strip()
                if objs:
                    methodology_text += f" Objectifs specifiques: {objs}."
                elements.append(para(methodology_text))
                elements.append(Spacer(1, 0.15 * cm))

                # Section 5 - Cadre theorique et formules
                elements.append(Paragraph("5. Cadre théorique et formules mathématiques", style_h1))
                elements.append(para("Les principales formules sont rendues ci-dessous en notation LaTeX."))
                formula_items = [
                    (r"\bar{x}=\frac{1}{n}\sum_{i=1}^{n}x_i", "Moyenne empirique des observations."),
                    (r"s^2=\frac{1}{n-1}\sum_{i=1}^{n}(x_i-\bar{x})^2", "Variance echantillonnale."),
                    (r"CV=\frac{s}{\bar{x}}\times 100", "Coefficient de variation (dispersion relative)."),
                    (r"IC_{95\%}=\bar{x}\pm t_{0.975,n-1}\cdot\frac{s}{\sqrt{n}}", "Intervalle de confiance a 95% de la moyenne."),
                    (r"IQR=Q_3-Q_1,\quad LB=Q_1-1.5\,IQR,\quad UB=Q_3+1.5\,IQR", "Bornes outliers par methode IQR."),
                    (r"r=\frac{\mathrm{cov}(X,Y)}{\sigma_X\sigma_Y}", "Correlation lineaire de Pearson."),
                    (r"\chi^2=\sum_{i,j}\frac{(O_{ij}-E_{ij})^2}{E_{ij}}", "Test du Chi-carre d'independance."),
                    (r"V=\sqrt{\frac{\chi^2}{n\cdot \min(r-1,c-1)}}", "V de Cramer (force d'association)."),
                    (r"y=\beta_0+\sum_{k=1}^{p}\beta_k x_k+\varepsilon", "Regression lineaire multiple."),
                    (r"R^2=1-\frac{SSE}{SST}", "Part de variance expliquee par le modele."),
                    (r"F=\frac{MS_{between}}{MS_{within}}", "Statistique du test ANOVA."),
                ]
                for latex_expr, interp in formula_items:
                    add_latex_formula(elements, latex_expr, interp)
                elements.append(para(
                    "Lecture: ces formules garantissent la traçabilité des résultats et facilitent la "
                    "validation scientifique des conclusions.",
                    style_note
                ))
                elements.append(Spacer(1, 0.2 * cm))

                # Section 6 - Qualité
                elements.append(Paragraph("6. Qualité des données", style_h1))
                quality_interp = (
                    f"Le taux de manquants est de {missing_pct:.2f}%: "
                    + ("niveau maîtrisé." if missing_pct < 5 else "niveau intermédiaire nécessitant ciblage." if missing_pct < 20 else "niveau critique à traiter prioritairement.")
                    + f" Le taux de doublons est de {duplicates_pct:.2f}%."
                )
                elements.append(para(quality_interp))

                top_missing_df = (
                    dict_df[["variable", "manquants", "%_manquants"]]
                    .sort_values("%_manquants", ascending=False)
                    .head(25)
                )
                add_df_table(
                    elements,
                    "Tableau 2 - Variables les plus incomplètes",
                    top_missing_df,
                    col_widths=[8.2 * cm, 4.2 * cm, 4.8 * cm],
                    interpretation=(
                        "Plus le pourcentage est élevé, plus l'incertitude analytique sur la variable augmente. "
                        "Ces colonnes doivent être priorisées dans une stratégie d'imputation ou de recollecte."
                    )
                )

                # Section 7 - Dictionnaire complet
                elements.append(PageBreak())
                elements.append(Paragraph("7. Dictionnaire complet des variables", style_h1))
                elements.append(para("Ce dictionnaire liste l'ensemble des variables avec type, manquants, cardinalité et exemples de modalités/valeurs."))
                add_df_table(
                    elements,
                    "Tableau 3 - Dictionnaire des variables (complet)",
                    dict_df,
                    col_widths=[4.0 * cm, 2.5 * cm, 2.2 * cm, 2.2 * cm, 1.8 * cm, 4.5 * cm],
                    max_rows_per_block=30,
                    interpretation="Usage ANSD: ce tableau sert de référentiel de métadonnées pour la traçabilité statistique."
                )

                # Section 8 - Numérique
                elements.append(PageBreak())
                elements.append(Paragraph("8. Analyse descriptive numérique", style_h1))
                if not num_summary_df.empty:
                    high_var = num_summary_df.sort_values("cv_%", ascending=False).head(1)
                    interpretation_num = "Les variables à CV élevé présentent une dispersion importante."
                    if not high_var.empty and pd.notna(high_var.iloc[0]["cv_%"]):
                        interpretation_num += f" Exemple principal: {high_var.iloc[0]['variable']} (CV={high_var.iloc[0]['cv_%']:.2f}%)."
                    if numeric_commentaries:
                        interpretation_num += " " + " ".join(numeric_commentaries[:3])
                    add_df_table(
                        elements,
                        "Tableau 4 - Statistiques numériques détaillées",
                        num_summary_df,
                        col_widths=[3.2 * cm, 1.3 * cm, 1.7 * cm, 1.7 * cm, 1.4 * cm, 1.3 * cm, 1.3 * cm, 1.3 * cm, 1.4 * cm, 1.4 * cm, 1.4 * cm],
                        max_rows_per_block=24,
                        interpretation=interpretation_num
                    )
                    add_df_table(
                        elements,
                        "Tableau 5 - Diagnostic outliers (méthode IQR)",
                        outlier_df,
                        col_widths=[3.3 * cm, 1.6 * cm, 1.6 * cm, 2.0 * cm, 2.0 * cm, 2.0 * cm, 2.3 * cm],
                        max_rows_per_block=24,
                        interpretation="Les colonnes avec un taux d'outliers élevé doivent faire l'objet de contrôles métier et de tests de robustesse."
                    )
                    if not numeric_infer_df.empty:
                        add_df_table(
                            elements,
                            "Tableau 5bis - Inference statistique (IC95 et normalite)",
                            numeric_infer_df,
                            col_widths=[3.2 * cm, 1.2 * cm, 1.7 * cm, 1.7 * cm, 1.7 * cm, 1.7 * cm, 2.1 * cm, 1.5 * cm],
                            max_rows_per_block=24,
                            interpretation=(
                                "Ce tableau quantifie l'incertitude sur les moyennes (IC95) et vérifie la normalité "
                                "via Shapiro/Jarque-Bera, utile pour choisir les tests paramétriques/non paramétriques."
                            )
                        )
                else:
                    elements.append(para("Aucune variable numérique exploitable dans le dataset."))

                # Section 9 - Catégoriel
                elements.append(PageBreak())
                elements.append(Paragraph("9. Analyse descriptive catégorielle", style_h1))
                if not cat_summary_df.empty:
                    concentrated = cat_summary_df[cat_summary_df["dominance_%"] > 80]
                    cat_interp = (
                        f"{len(concentrated)} variable(s) présente(nt) une modalité dominante > 80%, "
                        "ce qui peut signaler un déséquilibre structurel."
                    )
                    if categorical_commentaries:
                        cat_interp += " " + " ".join(categorical_commentaries[:3])
                    add_df_table(
                        elements,
                        "Tableau 6 - Statistiques catégorielles",
                        cat_summary_df,
                        col_widths=[4.2 * cm, 1.5 * cm, 1.8 * cm, 5.6 * cm, 2.2 * cm, 2.3 * cm],
                        max_rows_per_block=28,
                        interpretation=cat_interp
                    )
                else:
                    elements.append(para("Aucune variable catégorielle/booléenne exploitable."))

                # Section 10 - Bivarié / multivarié
                elements.append(PageBreak())
                elements.append(Paragraph("10. Analyses bivariées et multivariées", style_h1))
                if not corr_pairs_df.empty:
                    add_df_table(
                        elements,
                        "Tableau 7 - Corrélations les plus fortes",
                        corr_pairs_df.head(30)[["var1", "var2", "r", "|r|"]],
                        col_widths=[5.2 * cm, 5.2 * cm, 3.2 * cm, 3.2 * cm],
                        max_rows_per_block=26,
                        interpretation="Les corrélations élevées orientent l'analyse explicative, mais ne prouvent pas la causalité."
                    )
                else:
                    elements.append(para("Corrélations non calculables (moins de 2 variables numériques)."))

                if chi2_result:
                    chi2_text = (
                        f"Test Chi² sur ({chi2_result['var1']} x {chi2_result['var2']}): "
                        f"Chi²={chi2_result['chi2']:.4f}, p-value={chi2_result['pvalue']:.4e}, "
                        f"V de Cramer={chi2_result['cramers_v']:.4f}. "
                        + ("Association significative au seuil 5%." if chi2_result["pvalue"] < 0.05 else "Association non significative au seuil 5%.")
                    )
                    elements.append(para(chi2_text))
                    chi2_tab = chi2_result["table"].copy()
                    chi2_tab = chi2_tab.reset_index().rename(columns={"index": chi2_result["var1"]})
                    add_df_table(
                        elements,
                        "Tableau 8 - Contingence (catégorielle)",
                        chi2_tab,
                        max_rows_per_block=24,
                        interpretation="Ce tableau synthétise la structure conjointe entre les deux variables catégorielles les plus liées."
                    )

                if reg_result and "error" not in reg_result and not reg_coef_df.empty:
                    reg_eq_terms = [
                        f"{row['coefficient']:.4f}*{row['variable']}"
                        for _, row in reg_coef_df.iloc[1:].iterrows()
                    ]
                    reg_eq = f"{reg_target} = {reg_result['intercept']:.4f}" + (
                        " + " + " + ".join(reg_eq_terms) if reg_eq_terms else ""
                    )
                    reg_note = (
                        f"Modele de regression cible={reg_target}: R2={reg_result['r2']:.4f}, RMSE={reg_result['rmse']:.4f}. "
                        "Lecture: un R2 eleve indique une bonne explication de variance; "
                        "verifier en pratique les hypotheses residuelles. "
                        f"Equation estimee: {reg_eq}"
                    )
                    elements.append(para(reg_note))
                    add_df_table(
                        elements,
                        "Tableau 9 - Coefficients du modele de regression",
                        reg_coef_df,
                        col_widths=[7.8 * cm, 9.0 * cm],
                        max_rows_per_block=24,
                        interpretation=(
                            "Les signes des coefficients indiquent le sens de variation de la cible "
                            "quand les autres variables explicatives sont maintenues constantes."
                        ),
                    )

                if auto_anova and auto_anova.get("result"):
                    an_res = auto_anova["result"]
                    an_txt = (
                        f"ANOVA automatique: cible={auto_anova['target']} ; facteur={auto_anova['group']} ; "
                        f"F={an_res['f_statistic']:.4f} ; p-value={an_res['p_value']:.4e} ; eta2={an_res['eta_squared']:.4f}. "
                        + ("Resultat significatif au seuil 5%." if an_res["p_value"] < 0.05 else "Resultat non significatif au seuil 5%.")
                    )
                    elements.append(para(an_txt))

                # Section 11 - Visualisations
                elements.append(PageBreak())
                elements.append(Paragraph("11. Graphiques et interprétations", style_h1))

                # Graphique 1: manquants
                miss_plot = dict_df.sort_values("%_manquants", ascending=False).head(20)
                if not miss_plot.empty and miss_plot["%_manquants"].max() > 0:
                    fig, ax = plt.subplots(figsize=(10.5, 6))
                    vals = miss_plot.iloc[::-1]
                    ax.barh(vals["variable"], vals["%_manquants"], color="#0EA5E9", alpha=0.85)
                    ax.set_xlabel("% de valeurs manquantes")
                    ax.set_ylabel("Variables")
                    ax.set_title("Profil des valeurs manquantes (Top 20)")
                    ax.grid(axis="x", alpha=0.2)
                    add_figure(
                        elements,
                        fig,
                        "Figure 1 - Manquants par variable",
                        "Interprétation: les variables en tête doivent être priorisées pour imputation, exclusion ou collecte complémentaire."
                    )

                # Graphique 2: distributions numériques
                num_cols_plot = self.numeric_cols[:6]
                if num_cols_plot:
                    rows = math.ceil(len(num_cols_plot) / 2)
                    fig, axes = plt.subplots(rows, 2, figsize=(12, 3.2 * rows))
                    axes_arr = np.array(axes).reshape(-1)
                    for idx, col in enumerate(num_cols_plot):
                        ax = axes_arr[idx]
                        s = pd.to_numeric(df[col], errors="coerce").dropna()
                        values = np.asarray(s, dtype=float)
                        if values.size > 0:
                            ax.hist(values, bins=30, color="#2563EB", alpha=0.75, edgecolor="white")
                            ax.axvline(np.nanmean(values), color="#DC2626", linestyle="--", linewidth=1.3, label="Moyenne")
                            ax.axvline(np.nanmedian(values), color="#16A34A", linestyle="--", linewidth=1.3, label="Médiane")
                            ax.legend(fontsize=7, loc="best")
                        ax.set_title(col)
                        ax.set_xlabel("")
                        ax.set_ylabel("Fréquence")
                    for idx in range(len(num_cols_plot), len(axes_arr)):
                        axes_arr[idx].axis("off")
                    fig.suptitle("Distributions numériques (top variables)", y=1.02)
                    plt.tight_layout()
                    add_figure(
                        elements,
                        fig,
                        "Figure 2 - Histogrammes + densités",
                        "Interprétation: la forme des distributions (asymétrie, pics, queues) guide le choix des tests statistiques."
                    )

                # Graphique 3: boxplots outliers
                if self.numeric_cols:
                    sel_cols = self.numeric_cols[:8]
                    plot_df = df[sel_cols].copy()
                    fig, ax = plt.subplots(figsize=(12, 6))
                    series_list = [pd.to_numeric(plot_df[c], errors="coerce").dropna().to_numpy() for c in sel_cols]
                    series_list = [arr for arr in series_list if arr.size > 0]
                    valid_labels = [c for c in sel_cols if pd.to_numeric(plot_df[c], errors="coerce").dropna().shape[0] > 0]
                    if series_list:
                        bp = ax.boxplot(series_list, patch_artist=True, labels=valid_labels, showfliers=True)
                        for box in bp.get("boxes", []):
                            box.set_facecolor("#14B8A6")
                            box.set_alpha(0.65)
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right")
                    ax.set_title("Dispersion et outliers (variables numériques)")
                    ax.set_xlabel("")
                    ax.set_ylabel("Valeurs")
                    plt.tight_layout()
                    add_figure(
                        elements,
                        fig,
                        "Figure 3 - Boxplots multivariés",
                        "Interprétation: les points extrêmes indiquent des observations atypiques à vérifier contre les règles métier."
                    )

                # Graphique 4: heatmap corrélation
                if len(self.numeric_cols) >= 2:
                    corr_cols = self.numeric_cols[:15]
                    corr_m = df[corr_cols].corr()
                    fig, ax = plt.subplots(figsize=(10.5, 8.5))
                    corr_vals = corr_m.to_numpy(dtype=float)
                    im = ax.imshow(corr_vals, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
                    ax.set_xticks(np.arange(len(corr_cols)))
                    ax.set_yticks(np.arange(len(corr_cols)))
                    ax.set_xticklabels(corr_cols, rotation=45, ha="right", fontsize=8)
                    ax.set_yticklabels(corr_cols, fontsize=8)
                    for i in range(corr_vals.shape[0]):
                        for j in range(corr_vals.shape[1]):
                            v = corr_vals[i, j]
                            txt_color = "white" if abs(v) > 0.55 else "black"
                            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color=txt_color)
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Corrélation")
                    ax.set_title("Matrice de corrélation (variables numériques)")
                    plt.tight_layout()
                    interp = "Interprétation: les zones rouge/bleu intense signalent des associations fortes positives/négatives."
                    if strongest_pair is not None:
                        interp += f" Couple le plus corrélé: {strongest_pair['var1']} / {strongest_pair['var2']}."
                    add_figure(elements, fig, "Figure 4 - Heatmap de corrélation", interp)

                # Graphique 5: bar charts catégoriels
                cat_cols_plot = all_cat_cols[:4]
                if cat_cols_plot:
                    rows = len(cat_cols_plot)
                    fig, axes = plt.subplots(rows, 1, figsize=(11, 3.3 * rows))
                    if rows == 1:
                        axes = [axes]
                    for ax, col in zip(axes, cat_cols_plot):
                        counts = df[col].astype(str).fillna("NA").value_counts().head(10)
                        ax.bar(range(len(counts)), counts.values, color="#0EA5E9", alpha=0.85)
                        ax.set_title(col)
                        ax.set_xticks(range(len(counts)))
                        ax.set_xticklabels([str(v)[:18] for v in counts.index], rotation=30, ha="right")
                        ax.set_ylabel("Effectif")
                    fig.suptitle("Top modalités par variable catégorielle", y=1.01)
                    plt.tight_layout()
                    add_figure(
                        elements,
                        fig,
                        "Figure 5 - Distributions catégorielles",
                        "Interprétation: les catégories dominantes permettent d'identifier les segments majoritaires."
                    )

                # Graphique 6: scatter relation la plus forte
                if strongest_pair is not None:
                    x_col = strongest_pair["var1"]
                    y_col = strongest_pair["var2"]
                    pair_df = df[[x_col, y_col]].dropna()
                    if len(pair_df) > 2:
                        fig, ax = plt.subplots(figsize=(9.5, 6.5))
                        ax.scatter(pair_df[x_col], pair_df[y_col], alpha=0.5, color="#2563EB", edgecolors="white", linewidth=0.3)
                        z = np.polyfit(pair_df[x_col], pair_df[y_col], 1)
                        p = np.poly1d(z)
                        x_line = np.linspace(pair_df[x_col].min(), pair_df[x_col].max(), 100)
                        ax.plot(x_line, p(x_line), color="#DC2626", linewidth=2)
                        ax.set_title(f"Relation la plus forte: {x_col} vs {y_col}")
                        ax.set_xlabel(x_col)
                        ax.set_ylabel(y_col)
                        plt.tight_layout()
                        add_figure(
                            elements,
                            fig,
                            "Figure 6 - Nuage de points et tendance",
                            "Interprétation: la pente et la dispersion autour de la droite traduisent l'intensité de la relation linéaire."
                        )

                # Section 12 - Recommandations (avec synthèse IA intégrée)
                elements.append(PageBreak())
                elements.append(Paragraph("12. Recommandations opérationnelles", style_h1))
                recs = []
                if missing_pct >= 20:
                    recs.append("Priorité élevée: plan de traitement des valeurs manquantes (imputation guidée métier ou recollecte).")
                elif missing_pct >= 5:
                    recs.append("Priorité moyenne: traiter les colonnes partiellement manquantes avant modélisation.")
                else:
                    recs.append("Qualité de complétude satisfaisante: maintenir les contrôles de saisie.")

                if duplicates > 0:
                    recs.append("Mettre en place une règle de déduplication en amont pour stabiliser les agrégats.")

                if not outlier_df.empty and float(outlier_df["outliers_%"].max()) > 10:
                    recs.append("Contrôler les variables fortement atypiques (>10% d'outliers) et définir des seuils métier.")

                if strongest_pair is not None and strongest_pair["|r|"] > 0.8:
                    recs.append("Surveiller la multicolinéarité potentielle entre variables fortement corrélées avant régression.")

                if not cat_summary_df.empty and (cat_summary_df["dominance_%"] > 85).any():
                    recs.append("Équilibrer ou segmenter les analyses pour les variables catégorielles très déséquilibrées.")

                if not recs:
                    recs.append("Poursuivre l'analyse avec une segmentation métier et validation statistique croisée.")

                if self.report_brief.get("objectif_general", "").strip():
                    recs.append(
                        f"Aligner la feuille de route d'execution sur l'objectif general defini: "
                        f"{self.report_brief.get('objectif_general', '').strip()}."
                    )
                if self.report_brief.get("limitations_connues", "").strip():
                    recs.append(
                        f"Mettre en place un plan de mitigation des limites declarees: "
                        f"{self.report_brief.get('limitations_connues', '').strip()}."
                    )

                rec_txt = "\n".join([f"• {r}" for r in recs])
                elements.append(para(rec_txt))

                if ai_text:
                    elements.append(Spacer(1, 0.15 * cm))
                    elements.append(Paragraph("Synthèse IA intégrée", style_h2))
                    add_text_with_latex(elements, ai_text, default_style=style_note)
                elif self.enable_ai_report.get() and ai_error:
                    elements.append(Spacer(1, 0.1 * cm))
                    elements.append(para(f"Synthèse IA indisponible: {ai_error}", style_note))

                # Section 13 - Conclusion
                elements.append(PageBreak())
                elements.append(Paragraph("13. Conclusion générale", style_h1))
                outlier_focus = "Aucun signal atypique majeur n'a été identifié."
                if not outlier_df.empty:
                    top_out = outlier_df.iloc[0]
                    outlier_focus = (
                        f"La variable la plus sensible aux atypies est {top_out['variable']} "
                        f"avec {top_out['outliers_%']:.2f}% d'observations hors bornes IQR."
                    )
                assoc_text = "Aucune relation linéaire forte n'a été mise en évidence."
                if strongest_pair is not None:
                    assoc_text = (
                        f"La relation la plus marquée concerne {strongest_pair['var1']} et {strongest_pair['var2']} "
                        f"(r={strongest_pair['r']:.3f}), ce qui constitue un axe prioritaire d'explication."
                    )
                infer_text = ""
                if auto_anova and auto_anova.get("result"):
                    an = auto_anova["result"]
                    infer_text = (
                        f"L'inférence ANOVA (cible={auto_anova['target']}, facteur={auto_anova['group']}) "
                        f"retient p={an['p_value']:.4e} et eta2={an['eta_squared']:.4f}, "
                        "ce qui précise le niveau d'effet observé."
                    )
                objective_txt = self.report_brief.get("objectif_general", "").strip()
                limits_txt = self.report_brief.get("limitations_connues", "").strip()
                conclusion_parts = [
                    f"Le rapport confirme une base de données globalement exploitable, avec une complétude de {100 - missing_pct:.2f}% "
                    f"et un taux de doublons de {duplicates_pct:.2f}%.",
                    assoc_text,
                    outlier_focus,
                    infer_text,
                    ("L'objectif général défini a été adressé de manière structurée: " + objective_txt + ".") if objective_txt else "",
                    ("Les limites déclarées restent à surveiller: " + limits_txt + ".") if limits_txt else "",
                    "Sur le plan opérationnel, la suite recommandée est: fiabilisation des données, validation métier des signaux "
                    "statistiques, puis passage à des analyses explicatives/ prédictives ciblées.",
                ]
                elements.append(para(" ".join([p for p in conclusion_parts if p])))

                # Section 14 - Annexes
                elements.append(Spacer(1, 0.3 * cm))
                elements.append(Paragraph("14. Annexes techniques", style_h1))
                annex_txt = (
                    f"Source: {self.file_path or 'N/A'}\n"
                    f"Filtres actifs: {', '.join(self.active_filters) if self.active_filters else 'Aucun'}\n"
                    f"Theme du rapport: {self.report_brief.get('theme', '') or 'Non renseigne'}\n"
                    f"Public cible: {self.report_brief.get('public_cible', '') or 'Non renseigne'}\n"
                    f"Contexte IA exporté: {context_json_path if context_json_path else 'non exporté'}\n"
                    f"Config IA: {self.hf_config_path}\n"
                    f"Prompt IA: {self.hf_prompt_path}\n"
                    f"Variables numériques: {', '.join(self.numeric_cols) if self.numeric_cols else 'Aucune'}\n"
                    f"Variables catégorielles: {', '.join(all_cat_cols) if all_cat_cols else 'Aucune'}"
                )
                elements.append(para(annex_txt, style_note))

                doc.build(elements)

                done_msg = f"✅ Rapport complet généré: {output_path}"
                if context_json_path:
                    done_msg += f"\n📎 Contexte IA exporté: {context_json_path}"
                if self.enable_ai_report.get() and ai_error:
                    done_msg += f"\n⚠️ IA indisponible: {ai_error}"
                self.show_message(done_msg, "success")
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            
        except ImportError:
            self.show_message("❌ Installez reportlab: pip install reportlab", "error")
        except Exception as e:
            self.show_message(f"❌ Erreur: {str(e)}", "error")
            
    # ========================================================================
    # UTILITAIRES
    # ========================================================================
    
    def clear_results(self):
        """Efface les résultats"""
        self.txt_results.delete("1.0", "end")
        
    def print_header(self, title):
        """Affiche un en-tête"""
        self.txt_results.insert("end", "═" * 70 + "\n")
        self.txt_results.insert("end", f"   {title}\n")
        self.txt_results.insert("end", "═" * 70 + "\n")
        
    def show_message(self, msg, msg_type="info"):
        """Affiche un message"""
        prefix = {"success": "✅", "error": "❌", "warning": "⚠️", "info": "ℹ️"}
        self.clear_results()
        self.txt_results.insert("1.0", f"{prefix.get(msg_type, '')} {msg}\n")
        
    def show_error(self):
        """Affiche une erreur"""
        self.show_message("Veuillez charger un fichier", "error")


# ============================================================================
# POINT D'ENTRÉE
# ============================================================================
