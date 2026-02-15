#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EDA-Desk Pro - Exploratory Data Analysis Tool
Version Modern UI avec architecture subdivisée.
"""

import faulthandler
import os
import site
import subprocess
import sys
import tkinter as tk

# Chargés dynamiquement pour gérer les conflits numpy/matplotlib.
Colors = None
DataSourceManager = None
EDADeskPro = None
HuggingFaceReportAssistant = None
MultivariateAnalysis = None

faulthandler.enable()

__all__ = [
    "Colors",
    "DataSourceManager",
    "EDADeskPro",
    "HuggingFaceReportAssistant",
    "MultivariateAnalysis",
]


# ============================================================================
# POINT D'ENTRÉE
# ============================================================================

def _looks_like_numpy_matplotlib_conflict(exc: Exception) -> bool:
    """Détecte un conflit ABI numpy/matplotlib."""
    txt = f"{exc}"
    markers = (
        "_ARRAY_API",
        "numpy.core.multiarray failed to import",
        "A module that was compiled using NumPy 1.x cannot be run in NumPy 2",
    )
    return any(m in txt for m in markers)


def _move_user_site_after_dist_packages():
    """Place ~/.local après les dist-packages système pour prioriser la pile apt."""
    user_site = site.getusersitepackages()
    if not user_site or user_site not in sys.path:
        return
    sys.path.remove(user_site)
    dist_positions = [i for i, p in enumerate(sys.path) if p.endswith("/dist-packages")]
    insert_at = (max(dist_positions) + 1) if dist_positions else len(sys.path)
    sys.path.insert(insert_at, user_site)


def _purge_module_cache():
    """Nettoie les modules déjà importés avant un second essai."""
    prefixes = ("numpy", "matplotlib", "pandas", "seaborn", "eda_modern")
    for name in list(sys.modules.keys()):
        if any(name == p or name.startswith(f"{p}.") for p in prefixes):
            del sys.modules[name]


def _load_components():
    """Charge les composants applicatifs avec stratégie de fallback."""
    global Colors, DataSourceManager, EDADeskPro, HuggingFaceReportAssistant, MultivariateAnalysis

    try:
        from eda_modern import (
            Colors as _Colors,
            DataSourceManager as _DataSourceManager,
            EDADeskPro as _EDADeskPro,
            HuggingFaceReportAssistant as _HuggingFaceReportAssistant,
            MultivariateAnalysis as _MultivariateAnalysis,
        )
        Colors = _Colors
        DataSourceManager = _DataSourceManager
        EDADeskPro = _EDADeskPro
        HuggingFaceReportAssistant = _HuggingFaceReportAssistant
        MultivariateAnalysis = _MultivariateAnalysis
        return
    except Exception as first_exc:
        if not _looks_like_numpy_matplotlib_conflict(first_exc):
            raise

    _move_user_site_after_dist_packages()
    _purge_module_cache()

    try:
        from eda_modern import (
            Colors as _Colors,
            DataSourceManager as _DataSourceManager,
            EDADeskPro as _EDADeskPro,
            HuggingFaceReportAssistant as _HuggingFaceReportAssistant,
            MultivariateAnalysis as _MultivariateAnalysis,
        )
        Colors = _Colors
        DataSourceManager = _DataSourceManager
        EDADeskPro = _EDADeskPro
        HuggingFaceReportAssistant = _HuggingFaceReportAssistant
        MultivariateAnalysis = _MultivariateAnalysis
    except Exception as second_exc:
        raise RuntimeError(
            "Conflit d'environnement Python détecté.\n"
            "Action recommandée:\n"
            "  1) sudo apt install -y python3-pandas python3-seaborn python3-sklearn\n"
            "  2) Relancer: python3 eda_desk_modern.py\n"
            f"Détail: {second_exc}"
        ) from second_exc

def _customtkinter_preflight() -> bool:
    """Teste la stabilité des widgets CustomTkinter utilisés par l'application."""
    code = r"""
import customtkinter as ctk
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")
root = ctk.CTk()
root.geometry("220x120")
frame = ctk.CTkFrame(root)
frame.pack(fill="both", expand=True, padx=5, pady=5)
btn1 = ctk.CTkButton(frame, text="Open", font=("TkDefaultFont", 12, "bold"))
btn1.pack(fill="x", padx=5, pady=5)
btn2 = ctk.CTkButton(frame, text="API", width=110, font=("TkDefaultFont", 12, "bold"))
btn2.pack(fill="x", padx=5, pady=5)
theme_var = ctk.StringVar(value="Light")
theme_menu = ctk.CTkOptionMenu(frame, variable=theme_var, values=["Dark", "Light", "System"])
theme_menu.pack(fill="x", padx=5, pady=5)
root.update_idletasks()
root.update()
root.destroy()
"""
    try:
        result = subprocess.run(
            [sys.executable, "-X", "faulthandler", "-c", code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
            check=False,
        )
        return result.returncode == 0
    except Exception:
        return False


if __name__ == "__main__":
    # Priorise la pile scientifique système pour limiter les conflits ABI.
    _move_user_site_after_dist_packages()
    force_classic = os.environ.get("EDA_DESK_FORCE_CLASSIC", "0") == "1"
    if force_classic or not _customtkinter_preflight():
        print("⚠️ CustomTkinter instable sur ce système. Bascule vers eda_desk_pro.py")
        try:
            from eda_desk_pro import main as fallback_main
            fallback_main()
        except tk.TclError as e:
            print(f"Erreur Tkinter: {e}")
            print("Vérifiez que votre session graphique (DISPLAY) est active.")
        except Exception as e:
            print(f"❌ {e}")
            print("Fix recommandé:")
            print("  sudo apt install -y python3-pandas python3-seaborn python3-sklearn")
            print("Puis relancez: python3 eda_desk_modern.py")
    else:
        try:
            _load_components()
            app = EDADeskPro()
            app.mainloop()
        except RuntimeError as e:
            print(f"❌ {e}")
        except tk.TclError as e:
            print(f"Erreur Tkinter: {e}")
            print("Vérifiez que votre session graphique (DISPLAY) est active.")
