"""Widgets CustomTkinter réutilisables."""

import customtkinter as ctk

from .theme import Colors


class ModernCard(ctk.CTkFrame):
    """Carte moderne avec bordure arrondie"""

    def __init__(self, master, title: str = "", **kwargs):
        super().__init__(
            master,
            corner_radius=16,
            fg_color=Colors.BG_CARD,
            border_width=1,
            border_color=Colors.BORDER,
            **kwargs
        )

        if title:
            title_label = ctk.CTkLabel(
                self, text=title,
                font=ctk.CTkFont(size=14, weight="bold"),
                text_color=Colors.PRIMARY
            )
            title_label.pack(padx=15, pady=(15, 5), anchor="w")

        self.content_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.content_frame.pack(fill="both", expand=True, padx=15, pady=(5, 15))


class ModernButton(ctk.CTkButton):
    """Bouton moderne avec effets"""

    def __init__(self, master, text: str, command=None, style: str = "primary", **kwargs):
        colors = {
            "primary": (Colors.PRIMARY, Colors.PRIMARY_HOVER),
            "success": (Colors.SUCCESS, "#15803D"),
            "warning": (Colors.WARNING, "#D97706"),
            "danger": (Colors.DANGER, "#B91C1C"),
            "secondary": ("#E2ECF8", "#CFDFF2")
        }

        fg_color, hover_color = colors.get(style, colors["primary"])

        safe_text = "".join(ch for ch in str(text) if ch.isprintable() and ord(ch) < 0x10000).strip()

        super().__init__(
            master, text=(safe_text or "Action"), command=command,
            fg_color=fg_color, hover_color=hover_color,
            corner_radius=10, height=38,
            border_width=1,
            border_color=Colors.BORDER,
            font=("TkDefaultFont", 12, "bold"),
            text_color=Colors.TEXT_PRIMARY if style == "secondary" else "#FFFFFF",
            **kwargs
        )


class StatCard(ctk.CTkFrame):
    """Carte de statistique compacte"""

    def __init__(self, master, label: str, value: str, icon: str = "", **kwargs):
        super().__init__(
            master,
            corner_radius=14,
            fg_color=Colors.BG_CARD,
            border_width=1,
            border_color=Colors.BORDER,
            **kwargs
        )

        container = ctk.CTkFrame(self, fg_color="transparent")
        container.pack(fill="both", expand=True, padx=15, pady=12)

        header = ctk.CTkFrame(container, fg_color="transparent")
        header.pack(fill="x")

        if icon:
            ctk.CTkLabel(header, text=icon, font=ctk.CTkFont(size=18)).pack(side="left")

        ctk.CTkLabel(
            header, text=label,
            font=ctk.CTkFont(size=11),
            text_color=Colors.TEXT_SECONDARY
        ).pack(side="left", padx=(5 if icon else 0, 0))

        ctk.CTkLabel(
            container, text=value,
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color=Colors.TEXT_PRIMARY
        ).pack(anchor="w", pady=(5, 0))
