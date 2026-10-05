"""Єдиний стиль для графіків.

Палета взята з валідованого набору (перевірена на розрізнюваність при
дальтонізмі): перші три категоріальні слоти проходять усі перевірки для
всіх пар. Правила, яких тримаємось:
  - послідовна шкала (магнітуда) — один відтінок від світлого до темного;
  - розбіжна шкала (полярність, напр. кореляція) — два відтінки + нейтральний
    сірий у центрі, ніколи не "веселка";
  - текст завжди чорнильного кольору, ніколи кольором серії;
  - сітка й осі приглушені, дані — на першому плані;
  - жодних двох осей Y на одному графіку.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt

# Категоріальні слоти (фіксований порядок, не циклічний)
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
CATEGORICAL = [BLUE, ORANGE, AQUA]

# Розбіжна шкала для величин із полярністю (кореляції): синій <-> червоний
# через нейтральний сірий у нулі. Не "веселка" — середина має читатись як "нічого".
DIVERGING = "RdBu_r"

# Чорнило та поверхня
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8882"
GRID = "#e3e2de"

RED = "#e34948"   # статусний колір: лише для позначення проблемних значень


def set_style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "figure.dpi": 110,
        "savefig.dpi": 110,
        "savefig.bbox": "tight",

        "axes.edgecolor": GRID,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelcolor": INK_SECONDARY,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.titlepad": 10,
        "axes.labelsize": 9,

        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "grid.alpha": 1.0,
        "axes.axisbelow": True,

        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "xtick.direction": "out",
        "ytick.direction": "out",

        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 2.0,
        "lines.markersize": 5,
        "font.size": 9.5,
        "figure.titlesize": 12,
    })
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=CATEGORICAL)
