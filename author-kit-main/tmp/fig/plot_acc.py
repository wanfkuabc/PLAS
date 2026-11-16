import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

# ============================================================
# 0. CVPR-friendly global style
# ============================================================
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.5,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# ============================================================
# 1. RefCOCO results
#    Mean = (val + testA) / 2
# ============================================================
methods = [
    # name, year, MLLM size(B), val, testA
    ("LISA-7B",          2024,  7.0, 74.9, 79.1),
    ("GLaMM-7B",         2024,  7.0, 79.5, 83.2),
    ("OMG-LLaVA-7B",     2024,  7.0, 78.0, 80.3),

    ("Text4Seg-7B",      2025,  7.0, 79.3, 81.9),
    ("Text4Seg-13B",     2025, 13.0, 80.2, 82.7),
    ("SegLLM-7B",        2025,  7.0, 80.2, 81.5),
    ("UniPixel-7B",      2025,  7.0, 80.8, 83.0),
    ("LIRA-8B",          2025,  8.0, 81.8, 83.4),
    ("HiMTok-8B",        2025,  8.0, 85.9, 86.3),

    ("X-SAM-3.8B",       2026,  3.8, 85.1, 87.1),

    # Ours
    ("LLaVA6-3-8",       2026,  8.0, 85.66397966, 86.55751468),
]

data = []

for name, year, size_b, val, testa in methods:
    mean_score = (val + testa) / 2.0

    data.append({
        "name": name,
        "year": year,
        "size_b": size_b,
        "val": val,
        "testA": testa,
        "mean": mean_score,
    })

# ============================================================
# 2. IMPORTANT:
#    Replace with YOUR EXACT measured trainable parameters.
# ============================================================

OURS_TRAINABLE_M = 200.0
# !!! REPLACE 200.0 !!!
# Example:
# if your program prints 186.42 M:
# OURS_TRAINABLE_M = 186.42


# ============================================================
# 3. Trainable-parameter comparison
#
# exact:
# Text4Seg-7B = 160M
#
# lower-bound:
# GLaMM:
#   LoRA = 20M
#   text embedding = 262M
#   PLUS trainable region encoder / projection / pixel decoder
#   => actual > 282M
#
# X-SAM:
#   final mixed fine-tuning trains all parameters.
#   Phi-3 LLM alone = 3.8B
#   => actual system trainable count > 3800M
# ============================================================

efficiency = [
    {
        "name": "Text4Seg-7B",
        "params": 160,
        "score": (79.3 + 81.9) / 2,
        "bound": False,
    },
    {
        "name": "GLaMM-7B",
        "params": 282,
        "score": (79.5 + 83.2) / 2,
        "bound": True,       # lower bound
    },
    {
        "name": "X-SAM-3.8B",
        "params": 3800,
        "score": (85.1 + 87.1) / 2,
        "bound": True,       # LLM-only lower bound
    },
    {
        "name": "LLaVA6-3-8",
        "params": OURS_TRAINABLE_M,
        "score": (85.66397966 + 86.55751468) / 2,
        "bound": False,
    },
]

# ============================================================
# 4. Color palette
# ============================================================

COLORS = {
    2024: "#9CA3AF",    # gray
    2025: "#4C78A8",    # muted blue
    2026: "#E69F50",    # warm orange
}

OURS_COLOR = "#D62728"
OURS_EDGE = "#8B0000"

# ============================================================
# 5. Figure
# ============================================================

fig = plt.figure(figsize=(7.15, 3.05))

gs = fig.add_gridspec(
    1, 2,
    width_ratios=[1.48, 1.0],
    wspace=0.26
)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])


# ============================================================
# PANEL A
# Accuracy vs. MLLM size
# ============================================================

for d in data:

    if d["name"] == "LLaVA6-3-8":

        ax1.scatter(
            d["size_b"],
            d["mean"],
            marker="*",
            s=230,
            facecolor=OURS_COLOR,
            edgecolor=OURS_EDGE,
            linewidth=1.0,
            zorder=10,
        )

    else:

        ax1.scatter(
            d["size_b"],
            d["mean"],
            s=48,
            facecolor=COLORS[d["year"]],
            edgecolor="white",
            linewidth=0.7,
            alpha=0.95,
            zorder=4,
        )


# ------------------------------------------------------------
# label placement
# ------------------------------------------------------------

offsets = {
    "LISA-7B":       (-8, -12),
    "GLaMM-7B":      (-38, 6),
    "OMG-LLaVA-7B":  (-42, -3),

    "Text4Seg-7B":   (-48, 8),
    "Text4Seg-13B":  (-29, 8),
    "SegLLM-7B":     (5, -11),
    "UniPixel-7B":   (5, 5),
    "LIRA-8B":       (5, -10),
    "HiMTok-8B":     (6, 5),

    "X-SAM-3.8B":    (5, 5),

    "LLaVA6-3-8":    (7, 9),
}

for d in data:

    dx, dy = offsets.get(d["name"], (4, 4))

    if d["name"] == "LLaVA6-3-8":
        text = (
            f"Ours\n"
            f"{d['mean']:.2f}"
        )

        ax1.annotate(
            text,
            (d["size_b"], d["mean"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8.4,
            fontweight="bold",
            color=OURS_COLOR,
            ha="left",
            va="bottom",
        )

    else:
        ax1.annotate(
            d["name"],
            (d["size_b"], d["mean"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=6.7,
            color="#343A40",
        )


# ------------------------------------------------------------
# Highlight top-performing region
# ------------------------------------------------------------

ax1.axhspan(
    85.5, 86.6,
    alpha=0.055,
    color=OURS_COLOR,
    zorder=0
)

ax1.text(
    12.85,
    86.43,
    "Top-tier regime",
    fontsize=7,
    color=OURS_COLOR,
    ha="right",
    va="top",
)


# Better direction
ax1.annotate(
    "Better",
    xy=(3.9, 86.35),
    xytext=(5.15, 84.9),
    arrowprops=dict(
        arrowstyle="->",
        linewidth=0.8,
        color="#666666",
    ),
    fontsize=7,
    color="#666666",
    ha="center",
)


ax1.set_xlabel("Base MLLM size (B)")
ax1.set_ylabel(r"Mean RefCOCO cIoU  $\uparrow$")
ax1.set_title("(a) Performance vs. model scale", loc="left",
              fontweight="bold")

ax1.set_xlim(3.2, 13.7)
ax1.set_ylim(76.0, 87.0)

ax1.set_xticks([4, 7, 8, 13])

ax1.grid(
    axis="y",
    linestyle="--",
    linewidth=0.55,
    alpha=0.28
)

ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)


# ============================================================
# PANEL B
# Accuracy vs. trainable parameters
# ============================================================

for d in efficiency:

    is_ours = d["name"] == "LLaVA6-3-8"

    if is_ours:
        marker = "*"
        size = 245
        fc = OURS_COLOR
        ec = OURS_EDGE
        z = 10
    else:
        marker = "o"
        size = 60
        fc = "#4C78A8"
        ec = "white"
        z = 5

    ax2.scatter(
        d["params"],
        d["score"],
        marker=marker,
        s=size,
        facecolor=fc,
        edgecolor=ec,
        linewidth=0.9,
        zorder=z,
    )

    # --------------------------------------------------------
    # Lower-bound arrow
    # --------------------------------------------------------
    if d["bound"]:

        x1 = d["params"]
        x2 = d["params"] * 1.55

        ax2.annotate(
            "",
            xy=(x2, d["score"]),
            xytext=(x1 * 1.05, d["score"]),
            arrowprops=dict(
                arrowstyle="->",
                linewidth=0.9,
                color="#777777",
            ),
        )


# ------------------------------------------------------------
# labels panel B
# ------------------------------------------------------------

labels_eff = {
    "Text4Seg-7B":
        ("Text4Seg\n160M", (5, -18)),

    "GLaMM-7B":
        ("GLaMM\n≥282M", (5, 7)),

    "X-SAM-3.8B":
        ("X-SAM\n≥3.8B", (-10, 8)),

    "LLaVA6-3-8":
        (f"Ours\n{OURS_TRAINABLE_M:.0f}M", (7, 8)),
}

for d in efficiency:

    txt, off = labels_eff[d["name"]]

    ax2.annotate(
        txt,
        (d["params"], d["score"]),
        xytext=off,
        textcoords="offset points",
        fontsize=7.3 if d["name"] != "LLaVA6-3-8" else 8,
        fontweight="bold"
        if d["name"] == "LLaVA6-3-8"
        else "normal",
        color=OURS_COLOR
        if d["name"] == "LLaVA6-3-8"
        else "#343A40",
    )


ax2.set_xscale("log")

ax2.set_xlabel("Trainable parameters (M, log scale)")
ax2.set_ylabel(r"Mean RefCOCO cIoU  $\uparrow$")
ax2.set_title(
    "(b) Performance vs. training cost",
    loc="left",
    fontweight="bold",
)

ax2.set_xlim(100, 8000)
ax2.set_ylim(79.0, 87.0)

ax2.grid(
    axis="both",
    linestyle="--",
    linewidth=0.55,
    alpha=0.25
)

ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)


# ------------------------------------------------------------
# "Better" Pareto direction
# upper-left = higher accuracy + fewer trainable params
# ------------------------------------------------------------

ax2.annotate(
    "Better",
    xy=(130, 86.55),
    xytext=(400, 84.7),
    arrowprops=dict(
        arrowstyle="->",
        linewidth=0.8,
        color="#666666",
    ),
    fontsize=7,
    color="#666666",
    ha="center",
)


# ============================================================
# 6. Legend
# ============================================================

legend_elements = [
    Line2D(
        [0], [0],
        marker="o",
        color="none",
        markerfacecolor=COLORS[2024],
        markeredgecolor="white",
        markersize=6,
        label="2024",
    ),
    Line2D(
        [0], [0],
        marker="o",
        color="none",
        markerfacecolor=COLORS[2025],
        markeredgecolor="white",
        markersize=6,
        label="2025",
    ),
    Line2D(
        [0], [0],
        marker="o",
        color="none",
        markerfacecolor=COLORS[2026],
        markeredgecolor="white",
        markersize=6,
        label="2026",
    ),
    Line2D(
        [0], [0],
        marker="*",
        color="none",
        markerfacecolor=OURS_COLOR,
        markeredgecolor=OURS_EDGE,
        markersize=9,
        label="Ours",
    ),
]

ax1.legend(
    handles=legend_elements,
    frameon=False,
    ncol=4,
    loc="lower right",
    columnspacing=0.8,
    handletextpad=0.25,
)


# ============================================================
# 7. Main title / subtitle
# ============================================================

fig.suptitle(
    "LLaVA6-3-8: Strong RefCOCO Accuracy with Parameter-Efficient Adaptation",
    fontsize=10.6,
    fontweight="bold",
    y=1.005,
)

fig.text(
    0.5,
    0.955,
    r"Mean cIoU = (val + testA) / 2; arrows denote lower bounds on trainable parameters.",
    ha="center",
    fontsize=7.1,
    color="#555555",
)


# ============================================================
# 8. Export
# ============================================================

plt.subplots_adjust(
    left=0.075,
    right=0.985,
    bottom=0.19,
    top=0.84,
)

plt.savefig(
    "refcoco_accuracy_efficiency.pdf",
    bbox_inches="tight",
)

plt.savefig(
    "refcoco_accuracy_efficiency.png",
    dpi=600,
    bbox_inches="tight",
)

plt.show()
