"""
diagrams.py — Architecture & Result Diagrams
==============================================
Generates all architecture / pipeline / evaluation figures used in the
project report. Each function below writes one PNG into the current
directory.

Run this script after `python main.py` so that `results.json` exists.
"""

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np


OUTDIR = Path(".")


# ──────────────────────────────────────────────────────────────────────
# 1. Three-layer system architecture
# ──────────────────────────────────────────────────────────────────────

def diagram_system_architecture(path="fig_architecture.png"):
    fig, ax = plt.subplots(figsize=(13, 8.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.set_axis_off()

    def box(x, y, w, h, label, color, sub=None, fontsize=11):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
            boxstyle="round,pad=0.04,rounding_size=0.18",
            linewidth=1.4, edgecolor="#222", facecolor=color))
        ax.text(x + w / 2, y + h / 2 + (0.18 if sub else 0),
                label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold")
        if sub:
            ax.text(x + w / 2, y + h / 2 - 0.30, sub,
                    ha="center", va="center", fontsize=8.5, color="#333")

    def arrow(x1, y1, x2, y2, label=None):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
            arrowstyle="-|>", mutation_scale=18, color="#333", lw=1.4))
        if label:
            ax.text((x1 + x2) / 2 + 0.05, (y1 + y2) / 2 + 0.18,
                    label, fontsize=8.5, color="#444",
                    style="italic")

    # Title
    ax.text(5, 9.55, "Smart Agromet Advisory System — Three-Layer Architecture",
            ha="center", fontsize=15, fontweight="bold")

    # User layer
    box(3.7, 8.2, 2.6, 0.75, "Farmer (Natural Language Query)",
        "#FFFDE7", fontsize=11)

    # Layer 1: Knowledge Graph
    box(0.4, 5.9, 3.0, 1.7,
        "Layer 1 · Knowledge Graph",
        "#C8E6C9",
        sub="NetworkX DiGraph\nCrop · Soil · Weather · Advisory\naffects · requires · causes · recommended_for")

    # Layer 2: RAG
    box(3.7, 5.9, 2.6, 1.7,
        "Layer 2 · RAG",
        "#BBDEFB",
        sub="TF-IDF index over\nadvisory + KG-context docs\ncosine similarity, top-K")

    # Layer 3: LLM
    box(6.6, 5.9, 3.0, 1.7,
        "Layer 3 · LLM Generation",
        "#FFE0B2",
        sub="Gemini / OpenAI / template fallback\nconstrained system prompt\ngrounded composer offline")

    # Entity Extraction
    box(0.4, 4.2, 3.0, 0.95,
        "Entity Extraction (rule-based NER)",
        "#F8BBD0",
        sub="crop / weather / soil / category", fontsize=10)

    # Context Fusion
    box(3.7, 4.2, 2.6, 0.95,
        "Context Fusion",
        "#D1C4E9",
        sub="KG sub-graph + top-K docs", fontsize=10)

    # Evaluator
    box(6.6, 4.2, 3.0, 0.95,
        "Evaluator",
        "#CFD8DC",
        sub="Rel · Spec · Grd · P/R/F1 · Halluc · Lat", fontsize=10)

    # Output
    box(3.0, 2.2, 4.0, 1.4,
        "Grounded Advisory Response",
        "#FFCCBC",
        sub="specific dosages · timings · KG reasoning path",
        fontsize=11)

    # Dataset (foundation)
    box(2.0, 0.4, 6.0, 1.1,
        "Dataset (CROPS · SOILS · WEATHER · ADVISORIES · RELATIONS)",
        "#E0E0E0",
        sub="dataset.py — 6 crops · 4 soils · 6 weather conditions · 10 advisories · 38 relations")

    # Arrows
    arrow(5, 8.18, 5, 7.65, "query")
    arrow(2.0, 7.55, 2.0, 5.18, "subgraph")
    arrow(5.0, 7.55, 5.0, 5.18, "top-K")
    arrow(8.0, 7.55, 8.0, 5.18, "prompt")
    arrow(2.0, 4.18, 5.0, 4.18)
    arrow(5.0, 4.18, 6.6, 4.65)
    arrow(5.0, 4.18, 5.0, 3.65)
    arrow(5.0, 2.2, 5.0, 1.55, "stored")
    arrow(5.0, 1.5, 8.0, 4.18, "compare")

    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


# ──────────────────────────────────────────────────────────────────────
# 2. Pipeline / data-flow diagram
# ──────────────────────────────────────────────────────────────────────

def diagram_pipeline(path="fig_pipeline.png"):
    fig, ax = plt.subplots(figsize=(15, 5.5))
    ax.set_xlim(0, 15); ax.set_ylim(0, 6)
    ax.set_axis_off()

    steps = [
        ("Query",                  "user input",                "#FFE0B2"),
        ("Entity\nExtraction",     "crop · weather\nsoil · cat","#F8BBD0"),
        ("KG\nTraversal",          "multi-hop\n(affects, …)",   "#C8E6C9"),
        ("RAG\nRetrieval",         "TF-IDF + cosine\ntop-K=3",  "#BBDEFB"),
        ("Context\nFusion",        "KG ∪ RAG",                  "#D1C4E9"),
        ("LLM /\nComposer",        "grounded\ngeneration",      "#FFCCBC"),
        ("Advisory\nResponse",     "with reasoning\npath",      "#DCEDC8"),
    ]

    n = len(steps)
    box_w, box_h = 1.7, 1.6
    gap = (15 - n * box_w) / (n + 1)

    centers = []
    for i, (label, sub, color) in enumerate(steps):
        x = gap + i * (box_w + gap)
        y = 2.5
        ax.add_patch(FancyBboxPatch((x, y), box_w, box_h,
            boxstyle="round,pad=0.04,rounding_size=0.15",
            linewidth=1.3, edgecolor="#222", facecolor=color))
        ax.text(x + box_w / 2, y + box_h - 0.45, label,
                ha="center", va="center", fontsize=11, fontweight="bold")
        ax.text(x + box_w / 2, y + 0.45, sub,
                ha="center", va="center", fontsize=8.5, color="#333")
        centers.append((x + box_w / 2, y + box_h / 2))

    for (x1, y1), (x2, y2) in zip(centers[:-1], centers[1:]):
        ax.add_patch(FancyArrowPatch(
            (x1 + box_w / 2, y1), (x2 - box_w / 2, y2),
            arrowstyle="-|>", mutation_scale=16, color="#333", lw=1.5))

    # Annotation underneath
    notes = [
        "natural\nlanguage",
        "rule-based\nNER",
        "structured\nreasoning",
        "lexical\ngrounding",
        "fused\nprompt",
        "constrained\noutput",
        "explainable\nadvisory",
    ]
    for (cx, cy), note in zip(centers, notes):
        ax.text(cx, cy - box_h / 2 - 0.4, note,
                ha="center", fontsize=8, color="#555", style="italic")

    ax.text(7.5, 5.5, "Proposed Pipeline — End-to-End Data Flow",
            ha="center", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


# ──────────────────────────────────────────────────────────────────────
# 3. KG schema diagram (entity types & relations)
# ──────────────────────────────────────────────────────────────────────

def diagram_kg_schema(path="fig_kg_schema.png"):
    fig, ax = plt.subplots(figsize=(12, 7.5))
    ax.set_xlim(0, 12); ax.set_ylim(0, 8)
    ax.set_axis_off()

    entities = {
        "Weather":  (2.0, 6.0, "#1E88E5",
                     "id, condition, rainfall_mm,\nhumidity_pct, temp_c, impact"),
        "Crop":     (6.0, 6.0, "#2E7D32",
                     "id, name, season, water_need,\nsoil_preference, temp_range,\ndiseases, growth_stages"),
        "Soil":     (10.0, 6.0, "#6D4C41",
                     "id, name, type, ph_range,\nfertility, water_retention,\nsuitable_crops"),
        "Advisory": (4.0, 1.6, "#F4511E",
                     "id, crop, category, condition,\nsoil, advisory_text"),
        "Condition": (9.0, 1.6, "#8E24AA",
                     "id, name (waterlogging,\nfrost_damage, heat_stress, …)"),
    }
    centers = {}
    for name, (cx, cy, color, attrs) in entities.items():
        w, h = 2.6, 1.6
        ax.add_patch(FancyBboxPatch(
            (cx - w / 2, cy - h / 2), w, h,
            boxstyle="round,pad=0.04,rounding_size=0.18",
            facecolor=color, edgecolor="black", linewidth=1.4, alpha=0.92))
        ax.text(cx, cy + h / 2 - 0.32, name,
                ha="center", va="center", color="white",
                fontsize=14, fontweight="bold")
        ax.text(cx, cy - 0.2, attrs,
                ha="center", va="center", color="white",
                fontsize=8.5)
        centers[name] = (cx, cy)

    def rel(src, tgt, label, dx=0, dy=0, color="#222"):
        x1, y1 = centers[src]; x2, y2 = centers[tgt]
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
            arrowstyle="-|>", mutation_scale=20, color=color, lw=1.6,
            connectionstyle="arc3,rad=0.18"))
        mx, my = (x1 + x2) / 2 + dx, (y1 + y2) / 2 + dy
        ax.text(mx, my, label,
                ha="center", va="center", fontsize=11, fontweight="bold",
                color=color, bbox=dict(facecolor="white", edgecolor="none",
                                       alpha=0.85, pad=2.5))

    rel("Weather", "Crop",     "affects",         dy=0.4, color="#1565C0")
    rel("Crop",    "Soil",     "requires",        dy=0.4, color="#5D4037")
    rel("Weather", "Condition","causes",          dx=-0.5, color="#6A1B9A")
    rel("Advisory","Crop",     "recommended_for", dy=-0.2, color="#E65100")

    ax.text(6, 7.5, "Knowledge Graph Schema — Entities & Relations",
            ha="center", fontsize=15, fontweight="bold")
    ax.text(6, 0.3,
            "Cardinalities are many-to-many. The KG is a directed graph "
            "(`networkx.DiGraph`) but is acyclic on the present data.",
            ha="center", fontsize=9, style="italic", color="#555")

    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


# ──────────────────────────────────────────────────────────────────────
# 4. Module / component diagram
# ──────────────────────────────────────────────────────────────────────

def diagram_components(path="fig_components.png"):
    fig, ax = plt.subplots(figsize=(13, 7.5))
    ax.set_xlim(0, 13); ax.set_ylim(0, 8)
    ax.set_axis_off()

    modules = {
        "dataset.py":    (2.0, 1.0, "#FFE0B2",
                          "CROPS, SOILS, WEATHER\nADVISORIES, RELATIONS\nTEST_QUERIES"),
        "kg.py":         (5.5, 4.0, "#C8E6C9",
                          "AgroKnowledgeGraph\nextract_entities_from_query\nquery_context, format_context_for_llm"),
        "rag.py":        (9.5, 4.0, "#BBDEFB",
                          "TFIDFRetriever\nRAGSystem\nretrieve_context, format_retrieved_context"),
        "baseline.py":   (2.0, 6.5, "#FFCCBC",
                          "BaselineSystem\nget_llm_response\n(_template_response fallback)"),
        "proposed.py":   (7.0, 6.5, "#D1C4E9",
                          "ProposedSystem\n_compose_grounded_response\n(uses kg + rag)"),
        "evaluation.py": (11.0, 6.5, "#CFD8DC",
                          "AdvisoryEvaluator\nrelevance · specificity · grounding\nprecision · recall · F1 · latency"),
        "main.py":       (7.0, 1.0, "#FFFDE7",
                          "phase1_kg → phase2_rag\nphase3_run → phase4_eval\nphase5_paper_outputs"),
        "visualize.py":  (11.0, 1.0, "#F8BBD0",
                          "generate_kg_diagram"),
        "diagrams.py":   (11.0, 2.5, "#DCEDC8",
                          "all architecture +\nresult plots"),
    }
    centers = {}
    for name, (cx, cy, color, body) in modules.items():
        w, h = 2.6, 1.5
        ax.add_patch(FancyBboxPatch(
            (cx - w / 2, cy - h / 2), w, h,
            boxstyle="round,pad=0.04,rounding_size=0.15",
            facecolor=color, edgecolor="#222", linewidth=1.2))
        ax.text(cx, cy + h / 2 - 0.28, name,
                ha="center", va="center", fontsize=12, fontweight="bold")
        ax.text(cx, cy - 0.2, body,
                ha="center", va="center", fontsize=8.5, color="#333")
        centers[name] = (cx, cy)

    edges = [
        ("kg.py", "dataset.py"),
        ("rag.py", "dataset.py"),
        ("baseline.py", "dataset.py"),
        ("proposed.py", "kg.py"),
        ("proposed.py", "rag.py"),
        ("proposed.py", "baseline.py"),
        ("evaluation.py", "dataset.py"),
        ("main.py", "kg.py"),
        ("main.py", "rag.py"),
        ("main.py", "baseline.py"),
        ("main.py", "proposed.py"),
        ("main.py", "evaluation.py"),
        ("visualize.py", "kg.py"),
        ("diagrams.py", "main.py"),
    ]
    for src, tgt in edges:
        x1, y1 = centers[src]; x2, y2 = centers[tgt]
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
            arrowstyle="-|>", mutation_scale=12, color="#666",
            lw=1, alpha=0.8,
            connectionstyle="arc3,rad=0.06"))

    ax.text(6.5, 7.6, "Module Dependency Graph (imports)",
            ha="center", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


# ──────────────────────────────────────────────────────────────────────
# 5. Sequence diagram for a single query
# ──────────────────────────────────────────────────────────────────────

def diagram_sequence(path="fig_sequence.png"):
    fig, ax = plt.subplots(figsize=(13, 9))
    ax.set_xlim(0, 13); ax.set_ylim(0, 11)
    ax.set_axis_off()

    actors = ["User", "ProposedSystem", "Entity\nExtractor", "KG", "RAG", "LLM /\nComposer", "Evaluator"]
    n = len(actors)
    xs = np.linspace(1.0, 12.0, n)

    # Lifelines
    for x, name in zip(xs, actors):
        ax.add_patch(Rectangle((x - 0.7, 9.6), 1.4, 0.7,
            facecolor="#ECEFF1", edgecolor="#222", linewidth=1.0))
        ax.text(x, 9.95, name, ha="center", va="center", fontsize=10,
                fontweight="bold")
        ax.plot([x, x], [0.5, 9.6], color="#777", lw=0.6, ls="--")

    # Messages
    msgs = [
        (0, 1, 8.9,  "query"),
        (1, 2, 8.3,  "extract_entities(q)"),
        (2, 1, 7.7,  "{crop, weather, soil, category}"),
        (1, 3, 7.0,  "query_context(...)"),
        (3, 1, 6.4,  "kg_subgraph + reasoning_path"),
        (1, 4, 5.8,  "retrieve(q, k=3)"),
        (4, 1, 5.2,  "ranked docs (id, score)"),
        (1, 5, 4.5,  "fused prompt (kg+rag)"),
        (5, 1, 3.9,  "advisory text"),
        (1, 0, 3.2,  "advisory + reasoning_path"),
        (1, 6, 2.4,  "scoring(b_resp, p_resp)"),
        (6, 1, 1.8,  "metrics dict"),
    ]
    for src, tgt, y, label in msgs:
        x1, x2 = xs[src], xs[tgt]
        ax.add_patch(FancyArrowPatch((x1, y), (x2, y),
            arrowstyle="-|>", mutation_scale=14, color="#333", lw=1.3))
        mx = (x1 + x2) / 2
        ax.text(mx, y + 0.12, label, ha="center", fontsize=9, color="#222")

    ax.text(6.5, 10.6, "Sequence Diagram — Lifecycle of a Single Advisory Query",
            ha="center", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


# ──────────────────────────────────────────────────────────────────────
# 6. KG sub-graph for a worked example (Wheat + Drought + Alluvial)
# ──────────────────────────────────────────────────────────────────────

def diagram_kg_example(path="fig_kg_example.png"):
    import networkx as nx
    from kg import AgroKnowledgeGraph
    kg = AgroKnowledgeGraph()
    G = kg.graph

    seed = {"crop_wheat", "weather_drought", "soil_alluvial",
            "adv_wheat_irrigation", "soil_moisture_deficit", "crop_wilting"}
    nodes = set(seed)
    for s in seed:
        nodes.update(G.successors(s))
        nodes.update(G.predecessors(s))
    sub = G.subgraph(nodes)

    pos = nx.spring_layout(sub, k=2.0, seed=11)
    fig, ax = plt.subplots(figsize=(12, 7.5))

    color_map = {"crop": "#2E7D32", "soil": "#6D4C41", "weather": "#1E88E5",
                 "advisory": "#F4511E", "condition": "#8E24AA"}
    for t, c in color_map.items():
        sub_nodes = [n for n, d in sub.nodes(data=True) if d.get("type") == t]
        nx.draw_networkx_nodes(sub, pos, nodelist=sub_nodes,
            node_color=c, node_size=1500, alpha=0.92,
            edgecolors="black", linewidths=0.6, ax=ax)

    nx.draw_networkx_edges(sub, pos, ax=ax, alpha=0.55,
        arrows=True, arrowsize=18, edge_color="#333",
        connectionstyle="arc3,rad=0.08", width=1.3)
    edge_labels = {(u, v): d.get("relation", "")
                   for u, v, d in sub.edges(data=True)}
    nx.draw_networkx_edge_labels(sub, pos, edge_labels, font_size=8, ax=ax,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.85))
    labels = {n: G.nodes[n].get("name", G.nodes[n].get("condition", n))
              for n in sub.nodes()}
    nx.draw_networkx_labels(sub, pos, labels, font_size=9, font_weight="bold", ax=ax)

    ax.set_title("Worked Example — KG Sub-graph for ‘Wheat + Drought + Alluvial soil’",
                 fontsize=14, fontweight="bold")
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


# ──────────────────────────────────────────────────────────────────────
# 7. Result charts (read results.json)
# ──────────────────────────────────────────────────────────────────────

def _load_results():
    if not os.path.exists("results.json"):
        return None
    with open("results.json") as f:
        return json.load(f)


def diagram_results_summary(path="fig_results_summary.png"):
    data = _load_results()
    if data is None:
        print("results.json missing — run main.py first.")
        return

    evals = data["evaluations"]
    n = len(evals)
    metrics = ["relevance", "specificity", "grounding", "precision", "recall", "f1"]
    avg_b = [sum(ev["baseline"][m]    for ev in evals) / n for m in metrics]
    avg_p = [sum(ev["proposed"][m]    for ev in evals) / n for m in metrics]

    # Normalise relevance / specificity to 0-1 for the same axis
    avg_b_norm = [a / 5 if m in ("relevance", "specificity") else a
                  for a, m in zip(avg_b, metrics)]
    avg_p_norm = [a / 5 if m in ("relevance", "specificity") else a
                  for a, m in zip(avg_p, metrics)]

    x = np.arange(len(metrics))
    width = 0.36

    fig, ax = plt.subplots(figsize=(11, 5.6))
    bars_b = ax.bar(x - width / 2, avg_b_norm, width,
        color="#90CAF9", edgecolor="#1565C0", label="Baseline (LLM-only)")
    bars_p = ax.bar(x + width / 2, avg_p_norm, width,
        color="#A5D6A7", edgecolor="#2E7D32", label="Proposed (KG + RAG + LLM)")

    ax.set_xticks(x)
    ax.set_xticklabels([m.title() for m in metrics])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Normalised score (0–1)")
    ax.set_title("Average Evaluation Metrics — Baseline vs Proposed (n={} queries)".format(n),
                 fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper left")

    # Annotate raw values
    for bar, raw, m in zip(bars_b, avg_b, metrics):
        v = f"{raw:.2f}" if m not in ("relevance", "specificity") else f"{raw:.1f}/5"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                v, ha="center", fontsize=8.5, color="#0D47A1")
    for bar, raw, m in zip(bars_p, avg_p, metrics):
        v = f"{raw:.2f}" if m not in ("relevance", "specificity") else f"{raw:.1f}/5"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                v, ha="center", fontsize=8.5, color="#1B5E20")

    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def diagram_per_query_f1(path="fig_per_query_f1.png"):
    data = _load_results()
    if data is None:
        return
    evals = data["evaluations"]
    qids = [ev["query_id"] for ev in evals]
    b_f1 = [ev["baseline"]["f1"] for ev in evals]
    p_f1 = [ev["proposed"]["f1"] for ev in evals]

    x = np.arange(len(qids))
    width = 0.4
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.bar(x - width / 2, b_f1, width, color="#90CAF9",
           edgecolor="#1565C0", label="Baseline")
    ax.bar(x + width / 2, p_f1, width, color="#A5D6A7",
           edgecolor="#2E7D32", label="Proposed")

    ax.set_xticks(x); ax.set_xticklabels(qids)
    ax.set_ylabel("F1 vs gold advisory")
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-Query F1 — Match against Gold Advisory Specific Terms",
                 fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def diagram_grounding_scatter(path="fig_grounding.png"):
    data = _load_results()
    if data is None:
        return
    evals = data["evaluations"]
    qids = [ev["query_id"] for ev in evals]
    b = [ev["baseline"]["grounding"] for ev in evals]
    p = [ev["proposed"]["grounding"] for ev in evals]

    fig, ax = plt.subplots(figsize=(11, 4.8))
    x = np.arange(len(qids))
    ax.plot(x, b, marker="o", linewidth=2, color="#1565C0", label="Baseline")
    ax.plot(x, p, marker="s", linewidth=2, color="#2E7D32", label="Proposed")
    ax.fill_between(x, b, p, color="#A5D6A7", alpha=0.25,
                    label="Grounding gain")
    ax.set_xticks(x); ax.set_xticklabels(qids)
    ax.set_ylim(-0.05, 1.0)
    ax.set_ylabel("Grounding score (0–1)")
    ax.set_title("Per-Query Grounding Score — Word Overlap with Provided Context",
                 fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


# ──────────────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────────────

def main():
    diagram_system_architecture()
    diagram_pipeline()
    diagram_kg_schema()
    diagram_components()
    diagram_sequence()
    diagram_kg_example()
    diagram_results_summary()
    diagram_per_query_f1()
    diagram_grounding_scatter()
    # Also (re)generate the full KG diagram via visualize.py
    try:
        from visualize import generate_kg_diagram
        generate_kg_diagram("fig_kg_full.png")
    except Exception as e:
        print(f"KG full diagram skipped: {e}")


if __name__ == "__main__":
    main()
