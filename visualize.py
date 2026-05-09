"""
visualize.py — KG Visualization
=================================
Generates a layered, colour-coded knowledge graph diagram of the
agricultural KG. Nodes are arranged by type (crop / soil / weather /
condition / advisory) so the structure of the graph is legible.
"""
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx
    from kg import AgroKnowledgeGraph
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


COLOR_MAP = {
    "crop":     "#2E7D32",   # green
    "soil":     "#6D4C41",   # brown
    "weather":  "#1E88E5",   # blue
    "advisory": "#F4511E",   # orange
    "condition": "#8E24AA",  # purple
}

NODE_SIZE = {
    "crop":     1300,
    "soil":     1100,
    "weather":  1100,
    "advisory":  900,
    "condition": 700,
}


def _layered_layout(G):
    """Place nodes in vertical bands by type."""
    bands = {"weather": 4, "crop": 3, "soil": 2, "advisory": 1, "condition": 0}
    by_type = {t: [] for t in bands}
    for n, d in G.nodes(data=True):
        by_type.setdefault(d.get("type", "crop"), []).append(n)

    pos = {}
    for t, nodes in by_type.items():
        y = bands.get(t, 0)
        nodes = sorted(nodes)
        n = len(nodes)
        for i, node in enumerate(nodes):
            x = (i - (n - 1) / 2) * (1.4 if t in ("crop", "soil") else 1.0)
            pos[node] = (x, y * 1.6)
    return pos


def generate_kg_diagram(output_path="kg_diagram.png"):
    if not HAS_MPL:
        print("matplotlib not installed. Skipping visualization.")
        return

    kg = AgroKnowledgeGraph()
    G = kg.graph

    pos = _layered_layout(G)
    fig, ax = plt.subplots(figsize=(20, 11))

    # Draw nodes layer by layer for legible z-ordering
    for t, color in COLOR_MAP.items():
        sub = [n for n, d in G.nodes(data=True) if d.get("type") == t]
        nx.draw_networkx_nodes(
            G, pos, nodelist=sub,
            node_color=color, node_size=NODE_SIZE.get(t, 800),
            alpha=0.92, edgecolors="black", linewidths=0.6,
            ax=ax,
        )

    # Edge styling per relation
    relation_styles = {
        "affects":         {"edge_color": "#1565C0", "style": "solid"},
        "requires":        {"edge_color": "#5D4037", "style": "dashed"},
        "causes":          {"edge_color": "#6A1B9A", "style": "dotted"},
        "recommended_for": {"edge_color": "#E65100", "style": "solid"},
    }
    for rel, style in relation_styles.items():
        edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("relation") == rel]
        nx.draw_networkx_edges(
            G, pos, edgelist=edges, ax=ax,
            edge_color=style["edge_color"], style=style["style"],
            arrows=True, arrowsize=14, alpha=0.55, width=1.2,
            connectionstyle="arc3,rad=0.08",
        )

    labels = {n: G.nodes[n].get("name", G.nodes[n].get("condition", n))
              for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=7, font_weight="bold", ax=ax)

    # Legends
    node_legend = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=c, markersize=12, label=t.title())
        for t, c in COLOR_MAP.items()
    ]
    edge_legend = [
        plt.Line2D([0], [0], color=s["edge_color"], lw=2,
                   linestyle=s["style"], label=rel)
        for rel, s in relation_styles.items()
    ]
    leg1 = ax.legend(handles=node_legend, loc="upper left",
                     fontsize=10, title="Node types")
    ax.add_artist(leg1)
    ax.legend(handles=edge_legend, loc="upper right",
              fontsize=10, title="Edge relations")

    ax.set_title("Agricultural Knowledge Graph — Layered View by Entity Type",
                 fontsize=15, fontweight="bold", pad=14)
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"KG diagram saved to {output_path}")


if __name__ == "__main__":
    generate_kg_diagram()
