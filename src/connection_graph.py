
import networkx as nx
import plotly.graph_objects as go


def build_connection_graph(connections):
    """Create an interactive network graph of papers and shared concepts."""

    graph = nx.Graph()

    if not connections:
        return go.Figure().update_layout(
            title="No research connections found"
        )

    # Add paper and concept nodes
    for connection in connections:
        paper_1 = connection["paper_1"]
        paper_2 = connection["paper_2"]
        concept = connection["keyword"]

        graph.add_node(paper_1, node_type="paper")
        graph.add_node(paper_2, node_type="paper")
        graph.add_node(concept, node_type="concept")

        graph.add_edge(paper_1, concept)
        graph.add_edge(paper_2, concept)

    positions = nx.spring_layout(
        graph,
        seed=42,
        k=0.9,
        iterations=100
    )

    # Create graph edges
    edge_x = []
    edge_y = []

    for source, target in graph.edges():
        x0, y0 = positions[source]
        x1, y1 = positions[target]

        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=1.2, color="#465475"),
        hoverinfo="none"
    )

    # Create paper and concept nodes
    node_x = []
    node_y = []
    node_labels = []
    node_colors = []
    node_sizes = []
    node_hover = []

    for node in graph.nodes():
        x, y = positions[node]

        node_x.append(x)
        node_y.append(y)
        node_labels.append(node)

        if graph.nodes[node]["node_type"] == "paper":
            node_colors.append("#8b7cff")
            node_sizes.append(28)
            node_hover.append(f"Research Paper: {node}")
        else:
            node_colors.append("#45d6d0")
            node_sizes.append(17)
            node_hover.append(f"Shared Concept: {node}")

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=node_labels,
        textposition="top center",
        textfont=dict(
            color="#edf2ff",
            size=10
        ),
        hovertext=node_hover,
        hoverinfo="text",
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(
                width=1,
                color="#edf2ff"
            )
        )
    )

    figure = go.Figure(
        data=[edge_trace, node_trace]
    )

    figure.update_layout(
        title=dict(
            text="Research Connection Network",
            font=dict(color="#edf2ff", size=18)
        ),
        paper_bgcolor="#121a2d",
        plot_bgcolor="#121a2d",
        font=dict(color="#edf2ff"),
        showlegend=False,
        hovermode="closest",
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False
        )
    )

    return figure
