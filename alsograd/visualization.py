from graphviz import Digraph

from alsograd import Parameter


def create_graph(p: Parameter, render=False, f_name="compute_graph", backward_order=True,
                 show_edge_labels=False) -> Digraph:
    graph = Digraph("g", strict=False)

    color = lambda n: 'green' if n.requires_grad else 'red'

    nodes = p.topography()
    for node in nodes:
        if not node.builder:
            continue

        graph.node(node.label, style='filled', fillcolor=color(node))
        for parent in filter(lambda p: p is not None, node.builder.parents):
            if not parent in graph:
                graph.node(parent.label, style='filled', fillcolor=color(parent))

            s = (node.label, parent.label) if backward_order else (parent.label, node.label)

            edge_label = node.builder.__class__.__name__ if show_edge_labels else ""
            graph.edge(*s, label=edge_label)

    if render:
        graph.render(f_name, view=True)

    return graph
