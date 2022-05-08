from graphviz import Digraph

from alsograd import Parameter


def create_graph(p: Parameter, render=False, f_name="compute_graph", backward_order=True) -> Digraph:
    graph = Digraph("g", strict=False)

    nodes = p.topography()
    for node in nodes:
        if not node.builder:
            continue

        graph.node(node.label)
        for parent in filter(lambda p: p is not None, node.builder.parents):
            s = (node.label, parent.label) if backward_order else (parent.label, node.label)
            graph.edge(*s)

    if render:
        graph.render(f_name, view=True)

    return graph
