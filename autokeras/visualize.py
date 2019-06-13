import os
from graphviz import Digraph
from autokeras.utils import pickle_from_file
from autokeras.search import Searcher

def to_pdf(graph, path):
    dot = Digraph(comment='The Round Table')

    for index, node in enumerate(graph.node_list):
        dot.node(str(index), str(node.shape))

    for u in range(graph.n_nodes):
        for v, layer_id in graph.adj_list[u]:
            dot.edge(str(u), str(v), str(graph.layer_list[layer_id]))

    dot.render(path)


def visualize(path):
    cnn_module = pickle_from_file(os.path.join(path, 'module'))
    #cnn_module.searcher.path = path
    #cnn_module.searcher =
    for item in cnn_module.searcher.history:
        model_id = item['model_id']
        graph = cnn_module.searcher.load_model_by_id(model_id)



if __name__ == '__main__':
    #cnn_module = pickle_from_file(os.path.join("./deal-data/show_net", 'module'))
    for i in range(0,59):
        graph = pickle_from_file(os.path.join("deal-data/show_net/", str(i) + '.graph'))
        to_pdf(graph, os.path.join("deal-data/graph", str(i)))
    #visualize("./deal-data/show_net/")
    #graph = cnn_module.searcher.load_model_by_id(1)