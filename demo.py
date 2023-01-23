from graph_obj import my_graph

def demo_construct(is_lazy, file_name):
    tmp = my_graph.init_from_file(file_name, is_lazy)
    tmp.draw_graph(format = 'png')
    return tmp

def demo_list_edges(input_graph):
    return input_graph.get_edges()

def demo_DFS_edges(input_graph, beg_idx):
    return input_graph.DFS_traverse(beg_idx)

def demo_induce_sub_graph(input_graph, list_idx):
    tmp = input_graph.get_induced_subgraph(list_idx)
    tmp.draw_graph(format = 'png')
    return tmp

if __name__ == "__main__":
    f_g = demo_construct(False, "./demo_graph/first_graph.txt")
    s_g = demo_construct(True, "./demo_graph/second_graph.txt")
    print(demo_list_edges(f_g))
    node_idx, edge_traverse = demo_DFS_edges(s_g, 2)
    print(s_g.get_values_from_idx(node_idx))
    demo_induce_sub_graph(s_g, [7, 8, 9, 10, 11, 12])
    demo_induce_sub_graph(f_g, [1, 3, 4, 5, 0])