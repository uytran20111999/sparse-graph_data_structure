import numpy as np
import graphviz
from matplotlib import pyplot as plt
import os
from my_util import preprocess_line

class CSR:
    def __init__(self, dense_matrix = np.array([]), value_dtype = int):
        assert value_dtype in [int, float], f"edge value must be int or float"
        assert isinstance(dense_matrix, np.ndarray), f" input must be of type numpy array"
        assert len(dense_matrix.shape) == 2 or len(dense_matrix) == 0, \
                                            f"only support matrix of 2 dimensions or empty np array"
        self.type = value_dtype
        if len(dense_matrix) > 0:
            self.value = dense_matrix[dense_matrix != 0]
            _, self.cols = np.nonzero(dense_matrix)
            self.rows_cur = np.cumsum(np.concatenate((np.array([0]), 
                                        np.sum(dense_matrix != 0, axis=1))))
            self.NNZ = sum(dense_matrix != 0)
            self.orig_rows_len = dense_matrix.shape[0]
            self.orig_cols_len = dense_matrix.shape[1]
        else:
            self.value = np.array([], dtype = value_dtype)
            self.cols = np.array([], dtype = int)
            self.rows_cur = np.array([0], dtype = int)
            self.NNZ = 0
            self.orig_rows_len = 0
            self.orig_cols_len = 0

    def update_matrix(self, dense_rows):
        assert isinstance(dense_rows, np.ndarray), f" input must be of type numpy array"
        assert 0 < len(dense_rows.shape) <= 2, f"only support matrix of dim 1 and 2"
        if self.orig_cols_len > 0:
            assert dense_rows.shape[1] == self.orig_cols_len, f"columns shape mismatched, cannot append"
            assert self.orig_rows_len < self.orig_cols_len, \
                f"this function support solely for lazy construction, so that means a full graph must have cols = rows"
        self.value = np.concatenate((self.value, dense_rows[dense_rows != 0]))
        _, new_cols = np.nonzero(dense_rows)
        self.cols = np.concatenate((self.cols, new_cols))
        value = self.rows_cur[-1] if len(self.rows_cur) > 0 else 0
        new_rows = np.cumsum(value + np.sum(dense_rows != 0, axis=1))
        self.rows_cur = np.concatenate((self.rows_cur, new_rows))
        self.NNZ += sum(dense_rows!=0)
        self.orig_rows_len += len(dense_rows)
        self.orig_cols_len = dense_rows.shape[1]

    def __len__(self):
        return self.NNZ

    def dense_reconstruction(self):
        ret = []
        for i in range(len(self.rows_cur)-1):
            cur_row = np.zeros(self.orig_cols_len)
            num_nnz = self.rows_cur[i+1]-self.rows_cur[i]
            jump_pos = self.rows_cur[i]
            possible_values = self.value[jump_pos:jump_pos + num_nnz]
            possible_cols = self.cols[jump_pos:jump_pos + num_nnz]
            cur_row[possible_cols] = possible_values
            ret.append(cur_row)
        return np.array(ret,dtype =	self.type)

    def __getitem__(self, idx):
        assert isinstance(idx, tuple), "idx should be a tuple, ie instance[a,b]"
        row_idx, col_idx = idx
        assert 0 <= row_idx < self.orig_rows_len, \
        f"row_idx should be smaller than the origin row's len {self.orig_rows_len} and leq 0"
        assert 0 <= col_idx < self.orig_cols_len, \
        f"col_idx should be smaller than the origin col's len {self.orig_rows_len} and leq 0"
        num_nnz_at_row = self.rows_cur[row_idx+1] - self.rows_cur[row_idx]
        if not num_nnz_at_row:
            return 0
        traverse_jump = self.rows_cur[row_idx]
        possible_values = self.value[traverse_jump:traverse_jump + num_nnz_at_row]
        possible_cols = self.cols[traverse_jump:traverse_jump + num_nnz_at_row]
        sub_col_idx = np.searchsorted(possible_cols, col_idx)
        if sub_col_idx >= num_nnz_at_row or possible_cols[sub_col_idx] != col_idx:
            return 0
        else:
            return possible_values[sub_col_idx]

    def get_neightbor(self, row_idx):
        assert 0 <= row_idx < self.orig_rows_len, \
        f"row_idx should be smaller than the origin row's len {self.orig_rows_len} and leq 0"
        num_nnz_at_row = self.rows_cur[row_idx+1] - self.rows_cur[row_idx]
        if num_nnz_at_row == 0:
            return np.array([]), np.array([])
        traverse_jump = self.rows_cur[row_idx]
        return self.cols[traverse_jump:traverse_jump + num_nnz_at_row], \
                self.value[traverse_jump:traverse_jump + num_nnz_at_row]

    def check_full_graph(self):
        return self.orig_cols_len == self.orig_rows_len

class my_graph:
    def __init__(self, nodes_value, nodes_matrix, is_lazy = False):
        self.idx_to_value = np.array(nodes_value)
        self.sparse_edges = CSR(nodes_matrix)
        if not is_lazy:
            self.edges, self.orig_edge = my_graph.first_time_get_edges(self)
        else:
            self.edges, self.orig_edge = [], []

    @classmethod
    def init_from_csr_value(cls, csr, values):
        sub_graph = my_graph([], np.array([]))
        sub_graph.idx_to_value = values.copy()
        sub_graph.sparse_edges = csr
        sub_graph.edges, sub_graph.orig_edge = my_graph.first_time_get_edges(sub_graph)
        return sub_graph

    @classmethod
    def init_from_file(cls, file_path, is_lazy = False):
        with open(file_path, "r") as f:
            #first line is node type (any things that python support)
            #second line is edge type (float, int)
            #third line is node value
            #the rest is matrix
            support_type_node = {'int': int, 'float': float, 'string': str}
            support_type_edge = {'int': int, 'float': float}
            first_line = f.readline().strip('\n')
            assert first_line in support_type_node,\
                    f"Add your type and its constructor in support_type var."
            second_line = f.readline().strip('\n')
            assert second_line in support_type_edge,\
                    f"Edge weight only support in or float"
            node_values = preprocess_line(f.readline().strip('\n'), support_type_node[first_line])
            if not is_lazy:
                ans = []
                myline = f.readline().strip('\n')
                ans.append(preprocess_line(myline, support_type_edge[second_line]))
                while myline:
                    myline = f.readline().strip('\n')
                    if not myline: break
                    ans.append(preprocess_line(myline, support_type_edge[second_line]))
                return my_graph(node_values, 
                        np.array(ans,dtype = support_type_edge[second_line]))
            else:
                myline = f.readline().strip('\n')
                prep_line = preprocess_line(myline, support_type_edge[second_line])
                ret_graph = my_graph(node_values, np.array([prep_line]), is_lazy)
                while myline:
                    myline = f.readline().strip('\n')
                    if not myline: break
                    ret_graph.update_edges_information(
                                        np.array([preprocess_line(myline, 
                                        support_type_edge[second_line])]))
                return ret_graph

    def update_edges_information(self, nodes_matrix):
        #to do, finish lazy loading
        self.sparse_edges.update_matrix(nodes_matrix)
        if self.sparse_edges.check_full_graph(): # the get edges algor is costly, so it is best to run once.
            self.edges, self.orig_edge = my_graph.first_time_get_edges(self)

    def get_child(self, node_idx):
        return self.sparse_edges.get_neightbor(node_idx)

    def get_induced_subgraph(self, node_list):
        node_list = np.unique(node_list)
        new_csr = CSR()
        for i in range(len(node_list)):
            row = np.zeros((1,len(node_list)), dtype = int)
            for j in range(len(node_list)):
                if i==j:
                    continue
                row[0][j] = self.sparse_edges[node_list[i],node_list[j]]
            new_csr.update_matrix(row)
        value = self.idx_to_value[node_list]
        return my_graph.init_from_csr_value(new_csr, value)

    def BFS_traverse(self, begin_node_idx):
        next_nodes = [begin_node_idx]
        visited_nodes = [begin_node_idx]
        traverse_order = []
        edge_orders = []
        visited_edge = []
        while next_nodes:
            cur_node = next_nodes.pop(0)
            if visited_edge:
                edge_orders.append(visited_edge.pop(0))
            traverse_order.append(cur_node)
            neighbor_nodes, _ = self.get_child(cur_node)
            non_visit_neighbor = [node for node in neighbor_nodes if node not in visited_nodes]
            visited_edge.extend([(self.idx_to_value[cur_node], self.idx_to_value[node]) \
                            for node in neighbor_nodes if node not in visited_nodes])
            next_nodes.extend(non_visit_neighbor)
            visited_nodes.extend(non_visit_neighbor)
        return traverse_order, edge_orders
        
    def DFS_traverse(self, begin_node_idx):
        next_nodes = [begin_node_idx]
        visited_nodes = [begin_node_idx]
        traverse_order = []
        edge_orders = []
        visited_edge = []
        while next_nodes:
            if visited_edge:
                edge_orders.append(visited_edge.pop())
            cur_node = next_nodes.pop()
            traverse_order.append(cur_node)
            neighbor_nodes, _ = self.get_child(cur_node)
            non_visit_neighbor = [node for node in neighbor_nodes if node not in visited_nodes]
            visited_edge.extend([(self.idx_to_value[cur_node], self.idx_to_value[node]) \
                for node in neighbor_nodes if node not in visited_nodes])
            next_nodes.extend(non_visit_neighbor)
            visited_nodes.extend(non_visit_neighbor)
        return traverse_order, edge_orders
    
    def get_values_from_idx(self, node_idx_list):
        return [self.idx_to_value[i] for i in node_idx_list]

    def get_induced_subgraph_from_traverse(self, begin_node_idx, mode = "BFS"):
        assert mode in ["DFS", "BFS"], \
            f"for now, my_graph supports 2 types of sub graph generation: [DFS, BFS]"
        sub_graph_algor = {"DFS": self.DFS_traverse, "BFS": self.BFS_traverse}
        traverse_node, _ = sub_graph_algor[mode](begin_node_idx)
        return self.get_induced_subgraph(traverse_node)

    def get_vertices(self):
        return self.idx_to_value
    
    @classmethod
    def first_time_get_edges(cls, instance):
        ret = []
        orig_edge = []
        value = instance.idx_to_value
        for i in range(len(instance.idx_to_value)):
            neighbor, _ = instance.sparse_edges.get_neightbor(i)
            ret.extend([(value[i], value[j]) for j in neighbor if (value[i], value[j]) not in ret])
            orig_edge.extend([(i, j.item()) for j in neighbor if (i, j) not in orig_edge])
        return ret, orig_edge

    def get_edges(self):
        return self.edges

    def draw_graph(self, graph_name = "my_graph", render_path = "./graph_viz_render/", **kwargs):
        dot = graphviz.Digraph(name = graph_name)
        for i, virtex_value in enumerate(self.idx_to_value):
           dot.node(str(i), f"Index = {i} \n Value = {virtex_value}")
        for edge in self.orig_edge:
            end_1, end_2 = edge
            dot.edge(str(end_1), str(end_2), \
                    label = str(self.sparse_edges[end_1, end_2].item()))
        dot.render(os.path.join(render_path, graph_name), **kwargs)
        if 'format' in kwargs:
            format = kwargs['format']
            if format in ['png', 'jpeg']:
                img = plt.imread(os.path.join(render_path, graph_name) + f'.{format}')
                plt.imshow(img)
                plt.axis("off")
                plt.show()
            

if __name__ == "__main__":
    A = np.array([
         [0, 1, 0, 0, 0, 1],
         [0, 0, 1, 0, 0, 0],
         [0, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 1, 1],
         [1, 0, 0, 0, 0, 0],
         [0, 0, 1, 0, 1, 0]])
    value = [0, 1, 2, 3, 4, 5]
    tmp = my_graph(value, A)
    tmp2 = tmp.get_induced_subgraph([1, 2 , 5])
    print(tmp2.sparse_edges.dense_reconstruction())
    print(tmp2.get_edges())
    print(tmp2)
    graphviz_obj = tmp2.draw_graph(format = "png")
        
        