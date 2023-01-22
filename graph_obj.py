import numpy as np

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

class my_graph:
    def __init__(self, nodes_value, nodes_matrix):
        self.idx_to_value = np.array(nodes_value)
        self.sparse_edges = CSR(nodes_matrix)
        self.edges = my_graph.first_time_get_edges(self)

    @classmethod
    def init_from_csr_value(cls, csr, values):
        sub_graph = my_graph([], np.array([]))
        sub_graph.idx_to_value = values.copy()
        sub_graph.sparse_edges = csr
        sub_graph.edges = my_graph.first_time_get_edges(sub_graph)
        return sub_graph

    def update_edges_information(self, nodes_matrix):
        self.sparse_edges.update_matrix(nodes_matrix)

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
        value = instance.idx_to_value
        for i in range(len(instance.idx_to_value)):
            neighbor, _ = instance.sparse_edges.get_neightbor(i)
            ret.extend([(value[i], value[j]) for j in neighbor if (value[i], value[j]) not in ret])
        return ret

    def get_edges(self):
        return self.edges


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
    tmp2 = tmp.get_induced_subgraph([0, 5, 3 ,2])
    print(tmp2.sparse_edges.dense_reconstruction())
    print(tmp2.get_edges())
    # print(tmp.get_induced_subgraph([1]))
    #print(tmp.get_edges())
    # _, edge_order = tmp.DFS_traverse(0)
    # print(edge_order)
    # for i in range(5):
    #     _, edge_order = tmp.BFS_traverse(i)
    #     print(edge_order)
    # print(type(A))
    # tmp = CSR(np.array([]))

    # for i in range(0, 4):
    #     tmp.update_matrix(A[i][None, :])

    # print(tmp.dense_reconstruction())

    # tmp2 = CSR.from_instance_subrow(tmp, np.array([1,0,2, 3]))

    # print(tmp2.dense_reconstruction())

    # for i in range(4):
    #     for j in range(4):
    #         print(tmp[i,j], end = " ")
    #     print()
    # print(tmp.value)
    # print(tmp.cols)
    # print(tmp.rows_cur)
    # print(A.shape)
    # B = Test()
    # print(B[0,1])
    # # print(np.sum(A!=0))
    # # print(A[A!=0])
    # # R,C = np.nonzero(A)
    # # print(np.cumsum([0]+list(Counter(R).values())))
        
        