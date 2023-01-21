import numpy as np

class CSR:
    def __init__(self, dense_matrix):
        self.value = dense_matrix[dense_matrix != 0]
        _, self.cols = np.nonzero(dense_matrix)
        self.rows_cur = np.cumsum(np.concatenate((np.array([0]), 
                                    np.sum(dense_matrix != 0, axis=1))))
        self.NNZ = sum(dense_matrix!=0)
        self.orig_rows_len = dense_matrix.shape[0]
        self.orig_cols_len = dense_matrix.shape[1]

    def update_matrix(self, dense_rows):
        self.value = np.concatenate((self.value, dense_rows[dense_rows != 0]))
        _, new_cols = np.nonzero(dense_rows)
        self.cols = np.concatenate((self.cols, new_cols))
        value = self.rows_cur[-1] if len(self.rows_cur) > 0 else 0
        new_rows = np.cumsum(value +np.sum(dense_rows != 0, axis=1))
        self.rows_cur = np.concatenate((self.rows_cur, new_rows))
        self.NNZ += sum(dense_rows!=0)
        self.orig_rows_len += len(dense_rows)
    
    def __len__(self):
        return self.NNZ

    def dense_reconstruction(self):
        ret = []
        for i in range(len(self.rows_cur)-1):
            cur_row = np.zeros(4)
            num_nnz = self.rows_cur[i+1]-self.rows_cur[i]
            jump_pos = self.rows_cur[i]
            possible_values = self.value[jump_pos:jump_pos + num_nnz]
            possible_cols = self.cols[jump_pos:jump_pos + num_nnz]
            cur_row[possible_cols] = possible_values
            ret.append(cur_row)
        return np.array(ret)



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
            






if __name__ == "__main__":
    A = np.array([[0, 0, 0, 0],
         [5, 8, 0, 0],
         [0, 0, 3, 0],
         [0, 6, 0, 0]])
    tmp = CSR(A[0][None,:])

    for i in range(1, 4):
        tmp.update_matrix(A[i][None,:])

    print(tmp.dense_reconstruction())

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
        
        