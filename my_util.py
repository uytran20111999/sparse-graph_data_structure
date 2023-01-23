def preprocess_line(line, dtype):
    tmp = [dtype(d) for d in line.split()]
    return tmp
