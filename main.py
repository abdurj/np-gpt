from tensor.tensor import Tensor


if __name__ == "__main__":
    t1 = Tensor(
        [[1, 2, 3], [4, 5, 6]]
    )
    t2 = Tensor(
        [[7, 8, 9], [10, 11, 12]]
    )
    t = t1 + t2
    print(t.grad)