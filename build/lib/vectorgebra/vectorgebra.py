from graph import *


if __name__ == "__main__":
    m = Matrix.randMint(5, 5, -2, 2)
    Q, R = m.qr()
    print(Q)
    print()
    print(R)
    """t = Tensor(Matrix.randMint(2, 2, -2, 2),
               Matrix.randMint(2, 2, -2, 2),
               Matrix.randMint(2, 2, -2, 2))
    print(t.dimension)
    print(t)

    t -= Vector(1, 1)
    print(t)

    print()
    print(t)
    print(t.flatten())
    print(t.avg())

    m = Matrix(Vector(1, 2, 3), Vector(1, 2, 3))
    n = Matrix(Vector(1, 2, 3, 4), Vector(1, 2, 3, 4), Vector(1, 2, 3, 4))
    print(m.dimension)
    print(m * n)"""
    
