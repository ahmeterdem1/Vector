from graph import *

Vector(1, 2, 3)

if __name__ == "__main__":
    t = Tensor(Matrix.randMint(2, 2, -2, 2),
               Matrix.randMint(2, 2, -2, 2),
               Matrix.randMint(2, 2, -2, 2))
    print(t.dimension)
    print(t)

    t -= Vector(1, 1)
    print(t)

    print()
    print(t)
    print(t * Matrix(Vector(1, 2), Vector(3, 4)))
    print()
    print(t * t)


