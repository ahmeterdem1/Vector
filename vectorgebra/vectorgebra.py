from graph import *


if __name__ == "__main__":
    pass
    """ m = Matrix(Vector(1, 2, 3), Vector(4, 5, 6))
    print(m.echelon())
    print(m.cumsum())"""

    """
    # Compare tensorflow and our implementation on autograd!
    
    import tensorflow as tf
    from time import time


    x = Vector(*[Variable(k) for k in range(10000)])
    y = Vector.randVfloat(10000, 0, 2)
    z = x.dot(y)
    begin = time()
    gradient = grad(z, x.values)
    end = time()
    print(gradient[:10])
    print("Time in seconds: ", end - begin)


    x = tf.Variable([k for k in range(10000)], dtype=tf.float32)
    y = tf.Variable(y.values, dtype=tf.float32)
    with tf.GradientTape() as tape:
        z = tf.reduce_sum(tf.multiply(x, y))

    begin = time()
    gradient = tape.gradient(z, [x])
    end = time()
    print(gradient[0][:10])
    print("Time in seconds", end - begin)"""




    """v = Vector(lambda x: x, lambda x: x**2)
    w = Vector(lambda x: x, lambda x: x**2)
    l = v + w
    print(l(2))
    a = Vector(1, 2)
    b = Vector(1, 2)
    f = lambda x: x
    a = f - a
    print((f - l)(2))
    print(a(2))
    print((a - 2)(2))
    print(b + b, b - b, 1 - b, b + 2, 2 * b, b * 2)
    c = l * f
    print(c(2))
    c = f * l
    print(c(2))

    A = Vector(1, 2, 3)
    B = Vector(3, 2, 1)
    print(Vector.outer(A, B))
    v.append(f)
    print(v(2))
    v.pop()
    print(v(2))"""

    """m = Matrix.randMint(3, 2, -2, 2)
    n = Matrix.randMint(3, 2, -2, 2)
    print(m)
    print()
    print(n)
    print()
    print(m + n)
    print()
    print(m - n)
    print()
    print(m * n.transpose())
    print()
    print(Matrix.identity(3))"""


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
    
