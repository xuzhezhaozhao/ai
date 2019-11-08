
import tensorflow as tf


_MIN = -1e9


def dpp(L, K, epsilon=1e-7):
    """ Fast Greedy MAP Inference for Determinantal Point Process to Improve Recommendation Diversity
    Algorithm 1 Fast Greedy MAP Inference

    L: Tensor of shape (N, N)
    K: Return K elements

    Return:
      Y_g: Selected item ids

    TODO(zhezhaoxu): Add stopping criteria of dj^2 < epsilon
    """
    # Initialize
    c = None
    d_square = tf.diag_part(L)   # (N,)
    d_square = tf.reshape(d_square, (-1, 1))  # (N, 1)
    d_var = tf.Variable(d_square)

    j = tf.argmax(d_var)
    Y_g = []
    Y_g.append(j)

    # argmax mask
    d_var = tf.scatter_update(d_var, j, _MIN)

    # loop
    loop = 0
    while len(Y_g) < K:
        dj = tf.math.sqrt(tf.gather(d_square, j)[0])   # scalar
        Lj = tf.gather(L, j)    # (N,)
        Lj = tf.reshape(Lj, [-1, 1])  # [N, 1]
        # caculate e
        if loop > 0:
            cj = tf.gather(c, j)  # (loop,)
            cj = tf.reshape(cj, [1, -1])  # (1, loop)
            cdot = tf.reduce_sum(cj * c, axis=-1, keepdims=True)  # (N, 1)
            e = (Lj - cdot) / dj   # (N, 1)
        else:
            e = Lj / dj   # (N, 1)

        # update c
        if c is None:
            c = e   # (N, 1)
        else:
            c = tf.concat([c, e], axis=-1)   # (N, loop)
        # update d
        d_square = d_square - tf.square(e)   # (N, 1)
        d_var = tf.assign_sub(d_var, tf.square(e))   # (N, 1)

        # select one item
        j = tf.argmax(d_var)
        Y_g.append(j)

        # argmax mask
        d_var = tf.scatter_update(d_var, j, _MIN)

        loop += 1

    Y_g = tf.concat(Y_g, axis=-1)
    return Y_g
