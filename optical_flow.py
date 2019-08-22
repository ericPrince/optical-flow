import numpy as np
import scipy.ndimage


def poly_exp_v1(f, c, sigma, n):
    x = np.arange(-n, n + 1, dtype=np.int)

    # [n, n]
    X = np.stack(np.broadcast_arrays(x, x[:, None]), axis=-1)
    a = np.exp(-np.sum(X**2, axis=-1) / (2 * sigma**2))

    # [n, n, 6]
    b = np.stack([
        np.ones(a.shape),
        X[..., 1],
        X[..., 0],
        X[..., 1]**2,
        X[..., 0]**2,
        X[..., 0] * X[..., 1]
    ], axis=-1)

    ab = np.einsum('ij,ijk->ijk', a, b)
    abb = np.einsum('ijk,ijl->ijkl', ab, b)
    cf = c * f

    G = np.empty(list(f.shape) + [b.shape[-1]]*2)
    v = np.empty(list(f.shape) + [b.shape[-1]])
    for i in range(b.shape[-1]):
        for j in range(b.shape[-1]):
            G[..., i, j] = scipy.ndimage.correlate(c, abb[..., i, j], mode='constant', cval=0)
        v[..., i] = scipy.ndimage.correlate(cf, ab[..., i], mode='constant', cval=0)

    r = np.linalg.solve(G, v)

    A = np.empty(list(f.shape) + [2, 2])
    A[..., 0, 0] = r[..., 3]
    A[..., 0, 1] = r[..., 5] / 2
    A[..., 1, 0] = A[..., 0, 1]
    A[..., 1, 1] = r[..., 4]

    B = np.empty(list(f.shape) + [2])
    B[..., 0] = r[..., 1]
    B[..., 1] = r[..., 2]

    C = r[..., 0]

    # b: [n, n, 6]
    # r: [f, f, 6]
    # f: [f, f]
    # e = b*r - f

    return A, B, C


def poly_exp_v2(f, c, sigma, n):
    x = np.arange(-n, n + 1, dtype=np.int)

    # [n]
    a = np.exp(-x**2 / (2 * sigma**2))

    # [n, 6]
    bx = np.stack([
        np.ones(a.shape),
        x,
        np.ones(a.shape),
        x**2,
        np.ones(a.shape),
        x
    ], axis=-1)
    by = np.stack([
        np.ones(a.shape),
        np.ones(a.shape),
        x,
        np.ones(a.shape),
        x**2,
        x,
    ], axis=-1)

    cf = c * f

    G = np.empty(list(f.shape) + [bx.shape[-1]]*2)
    v = np.empty(list(f.shape) + [bx.shape[-1]])

    ab = np.einsum('i,ij->ij', a, bx)
    abb = np.einsum('ij,ik->ijk', ab, bx)

    for i in range(bx.shape[-1]):
        for j in range(bx.shape[-1]):
            G[..., i, j] = scipy.ndimage.correlate1d(c, abb[..., i, j], axis=0, mode='constant', cval=0)
        v[..., i] = scipy.ndimage.correlate1d(cf, ab[..., i], axis=0, mode='constant', cval=0)

    ab = np.einsum('i,ij->ij', a, by)
    abb = np.einsum('ij,ik->ijk', ab, by)

    for i in range(bx.shape[-1]):
        for j in range(bx.shape[-1]):
            G[..., i, j] = scipy.ndimage.correlate1d(G[..., i, j], abb[..., i, j], axis=1, mode='constant', cval=0)
        v[..., i] = scipy.ndimage.correlate1d(v[..., i], ab[..., i], axis=1, mode='constant', cval=0)

    r = np.linalg.solve(G, v)

    A = np.empty(list(f.shape) + [2, 2])
    A[..., 0, 0] = r[..., 3]
    A[..., 0, 1] = r[..., 5] / 2
    A[..., 1, 0] = A[..., 0, 1]
    A[..., 1, 1] = r[..., 4]

    B = np.empty(list(f.shape) + [2])
    B[..., 0] = r[..., 1]
    B[..., 1] = r[..., 2]

    C = r[..., 0]

    # b: [n, n, 6]
    # r: [f, f, 6]
    # f: [f, f]
    # e = b*r - f

    return A, B, C


poly_exp = poly_exp_v2


def flow_iterative(f1, f2, n, sigma, c1, c2, sigma_flow, n_flow, num_iter=3, d=None):
    A1, B1, C1 = poly_exp(f1, c1, n, sigma)
    A2, B2, C2 = poly_exp(f2, c2, n, sigma)

    x = np.stack(np.broadcast_arrays(
        np.arange(f1.shape[0]),
        np.arange(f1.shape[1])[:, None]
    ), axis=-1).astype(np.int)

    if d is None:
        d = np.zeros(list(f1.shape) + [2])

    xw = np.arange(-n_flow, n_flow + 1)
    w = np.exp(-xw**2 / (2 * sigma_flow**2))

    for _ in range(num_iter):
        d_ = d.astype(np.int)
        x_ = x + d_

        # TODO: add certainty weight to A and db?
        A = (A1 + A2[x_]) / 2
        db = -1/2 * (B2[x_] - B1) + A @ d_

        A_T = A.swapaxes(-1, -2)
        ATA = A_T @ A
        ATb = A_T @ db
        # btb = db.swapaxes(-1, -2) @ db

        M = scipy.ndimage.correlate1d(ATA, w, axis=0, mode='reflect')
        M = scipy.ndimage.correlate1d(M, w, axis=1, mode='reflect')

        y = scipy.ndimage.correlate1d(ATb, w, axis=0, mode='reflect')
        y = scipy.ndimage.correlate1d(y, w, axis=1, mode='reflect')

        d = np.linalg.solve(M, y)

    return d
