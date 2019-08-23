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


def poly_exp_v2(f, c, sigma):
    n = int(4*sigma + 1)
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


def flow_iterative(f1, f2, sigma, c1, c2, sigma_flow, num_iter=3, d=None, model='constant'):
    A1, B1, C1 = poly_exp(f1, c1, sigma)
    A2, B2, C2 = poly_exp(f2, c2, sigma)

    x = np.stack(np.broadcast_arrays(
        np.arange(f1.shape[0])[:, None],
        np.arange(f1.shape[1])
    ), axis=-1).astype(np.int)

    if d is None:
        d = np.zeros(list(f1.shape) + [2])

    n_flow = int(4*sigma_flow + 1)
    xw = np.arange(-n_flow, n_flow + 1)
    w = np.exp(-xw**2 / (2 * sigma_flow**2))

    if model == 'constant':
        S = np.eye(2)

    elif model == 'projective':
        # S: [h, w, 2, 8]
        S = np.empty(list(x.shape) + [8])

        S[..., 0, 0] = 1
        S[..., 0, 1] = x[..., 0]
        S[..., 0, 2] = x[..., 1]
        S[..., 0, 3] = 0
        S[..., 0, 4] = 0
        S[..., 0, 5] = 0
        S[..., 0, 6] = x[..., 0]**2
        S[..., 0, 7] = x[..., 0]*x[..., 1]

        S[..., 1, 0] = 0
        S[..., 1, 1] = 0
        S[..., 1, 2] = 0
        S[..., 1, 3] = 1
        S[..., 1, 4] = x[..., 0]
        S[..., 1, 5] = x[..., 1]
        S[..., 1, 6] = x[..., 0]*x[..., 1]
        S[..., 1, 7] = x[..., 1]**2

    else:
        raise ValueError('Invalid parametrization model')

    S_T = S.swapaxes(-1, -2)

    for _ in range(num_iter):
        d_ = d.astype(np.int)
        x_ = x + d_

        # x_ = np.maximum(np.minimum(x_, np.array(f1.shape) - 1), 0)

        x_2 = np.maximum(np.minimum(x_, np.array(f1.shape) - 1), 0)
        off_f = np.any(x_ != x_2, axis=-1)
        x_ = x_2

        c_ = c1[x_[..., 0], x_[..., 1]]  # TODO: set certainty to 0 for off im
        c_[off_f] = 0

        # TODO: add certainty weight to A and db?
        A = (A1 + A2[x_[..., 0], x_[..., 1]]) / 2
        A *= c_[..., None, None]

        db = -1/2 * (B2[x_[..., 0], x_[..., 1]] - B1) + (A @ d_[..., None])[..., 0]
        db *= c_[..., None]

        A_T = A.swapaxes(-1, -2)
        ATA = S_T @ A_T @ A @ S
        ATb = (S_T @ A_T @ db[..., None])[..., 0]
        # btb = db.swapaxes(-1, -2) @ db

        M = scipy.ndimage.correlate1d(ATA, w, axis=0, mode='constant', cval=0)
        M = scipy.ndimage.correlate1d(M, w, axis=1, mode='constant', cval=0)

        y = scipy.ndimage.correlate1d(ATb, w, axis=0, mode='constant', cval=0)
        y = scipy.ndimage.correlate1d(y, w, axis=1, mode='constant', cval=0)

        d = (S @ np.linalg.solve(M, y)[..., None])[..., 0]

    return d, x + d


def main():
    fn1 = r"C:\Users\Prince\Documents\projects\spatial_domain_toolbox\yosemite_sequence\yos2.tif"
    fn2 = r"C:\Users\Prince\Documents\projects\spatial_domain_toolbox\yosemite_sequence\yos3.tif"

    from PIL import Image
    import skimage.transform

    f1 = np.array(Image.open(fn1), dtype=np.double)
    f2 = np.array(Image.open(fn2), dtype=np.double)

    # c1 = np.ones_like(f1)
    # c2 = np.ones_like(f2)

    c1 = np.minimum(1, 1/5*np.minimum(np.arange(f1.shape[0])[:, None], np.arange(f1.shape[1])))
    c1 = np.minimum(c1, 1/5*np.minimum(
        f1.shape[0] - 1 - np.arange(f1.shape[0])[:, None],
        f1.shape[1] - 1 - np.arange(f1.shape[1])
    ))
    c2 = c1

    d = None

    n_pyr = 5
    opts = dict(
        sigma=2.0,
        sigma_flow=1.5,
        num_iter=10,
        model='constant',
    )

    for pyr1, pyr2, c1_, c2_ in reversed(list(zip(
        skimage.transform.pyramid_gaussian(f1, n_pyr),
        skimage.transform.pyramid_gaussian(f2, n_pyr),
        skimage.transform.pyramid_gaussian(c1, n_pyr),
        skimage.transform.pyramid_gaussian(c2, n_pyr),
    ))):
        if d is not None:
            # TODO: account for shapes not quite matching
            d = skimage.transform.pyramid_expand(d)
            d = d[:pyr1.shape[0], :pyr2.shape[1]]

        d, xw = flow_iterative(pyr1, pyr2, c1=c1_, c2=c2_, d=d, **opts)

    import cv2

    d2 = cv2.calcOpticalFlowFarneback(
        f2.astype(np.uint8),
        f1.astype(np.uint8),
        None,
        pyr_scale=0.5,
        levels=6,
        winsize=25,
        iterations=10,
        poly_n=25,
        poly_sigma=3.0,
        # flags=0
        flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN
    )
    d2 = -d2[..., (1, 0)]

    xw2 = d2 + np.stack(np.broadcast_arrays(
        np.arange(f1.shape[0])[:, None],
        np.arange(f1.shape[1])
    ), axis=-1).astype(np.int)

    f2_w2 = skimage.transform.warp(f2, np.moveaxis(xw2, -1, 0), cval=np.nan)

    f2_w = skimage.transform.warp(f2, np.moveaxis(xw, -1, 0), cval=np.nan)

    import matplotlib.pyplot as plt
    from matplotlib import cm

    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)

    vmin, vmax = np.nanpercentile(f1 - f2, [2, 98])

    axes[0, 0].imshow(f1, cmap=cm.gray)
    axes[0, 1].imshow(f2, cmap=cm.gray)
    axes[1, 0].imshow(f1 - f2_w2, cmap=cm.gray, vmin=vmin, vmax=vmax)
    axes[1, 1].imshow(f1 - f2_w, cmap=cm.gray, vmin=vmin, vmax=vmax)

    plt.show()


if __name__ == '__main__':
    main()
