import numpy as np
import scipy.ndimage


def poly_exp(f, c, sigma):
    """
    Calculate local polynomial expansion of a 2D signal, as described by Farneback

    Uses separable normalized correlation

    $f ~ x^T A x + B^T x + C$

    Parameters
    ----------
    f
        Input signal
    c
        Certainty of signal
    sigma
        Standard deviation of applicability Gaussian kernel

    Returns
    -------
    A
        Quadratic term of polynomial expansion
    B
        Linear term of polynomial expansion
    C
        Constant term of polynomial expansion
    """
    # Calculate applicability kernel (1D because it is separable)
    n = int(4*sigma + 1)
    x = np.arange(-n, n + 1, dtype=np.int)
    a = np.exp(-x**2 / (2 * sigma**2))  # a: applicability kernel [n]

    # b: calculate b from the paper. Calculate separately for X and Y dimensions
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

    # Pre-calculate product of certainty and signal
    cf = c * f

    # G and v are used to calculate "r" from the paper: v = G*r
    # r is the parametrization of the 2nd order polynomial for f
    G = np.empty(list(f.shape) + [bx.shape[-1]]*2)
    v = np.empty(list(f.shape) + [bx.shape[-1]])

    # Apply separable cross-correlations

    # Pre-calculate quantities recommended in paper
    ab = np.einsum('i,ij->ij', a, bx)
    abb = np.einsum('ij,ik->ijk', ab, bx)

    # Calculate G and v for each pixel with cross-correlation
    for i in range(bx.shape[-1]):
        for j in range(bx.shape[-1]):
            G[..., i, j] = scipy.ndimage.correlate1d(c, abb[..., i, j], axis=0, mode='constant', cval=0)

        v[..., i] = scipy.ndimage.correlate1d(cf, ab[..., i], axis=0, mode='constant', cval=0)

    # Pre-calculate quantities recommended in paper
    ab = np.einsum('i,ij->ij', a, by)
    abb = np.einsum('ij,ik->ijk', ab, by)

    # Calculate G and v for each pixel with cross-correlation
    for i in range(bx.shape[-1]):
        for j in range(bx.shape[-1]):
            G[..., i, j] = scipy.ndimage.correlate1d(G[..., i, j], abb[..., i, j], axis=1, mode='constant', cval=0)

        v[..., i] = scipy.ndimage.correlate1d(v[..., i], ab[..., i], axis=1, mode='constant', cval=0)

    # Solve r for each pixel
    r = np.linalg.solve(G, v)

    # Quadratic term
    A = np.empty(list(f.shape) + [2, 2])
    A[..., 0, 0] = r[..., 3]
    A[..., 0, 1] = r[..., 5] / 2
    A[..., 1, 0] = A[..., 0, 1]
    A[..., 1, 1] = r[..., 4]

    # Linear term
    B = np.empty(list(f.shape) + [2])
    B[..., 0] = r[..., 1]
    B[..., 1] = r[..., 2]

    # constant term
    C = r[..., 0]

    # b: [n, n, 6]
    # r: [f, f, 6]
    # f: [f, f]
    # e = b*r - f

    return A, B, C


def flow_iterative(f1, f2, sigma, c1, c2, sigma_flow, num_iter=1, d=None, p=None, model='constant', mu=None):
    """
    Calculate optical flow described by Gunnar Farneback

    Parameters
    ----------
    f1
        First image
    f2
        Second image
    sigma
        Polynomial expansion applicability Gaussian kernel sigma
    c1
        Certainty of first image
    c2
        Certainty of second image
    sigma_flow
        Applicability window Gaussian kernel sigma for polynomial matching
    num_iter
        Number of iterations to run
    d: (optional)
        Initial displacement field
    p: (optional)
        Initial global displacement model parameters
    model: ['constant', 'affine', 'eight_param']
        Optical flow parametrization to use
    mu: (optional)
        Weighting term for usage of global parametrization. Defaults to
        using value recommended in Farneback's thesis

    Returns
    -------
    d
        Optical flow field
    x
        Pixel coordinate map between f1 and f2
    """

    # Calculate the polynomial expansion at each point in the images
    A1, B1, C1 = poly_exp(f1, c1, sigma)
    A2, B2, C2 = poly_exp(f2, c2, sigma)

    # Pixel coordinates of each point in the images
    x = np.stack(np.broadcast_arrays(
        np.arange(f1.shape[0])[:, None],
        np.arange(f1.shape[1])
    ), axis=-1).astype(np.int)

    # Initialize displacement field
    if d is None:
        d = np.zeros(list(f1.shape) + [2])

    # Set up applicability convolution window
    n_flow = int(4*sigma_flow + 1)
    xw = np.arange(-n_flow, n_flow + 1)
    w = np.exp(-xw**2 / (2 * sigma_flow**2))

    # Evaluate warp parametrization model at pixel coordinates
    if model == 'constant':
        S = np.eye(2)

    elif model in ('affine', 'eight_param'):
        S = np.empty(list(x.shape) + [6 if model == 'affine' else 8])

        S[..., 0, 0] = 1
        S[..., 0, 1] = x[..., 0]
        S[..., 0, 2] = x[..., 1]
        S[..., 0, 3] = 0
        S[..., 0, 4] = 0
        S[..., 0, 5] = 0

        S[..., 1, 0] = 0
        S[..., 1, 1] = 0
        S[..., 1, 2] = 0
        S[..., 1, 3] = 1
        S[..., 1, 4] = x[..., 0]
        S[..., 1, 5] = x[..., 1]

        if model == 'eight_param':
            S[..., 0, 6] = x[..., 0] ** 2
            S[..., 0, 7] = x[..., 0] * x[..., 1]

            S[..., 1, 6] = x[..., 0] * x[..., 1]
            S[..., 1, 7] = x[..., 1] ** 2

    else:
        raise ValueError('Invalid parametrization model')

    # if p is not None:
    #     d = S @ p

    S_T = S.swapaxes(-1, -2)

    # Iterate convolutions to estimate the optical flow
    for _ in range(num_iter):
        # Set d~ as displacement field fit to nearest pixel (and constrain to not
        # being off image). Note we are setting certainty to 0 for points that
        # would have been off-image had we not constrained them
        d_ = d.astype(np.int)
        x_ = x + d_

        # x_ = np.maximum(np.minimum(x_, np.array(f1.shape) - 1), 0)

        # Constrain d~ to be on-image, and find points that would have
        # been off-image
        x_2 = np.maximum(np.minimum(x_, np.array(f1.shape) - 1), 0)
        off_f = np.any(x_ != x_2, axis=-1)
        x_ = x_2

        # Set certainty to 0 for off-image points
        c_ = c1[x_[..., 0], x_[..., 1]]
        c_[off_f] = 0

        # Calculate A and delB for each point, according to paper
        A = (A1 + A2[x_[..., 0], x_[..., 1]]) / 2
        A *= c_[..., None, None]  # recommendation in paper: add in certainty by applying to A and delB

        delB = -1/2 * (B2[x_[..., 0], x_[..., 1]] - B1) + (A @ d_[..., None])[..., 0]
        delB *= c_[..., None]  # recommendation in paper: add in certainty by applying to A and delB

        # Pre-calculate quantities recommended by paper
        A_T = A.swapaxes(-1, -2)
        ATA = S_T @ A_T @ A @ S
        ATb = (S_T @ A_T @ delB[..., None])[..., 0]
        # btb = delB.swapaxes(-1, -2) @ delB

        # If mu is 0, it means the global/average parametrized warp should not be
        # calculated, and the parametrization should apply to the local calculations
        if mu == 0:
            # Apply separable cross-correlation to calculate linear equation
            # for each pixel: G*d = h
            G = scipy.ndimage.correlate1d(ATA, w, axis=0, mode='constant', cval=0)
            G = scipy.ndimage.correlate1d(G, w, axis=1, mode='constant', cval=0)

            h = scipy.ndimage.correlate1d(ATb, w, axis=0, mode='constant', cval=0)
            h = scipy.ndimage.correlate1d(h, w, axis=1, mode='constant', cval=0)

            d = (S @ np.linalg.solve(G, h)[..., None])[..., 0]

        # if mu is not 0, it should be used to regularize the least squares problem
        # and "force" the background warp onto uncertain pixels
        else:
            # Calculate global parametrized warp
            G_avg = np.mean(ATA, axis=(0, 1))
            h_avg = np.mean(ATb, axis=(0, 1))
            p_avg = np.linalg.solve(G_avg, h_avg)
            d_avg = (S @ p_avg[..., None])[..., 0]

            # Default value for mu is to set mu to 1/2 the trace of G_avg
            if mu is None:
                mu = 1/2 * np.trace(G_avg)

            # Apply separable cross-correlation to calculate linear equation
            G = scipy.ndimage.correlate1d(A_T @ A, w, axis=0, mode='constant', cval=0)
            G = scipy.ndimage.correlate1d(G, w, axis=1, mode='constant', cval=0)

            h = scipy.ndimage.correlate1d((A_T @ delB[..., None])[..., 0], w, axis=0, mode='constant', cval=0)
            h = scipy.ndimage.correlate1d(h, w, axis=1, mode='constant', cval=0)

            # Refine estimate of displacement field
            d = np.linalg.solve(G + mu*np.eye(2), h + mu*d_avg)

    # TODO: return global displacement parameters and/or global displacement if mu != 0

    return d, x + d


def main():
    fn1 = r"C:\Users\Prince\Documents\projects\spatial_domain_toolbox\yosemite_sequence\yos2.tif"
    fn2 = r"C:\Users\Prince\Documents\projects\spatial_domain_toolbox\yosemite_sequence\yos4.tif"

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
        # model='constant',
        model='eight_param',
        mu=None,
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
