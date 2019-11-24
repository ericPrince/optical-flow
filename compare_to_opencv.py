# note: the conda environment defined in test-environment.yml contains all
# dependencies needed to run this script

from functools import partial

import numpy as np
from PIL import Image
import skimage.transform
import cv2
import matplotlib.pyplot as plt

from optical_flow import flow_iterative


def main():
    """
    Compares this implementation of Farneback's algorithms to OpenCV's implementation
    of a similar version of the algorithm
    """

    # ---------------------------------------------------------------
    # get images to calculate flow for
    # ---------------------------------------------------------------

    fn1 = r"C:\Users\Prince\Documents\projects\spatial_domain_toolbox\yosemite_sequence\yos2.tif"
    fn2 = r"C:\Users\Prince\Documents\projects\spatial_domain_toolbox\yosemite_sequence\yos4.tif"

    f1 = np.array(Image.open(fn1), dtype=np.double)
    f2 = np.array(Image.open(fn2), dtype=np.double)

    # certainties for images - certainty is decreased for pixels near the edge
    # of the image, as recommended by Farneback

    # c1 = np.ones_like(f1)
    # c2 = np.ones_like(f2)

    c1 = np.minimum(1, 1/5*np.minimum(np.arange(f1.shape[0])[:, None], np.arange(f1.shape[1])))
    c1 = np.minimum(c1, 1/5*np.minimum(
        f1.shape[0] - 1 - np.arange(f1.shape[0])[:, None],
        f1.shape[1] - 1 - np.arange(f1.shape[1])
    ))
    c2 = c1

    # ---------------------------------------------------------------
    # calculate optical flow with this algorithm
    # ---------------------------------------------------------------

    n_pyr = 4

    # # version using perspective warp regularization
    # # to clean edges
    # opts = dict(
    #     sigma=4.0,
    #     sigma_flow=4.0,
    #     num_iter=3,
    #     model='eight_param',
    #     mu=None,
    # )

    # version using no regularization model
    opts = dict(
        sigma=4.0,
        sigma_flow=4.0,
        num_iter=3,
        model='constant',
        mu=0,
    )

    # optical flow field
    d = None

    # calculate optical flow using pyramids
    # note: reversed(...) because we start with the smallest pyramid
    for pyr1, pyr2, c1_, c2_ in reversed(list(zip(
        *list(map(
            partial(skimage.transform.pyramid_gaussian, max_layer=n_pyr),
            [f1, f2, c1, c2]
        ))
    ))):
        if d is not None:
            # TODO: account for shapes not quite matching
            d = skimage.transform.pyramid_expand(d)
            d = d[:pyr1.shape[0], :pyr2.shape[1]]

        d = flow_iterative(pyr1, pyr2, c1=c1_, c2=c2_, d=d, **opts)

    xw = d + np.moveaxis(np.indices(f1.shape), 0, -1)

    # ---------------------------------------------------------------
    # calculate optical flow with opencv
    # ---------------------------------------------------------------

    opts_cv = dict(
        pyr_scale=0.5,
        levels=6,
        winsize=25,
        iterations=10,
        poly_n=25,
        poly_sigma=3.0,
        # flags=0
        flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN
    )

    d2 = cv2.calcOpticalFlowFarneback(
        f2.astype(np.uint8),
        f1.astype(np.uint8),
        None,
        **opts_cv
    )
    d2 = -d2[..., (1, 0)]

    xw2 = d2 + np.moveaxis(np.indices(f1.shape), 0, -1)

    # ---------------------------------------------------------------
    # use calculated optical flow to warp images
    # ---------------------------------------------------------------

    # opencv warped frame
    f2_w2 = skimage.transform.warp(f2, np.moveaxis(xw2, -1, 0), cval=np.nan)

    # warped frame
    f2_w = skimage.transform.warp(f2, np.moveaxis(xw, -1, 0), cval=np.nan)

    # ---------------------------------------------------------------
    # visualize results
    # ---------------------------------------------------------------

    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)

    p = 2.0  # percentile of histogram edges to chop off
    vmin, vmax = np.nanpercentile(f1 - f2, [p, 100 - p])
    cmap = 'gray'

    axes[0, 0].imshow(f1, cmap=cmap)
    axes[0, 0].set_title('f1 (fixed image)')
    axes[0, 1].imshow(f2, cmap=cmap)
    axes[0, 1].set_title('f2 (moving image)')
    axes[1, 0].imshow(f1 - f2_w2, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1, 0].set_title('difference f1 - f2 warped: opencv implementation')
    axes[1, 1].imshow(f1 - f2_w, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1, 1].set_title('difference f1 - f2 warped: this implementation')

    plt.show()


if __name__ == '__main__':
    main()
