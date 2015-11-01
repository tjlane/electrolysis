
"""
align.py
"""


import numpy as np
from scipy import ndimage


def blob_align(images, init_peak_loc=None):
    """
    Align a series of images on the strongest peak.

    Pad your image beforehand if you want to retain information at the
    very edges of the image.
    """

    print images.shape

    # initial condition for peak placement
    if init_peak_loc is None:
        anchor_pos = np.array([images.shape[1], images.shape[2]]) / 2.0
    else:
        anchor_pos = init_peak_loc


    for i in range(images.shape[0]):

        # find the beam in the image by blurring w/Gaussian and peakfinding
        # parameters below were chosen to work well with Oct run on UED
        gi = ndimage.filters.gaussian_filter(images[i], 2.0)
        pos, shp = find_blobs(gi, discard_border=10,
                              discard_small_blobs=20)

        if len(pos) == 0:
            raise RuntimeError('No peaks found in image %d' % i)

        # choose the closest peak to the previous one
        pos = np.array(pos)
        diffs = pos - anchor_pos[None,:]
        new_pos = pos[ np.argmin(np.square(diffs).sum(1)) ] # L2 norm
        print 'is:', new_pos - anchor_pos

        # shift image
        images[i] = ndimage.interpolation.shift(images[i], anchor_pos - new_pos)
        
        # reset
        #prev_pos = new_pos


    return images


def find_blobs(image, sigma_threshold=5.0, discard_border=1,
               discard_small_blobs=0):
    """
    Find peaks, or `blobs`, in a 2D image.
    
    This algorithm works based on a simple threshold. It finds continuous
    regions of intensity that are greater than `sigma_threshold` standard
    deviations over the mean, and returns each of those regions as a single
    blob.
    
    Parameters
    ----------
    image : np.ndarray, two-dimensional
        An image to peakfind on.
        
    Returns
    -------
    centers : list of tuples of floats
        A list of the (x,y)/(col,row) positions of each peak, in pixels.
        
    widths : list of tuples of floats
        A list of the (x,y)/(col,row) size of each peak, in pixels.
        
    Optional Parameters
    -------------------
    sigma_threshold : float
        How many standard deviations above the mean to set the binary threshold.
    
    discard_border : int
        The size of a border region to ignore. In many images, the borders are
        noisy or systematically erroneous.
    
    discard_small_blobs : int
        Discard few-pixel blobs, which are the most common false positives
        for the blob finder. The argument specifies the minimal area
        (in pixels) a blob must encompass to be counted. Default: no
        rejections (0 pixels).

    Notes
    -----
    Tests indicate this algorithm takes ~200 ms to process a single image, so
    can run at ~5 Hz on a single processor.
    """
    
    if not len(image.shape) == 2:
        raise ValueError('Can only process 2-dimensional images')
    
    # discard the borders, which can be noisy...
    image[ :discard_border,:] = 0
    image[-discard_border:,:] = 0
    image[:, :discard_border] = 0
    image[:,-discard_border:] = 0
    
    # find the center of blobs above `sigma_threshold` STDs
    binary = (image > (image.mean() + image.std() * sigma_threshold))
    labeled, num_labels = ndimage.label(binary)
    centers = ndimage.measurements.center_of_mass(binary, 
                                                  labeled,
                                                  range(1,num_labels+1))
                                                    
                                                  
    # for each peak, find it's x- & y-width
    #   we do this by measuring how many pixels are above 5-sigma in both the
    #   x and y direction at the center of each blob
    
    widths = []
    warning_printed = False

    for i in range(num_labels)[::-1]: # backwards so pop works below
        
        c = centers[i]
        r_slice = labeled[int(c[0]),:]
        zy = np.where( np.abs(r_slice - np.roll(r_slice, 1)) == i+1 )[0]
        
        c_slice = labeled[:,int(c[1])]
        zx = np.where( np.abs(c_slice - np.roll(c_slice, 1)) == i+1 )[0]
        
        
        if not (len(zx) == 2) or not (len(zy) == 2):
            if not warning_printed:
                print "WARNING: Peak algorithm confused about width of peak at", c
                print "         Setting default peak width (5,5). This warning"
                print "         will only be printed ONCE. Proceed w/caution!"
                warning_printed = True
            widths.append( (5.0, 5.0) )
        else:
            x_width = zx[1] - zx[0]
            y_width = zy[1] - zy[0]

            # if the blob is a "singleton" and we want to get rid
            # of it, we do so, otherwise we add the widths
            if (x_width * y_width) < discard_small_blobs:
                #print "Discarding small blob %d, area %d" % (i, (x_width * y_width))
                centers.pop(i)
            else:
                widths.append( (x_width, y_width) )
        
    assert len(centers) == len(widths), 'centers and widths not same len'

    return centers, widths
    
    
def draw_blobs(image, centers, widths):
    """
    Draw blobs (peaks) on an image.
    
    Parameters
    ----------
    image : np.ndarray, two-dimensional
        An image to render.
    
    centers : list of tuples of floats
        A list of the (x,y) positions of each peak, in pixels.
        
    widths : list of tuples of floats
        A list of the (x,y) size of each peak, in pixels.
    """
    
    plt.figure()
    plt.imshow(image.T, interpolation='nearest')
    
    centers = np.array(centers)
    widths = np.array(widths)
    
    # flip the y-sign to for conv. below
    diagonal_widths = widths.copy()
    diagonal_widths[:,1] *= -1

    for i in range(len(centers)):
       
        # draw a rectangle around the center 
        pts = np.array([
               centers[i] - widths[i] / 2.0,          # bottom left
               centers[i] - diagonal_widths[i] / 2.0, # top left
               centers[i] + widths[i] / 2.0,          # top right
               centers[i] + diagonal_widths[i] / 2.0, # bottom right
               centers[i] - widths[i] / 2.0           # bottom left
              ])
        
        plt.plot(pts[:,0], pts[:,1], color='orange', lw=3)
        
    plt.xlim([0, image.shape[0]])
    plt.ylim([0, image.shape[1]])
    plt.show()
    
    return


def test_alignment():

    #from skimage import data
    #i0 = data.camera()

    i0 = np.zeros((100, 100))
    i0[50:55,50:55] = 1000.0

    s1 = np.random.randn(2)*10.0
    i1 = ndimage.interpolation.shift(i0, s1)
    print s1
    i2 = ndimage.interpolation.shift(i0, np.random.randn(2)*10.0)

    images = np.array([i0, i1, i2])
    unaligned = np.copy(images)

    blob_align(images)

    from matplotlib import pyplot as plt

    plt.figure()
    ax = plt.subplot(121)
    ax.imshow(unaligned.sum(0))
    ax = plt.subplot(122)
    ax.imshow(images.sum(0))
    plt.show()



if __name__ == "__main__":
    test_alignment()

