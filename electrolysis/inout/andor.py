
"""
andor images
"""

import struct
import numpy as np


class SifFile(object):
    """
    SifFile is the Python representation of an Andor SIF image
    file. Image data is stored in a numpy array indexed as [row,
    column] instead of [x, y].

    original code:
    November 2008
    Michael V. DePalatis <surname at gmail dot com>

    modifications:
    November 2015
    TJ Lane <tjlane@slac.stanford.edu>

    
    """

    def __init__(self, path):
        self.open(path)
        return


    def open(self, path):
        """Opens the SIF file at path and stores the data in
        self.data."""

        if path.lower().endswith('.sif'):
            opener = open
        elif path.lower().endswith('.sif.bz2'): # bz2 compressed sif files
            import bz2
            opener = bz2.BZ2File
        else:
            raise ValueError('Wrong extension.')

        sif = opener(path, 'rb')

        # Verify we have a SIF file
        if sif.readline().strip() != b"Andor Technology Multi-Channel File":
            sif.close()
            raise Exception("File %s is not an Andor SIF file." % path)

        # Ignore lines until we get to camera model
        for i in range(2):
            sif.readline()

        # Get camera model
        self.cammodel = sif.readline().strip()

        # Get CCD dimension in pixels
        shape = sif.readline().split()
        self.ccd_size = (int(shape[0]), int(shape[1]))

        # Read superpixeling data
        line = sif.readline().split()

        # Read data
        raw_data = sif.read()

        #raw_data = raw_data[len(raw_data) - 4*np.prod(self.ccd_size):]
        #self.data = np.fromstring(raw_data, dtype=np.float32)
        #self.data = np.reshape(self.data, self.ccd_size)

        off = 4 * 105
        raw_data = raw_data[len(raw_data) - 4*np.prod(self.ccd_size) - off:-off]
        self.data = np.fromstring(raw_data, dtype=np.float32)
        self.data = np.reshape(self.data, self.ccd_size)

        return




def test_sif():
    import pylab
    sif = SifFile("sum.sif")
    sif2 = 2*sif
    print sif2.data
    pylab.imshow(sif.data, interpolation="nearest")
    pylab.show()    
