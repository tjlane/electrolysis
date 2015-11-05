
"""
Input/Output
"""

import struct
import numpy as np


def load(filename, sequential_darks=False):
    d = BerkeleyDat(filename).images.astype(np.float)
    if sequential_darks:
        dark = np.mean(d[0::2,:,:], axis=0)
        images = d[1::2,:,:] - dark
    else:
        images = d
    return images


class BerkeleyDat(object):
    """
    Files contain 36 bytes of metadata followed by data from 32 images.

    -- metadata: nine 32-bit integers: one unsigned 32-bit slow-adc-timestamp 
       (on same 20 us per-tick image clk see 'pixel1' below), followed by the
       eight 32-bit signed slow-adc-values
    -- image data: 16-bit unsigned integers: first four pixels contain 
       image-metadata

       pixel0 = readout information should be 0x003f
       pixel1 = image timestamp (since the system was powered up, 
                rolls over...) , provides a relative time of when 
                an image was acquired (20 us per tick)
       pixel3 & pixel4 = 32-bit unsigned int frame number (since the 
                         enabling of file writing or the restart of 
                         the daq)
    """


    def __init__(self, filename):

        self._filename = filename
        self._read(self._filename)


        return


    def _read(self, filename):

        # ---> image size
        # 36 bytes metadata +
        # 1024 x 1024 pixels x 2 bytes x 32 images

        handle = open(filename, 'rb')

        # ---> read metadata header
        self._slow_adc_timestamp = struct.unpack("I", handle.read(4))
        self._slow_adc_values = [ struct.unpack("i", handle.read(4)) \
                                  for x in range(8) ]

        # ---> read the image data (bytes * pixels * num images)
        img = np.fromstring( handle.read(2 * 1024 * 1024 * 32), np.uint16 )
        img = img.reshape(32, 1024, 1024)

        # ---> get the per-image headers out (first 4 pixels are overwritten)
        self._img_headers = np.copy(img[:,0,0:4])
        img[:,0,0:4] = 0

        self._images = img

        return


    @property
    def images(self):
        return self._images

"""andor.py version 1.0
November 2008
Michael V. DePalatis <surname at gmail dot com>

Contains functions to handle Andor SIF image files including
conversion to other formats and reading basic information from the SIF
file. Little is done in the way of error checking, but this is
probably not a real issue as long as you actually pass it a valid SIF
file.
"""

import sys
import numpy as np

class SifFile:
    """SifFile is the Python representation of an Andor SIF image
    file. Image data is stored in a numpy array indexed as [row,
    column] instead of [x, y]."""

    def __init__(self, path=""):
        self.data = 0
        if path != "":
            self.open(path)

    def __add__(self, other):
        new_sif = self.__class__()
        new_sif.__dict__ = self.__dict__
        if isinstance(other, (SifFile, np.ndarray)):
            new_sif.data = new_sif.data + other.data
        elif isinstance(other, (int, float)):
            new_sif.data = new_sif.data + other
        else:
            raise TypeError("Addition of SIF data requires another SifFile instance, a numpy array, or a scalar.")
        return new_sif

    def __sub__(self, other):
        new_sif = self.__class__()
        new_sif.__dict__ = self.__dict__
        if isinstance(other, (SifFile, np.ndarray)):
            new_sif.data = new_sif.data - other.data
        elif isinstance(other, (int, float)):
            new_sif.data = new_sif.data - other
        else:
            raise TypeError("Subtraction of SIF data requires another SifFile instance, a numpy array, or a scalar.")
        return new_sif

    def __mul__(self, other):
        new_sif = self.__class__()
        new_sif.__dict__ = self.__dict__
        if isinstance(other, (SifFile, np.ndarray)):
            new_sif.data = new_sif.data * other.data
        elif isinstance(other, (int, float)):
            new_sif.data = new_sif.data * other
        else:
            raise TypeError("Multiplcation of SIF data requires another SifFile instance, a numpy array, or a scalar.")
        return new_sif

    def __rmul__(self, other):
        return self.__mul__(other)

    def open(self, path):
        """Opens the SIF file at path and stores the data in
        self.data."""
        sif = open(path, "rb")

        # Verify we have a SIF file
        if sif.readline().strip() != "Andor Technology Multi-Channel File":
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

        # Ignore next 17 lines prior to superpixeling information
        for i in range(17):
            sif.readline()

        # Read superpixeling data
        line = sif.readline().split()
        #self.shape = (self.ccd_size[1]/int(line[5]), self.ccd_size[0]/int(line[6]))

        # Skip next line
        for i in range(1):
            sif.readline()

        # Read data
        self.data = np.fromstring(sif.read(), dtype=np.float32)
        self.data = self.data[:len(self.data)-2]
        if line[3] < line[2]:
            self.shape = (len(self.data)/int(line[3]), int(line[3]))
        else:
            # I'm not sure if this is correct...
            # Needs more testing.
            self.shape = (int(line[2]), len(self.data)/int(line[2]))
        self.data = np.reshape(self.data, self.shape)



def test_sif():
    import pylab
    sif = SifFile("sum.sif")
    sif2 = 2*sif
    print sif2.data
    pylab.imshow(sif.data, interpolation="nearest")
    pylab.show()    




def test_lbl():

    #from matplotlib import pyplot as plt

    dat = BerkeleyDat('/reg/data/ana03/ued/20151026/scan_0000000595/'
                      'team1k_0000000595_0000000025.dat')
    #print dat.images
    np.save('test.npy', dat.images[0])
    #plt.imshow(dat.images[0])
    #plt.show()

if __name__ == '__main__':
    test()
