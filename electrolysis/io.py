
"""
Input/Output
"""

import struct
import numpy as np


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



def test():

    #from matplotlib import pyplot as plt

    dat = BerkeleyDat('/reg/data/ana03/ued/20151026/scan_0000000595/'
                      'team1k_0000000595_0000000025.dat')
    #print dat.images
    np.save('test.npy', dat.images[0])
    #plt.imshow(dat.images[0])
    #plt.show()

if __name__ == '__main__':
    test()
