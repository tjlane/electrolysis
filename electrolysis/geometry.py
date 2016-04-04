
import numpy as np

def rotate(pitch, yaw, roll):

    # ccw around z 
    R_roll  = np.array([[ np.cos(roll), -np.sin(roll),  0.0 ],
                        [ np.sin(roll),  np.cos(roll),  0.0 ],
                        [         0.0,            0.0,  1.0 ] ])

    # ccw around y 
    R_yaw   = np.array([[ np.cos(yaw), 0.0, np.sin(yaw) ],
                        [         0.0, 1.0,         0.0 ],
                        [-np.sin(yaw), 0.0, np.cos(yaw) ] ])

    # ccw around x
    R_pitch = np.array([[ 1.0,           0.0,            0.0 ],
                        [ 0.0, np.cos(pitch), -np.sin(pitch) ],
                        [ 0.0, np.sin(pitch),  np.cos(pitch) ] ])

    # yaw 2nd, pitch last
    return np.dot(R_pitch, np.dot(R_yaw, R_roll))


def xyz_grid(shape, center, z_offset):

    center = np.array(center)

    mg = np.mgrid[0:shape[0]-1:1j*shape[0], 0:shape[1]-1:1j*shape[1]]
    xy = mg - np.array( center[:,None,None] )
    z  = np.ones((1, shape[0], shape[1])) * z_offset

    return np.concatenate((xy,z), axis=0)



def real_to_recip(xyz, wavelength):
    """
    assume xyz is (N,3)
    """
    beam_vector = np.array([0.0, 0.0, 1.0])
    norm = np.sqrt( np.sum( np.power(xyz, 2), axis=1 ) )
    S = xyz / norm[:,None]
    q = (2.0 * np.pi / wavelength) * (S - beam_vector)
    assert q.shape == xyz.shape, 'q-shape incorrect'
    return q


def polar_to_cart(polar):

    if len(polar.shape) == 2: # assume r = 1
        r     = 1.0
        theta = polar[:,0]
        phi   = polar[:,1]
    else:
        r     = polar[:,0]
        theta = polar[:,1]
        phi   = polar[:,2]

    xyz = r * np.array([ np.sin(theta) * np.cos(phi),
                         np.sin(theta) * np.sin(phi),
                         np.cos(theta) ])

    return xyz

def cart_to_polar(xyz):

    polar = np.zeros_like(xyz)
    polar[:,0] = np.sqrt( np.sum( np.square(xyz), axis=1) )
    polar[:,1] = np.arccos(xyz[:,2] / r)
    polar[:,2] = np.arctan2(xyz[:,1], xyz[:,0])

    return polar


