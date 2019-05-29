#
# ICRAR - International Centre for Radio Astronomy Research
# (c) UWA - The University of Western Australia, 2018
# Copyright by UWA (in the framework of the ICRAR)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
"""HMF plots"""

import numpy as np
import common
import os

##################################

def prepare_data(phot_data, ids_sed, hdf5_data, subvols, lightcone_dir, ncounts, nbands):

    (dec, ra, zobs, idgal) = hdf5_data
   
    #(SCO, id_cos) = co_hdf5_data
    #components of apparent magnitudes:
    #(len(my_data), 2, 2, 5, nbands)
    #0: disk instability bulge
    #1: galaxy merger bulge
    #2: total bulge
    #3: disk
    #4: total
    SEDs_dust   = phot_data[0]
    SEDs_dust_disk = phot_data[1]
    SEDs_dust_bulge = phot_data[2]

    bands = (2, 4, 10)
    with open('/mnt/su3ctm/clagos/Lightcones/DriverNumberCounts/Shark/Shark-Deep-GAMA-Lightcone.txt', 'wb') as f:
         f.write("#Galaxies from Shark (Lagos et al. 2018) in the GAMA-deep lightcone\n")
         f.write("#area of lightcone 107.889deg2\n")
         f.write("#dec ra z u r K\n")
         for a,b,c,d,e,g in zip(dec, ra, zobs, SEDs_dust[2], SEDs_dust[4], SEDs_dust[10]):
             if(d > 0 and d < 40):
                f.write("%5.2f %5.2f %5.2f %5.2f %5.2f %5.2f\n" % (a,b,c,d,e,g))

def main():

    lightcone_dir = '/mnt/su3ctm/clagos/Stingray/output/medi-SURFS/Shark-TreeFixed-ReincPSO-kappa0p002/deep-optical/'
    outdir= '/mnt/su3ctm/clagos/Stingray/output/medi-SURFS/Shark-TreeFixed-ReincPSO-kappa0p002/deep-optical/Plots/'
    obsdir= '/home/clagos/git/shark/data/'

    subvols = range(64)

    # Loop over redshift and subvolumes
    plt = common.load_matplotlib()
    totarea = 107.8890011908422 #deg2
    areasub = totarea/64.0 * len(subvols)  #deg2

    #fields_sed = {'SED/ab_dust': ('total', 'disk', 'bulge_t')}

    #ids_sed_ab, seds_ab = common.read_photometry_data_hdf5(lightcone_dir, fields_sed, subvols)

    fields_sed = {'SED/ap_dust': ('total', 'disk', 'bulge_t')}

    ids_sed, seds = common.read_photometry_data_hdf5(lightcone_dir, fields_sed, subvols)
 
    fields = {'galaxies': ('dec', 'ra', 'zobs',
                           'id_galaxy_sky')}

    hdf5_data = common.read_lightcone(lightcone_dir, fields, subvols)

    nbands = len(seds[0])
    ncounts = np.zeros(shape = (3, nbands, len(mbins)))
 
    prepare_data(seds, ids_sed, hdf5_data, subvols, lightcone_dir, ncounts, nbands)


if __name__ == '__main__':
    main()
