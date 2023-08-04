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

from astropy.io import ascii
import numpy as np
import common
import os

##################################
h0 = 0.6751

def prepare_data(phot_data, phot_dataab, ids_sed, hdf5_data, subvols, lightcone_dir,  nbands):

    (dec, ra, zobs, idgal, sfrb, sfrd, mstarb, mstard, rsb, rsd) = hdf5_data
   
    SEDs_dust   = phot_data[0]
    SEDs_nodustab   = phot_dataab[0]

    mstartot = np.log10((mstarb+mstard)/h0)

    bands = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)

   #(0): "FUV_GALEX", "NUV_GALEX", "u_SDSS", "g_SDSS", "r_SDSS", "i_SDSS",
   #(6): "z_SDSS", "Y_VISTA", "J_VISTA", "H_VISTA", "K_VISTA", "W1_WISE",
   #(12): "I1_Spitzer", "I2_Spitzer", "W2_WISE", "I3_Spitzer", "I4_Spitzer",
   #(17): "W3_WISE", "W4_WISE", "P70_Herschel", "P100_Herschel",
   #(21): "P160_Herschel", "S250_Herschel", "S350_Herschel", "S450_JCMT",
   #(25): "S500_Herschel", "S850_JCMT", "Band9_ALMA", "Band8_ALMA",
   #(29): "Band7_ALMA", "Band6_ALMA", "Band5_ALMA", "Band4_ALMA"

    ind = np.where((zobs <= 6) & (SEDs_dust[6,:] <= 25) & (SEDs_dust[6,:] > 0))
    zin = zobs[ind]
    rain = ra[ind]
    decin = dec[ind]
    smin = mstartot[ind]
    sedsin = SEDs_dust[:,ind]
    sedsabin = SEDs_nodustab[:,ind]
    sedsin = sedsin[:,0,:]
    sedsabin = sedsabin[:,0,:]
    print(sedsin.shape)

    data_to_write = np.zeros(shape = (len(zin), 34))
    for i in range(0, len(zin)):
        data_to_write[i,0] = decin[i]
        data_to_write[i,1] = rain[i]
        data_to_write[i,2] = zin[i]
        data_to_write[i,3] = smin[i]
        for j, b in enumerate(bands):
            data_to_write[i,j+4] = sedsin[b,i]
            data_to_write[i,j+4+len(bands)] = sedsabin[b,i]

    print(data_to_write.shape)
    print(data_to_write[0,:])
    writeon = True
    if(writeon == True):
       ascii.write(data_to_write, '/group/pawsey0119/clagos/Stingray/output/medi-SURFS/Shark-TreeFixed-ReincPSO-kappa0p002/deep-optical-final/Shark-Lightcone-4MOST-extragalactic.csv', names = ['dec', 'ra', 'redshift', 'log10(mstar)', 'app_FUV', 'app_NUV', 'app_u_VST', 'app_g_VST', 'app_r_VST', 'app_i_VST', 'app_VISTAZ', 'app_VISTAY', 'app_VISTAJ', 'app_VISTAH', 'app_VISTAK', 'app_W1', 'app_Spitzer1', 'app_Spitzer2', 'app_W2', 'abs_FUV', 'abs_NUV', 'abs_u', 'abs_g', 'abs_r', 'abs_i', 'abs_z', 'abs_VISTAY', 'abs_VISTAJ', 'abs_VISTAH', 'abs_VISTAK', 'abs_W1', 'abs_Spitzer1', 'abs_Spitzer2', 'abs_W2'], overwrite = True, format='csv', fast_writer=False)
     
def main():

    lightcone_dir = '/group/pawsey0119/clagos/Stingray/output/medi-SURFS/Shark-TreeFixed-ReincPSO-kappa0p002/deep-optical-final/'
    outdir= '/group/pawsey0119/clagos/Stingray/output/medi-SURFS/Shark-TreeFixed-ReincPSO-kappa0p002/deep-optical-final/Plots/'
    obsdir= '/home/clagos/shark/data/'

    subvols = range(64) #(0,1,2,3,4,5,6,7,8,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35)
    #(9,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63) #0,1,2,3,4,5,6,7,8,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35) #range(64)
    sed_file = "Sting-SED-VST-eagle-rr14"

    # Loop over redshift and subvolumes
    plt = common.load_matplotlib()
    totarea = 107.8890011908422 #deg2
    areasub = totarea/64.0 * len(subvols)  #deg2

    #fields_sed = {'SED/ab_dust': ('total', 'disk', 'bulge_t')}

    #ids_sed_ab, seds_ab = common.read_photometry_data_hdf5(lightcone_dir, fields_sed, subvols)

    fields_sed = {'SED/ap_dust': ('total', 'bulge_t')}
    fields_absed = {'SED/ab_nodust': ('total', 'bulge_t')}

    ids_sed, seds = common.read_photometry_data_hdf5(lightcone_dir, fields_sed, subvols, sed_file)
    ids_sed, sedsab = common.read_photometry_data_hdf5(lightcone_dir, fields_absed, subvols, sed_file)

    fields = {'galaxies': ('dec', 'ra', 'zobs',
                           'id_galaxy_sky','sfr_burst','sfr_disk','mstars_bulge','mstars_disk','rstar_bulge_apparent','rstar_disk_apparent')}

    hdf5_data = common.read_lightcone(lightcone_dir, fields, subvols, "mock")

    nbands = len(seds[0])
    prepare_data(seds, sedsab, ids_sed, hdf5_data, subvols, lightcone_dir, nbands)


if __name__ == '__main__':
    main()
