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
h0 = 0.6751

def prepare_data(phot_data, phot_dataab, ids_sed, hdf5_data, subvols, lightcone_dir,  nbands):

    (dec, ra, zobs, idgal, sfrb, sfrd, mstarb, mstard, rsb, rsd) = hdf5_data
   
    #(SCO, id_cos) = co_hdf5_data
    #components of apparent magnitudes:
    #(len(my_data), 2, 2, 5, nbands)
    #0: disk instability bulge
    #1: galaxy merger bulge
    #2: total bulge
    #3: disk
    #4: total
    SEDs_dust   = phot_data[0]
    SEDs_dust_bulge = phot_data[1]
    SEDs_dustab   = phot_dataab[0]
    SEDs_dust_bulgeab = phot_dataab[1]

    mstartot = np.log10((mstarb+mstard)/h0)
    sfrtot = np.log10((sfrb+sfrd)/h0/1e9)
    re = (rsb*mstarb + mstard*rsd) / (mstarb+mstard)
    BT = mstarb / (mstarb+mstard)

    bands = (2, 3, 4, 5, 6, 10, 12, 13)
    print SEDs_dust[13]
    with open('/group/pawsey0119/clagos/Stingray/output/medi-SURFS/Shark-TreeFixed-ReincPSO-kappa0p002/deep-optical/Shark-deep-opticalLightcone-SERVS.txt', 'wb') as fil:
         fil.write("#Galaxies from Shark (Lagos et al. 2018) in the optical-deep lightcone\n")
         fil.write("#SED modelling as described in Lagos et al. (2019).\n")
         fil.write("#area 107.889 deg2\n")
         fil.write("#S_3.6microns > 0.575 micro Jy\n")
         fil.write("#units\n")
         fil.write("#mstar[Msun]\n")
         fil.write("#sfr[Msun/yr]\n")
         fil.write("#re[arcsec]\n")
         fil.write("#magnitudes AB\n")
         fil.write("#\n")
         fil.write("#dec ra redshift log10(mstar) log10(sfr) re B/T app_u app_g app_r app_i app_z app_VISTAK app_Spitzer1 app_Spitzer2 abs_u abs_g abd_r abs_i abs_z abs_VISTAK abs_Spitzer1 abs_Spitzer2\n")
         for a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w in zip(dec, ra, zobs, mstartot, sfrtot, re, BT, SEDs_dust[2], SEDs_dust[3], SEDs_dust[4], SEDs_dust[5], SEDs_dust[6], SEDs_dust[10], SEDs_dust[12], SEDs_dust[13], SEDs_dustab[2], SEDs_dustab[3], SEDs_dustab[4], SEDs_dustab[5], SEDs_dustab[6], SEDs_dustab[10], SEDs_dustab[12], SEDs_dustab[13]):
             if(h > 0 and n <= 24.5):
                fil.write("%5.10f %5.10f %5.7f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f \n" % (a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w))

def main():

    lightcone_dir = '/group/pawsey0119/clagos/Stingray/output/medi-SURFS/Shark-TreeFixed-ReincPSO-kappa0p002/deep-optical/'
    outdir= '/group/pawsey0119/clagos/Stingray/output/medi-SURFS/Shark-TreeFixed-ReincPSO-kappa0p002/deep-optical/Plots/'
    obsdir= '/home/clagos/shark/data/'

    subvols = range(64) #(0,1,2,3,4,5,6,7,8,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35)
    #(9,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63) #0,1,2,3,4,5,6,7,8,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35) #range(64)
    sed_file = "Sting-SED-eagle-rr14-steep"

    # Loop over redshift and subvolumes
    plt = common.load_matplotlib()
    totarea = 107.8890011908422 #deg2
    areasub = totarea/64.0 * len(subvols)  #deg2

    #fields_sed = {'SED/ab_dust': ('total', 'disk', 'bulge_t')}

    #ids_sed_ab, seds_ab = common.read_photometry_data_hdf5(lightcone_dir, fields_sed, subvols)

    fields_sed = {'SED/ap_dust': ('total', 'bulge_t')}
    fields_absed = {'SED/ab_dust': ('total', 'bulge_t')}

    ids_sed, seds = common.read_photometry_data_hdf5(lightcone_dir, fields_sed, subvols, sed_file)
    ids_sed, sedsab = common.read_photometry_data_hdf5(lightcone_dir, fields_absed, subvols, sed_file)

    fields = {'galaxies': ('dec', 'ra', 'zobs',
                           'id_galaxy_sky','sfr_burst','sfr_disk','mstars_bulge','mstars_disk','rstar_bulge_apparent','rstar_disk_apparent')}

    hdf5_data = common.read_lightcone(lightcone_dir, fields, subvols)

    nbands = len(seds[0])
    prepare_data(seds, sedsab, ids_sed, hdf5_data, subvols, lightcone_dir, nbands)


if __name__ == '__main__':
    main()
