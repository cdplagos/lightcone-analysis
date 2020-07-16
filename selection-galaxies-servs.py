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

   #(0): "FUV_GALEX", "NUV_GALEX", "u_SDSS", "g_SDSS", "r_SDSS", "i_SDSS",
   #(6): "z_SDSS", "Y_VISTA", "J_VISTA", "H_VISTA", "K_VISTA", "W1_WISE",
   #(12): "I1_Spitzer", "I2_Spitzer", "W2_WISE", "I3_Spitzer", "I4_Spitzer",
   #(17): "W3_WISE", "W4_WISE", "P70_Herschel", "P100_Herschel",
   #(21): "P160_Herschel", "S250_Herschel", "S350_Herschel", "S450_JCMT",
   #(25): "S500_Herschel", "S850_JCMT", "Band9_ALMA", "Band8_ALMA",
   #(29): "Band7_ALMA", "Band6_ALMA", "Band5_ALMA", "Band4_ALMA"

    with open('/group/pawsey0119/clagos/Stingray/output/medi-SURFS/Shark-TreeFixed-ReincPSO-kappa0p002/deep-optical/Shark-deep-opticalLightcone-Hosein2020.txt', 'wb') as fil:
         fil.write("#Galaxies from Shark (Lagos et al. 2018) in the optical-deep lightcone\n")
         fil.write("#SED modelling as described in Lagos et al. (2019).\n")
         fil.write("#area 107.889 deg2\n")
         fil.write("#mag_y < 23.5\n")
         fil.write("#redshift < 1.5\n")
         fil.write("#units\n")
         fil.write("#mstar[Msun]\n")
         fil.write("#sfr[Msun/yr]\n")
         fil.write("#re[arcsec]\n")
         fil.write("#magnitudes AB\n")
         fil.write("#\n")
         fil.write("#id_galaxy_sky dec ra redshift log10(mstar) log10(sfr) re B/T app_u app_g app_r app_i app_z app_VISTAY app_VISTAJ app_VISTAH app_VISTAK app_W1 app_Spitzer1 app_Spitzer2 app_W2 abs_u abs_g abs_r abs_i abs_z abs_VISTAY abs_VISTAJ abs_VISTAH abs_VISTAK abs_W1 abs_Spitzer1 abs_Spitzer2 abs_W2\n")
         for ids, a,b,c,d,e,f,g,h1,i1,j1,k1,l1,m1,n1,o1,p1,q1,r1,s1,t1,h2,i2,j2,k2,l2,m2,n2,o2,p2,q2,r2,s2,t2 in zip(idgal, dec, ra, zobs, mstartot, sfrtot, re, BT, SEDs_dust[2], SEDs_dust[3], SEDs_dust[4], SEDs_dust[5], SEDs_dust[6], SEDs_dust[7], SEDs_dust[8], SEDs_dust[9], SEDs_dust[10], SEDs_dust[11], SEDs_dust[12], SEDs_dust[13],SEDs_dust[14],  SEDs_dustab[2], SEDs_dustab[3], SEDs_dustab[4], SEDs_dustab[5], SEDs_dustab[6], SEDs_dustab[7], SEDs_dustab[8], SEDs_dustab[9], SEDs_dustab[10], SEDs_dustab[11], SEDs_dustab[12], SEDs_dustab[13], SEDs_dustab[14]):
             if(l1 > 0 and m1 < 23.5 and c < 1.5):
                fil.write("%20.0f %5.10f %5.10f %5.7f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f\n" % (ids, a,b,c,d,e,f,g,h1,i1,j1,k1,l1,m1,n1,o1,p1,q1,r1,s1,t1,h2,i2,j2,k2,l2,m2,n2,o2,p2,q2,r2,s2,t2))

def main():

    lightcone_dir = '/group/pawsey0119/clagos/Stingray/output/medi-SURFS/Shark-TreeFixed-ReincPSO-kappa0p002/deep-optical/'
    outdir= '/group/pawsey0119/clagos/Stingray/output/medi-SURFS/Shark-TreeFixed-ReincPSO-kappa0p002/deep-optical/Plots/'
    obsdir= '/home/clagos/shark/data/'

    subvols = range(64) #(0,1,2,3,4,5,6,7,8,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35)
    #(9,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63) #0,1,2,3,4,5,6,7,8,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35) #range(64)
    sed_file = "Sting-SED-eagle-rr14"

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
