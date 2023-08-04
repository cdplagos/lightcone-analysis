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
import functools
import common
import os
import utilities_statistics as us
 
##################################
h0 = 0.6751
PI = 3.141592654

#the parameter below considers the difference between 
#the peak flux and the maximum in a box-shaped emission line
boost_box_profile = 1.5

# Mass function initialization

vlow = 1.5
vupp = 3.5
dv = 0.25
vbins = np.arange(vlow,vupp,dv)
xvf   = vbins + dv/2.0


def prepare_data(phot_data, ids_sed, hdf5_data, hdf5_co_data, hdf5_data_groups, subvols, lightcone_dir,  nbands, lco_vel_scaling):

    bin_it = functools.partial(us.wmedians, xbins=xvf)

    (dec, ra, zobs, idgal, sfrb, sfrd, mstarb, mstard, rsb, rsd, id_group, dc, mvir_host) = hdf5_data
    (SCO, SCO_peak) = hdf5_co_data
    (id_group_sky, mvir, n_selec, n_all, ra_group, dec_group, zobs_group) = hdf5_data_groups
    #(SCO, id_cos) = co_hdf5_data

    vline = SCO[:,0] / SCO_peak[:,0] / boost_box_profile

    #components of apparent magnitudes:
    #(len(my_data), 2, 2, 5, nbands)
    #0: disk instability bulge
    #1: galaxy merger bulge
    #2: total bulge
    #3: disk
    #4: total
    SEDs_dust   = phot_data[0]
    SEDs_dust_bulge = phot_data[1]

    mstartot = np.log10((mstarb+mstard)/h0)
    sfrtot = np.log10((sfrb+sfrd)/h0/1e9)
    re = (rsb*mstarb + mstard*rsd) / (mstarb+mstard)
    BT = mstarb / (mstarb+mstard)

   #(0): "FUV_GALEX", "NUV_GALEX", "u_SDSS", "g_SDSS", "r_SDSS", "i_SDSS",
   #(6): "z_SDSS", "Y_VISTA", "J_VISTA", "H_VISTA", "K_VISTA", "W1_WISE",
   #(12): "I1_Spitzer", "I2_Spitzer", "W2_WISE", "I3_Spitzer", "I4_Spitzer",
   #(17): "W3_WISE", "W4_WISE", "P70_Herschel", "P100_Herschel",
   #(21): "P160_Herschel", "S250_Herschel", "S350_Herschel", "S450_JCMT",
   #(25): "S500_Herschel", "S850_JCMT", "Band9_ALMA", "Band8_ALMA",
   #(29): "Band7_ALMA", "Band6_ALMA", "Band5_ALMA", "Band4_ALMA"

    bands = (2, 3, 4, 5, 6, 10, 12, 13, 19, 20, 21, 22, 23, 24, 25, 26, 30, 31, 32)
    fluxSEDs = 10.0**(SEDs_dust/(-2.5)) * 3631.0 * 1e3 #in mJy

    dgal = 4.0 * PI * pow((1.0+zobs) * dc/h0, 2.0)

    LCO = SCO[:, 0] * dgal * 3.25e7 / (115.2712)**2.0 / (4.0*PI)
    ind = np.where((zobs < 2.1) & (LCO > 1e6))
    lco_vel_scaling[0, :] = bin_it(x=np.log10(vline[ind]), y=np.log10(LCO[ind]))
    ind = np.where((zobs < 2.1) & (LCO > 1e6) & (mvir_host > 1e13))
    lco_vel_scaling[1, :] = bin_it(x=np.log10(vline[ind]), y=np.log10(LCO[ind]))

    ind = np.where((zobs >1.9) & (zobs < 2.1) & (LCO > 1e10))
    lcobright = np.log10(LCO[ind]) 
    vcobright = np.log10(vline[ind])

    #450mu JCMT selected to have a flux >0.1mJy
    ind = np.where((SEDs_dust[24,:] > 0) & (SEDs_dust[24,:] < 18.900065622282231))
    SEDs_dustin = SEDs_dust[:,ind]
    fluxSEDsin = fluxSEDs[:,ind]
    SCOin = SCO[ind,:]
    SEDs_dustin = SEDs_dustin[:,0,:]
    fluxSEDsin = fluxSEDsin[:,0,:]
    SCOin = SCOin[0,:,:]
    SCOin = SCOin.reshape((10,len(dec[ind])))
    #select unique group ids and match them with the data in /groups/
    group_ids_unique = np.unique(id_group[ind])
    matching_groups = np.in1d(id_group_sky, group_ids_unique)

    #select only those groups that host the galaxies selected
    ingroup = np.where(matching_groups)
    mvirin = mvir[ingroup]
    n_allin = n_all[ingroup]
    ra_groupin = ra_group[ingroup]
    dec_groupin = dec_group[ingroup]
    id_group_skyin = id_group_sky[ingroup]
    zobs_groupin = zobs_group[ingroup]

    writeon = True
    if(writeon == True):
       with open('/group/pawsey0119/clagos/Stingray/output/medi-SURFS/Shark-TreeFixed-ReincPSO-kappa0p002/deep-optical-final/Shark-deep-opticalLightcone-SMGsAt450-p1.txt', 'wt') as fil:
            fil.write("#Galaxies from Shark (Lagos et al. 2018) in the optical-deep lightcone\n")
            fil.write("#SED modelling as described in Lagos et al. (2019) and used in Lagos et al. (2020) to analyse SMGs.\n")
            fil.write("#area 107.889 deg2\n")
            fil.write("#S_450mu > 0.1 milli Jy\n")
            fil.write("#units\n")
            fil.write("#mstar[Msun]\n")
            fil.write("#sfr[Msun/yr]\n")
            fil.write("#fluxes in mJy\n")
            fil.write("#\n")
            fil.write("#dec ra redshift log10(mstar) log10(sfr) S_450mu_JCMT S_850mu_JCMT S_870mu_ALMAB7 S_1mm_ALMAB6 S_2mm_ALMAB4\n")
            for a,b,c,d,e,f,g,h,i,j in zip(dec[ind], ra[ind], zobs[ind], mstartot[ind], sfrtot[ind], fluxSEDsin[24,:], fluxSEDsin[26,:], fluxSEDsin[29,:], fluxSEDsin[30,:], fluxSEDsin[32,:]):
                fil.write("%5.10f %5.10f %5.7f %5.3f %5.3f %5.3f %5.3f %5.3f %5.3f %5.3f\n" % (a,b,c,d,e,f,g,h,i,j))
   

def main():

    lightcone_dir = '/group/pawsey0119/clagos/Stingray/output/medi-SURFS/Shark-TreeFixed-ReincPSO-kappa0p002/deep-optical-final/'
    outdir= '/group/pawsey0119/clagos/Stingray/output/medi-SURFS/Shark-TreeFixed-ReincPSO-kappa0p002/deep-optical-final/Plots/'
    obsdir= '/home/clagos/shark/data/'


    #initialize relevant matrices
    lco_vel_scaling = np.zeros(shape = (2, 4, len(vbins)))

    subvols = [1,6,8,9,11,13,15,16,17,18,20,21,22,25,29,31,33,36,39,40,43,46,47,48,49,50,52,55,56,57,59,60,61,62]
    #[2,3,4,5,7,10,12,14,19,23,26,27,28,30,32,34,37,38,41,42,44,45,51,53,54,58,63]
    #[0,24,35] 
    #[1,6,8,9,11,13,15,16,17,18,20,21,22,25,29,31,33,36,39,40,43,46,47,48,49,50,52,55,56,57,59,60,61,62]
    #range(64) 
    #(20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63) 
    #range(20) #(0,1,2,3,4,5,6,7,8,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35)
    #(9,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63) #0,1,2,3,4,5,6,7,8,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35) #range(64)
    sed_file = "Sting-SED-eagle-rr14"

    # Loop over redshift and subvolumes
    plt = common.load_matplotlib()
    totarea = 107.8890011908422 #deg2
    areasub = totarea/64.0 * len(subvols)  #deg2

    #fields_sed = {'SED/ab_dust': ('total', 'disk', 'bulge_t')}

    #ids_sed_ab, seds_ab = common.read_photometry_data_hdf5(lightcone_dir, fields_sed, subvols)

    fields_sed = {'SED/ap_dust': ('total', 'bulge_t')}

    ids_sed, seds = common.read_photometry_data_hdf5(lightcone_dir, fields_sed, subvols, sed_file)

    fields = {'galaxies': ('dec', 'ra', 'zobs',
                           'id_galaxy_sky','sfr_burst','sfr_disk','mstars_bulge','mstars_disk','rstar_bulge_apparent',
                           'rstar_disk_apparent','id_group_sky','dc', 'mvir_hosthalo')}

    hdf5_data = common.read_lightcone(lightcone_dir, fields, subvols, "mock")

    fields = {'groups': ('id_group_sky', 'mvir', 'n_galaxies_selected',
                         'n_galaxies_total','ra','dec','zobs')}

    hdf5_data_groups = common.read_lightcone(lightcone_dir, fields, subvols, "mock")

    fields = {'galaxies': ('SCO','SCO_peak')}
    hdf5_co_data = common.read_co_lightcone(lightcone_dir, fields, subvols)

    nbands = len(seds[0])
    prepare_data(seds, ids_sed, hdf5_data, hdf5_co_data, hdf5_data_groups, subvols, lightcone_dir, nbands, lco_vel_scaling)


if __name__ == '__main__':
    main()
