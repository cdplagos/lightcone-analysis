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
#the parameter below considers the difference between 
#the peak flux and the maximum in a box-shaped emission line
boost_box_profile = 1.5

def prepare_data(phot_data, ids_sed, hdf5_data, hdf5_co_data, hdf5_data_groups, subvols, lightcone_dir,  nbands):

    (dec, ra, zobs, idgal, sfrb, sfrd, mstarb, mstard, rsb, rsd, id_group) = hdf5_data
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

    #S850 microns selected to have a flux >0.05mJy
    ind = np.where((SEDs_dust[26,:] > 0) & (SEDs_dust[26,:] < 19.652640611442184))
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
       with open('/group/pawsey0119/clagos/Stingray/output/medi-SURFS/Shark-TreeFixed-ReincPSO-kappa0p002/deep-optical/Shark-deep-opticalLightcone-AtLAST.txt', 'wb') as fil:
            fil.write("#Galaxies from Shark (Lagos et al. 2018) in the optical-deep lightcone\n")
            fil.write("#SED modelling as described in Lagos et al. (2019).\n")
            fil.write("#area 107.889 deg2\n")
            fil.write("#S_850microns > 0.05 milli Jy\n")
            fil.write("#units\n")
            fil.write("#mstar[Msun]\n")
            fil.write("#sfr[Msun/yr]\n")
            fil.write("#re[arcsec]\n")
            fil.write("#magnitudes AB\n")
            fil.write("#\n")
            fil.write("#dec ra redshift log10(mstar) log10(sfr) re B/T app_1500Angs app_u app_g app_r app_i app_z app_VISTAK app_Spitzer1 app_Spitzer2 id_group_sky\n")
            for a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,r in zip(dec[ind], ra[ind], zobs[ind], mstartot[ind], sfrtot[ind], re[ind], BT[ind], SEDs_dustin[0,:], SEDs_dustin[2,:], SEDs_dustin[3,:], SEDs_dustin[4,:], SEDs_dustin[5,:], SEDs_dustin[6,:], SEDs_dustin[10,:], SEDs_dustin[12,:], SEDs_dustin[13,:], id_group[ind]):
                fil.write("%5.10f %5.10f %5.7f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %10.0f\n" % (a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,r))
   
       with open('/group/pawsey0119/clagos/Stingray/output/medi-SURFS/Shark-TreeFixed-ReincPSO-kappa0p002/deep-optical/Shark-deep-opticalLightcone-AtLAST-FIR.txt', 'wb') as fil:
            fil.write("#Galaxies from Shark (Lagos et al. 2018) in the optical-deep lightcone\n")
            fil.write("#SED modelling as described in Lagos et al. (2019).\n")
            fil.write("#area 107.889 deg2\n")
            fil.write("#S_850microns > 0.05 milli Jy\n")
            fil.write("#units\n")
            fil.write("#fluxes in units of mJy\n")
            fil.write("#\n")
            fil.write("#flux_P70_Herschel flux_P100_Herschel flux_P160_Herschel flux_S250_Herschel flux_S350_Herschel flux_S450_JCMT flux_S500_Herschel flux_S850_JCMT flux_Band7_ALMA flux_Band6_ALMA flux_Band5_ALMA flux_Band4_ALMA\n")
            for a,b,c,d,e,f,g,h,i,j,k,l in zip(fluxSEDsin[19,:], fluxSEDsin[20,:], fluxSEDsin[21,:], fluxSEDsin[22,:], fluxSEDsin[23,:], fluxSEDsin[24,:], fluxSEDsin[25,:], fluxSEDsin[26,:], fluxSEDsin[29,:], fluxSEDsin[30,:], fluxSEDsin[31,:], fluxSEDsin[32,:]):
                fil.write("%5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f \n" % (a,b,c,d,e,f,g,h,i,j,k,l))
   
       with open('/group/pawsey0119/clagos/Stingray/output/medi-SURFS/Shark-TreeFixed-ReincPSO-kappa0p002/deep-optical/Shark-deep-opticalLightcone-AtLAST-CO.txt', 'wb') as fil:
            fil.write("#Galaxies from Shark (Lagos et al. 2018) in the optical-deep lightcone\n")
            fil.write("#SED modelling as described in Lagos et al. (2019).\n")
            fil.write("#area 107.889 deg2\n")
            fil.write("#S_850microns > 0.05 milli Jy\n")
            fil.write("#units\n")
            fil.write("#CO integrated line fluxes in Jy km/s\n")
            fil.write("#FWHM in km/s\n")
            fil.write("#\n")
            fil.write("#SCO(1-0) SCO(2-1) SCO(3-2) SCO(4-3) SCO(5-4) SCO(6-5) SCO(7-6) SCO(8-7) SCO(9-8) SCO(10-9) FWHM\n")
            for a,b,c,d,e,f,g,h,i,j,k in zip(SCOin[0,:], SCOin[1,:], SCOin[2,:], SCOin[3,:], SCOin[4,:], SCOin[5,:], SCOin[6,:], SCOin[7,:], SCOin[8,:], SCOin[9,:], vline[ind]):
                fil.write("%5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f\n" % (a,b,c,d,e,f,g,h,i,j,k))

       with open('/group/pawsey0119/clagos/Stingray/output/medi-SURFS/Shark-TreeFixed-ReincPSO-kappa0p002/deep-optical/Shark-deep-opticalLightcone-AtLAST-Groups.txt', 'wb') as fil:
            fil.write("#Galaxies from Shark (Lagos et al. 2018) in the optical-deep lightcone\n")
            fil.write("#SED modelling as described in Lagos et al. (2019).\n")
            fil.write("#area 107.889 deg2\n")
            fil.write("#S_850microns > 0.05 milli Jy\n")
            fil.write("#units\n")
            fil.write("#mvir in [Msun]\n")
            fil.write("#\n")
            fil.write("#id_group_galaxy mvir ra_group dec_group num_all_satellites\n")
            for a,b,c,d,e,f in zip(id_group_skyin, mvirin, ra_groupin, dec_groupin, n_allin, zobs_groupin):
                fil.write("%10.0f %5.2f %5.10f %5.10f %5.0f %5.2f\n" % (a,b,c,d,e,f))

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

    ids_sed, seds = common.read_photometry_data_hdf5(lightcone_dir, fields_sed, subvols, sed_file)

    fields = {'galaxies': ('dec', 'ra', 'zobs',
                           'id_galaxy_sky','sfr_burst','sfr_disk','mstars_bulge','mstars_disk','rstar_bulge_apparent',
                           'rstar_disk_apparent','id_group_sky')}

    hdf5_data = common.read_lightcone(lightcone_dir, fields, subvols)

    fields = {'groups': ('id_group_sky', 'mvir', 'n_galaxies_selected',
                         'n_galaxies_total','ra','dec','zobs')}

    hdf5_data_groups = common.read_lightcone(lightcone_dir, fields, subvols)

    fields = {'galaxies': ('SCO','SCO_peak')}
    hdf5_co_data = common.read_co_lightcone(lightcone_dir, fields, subvols)

    nbands = len(seds[0])
    prepare_data(seds, ids_sed, hdf5_data, hdf5_co_data, hdf5_data_groups, subvols, lightcone_dir, nbands)


if __name__ == '__main__':
    main()
