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

def prepare_data(phot_data, ids_sed, hdf5_data, hdf5_data_mvir, subvols, lightcone_dir,  nbands):

    (dec, ra, zobs, idgal, sfrb, sfrd, mstarb, mstard, rsb, rsd, id_group, dc, mvir_host, typeg) = hdf5_data
    (mvirz0, idgal, snap, subv) = hdf5_data_mvir

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
    mvir = np.log10(mvir_host/h0)
    mvirz0 = np.log10(mvirz0/h0)

   #(0): "FUV_GALEX", "NUV_GALEX", "u_VST", "g_VST", "r_VST", "i_VST",
   #(6): "Z_VISTA", "Y_VISTA", "J_VISTA", "H_VISTA", "K_VISTA", "W1_WISE",
   #(12): "I1_Spitzer", "I2_Spitzer", "W2_WISE", "I3_Spitzer", "I4_Spitzer",
   #(17): "W3_WISE", "W4_WISE", "P70_Herschel", "P100_Herschel",
   #(21): "P160_Herschel", "S250_Herschel", "S350_Herschel", "S450_JCMT",
   #(25): "S500_Herschel", "S850_JCMT", "Band9_ALMA", "Band8_ALMA",
   #(29): "Band7_ALMA", "Band6_ALMA", "Band5_ALMA", "Band4_ALMA"

    fluxSEDs = 10.0**(SEDs_dust/(-2.5)) * 3631.0 * 1e3 #in mJy

    dgal = 4.0 * PI * pow((1.0+zobs) * dc/h0, 2.0)

    #S850 microns selected to have a flux >0.05mJy
    ind = np.where((SEDs_dust[6,:] > 0) & (SEDs_dust[6,:] <= 23) & (zobs <= 3))
    SEDs_dustin = SEDs_dust[:,ind]
    SEDs_dustin = SEDs_dustin[:,0,:]
   

    writeon = True
    if(writeon == True):
       with open('/scratch/pawsey0119/clagos/Stingray/output/medi-SURFS/Shark-TreeFixed-ReincPSO-kappa0p002/deep-optical/Shark-deep-opticalLightcone-WAVES.txt', 'w') as fil:
            fil.write("#Galaxies from Shark (Lagos et al. 2018) in the optical-deep lightcone\n")
            fil.write("#SED modelling as described in Lagos et al. (2019).\n")
            fil.write("#\n")
            fil.write("#\n")
            fil.write("#\n")
            fil.write("#If using this lightcone for your science please include the following information (or similar) in your paper:\n")
            fil.write("#\n")
            fil.write("#The Shark semi-analytic model of galaxy formation (Lagos et al. 2018) generates self-consistent predictions from the far-ultraviolet to the far-infrared. Shark includes a wide range of baryon physics that are thought to play an important role in the formation and evolution of galaxies, including: (i) the collapse and merging of dark matter (DM) halos; (ii) the accretion of gas onto halos, which is modulated by the DM accretion rate; (iii) the shock heating and radiative cooling of gas inside DM halos, leading to the formation of galactic discs via conservation of specific angular momentum of the cooling gas; (iii) star formation in galaxy discs; (iv) stellar feedback from the evolving stellar populations; (v) chemical enrichment of stars and gas; (vi) the growth via gas accretion and merging of supermassive black holes; (vii) heating by active galactic nuclei (AGN); (viii) photoionization of the intergalactic medium; (ix) galaxy mergers driven by dynamical friction within common DM halos which can trigger bursts of SF and the formation and/or growth of spheroids; (x) collapse of globally unstable disks that also lead to SF bursts and the formation or growth of bulges. For this paper, we use a lightcone generated from the default model presented in Lagos et al. (2018) using the method described in Chauhan et al. (2019).\n")
            fil.write("#\n")
            fil.write("#Lagos et al. (2019) and Lagos et al. (2020) presented the FUV-to-FIR multi-wavelength predictions of Shark, by combining the model with the generative SED model ProSpect (Robotham et al. 2020), and using a novel method to compute dust attenuation parameters based on the radiative transfer analysis of the EAGLE hydrodynamical simulations of Trayford et al. (2020). The model was shown to produce excellent agreement with the observed number counts from the FUV to the FIR (Lagos et al. 2019), and the redshift distribution of FIR-selected sources (Lagos et al. 2020). Here, we use a lightcone of area 107.889deg^2 which includes all galaxies at 0<=z<=6 with an 850microns flux > 0.01mJy.\n")
            fil.write("#\n")
            fil.write("#Papers to reference:\n")
            fil.write("#Lagos et al. (2020) https://ui.adsabs.harvard.edu/abs/2020MNRAS.499.1948L/abstract\n")
            fil.write("#Lagos et al. (2019) https://ui.adsabs.harvard.edu/abs/2019MNRAS.489.4196L/abstract\n")
            fil.write("#Lagos et al. (2018) https://ui.adsabs.harvard.edu/abs/2018MNRAS.481.3573L/abstract\n"
            fil.write("#Trayford et al. (2020) https://ui.adsabs.harvard.edu/abs/2020MNRAS.491.3937T/abstract\n")
            fil.write("#Robotham et al. (2020) https://ui.adsabs.harvard.edu/abs/2020MNRAS.495..905R/abstract\n")
            fil.write("#Chauhan et al. (2019) https://ui.adsabs.harvard.edu/abs/2019MNRAS.488.5898C/abstract\n")
            fil.write("#\n")
            fil.write("#\n")
            fil.write("#\n")
            fil.write("#area 107.889 deg2\n")
            fil.write("#app_Z_VISTA <= 23 and redshift <= 3\n")
            fil.write("#units\n")
            fil.write("#mstar and mvir in [Msun]\n")
            fil.write("#sfr[Msun/yr]\n")
            fil.write("#re[arcsec]\n")
            fil.write("#magnitudes AB\n")
            fil.write("#type_galaxy: =0 for central, and >0 for satellites\n")
            fil.write("#id_group_sky = -1 if a galaxy is the only one in its host halo\n")
            fil.write("#\n")
            fil.write("#dec ra redshift log10(mstar) log10(sfr) re B/T app_u_VST app_g_VST app_r_VST app_i_VST app_Z_VISTA app_Y_VISTA app_J_VISTA app_H_VISTA app_K_VISTA id_group_sky log10(mvir) log10(mvir_z0) type_galaxy\n")
            for a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,r,s,t,u in zip(dec[ind], ra[ind], zobs[ind], mstartot[ind], sfrtot[ind], re[ind], BT[ind], SEDs_dustin[2,:], SEDs_dustin[3,:], SEDs_dustin[4,:], SEDs_dustin[5,:], SEDs_dustin[6,:], SEDs_dustin[7,:], SEDs_dustin[8,:], SEDs_dustin[9,:], SEDs_dustin[10,:], id_group[ind], mvir[ind], mvirz0[ind], typeg[ind]):
                fil.write("%5.10f %5.10f %5.10f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f  %5.2f %5.2f %5.2f %5.2f %5.2f %10.0f %5.2f %5.2f %5.2f\n" % (a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,r,s,t,u))

def main():

    lightcone_dir = '/scratch/pawsey0119/clagos/Stingray/output/medi-SURFS/Shark-TreeFixed-ReincPSO-kappa0p002/deep-optical-final/'
    outdir= '/sratch/pawsey0119/clagos/Stingray/output/medi-SURFS/Shark-TreeFixed-ReincPSO-kappa0p002/deep-optical-final/Plots/'
    obsdir= '/software/projects/pawsey0119/clagos/shark/data/'


    subvols = range(0,64)
    #[0,24]
    #[9,34,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63]
    #[1,2,3,4,5,6,7,8,10,11,12,13,14,15,16,17,18,19,20,21,22,23,25,26,27,28,29,30,31,32,33,34]

    sed_file = "Sting-SED-VST-eagle-rr14"

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
                           'rstar_disk_apparent','id_group_sky','dc', 'mvir_hosthalo', 'type')}
    fields_mvir = {'galaxies': ('mvir_z0','id_galaxy_sam','snapshot','subvolume')}

    hdf5_data = common.read_lightcone(lightcone_dir, 'split/', fields, subvols, "mock")

    hdf5_data_mvir = common.read_lightcone(lightcone_dir, 'split/', fields_mvir, subvols, "final_mvir")

    nbands = len(seds[0])
    prepare_data(seds, ids_sed, hdf5_data, hdf5_data_mvir, subvols, lightcone_dir, nbands)


if __name__ == '__main__':
    main()
