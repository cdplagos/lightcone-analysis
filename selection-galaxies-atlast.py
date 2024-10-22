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

zsun = 0.0189

def plot_s24_s250_correlation(plt, outdir, s24, s250):

    xtit="$\\rm log_{10}(S_{24}/mJy)$"
    ytit="$\\rm log_{10}(S_{250}/mJy)$"
   
    xmin, xmax, ymin, ymax = -4, 2, -4, 2
   
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111)
   
    common.prepare_ax(ax, xmin, xmax, ymin, ymax, xtit, ytit, locators=(1, 1, 1, 1))
   
    im = ax.hexbin(s24, s250, xscale='linear', yscale='linear', gridsize=(25,25), cmap='magma', mincnt=10)
   
    namefig = "s24_s250_correlation.pdf"
   
    common.savefig(outdir, fig, namefig)

def plot_lco_vel(plt, outdir, lco_vel_scaling, vcobright, lcobright, mvir_hostcobright):

    fig = plt.figure(figsize=(5,4.5))
    ytit = "$\\rm log_{10} (\\rm L_{\\rm CO(1-0)}/K\\, km\\, s^{-1}\\, pc^2)$"
    xtit = "$\\rm log_{10}(\\rm FWHM/\\rm km \\, s^{-1})$"
    xmin, xmax, ymin, ymax = 1.5, 3.5, 6, 12
    xleg = xmax - 0.2 * (xmax - xmin)
    yleg = ymax - 0.1 * (ymax - ymin)

    ax = fig.add_subplot(111)
    common.prepare_ax(ax, xmin, xmax, ymin, ymax, xtit, ytit, locators=(0.1, 1, 0.1, 1))
    ax.text(xleg, yleg, 'z=2')

    #Predicted relation
    ind = np.where(lco_vel_scaling[0,0,:] != 0)
    yplot = lco_vel_scaling[0, 0,ind]
    errdn = lco_vel_scaling[0, 1,ind]
    errup = lco_vel_scaling[0, 2,ind]

    xplot = xvf[ind]
    ax.errorbar(xplot,yplot[0],yerr=[errdn[0],errup[0]], ls='None', mfc='None', ecolor = 'b', mec='b',marker='o',label="all galaxies")
    print("relation for all galaxies")
    for a,b,c,d in zip(xplot,yplot[0],errdn[0],errup[0]):
        print (a,b,c,d)


    ind = np.where(lco_vel_scaling[1,0,:] != 0)
    yplot = lco_vel_scaling[1, 0,ind]
    errdn = lco_vel_scaling[1, 1,ind]
    errup = lco_vel_scaling[1, 2,ind]

    xplot = xvf[ind]
    ax.errorbar(xplot,yplot[0],yerr=[errdn[0],errup[0]], ls='None', mfc='None', ecolor = 'r', mec='r',marker='s',label="$M_{\\rm halo}> 10^{13}\\, M_{\\odot}$")
    print("relation for cluster galaxies")
    for a,b,c,d in zip(xplot,yplot[0],errdn[0],errup[0]):
        print (a,b,c,d)

    ind = np.where(mvir_hostcobright <= 13)
    ax.plot(vcobright[ind], lcobright[ind], 'bo', markersize=1, alpha=0.5)

    ind = np.where(mvir_hostcobright > 13)
    ax.plot(vcobright[ind], lcobright[ind], 'ro', markersize=3, alpha=0.5)

    print("individual galaxies")
    for a,b,c in zip(vcobright, lcobright, mvir_hostcobright):
        print(a,b,c)
    common.prepare_legend(ax, ['b','r'], loc=4)
    common.savefig(outdir, fig, 'lco_fwhm_z2.pdf')


def prepare_data(phot_data, ids_sed, seds_rad, hdf5_data, hdf5_co_data, hdf5_data_groups, hdf5_data_mvir, lir, lir_bc_cont, subvols, lightcone_dir,  nbands, lco_vel_scaling):

    bin_it = functools.partial(us.wmedians, xbins=xvf)

    (dec, ra, zobs, zcos, idgal, sfrb, sfrd, mstarb, mstard, rsb, rsd, id_group, dc, mvir_host, mmold, mmolb, idgal_sam, zb, zd) = hdf5_data
    (SCO, SCO_peak) = hdf5_co_data
    (id_group_sky, mvir, n_selec, n_all, ra_group, dec_group, zobs_group) = hdf5_data_groups
    (mvirz0, idgalm, snap, subv) = hdf5_data_mvir

    #(SCO, id_cos) = co_hdf5_data

    print(zd)
    vline = SCO[:,0] / SCO_peak[:,0] / boost_box_profile
    lir_tot = lir[0]
    lir_tot = lir_tot[0,:]

    lir_disk = lir[1]
    lir_bc_cont_total = lir_bc_cont[0]
    lir_bc_cont_total = lir_bc_cont_total[0,:]
    temp_bc = 50.0
    temp_diff = 23.0
    temp_eff = temp_bc * lir_bc_cont_total + temp_diff * (1 - lir_bc_cont_total)

    #components of apparent magnitudes:
    #(len(my_data), 2, 2, 5, nbands)
    #0: disk instability bulge
    #1: galaxy merger bulge
    #2: total bulge
    #3: disk
    #4: total
    SEDs_dust   = phot_data[0]
    SEDs_dust_bulge = phot_data[1]

    SEDs_dust_radio = seds_rad[0]
    SEDs_dust_radio_bulge = seds_rad[0]


    mstartot = np.log10((mstarb+mstard)/h0)
    sfrtot = np.log10((sfrb+sfrd)/h0/1e9)
    fracsb = sfrb / (sfrb+sfrd)

    re = (rsb*mstarb + mstard*rsd) / (mstarb+mstard)
    BT = mstarb / (mstarb+mstard)
    mmol_tot = (mmold + mmolb)/h0
    zgas_eff = (zd * mmold + zb * mmolb) / (mmold + mmolb) / zsun

    ind = np.where(mmol_tot <= 10)
    mmol_tot[ind] = 10

    ind = np.where((mmold + mmolb) <= 0)
    mmol_tot[ind] = 1
    zgas_eff[ind] = 1e-6
   #(0): "FUV_GALEX", "NUV_GALEX", "u_SDSS", "g_SDSS", "r_SDSS", "i_SDSS",
   #(6): "z_SDSS", "Y_VISTA", "J_VISTA", "H_VISTA", "K_VISTA", "W1_WISE",
   #(12): "I1_Spitzer", "I2_Spitzer", "W2_WISE", "I3_Spitzer", "I4_Spitzer",
   #(17): "W3_WISE", "W4_WISE", "P70_Herschel", "P100_Herschel",
   #(21): "P160_Herschel", "S250_Herschel", "S350_Herschel", "S450_JCMT",
   #(25): "S500_Herschel", "S850_JCMT", "Band9_ALMA", "Band8_ALMA",
   #(29): "Band7_ALMA", "Band6_ALMA", "Band5_ALMA", "Band4_ALMA"

    bands = (2, 3, 4, 5, 6, 10, 12, 13, 19, 20, 21, 22, 23, 24, 25, 26, 30, 31, 32)
    bands_radio = (1, 2)
    fluxSEDs = 10.0**(SEDs_dust/(-2.5)) * 3631.0 * 1e3 #in mJy
    fluxSEDs_radio = 10.0**(SEDs_dust_radio/(-2.5)) * 3631.0 * 1e3 #in mJy

    dgal = 4.0 * PI * pow((1.0+zobs) * dc/h0, 2.0)

    LCO = SCO[:, 0] * dgal * 3.25e7 / (115.2712)**2.0 / (4.0*PI)
    ind = np.where((zobs < 2.1) & (LCO > 1e6))
    lco_vel_scaling[0, :] = bin_it(x=np.log10(vline[ind]), y=np.log10(LCO[ind]))
    ind = np.where((zobs < 2.1) & (LCO > 1e6) & (mvir_host/h0 > 1e13))
    lco_vel_scaling[1, :] = bin_it(x=np.log10(vline[ind]), y=np.log10(LCO[ind]))

    ind = np.where((zobs >1.9) & (zobs < 2.1) & (LCO > 1e10))
    lcobright = np.log10(LCO[ind]) 
    vcobright = np.log10(vline[ind])
    mvir_hostcobright = np.log10(mvir_host[ind]/h0)

    #S850 microns selected to have a flux >0.01mJy
    ind = np.where((SEDs_dust[26,:] > 0) & (SEDs_dust[26,:] < 21.4))
    SEDs_dustin = SEDs_dust[:,ind]
    SEDs_dust_radio_in = SEDs_dust_radio[:,ind]
    fluxSEDsin = fluxSEDs[:,ind]
    fluxSEDs_radioin = fluxSEDs_radio[:,ind]
    s24 = np.log10(fluxSEDs_radioin[2,:])
    s250 = np.log10(fluxSEDsin[22,:])
    print("Correlation coeff 24 vs 250 microns", np.polyfit(s24[0,:], s250[0,:], 1), np.corrcoef(s24[0,:], y=s250[0,:]))
    s24 = s24[0,:]
    s250 = s250[0,:]

    SCOin = SCO[ind,:]
    SEDs_dustin = SEDs_dustin[:,0,:]
    fluxSEDsin = fluxSEDsin[:,0,:]
    fluxSEDs_radioin = fluxSEDs_radioin[:,0,:]
    SCOin = SCOin[0,:,:]
    SCOin = SCOin.reshape((10,len(dec[ind])))
    #select unique group ids and match them with the data in /groups/
    group_ids_unique = np.unique(id_group[ind])
    matching_groups = np.in1d(id_group_sky, group_ids_unique)
    mvirz0_in = np.log10(mvirz0[ind]/h0)
    mvir_host_in = np.log10(mvir_host[ind]/h0)
    lir_in = np.log10(lir_tot[ind])
    tdust_effin = temp_eff[ind]
    #select only those groups that host the galaxies selected
    ingroup = np.where(matching_groups)
    mvirin = mvir[ingroup]
    n_allin = n_all[ingroup]
    ra_groupin = ra_group[ingroup]
    dec_groupin = dec_group[ingroup]
    id_group_skyin = id_group_sky[ingroup]
    zobs_groupin = zobs_group[ingroup]

    writeon = True
    write_study = False
    if(writeon == True):
       with open('/scratch/pawsey0119/clagos/Stingray/output/medi-SURFS/Shark-TreeFixed-ReincPSO-kappa0p002/deep-optical-final/Shark-deep-opticalLightcone-AtLAST-CO_p3.txt', 'w') as fil:
            fil.write("#Galaxies from Shark (Lagos et al. 2018) in the optical-deep lightcone\n")
            fil.write("#SED modelling as described in Lagos et al. (2019).\n")
            fil.write("#area 107.889 deg2\n")
            fil.write("#S_850microns > 0.01 milli Jy\n")
            fil.write("#units\n")
            fil.write("#CO integrated line fluxes in Jy km/s\n")
            fil.write("#FWHM in km/s\n")
            fil.write("#\n")
            fil.write("#SCO(1-0) SCO(2-1) SCO(3-2) SCO(4-3) SCO(5-4) SCO(6-5) SCO(7-6) SCO(8-7) SCO(9-8) SCO(10-9) FWHM\n")
            for a,b,c,d,e,f,g,h,i,j,k in zip(SCOin[0,:], SCOin[1,:], SCOin[2,:], SCOin[3,:], SCOin[4,:], SCOin[5,:], SCOin[6,:], SCOin[7,:], SCOin[8,:], SCOin[9,:], vline[ind]):
                fil.write("%5.7f %5.7f %5.7f %5.7f %5.7f %5.7f %5.7f %5.7f %5.7f %5.7f %5.5f\n" % (a,b,c,d,e,f,g,h,i,j,k))
   
       with open('/scratch/pawsey0119/clagos/Stingray/output/medi-SURFS/Shark-TreeFixed-ReincPSO-kappa0p002/deep-optical-final/Shark-deep-opticalLightcone-AtLAST-Groups_p3.txt', 'w') as fil:
            fil.write("#Galaxies from Shark (Lagos et al. 2018) in the optical-deep lightcone\n")
            fil.write("#SED modelling as described in Lagos et al. (2019).\n")
            fil.write("#area 107.889 deg2\n")
            fil.write("#S_850microns > 0.01 milli Jy\n")
            fil.write("#units\n")
            fil.write("#mvir in [Msun]\n")
            fil.write("#\n")
            fil.write("#id_group_galaxy mvir ra_group dec_group num_all_satellites\n")
            for a,b,c,d,e,f in zip(id_group_skyin, mvirin, ra_groupin, dec_groupin, n_allin, zobs_groupin):
                fil.write("%10.0f %5.5f %5.10f %5.10f %5.0f %5.5f\n" % (a,b,c,d,e,f))
   

       if(write_study == True):
           with open('/scratch/pawsey0119/clagos/Stingray/output/medi-SURFS/Shark-TreeFixed-ReincPSO-kappa0p002/deep-optical-final/Shark-deep-opticalLightcone-dust_cont_study.txt', 'w') as fil:
                fil.write("#Galaxies from Shark (Lagos et al. 2018) in the optical-deep lightcone\n")
                fil.write("#SED modelling as described in Lagos et al. (2019).\n")
                fil.write("#area 107.889 deg2\n")
                fil.write("#S_850microns > 0.01 milli Jy\n")
                fil.write("#units\n")
                fil.write("#mstar[Msun]\n")
                fil.write("#sfr[Msun/yr]\n")
                fil.write("#re[arcsec]\n")
                fil.write("#magnitudes AB\n")
                fil.write("#fluxes in units of mJy\n")
                fil.write("#dec ra redshift log10(mstar) log10(sfr) flux_S850_JCMT flux_Band7_ALMA flux_Band6_ALMA flux_Band5_ALMA flux_Band4_ALMA SCO(1-0) SCO(2-1) SCO(3-2) SCO(4-3) SCO(5-4) SCO(6-5) log10(Mmol) log10(LIR[Lsun]) Tdust_eff[K]\n")
                for a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,r,s,t in zip(dec[ind], ra[ind], zobs[ind], mstartot[ind], sfrtot[ind], fluxSEDsin[26,:], fluxSEDsin[29,:], fluxSEDsin[30,:], fluxSEDsin[31,:], fluxSEDsin[32,:], SCOin[0,:], SCOin[1,:], SCOin[2,:], SCOin[3,:], SCOin[4,:], SCOin[5,:], np.log10(mmol_tot[ind]), lir_in[:], tdust_effin[:]):
                       fil.write("%5.10f %5.7f %5.7f %5.7f %5.7f %5.7f %5.7f %5.7f %5.7f %5.7f %5.7f %5.7f %5.7f %5.7f %5.7f %5.7f %5.7f %5.7f %5.7f\n" % (a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,r,s,t))

   

       with open('/scratch/pawsey0119/clagos/Stingray/output/medi-SURFS/Shark-TreeFixed-ReincPSO-kappa0p002/deep-optical-final/Shark-deep-opticalLightcone-AtLAST_p3.txt', 'w') as fil:
               fil.write("#Galaxies from Shark (Lagos et al. 2018) in the optical-deep lightcone\n")
               fil.write("#SED modelling as described in Lagos et al. (2019).\n")
               fil.write("#area 107.889 deg2\n")
               fil.write("#S_850microns > 0.01 milli Jy\n")
               fil.write("#units\n")
               fil.write("#mstar[Msun]\n")
               fil.write("#sfr[Msun/yr]\n")
               fil.write("#re[arcsec]\n")
               fil.write("#zgas[Zsun]\n")
               fil.write("#magnitudes AB (1500Angs and 2400Angs correspond to GALEX FUV and NUV, while ugriz are SDSS bands)\n")
               fil.write("#\n")
               fil.write("#dec ra redshift[obs] redshift[cos] log10(mstar) log10(sfr) re B/T app_1500Angs app_2400Angs app_u app_g app_r app_i app_z app_VISTAY app_VISTAJ app_VISTAH app_VISTAK app_Spitzer1 app_Spitzer2 app_Spitzer3 app_Spitzer4 id_group_sky log10(mvir_current) log10(mvir_z0) log10(mmol) frac_sfr_starburst log10(zgas)\n")
               for a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,r,s,t,u,v,w,x,a1,a2,a3,a4,a5,a6 in zip(dec[ind], ra[ind], zobs[ind], zcos[ind], mstartot[ind], sfrtot[ind], re[ind], BT[ind], SEDs_dustin[0,:], SEDs_dustin[1,:], SEDs_dustin[2,:], SEDs_dustin[3,:], SEDs_dustin[4,:], SEDs_dustin[5,:], SEDs_dustin[6,:], SEDs_dustin[7,:], SEDs_dustin[8,:], SEDs_dustin[9,:], SEDs_dustin[10,:], SEDs_dustin[12,:], SEDs_dustin[13,:], SEDs_dustin[15,:], SEDs_dustin[16,:], id_group[ind], mvir_host_in[:], mvirz0_in[:], np.log10(mmol_tot[ind]), fracsb[ind], np.log10(zgas_eff[ind])):
                   fil.write("%5.10f %5.10f %5.7f %5.7f %5.7f %5.5f %5.5f %5.5f %5.5f %5.5f %5.5f %5.5f %5.5f %5.5f %5.5f %5.5f %5.5f %5.5f %5.5f %5.5f %5.5f %5.5f %5.5f %10.0f %5.5f %5.5f %5.5f %5.5f %5.5f\n" % (a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,r,s,t,u,v,w,x,a1,a2,a3,a4,a5,a6))
      
       with open('/scratch/pawsey0119/clagos/Stingray/output/medi-SURFS/Shark-TreeFixed-ReincPSO-kappa0p002/deep-optical-final/Shark-deep-opticalLightcone-AtLAST-FIR_p3.txt', 'w') as fil:
               fil.write("#Galaxies from Shark (Lagos et al. 2018) in the optical-deep lightcone\n")
               fil.write("#SED modelling as described in Lagos et al. (2019).\n")
               fil.write("#area 107.889 deg2\n")
               fil.write("#S_850microns > 0.01 milli Jy\n")
               fil.write("#units\n")
               fil.write("#fluxes in units of mJy\n")
               fil.write("#\n")
               fil.write("#flux_24mu_Spitzer flux_P70_Herschel flux_P100_Herschel flux_P160_Herschel flux_S250_Herschel flux_S350_Herschel flux_S450_JCMT flux_S500_Herschel flux_S850_JCMT flux_Band7_ALMA flux_Band6_ALMA flux_Band5_ALMA flux_Band4_ALMA\n")
               for a,b,c,d,e,f,g,h,i,j,k,l,o in zip(fluxSEDs_radioin[2,:], fluxSEDsin[19,:], fluxSEDsin[20,:], fluxSEDsin[21,:], fluxSEDsin[22,:], fluxSEDsin[23,:], fluxSEDsin[24,:], fluxSEDsin[25,:], fluxSEDsin[26,:], fluxSEDsin[29,:], fluxSEDsin[30,:], fluxSEDsin[31,:], fluxSEDsin[32,:]):
                   fil.write("%5.7f %5.7f %5.7f %5.7f %5.7f %5.7f %5.7f %5.7f %5.7f %5.7f %5.7f %5.7f %5.7f \n" % (a,b,c,d,e,f,g,h,i,j,k,l,o))
   

    return (vcobright, lcobright, mvir_hostcobright, s24, s250)

def main():

    lightcone_dir = '/scratch/pawsey0119/clagos/Stingray/output/medi-SURFS/Shark-TreeFixed-ReincPSO-kappa0p002/deep-optical-final/'
    outdir= '/scratch/pawsey0119/clagos/Stingray/output/medi-SURFS/Shark-TreeFixed-ReincPSO-kappa0p002/deep-optical-final/Plots/'
    obsdir= '/software/projects/pawsey0119/clagos/shark/data/'

    #initialize relevant matrices
    lco_vel_scaling = np.zeros(shape = (2, 4, len(vbins)))

    subvols = [1,2,3,4,5,6,7,8,10,11,12,13,14,15,16,17,18,19,20,21,22,23,25,26,27,28,29,30,31,32,33,34] 
    #[9,34,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63] #p2
    #[1,2,3,4,5,6,7,8,10,11,12,13,14,15,16,17,18,19,20,21,22,23,25,26,27,28,29,30,31,32,33,34] #p3
    #
    #[0,24] #p1

    sed_file = "Sting-SED-eagle-rr14"
    sed_file_radio = "Sting-SED-eagle-rr14-radio-only-v0"

    # Loop over redshift and subvolumes
    plt = common.load_matplotlib()
    totarea = 107.8890011908422 #deg2
    areasub = totarea/64.0 * len(subvols)  #deg2

    #fields_sed = {'SED/ab_dust': ('total', 'disk', 'bulge_t')}

    #ids_sed_ab, seds_ab = common.read_photometry_data_hdf5(lightcone_dir, fields_sed, subvols)

    fields_sed = {'SED/ap_dust': ('total', 'bulge_t')}
    fields_sed_lir = {'SED/lir_dust': ('total', 'disk')}
    fields_sed_lir_cont = {'SED/lir_dust_contribution_bc': ('total', 'disk')}


    ids_sed, seds = common.read_photometry_data_hdf5(lightcone_dir, fields_sed, subvols, sed_file)
    ids_sed_lir, lir = common.read_photometry_data_hdf5(lightcone_dir, fields_sed_lir, subvols, sed_file)
    ids_sed_lir, lir_bc_cont = common.read_photometry_data_hdf5(lightcone_dir, fields_sed_lir_cont, subvols, sed_file)
    ids_sed_rad, seds_rad = common.read_photometry_data_hdf5(lightcone_dir, fields_sed, subvols, sed_file_radio)

    fields = {'galaxies': ('dec', 'ra', 'zobs', 'zcos',
                           'id_galaxy_sky','sfr_burst','sfr_disk','mstars_bulge','mstars_disk','rstar_bulge_apparent',
                           'rstar_disk_apparent','id_group_sky','dc', 'mvir_hosthalo', 'mmol_disk', 'mmol_bulge','id_galaxy_sam',
                           'zgas_bulge','zgas_disk')}

    hdf5_data = common.read_lightcone(lightcone_dir, 'split/', fields, subvols, "mock")

    fields = {'groups': ('id_group_sky', 'mvir', 'n_galaxies_selected',
                         'n_galaxies_total','ra','dec','zobs')}

    hdf5_data_groups = common.read_lightcone(lightcone_dir, 'split/', fields, subvols, "mock")

    fields = {'galaxies': ('SCO','SCO_peak')}
    hdf5_co_data = common.read_co_lightcone(lightcone_dir, fields, subvols)

    fields_mvir = {'galaxies': ('mvir_z0','id_galaxy_sam','snapshot','subvolume')}
    hdf5_data_mvir = common.read_lightcone(lightcone_dir, 'split/', fields_mvir, subvols, "final_mvir")

    nbands = len(seds[0])
    (vcobright, lcobright, mvir_hostcobright, s24, s250) = prepare_data(seds, ids_sed, seds_rad, hdf5_data, hdf5_co_data, hdf5_data_groups, hdf5_data_mvir, lir, lir_bc_cont, subvols, lightcone_dir, nbands, lco_vel_scaling)
    #plot_lco_vel(plt, outdir, lco_vel_scaling, vcobright, lcobright, mvir_hostcobright)
    #plot_s24_s250_correlation(plt, outdir, s24, s250)

if __name__ == '__main__':
    main()
