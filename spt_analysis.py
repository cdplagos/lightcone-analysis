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
import utilities_statistics as us
import os

##################################
flux_threshs = np.array([1e-3, 1e-2, 2.5e-2, 5e-2, 7.5e-2, 1e-1, 2.5e-1, 3.5e-1, 4.5e-1, 5.5e-1, 6.5e-1, 7.5e-1, 8.5e-1, 9.5e-1, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]) #in milli Jy
flux_threshs_log = np.log10(flux_threshs[:])

flux_threshs_compact = np.array([1e-2, 1e-1, 1.0])
flux_threshs_compactab = -2.5 * np.log10(flux_threshs_compact*1e-3 / 3631.0)

temp_screen = 22.0
temp_bc = 57.0 #49.0 #37.0 

mlowab = 0
muppab = 30
dmab = 0.5
mbinsab = np.arange(mlowab,muppab,dmab)
xlfab   = mbinsab + dmab/2.0

zlow = 0
zupp = 7.0
dz = 0.25
zbins = np.arange(zlow,zupp,dz)
xz   = zbins + dz/2.0

flow = 0
fupp = 50.0
df = 7.5
fbins = np.arange(flow,fupp,df)
xf   = fbins + df/2.0


# Constants
GyrToYr = 1e9
Zsun = 0.0127
XH = 0.72
PI = 3.141592654
MpcToKpc = 1e3
c_light = 299792458.0 #m/s
h0 = 0.6751
zsun = 0.0189
min_metallicity = zsun * 1e-3

# Function initialization
mlow = -3.0
mupp = 3.0
dm = 0.25
mbins = np.arange(mlow,mupp,dm)
xlf   = mbins + dm/2.0

mlow2 = -2
mupp2 = 3
dm2 = 0.15
mbins2 = np.arange(mlow2,mupp2,dm2)
xlf2   = mbins2 + dm2/2.0

zlow = 0.5
zupp = 6
dz = 1.0
zbins = np.arange(zlow,zupp,dz)
xz   = zbins + dz/2.0

dzobs = 0.5
zbinsobs = np.arange(zlow,zupp,dzobs)
xzobs   = zbinsobs + dzobs/2.0

mslow = 7.0
msupp = 13.0
dms   = 0.35
msbins = np.arange(mslow,msupp,dms)
xmsf   = msbins + dms/2.0

msfrlow = -3.0
msfrupp = 3.0
dmsfr   = 0.35
msfrbins = np.arange(msfrlow,msfrupp,dmsfr)
xmsfr   = msfrbins + dmsfr/2.0


def sample_spt_2020(seds, z):

    fluxes = 10.0**(seds[32,:]/(-2.5)) * 3631.0 * 1e3 #mJy
    sptsources = np.loadtxt('S2mm_SPT.dat')
    tot_gals = np.sum(sptsources[:,2])
    nbins = len(sptsources[:,0])
    Nsampling = 50

    zmeds = np.zeros(shape = (Nsampling, len(xz)))
    for s in range(0,Nsampling):
        g = 0
        z_gals =  np.zeros(shape = (int(tot_gals)))
        for i in range(0,nbins):
            ind = np.where((fluxes[:] > sptsources[i,0]) & (fluxes[:] < sptsources[i,1]))
            nselected = int(sptsources[i,2])
            if(nselected > len(fluxes[ind])):
               nselected = len(fluxes[ind])
            if(nselected > 0):
               fluxesin = fluxes[ind]
               zin = z[ind]
               ids = np.arange(len(fluxes[ind]))
               selected = np.random.choice(ids, size=int(sptsources[i,2]))
               for j in range(0,len(selected)):
                   z_gals[g+j] = zin[selected[j]]
               g = g + nselected
        ind = np.where(z_gals != 0)
        H, bins_edges = np.histogram(z_gals[ind],bins=np.append(zbins,zupp))
        zmeds[s,:] = H

    zmed_spt = np.zeros(shape = (3, len(xz)))
    for i in range(0,len(xz)):
        zmed_spt[:,i] = us.gpercentiles(zmeds[:,i])
        zmed_spt[0,i] = np.max(zmeds[:,i])

    print(zmed_spt[0,:], xz, np.sum(zmed_spt[0,:]))
    return (zmed_spt)


def plot_props_z_spt(plt, outdir, obsdir, zmed_spt, area):

    #plot evolution of SMGs in the UVJ plane
    xtit="$\\rm redshift$"
    ytit="$\\rm N/dz$"

    xmin, xmax, ymin, ymax = 0, 6, 0, max(zmed_spt[0,:])/dz + 1.0
    xleg = xmax - 0.18 * (xmax-xmin)
    yleg = ymin + 0.1 * (ymax-ymin)

    fig = plt.figure(figsize=(5,4))
    ax = fig.add_subplot(111)
    common.prepare_ax(ax, xmin, xmax, ymin, ymax, xtit, ytit, locators=(1, 1, 5, 5))
    fig.subplots_adjust(left=0.15, bottom=0.15)
    ind = np.where(zmed_spt[0,:] != 0)
    x = xz[ind]
    y = zmed_spt[0,ind] / dz
    yerr_dn = zmed_spt[1,ind]
    yerr_up = zmed_spt[2,ind]
       
    ax.fill_between(x,y[0]-yerr_dn[0], y[0]+yerr_up[0],facecolor='red', alpha=0.5,interpolate=True)
    ax.plot(x,y[0], linestyle='solid', color='red')
    for a,b,c,d in zip(x,y[0],yerr_dn[0],yerr_up[0]):
        print(a,b,c,d)
    namefig = "spt_redshift_distribution.pdf"
    common.savefig(outdir, fig, namefig)

def prepare_data(phot_data, seds_lir, hdf5_data):

    (dec, ra, zobs, idgal, msb, msd, mhalo, sfrb, sfrd, typeg, mgd, mgb, mmold, mmolb, zd, zb, dc, rgd, rgb) = hdf5_data

    print(max(zobs))
    LIR_total = seds_lir[0]
    print(LIR_total)
    ind = np.where((LIR_total > 3e12) & (zobs >= 4) & (zobs <= 8))
    print('number of galaxies with LIR>3e12Lsun at z>4', len(LIR_total[ind]))
    ind = np.where((LIR_total > 3e12) & (zobs >= 4.5) & (zobs <= 8))
    print('number of galaxies with LIR>3e12Lsun at z>4.5', len(LIR_total[ind]))
    ind = np.where((LIR_total > 3e12) & (zobs >= 5) & (zobs <= 8))
    print('number of galaxies with LIR>3e12Lsun at z>5', len(LIR_total[ind]))
    ind = np.where((LIR_total > 3e12) & (zobs >= 5.5) & (zobs <= 8))
    print('number of galaxies with LIR>3e12Lsun at z>5.5', len(LIR_total[ind]))
    ind = np.where((LIR_total > 3e12) & (zobs >= 6) & (zobs <= 8))
    print('number of galaxies with LIR>3e12Lsun at z>6', len(LIR_total[ind]))

    
    #components of apparent magnitudes:
    #(len(my_data), 2, 2, 5, nbands)
    #0: disk instability bulge
    #1: galaxy merger bulge
    #2: total bulge
    #3: disk
    #4: total
    SEDs_dust   = phot_data[0]
    #analysis spt 
    (zmed_spt) = sample_spt_2020(SEDs_dust, zobs)
    return (zmed_spt)

def main():

    lightcone_dir = '/group/pawsey0119/clagos/Stingray/output/medi-SURFS/Shark-TreeFixed-ReincPSO-kappa0p002/deep-optical/'
    outdir= '/group/pawsey0119/clagos/Stingray/output/medi-SURFS/Shark-TreeFixed-ReincPSO-kappa0p002/deep-optical/Plots/'
    #'/mnt/su3ctm/clagos/Stingray/output/medi-SURFS/Shark-Lagos18-final/deep-optical/'
    obsdir= '/home/clagos/shark/data/'


    Variable_Ext = True
    sed_file = "Sting-SED-eagle-rr14"

    subvols = range(64) #[0,1,2,3,4,5,6,7,8,9,10,60,61,62,63] #11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 60, 61, 62, 63]#,11,12,13,14,15,16,17,18,19,20,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63) #(0,1) #range(20) #(40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63) 
    #0,1,2,3,4,5,6,7,8,9,10,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63) # #(0,10,11,12,13,14,15,16,17) #2,3,4) #range(64) 

    # Loop over redshift and subvolumes
    plt = common.load_matplotlib()

    totarea =  107.8890011908422 #286 #10.0 #deg2 107.8890011908422 #deg2

    areasub = totarea/64.0 * len(subvols)  #deg2
    print ("Area of survey in deg2 %f" % areasub)
    #100, 250, 450, 850, band-7, band-6, band-5, band-4
    bands = (21, 23, 25, 26, 29, 30, 31, 32)


    fields_sed = {'SED/ap_dust': ('total', 'disk')}
    ids_sed, seds = common.read_photometry_data_hdf5(lightcone_dir, fields_sed, subvols, sed_file)

    fields_sed = {'SED/lir_dust': ('total', 'disk')}
    ids_sed, seds_lir = common.read_photometry_data_hdf5(lightcone_dir, fields_sed, subvols, sed_file)

    fields = {'galaxies': ('dec', 'ra', 'zobs',
                           'id_galaxy_sky', 'mstars_bulge', 'mstars_disk', 
                           'mvir_hosthalo', 'sfr_burst', 'sfr_disk', 'type',
                           'mgas_disk','msgas_bulge','mmol_disk','mmol_bulge',
                           'zgas_bulge','zgas_disk', 'dc', 'rgas_disk_intrinsic', 'rgas_bulge_intrinsic')}

    hdf5_data = common.read_lightcone(lightcone_dir, fields, subvols, "mocksky")
  
    nbands = len(seds[0])


   #(0): "FUV_GALEX", "NUV_GALEX", "u_SDSS", "g_SDSS", "r_SDSS", "i_SDSS",
   #(6): "z_SDSS", "Y_VISTA", "J_VISTA", "H_VISTA", "K_VISTA", "W1_WISE",
   #(12): "I1_Spitzer", "I2_Spitzer", "W2_WISE", "I3_Spitzer", "I4_Spitzer",
   #(17): "W3_WISE", "W4_WISE", "P70_Herschel", "P100_Herschel",
   #(21): "P160_Herschel", "S250_Herschel", "S350_Herschel", "S450_JCMT",
   #(25): "S500_Herschel", "S850_JCMT", "Band9_ALMA", "Band8_ALMA",
   #(29): "Band7_ALMA", "Band6_ALMA", "Band5_ALMA", "Band4_ALMA"

    #process data
    (zmed_spt) = prepare_data(seds, seds_lir, hdf5_data)


    if(Variable_Ext):
       outdir = os.path.join(outdir, 'eagle-rr14')

    plot_props_z_spt(plt, outdir, obsdir, zmed_spt, areasub)

if __name__ == '__main__':
    main()
