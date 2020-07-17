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

zlow = 0
zupp = 6
dz = 0.2
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


def sample_bothwell(seds, co, z, Lxray):

    fluxes = 10.0**(seds[29,:]/(-2.5)) * 3631.0 * 1e3 #mJy
    b13sources = np.loadtxt('S850mu_Bothwell13.dat')
    b13sources[:,2] = b13sources[:,2]*3
    tot_gals = np.sum(b13sources[:,2])
    nbins = len(b13sources[:,0])
    Nsampling = 50
    cosleds_survey =  np.zeros(shape = (Nsampling, 10, 3))
    co_gals = np.zeros(shape = (Nsampling, 10, int(tot_gals)))
    lx_gals = np.zeros(shape = (Nsampling, int(tot_gals)))

    zmeds = np.zeros(shape = (Nsampling))
    for s in range(0,Nsampling):
        g = 0
        z_gals =  np.zeros(shape = (int(tot_gals)))
        s850gals = np.zeros(shape = (int(tot_gals)))
        for i in range(0,nbins):
            ind = np.where((fluxes[:] > b13sources[i,0]) & (fluxes[:] < b13sources[i,1]))
            if(len(fluxes[ind]) >= int(b13sources[i,2])):
               fluxesin = fluxes[ind]
               zin = z[ind]
               lxin = Lxray[ind]
               ids = np.arange(len(fluxes[ind]))
               cosledsin = np.zeros(shape = (len(fluxes[ind]),10))
               for l in range(0,10):
                   cosledsin[:,l] = co[ind,l]
               selected = np.random.choice(ids, size=int(b13sources[i,2]))
               for j in range(0,len(selected)):
                   s850gals[g+j] = fluxesin[selected[j]]
                   z_gals[g+j] = zin[selected[j]]
                   lx_gals[s,g+j] = lxin[selected[j]]
                   for l in range(0,10):
                       co_gals[s,l,g+j] = cosledsin[selected[j],l]
               g = g + int(b13sources[i,2])

        for l in range(0,10):
            cosleds_survey[s,l,:] = us.gpercentiles(co_gals[s,l,:])

        zmeds[s] = np.median(z_gals)

    print(np.median(zmeds))

    cosleds =  np.zeros(shape = (3, 10))
    #ind = np.where(cosleds_survey[:,0,0] == np.max(cosleds_survey[:,0,0]))
    ind = np.random.randint(0,49)
    co_gals_selected = co_gals[ind, :, :]
    lx_gals_selected = lx_gals[ind, :] 
    print(zmeds[ind])

    for l in range(0,10):
        cosleds[0,l] = cosleds_survey[ind,l,0]
        cosleds[1,l] = cosleds_survey[ind,l,1]
        cosleds[2,l] = cosleds_survey[ind,l,2]

    print(co_gals_selected.shape)
    return (co_gals_selected, lx_gals_selected, s850gals, cosleds)


def plot_co_sleds_smgs(plt, outdir, obsdir, co_sleds_b13_gals, lx_b13_gals, co_sleds_b13, s850_b13):

    xj = np.array([1,2,3,4,5,6,7,8,9,10])
    rangeflux = [4,6,10,25]
    colors = ['Orange', 'SeaGreen', 'Purple']
    #plot evolution of SMGs in the UVJ plane
    xtit="$\\rm J_{\\rm upper}$"
    ytit="$\\rm I_{\\rm CO}/Jy\\, km\\, s^{-1}$"

    xmin, xmax, ymin, ymax = 0.5, 9, 0.05, 15
    xleg = xmax - 0.18 * (xmax-xmin)
    yleg = ymin + 0.1 * (ymax-ymin)

    fig = plt.figure(figsize=(5,4))
    ax = fig.add_subplot(111)
    common.prepare_ax(ax, xmin, xmax, ymin, ymax, xtit, ytit, locators=(1, 1, 1, 1))
    plt.yscale('log')
    xobs, yobs, yobserr, s850obs = np.loadtxt(obsdir + 'Gas/ICO_Bothwell13.dat', usecols=[0,1,2,3], unpack = True)
    Valentino20x = [2,5,7]
    Valentino20y = [0.3672050817314368,0.6037942140568486,0.5205529850302092]
    Valentino20yerrdn = [0.2048617824396437,0.12075884281136973,0.23661499319554968]
    Valentino20yerrup = [0.07344101634628725,0.24151768562273956,0.14196899591733003]
        

    fig.subplots_adjust(left=0.15, bottom=0.15)

    shifts = (np.random.rand(len(xobs))-0.5)*0.1
    labels=['[4,6]', '[6,10]', '>10']
    for c in range(0,len(colors)):
        galsin = np.where((s850obs > rangeflux[c]) & (s850obs < rangeflux[c+1]))
        if(len(s850obs[galsin]) > 0):
           col = colors[c]
           xin = xobs[galsin]+shifts[galsin]
           yin = yobs[galsin]
           yer = yobserr[galsin]
           ind=np.where(yer != 0)
           ax.errorbar(xin[ind],yin[ind],yerr=[yer[ind],yer[ind]], ls='None', color=col, marker='s',fill=None, alpha=0.5, label=labels[c])
           ind=np.where(yer == 0)
           ax.plot(xin[ind],yin[ind],'v', color=col)

    for c in range(0,len(colors)):
        galsin = np.where((s850_b13 > rangeflux[c]) & (s850_b13 < rangeflux[c+1]))
        if(len(s850_b13[galsin]) > 0):
           col = colors[c]
           co_sleds_b13_gals_in = co_sleds_b13_gals[:,galsin]
           for g in range(0,len(co_sleds_b13_gals_in[0,:])):
               ax.plot(xj,co_sleds_b13_gals_in[:,g], linestyle='solid', linewidth = 0.3, color=col)

    common.prepare_legend(ax, colors, loc='upper right')
    namefig = "example_co_sleds_850microns.pdf"
    common.savefig(outdir, fig, namefig)

    #plot evolution of SMGs in the UVJ plane
    fig = plt.figure(figsize=(5,4))
    ax = fig.add_subplot(111)
    common.prepare_ax(ax, xmin, xmax, ymin, ymax, xtit, ytit, locators=(1, 1, 1, 1))
    plt.yscale('log')
    xobs, yobs, yobserr, s850obs = np.loadtxt(obsdir + 'Gas/ICO_Bothwell13.dat', usecols=[0,1,2,3], unpack = True)
    fig.subplots_adjust(left=0.15, bottom=0.15)

    for g in range(0,len(s850_b13)):
        ax.plot(xj,co_sleds_b13_gals[:,g], linestyle='solid', linewidth = 0.3, color='Salmon')

    ind = np.where((co_sleds_b13_gals[3,:] > 0.3) & (co_sleds_b13_gals[3,:] < 5))
    co_selected = co_sleds_b13_gals[:,ind]
    lumratio = lx_b13_gals[ind]/co_sleds_b13_gals[3,ind]
    lxrayin = lx_b13_gals[ind]
    co_selected = co_selected[:,0,:]
    lumratio = lumratio[0,:]
    ind = np.where(lumratio == min(lumratio))
    print(lxrayin[ind])
    co_sleds_b13_gals_selec = co_selected[:,ind]
    for i in range(0,len(lumratio[ind])):
        ax.plot(xj,co_sleds_b13_gals_selec[:,0,i], linestyle='dashed', linewidth = 3, color='YellowGreen')
    ind = np.where(lumratio == max(lumratio))
    print(lxrayin[ind])
    co_sleds_b13_gals_selec = co_selected[:,ind]
    for i in range(0,len(lumratio[ind])):
        ax.plot(xj,co_sleds_b13_gals_selec[:,0,i], linestyle='dashed', linewidth = 3, color='MediumBlue')

    ax.plot(xj,co_sleds_b13[0,:], linestyle='solid', linewidth=2, color='k')
    ax.plot(xj,co_sleds_b13[0,:]-co_sleds_b13[1,:], linestyle='dotted', linewidth=3, color='k')
    ax.plot(xj,co_sleds_b13[2,:]+co_sleds_b13[0,:], linestyle='dotted', linewidth=3, color='k')

    shifts = (np.random.rand(len(xobs))-0.5)*0.1
    ind=np.where(yobserr != 0)
    ax.errorbar(xobs[ind]+shifts[ind],yobs[ind],yerr=[yobserr[ind],yobserr[ind]], ls='None', color='grey', marker='s',fill=None, alpha=0.5)
    ind=np.where(yobserr == 0)
    ax.plot(xobs[ind]+shifts[ind],yobs[ind],'v', color='blue')
    ax.errorbar(Valentino20x, Valentino20y, yerr=[Valentino20yerrdn, Valentino20yerrup], ls='None', color='indigo', marker='D',alpha=0.8)

    #common.prepare_legend(ax, 'k', loc='upper left')
    namefig = "example_co_sleds_850microns_simple.pdf"
    common.savefig(outdir, fig, namefig)

def prepare_data(phot_data, ids_sed, hdf5_data, hdf5_co_data, subvols, lightcone_dir, nbands, area):

    (dec, ra, zobs) = hdf5_data
    (SCO, SCO_peak, Lxray) = hdf5_co_data

    #components of apparent magnitudes:
    #(len(my_data), 2, 2, 5, nbands)
    #0: disk instability bulge
    #1: galaxy merger bulge
    #2: total bulge
    #3: disk
    #4: total
    SEDs_dust   = phot_data[0]

    #analyse sample of as2uds-like galaxies
    (co_sleds_b13_gals, lx_b13_gals, s850_b13, co_sleds_b13) = sample_bothwell(SEDs_dust, SCO, zobs, Lxray)

    return (co_sleds_b13_gals, lx_b13_gals, s850_b13, co_sleds_b13)

def main():

    lightcone_dir = '/group/pawsey0119/clagos/Stingray/output/medi-SURFS/Shark-TreeFixed-ReincPSO-kappa0p002/deep-optical/'
    outdir= '/group/pawsey0119/clagos/Stingray/output/medi-SURFS/Shark-TreeFixed-ReincPSO-kappa0p002/deep-optical/Plots/'
    #'/mnt/su3ctm/clagos/Stingray/output/medi-SURFS/Shark-Lagos18-final/deep-optical/'
    obsdir= '/home/clagos/shark/data/'


    Variable_Ext = True
    sed_file = "Sting-SED-eagle-rr14"

    subvols = [0,1,2,3,4,5,6,7,8,9,10] #range(64) #[0,1,2,3,4,5] #,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63]
#[0] #range(64) #[0, 1, 6, 7, 8, 9, 11, 12, 13, 15, 16, 21, 22, 23, 24, 25, 26, 28, 29, 30, 31, 32, 33, 34, 36, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 52, 54, 55, 56, 57, 59, 60, 61, 62]
    #[0,1]#,3,4,5,6,7,8,9,10] #range(64) #[0,1,2,3,4,5,6,7,8,9,10] #,11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 60, 61, 62, 63]#,11,12,13,14,15,16,17,18,19,20,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63) #(0,1) #range(20) #(40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63) 
    #0,1,2,3,4,5,6,7,8,9,10,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63) # #(0,10,11,12,13,14,15,16,17) #2,3,4) #range(64) 

    # Loop over redshift and subvolumes
    plt = common.load_matplotlib()

    totarea =  107.8890011908422 #286 #10.0 #deg2 107.8890011908422 #deg2

    areasub = totarea/64.0 * len(subvols)  #deg2
    print ("Area of survey in deg2 %f" % areasub)
    #100, 250, 450, 850, band-7, band-6, band-5, band-4
    bands = (21, 23, 25, 26, 29, 30, 31, 32)

    fields_sed = {'SED/ap_dust': ('total', 'disk', 'bulge_t',  'bulge_d', 'bulge_m')}
    ids_sed, seds = common.read_photometry_data_hdf5(lightcone_dir, fields_sed, subvols, sed_file)
 
    fields = {'galaxies': ('dec', 'ra', 'zobs')}

    hdf5_data = common.read_lightcone(lightcone_dir, fields, subvols, "mocksky")
  
    fields = {'galaxies': ('SCO','SCO_peak', 'Lum_AGN_HardXray')}
    hdf5_co_data = common.read_co_lightcone(lightcone_dir, fields, subvols)
    nbands = len(seds[0])

   #(0): "FUV_GALEX", "NUV_GALEX", "u_SDSS", "g_SDSS", "r_SDSS", "i_SDSS",
   #(6): "z_SDSS", "Y_VISTA", "J_VISTA", "H_VISTA", "K_VISTA", "W1_WISE",
   #(12): "I1_Spitzer", "I2_Spitzer", "W2_WISE", "I3_Spitzer", "I4_Spitzer",
   #(17): "W3_WISE", "W4_WISE", "P70_Herschel", "P100_Herschel",
   #(21): "P160_Herschel", "S250_Herschel", "S350_Herschel", "S450_JCMT",
   #(25): "S500_Herschel", "S850_JCMT", "Band9_ALMA", "Band8_ALMA",
   #(29): "Band7_ALMA", "Band6_ALMA", "Band5_ALMA", "Band4_ALMA"

    #process data
    (co_sleds_b13_gals, lx_b13_gals, s850_b13, co_sleds_b13) = prepare_data(seds, ids_sed, hdf5_data, hdf5_co_data, subvols, 
                                                                            lightcone_dir, nbands, areasub)

    if(Variable_Ext):
       outdir = os.path.join(outdir, 'eagle-rr14')

    plot_co_sleds_smgs(plt, outdir, obsdir, co_sleds_b13_gals, lx_b13_gals, co_sleds_b13, s850_b13)

if __name__ == '__main__':
    main()
