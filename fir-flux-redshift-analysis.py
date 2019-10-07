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

# Constants
GyrToYr = 1e9
Zsun = 0.0127
XH = 0.72
PI = 3.141592654
MpcToKpc = 1e3
c_light = 299792458.0 #m/s

# Mass function initialization

mlow = -2.0
mupp = 3.0
dm = 0.25
mbins = np.arange(mlow,mupp,dm)
xlf   = mbins + dm/2.0

mlow2 = -2
mupp2 = 2
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

def plot_observations(ax):

    # Add in points from Bethermin (he sent me on 15 May 2019)
    ax.plot(9.0, 0.52, 'o', mfc='black',markersize=9) #Berta+11
    ax.plot(20.0, 0.97, 'o', mfc='purple',markersize=9) #Bethermin+12b
    ax.plot(5.0, 1.4, 'o', mfc='blue',markersize=9) #Geach+13
    ax.plot(13.0, 1.95, 'o', mfc='blue',markersize=9) #Casey+13
    ax.plot(4.1, 2.5, 'o', mfc='aqua',markersize=9) #Wardlow+11
    ax.plot(3.0, 2.2, 'o', mfc='aqua',markersize=9) #Chapman+05
    ax.plot(4.2, 3.1, 'x', mec='green',markersize=9,mew=3) #Smolcic+12
    ax.plot(1.0, 2.2, 'o', mfc='green',markersize=9) #Michalowski+12
    ax.plot(2.0, 2.6, 'o', mfc='green',markersize=9) #Yun+12
    ax.plot(0.25, 2.91, 'o', mfc='red',markersize=9) #Staguhn+14
    # Add on more recent points for observational constraints (interferometric only?)
    ax.plot(3.6, 2.7, 'x', mec='aqua',markersize=9,mew=3)  # da Cunha+15
    ax.plot(3.6, 2.3, 'x', mec='aqua',markersize=9,mew=3)  # Simpson+14
    ax.plot(8.0, 2.65, 'x', mec='aqua',markersize=9,mew=3) # Simpson+17
    ax.plot(2.25, 2.74, 'x', mec='aqua',markersize=9, mew=3)    # Cowie+18
    ax.plot(3.5, 2.48, 'x', mec='green',markersize=9,mew=3)  # Brisbin+17
    ax.plot(3.3, 3.1, 'x', mec='green',markersize=9,mew=3) # Miettinen15 as revised by Brisbin17
    ax.plot(25.0, 3.9, 's', mfc='none',mec='orange',markersize=9,mew=3) # Strandet+16


def plot_redshift(plt, outdir, obsdir, zdist_flux_cuts, zdist_flux_cuts_scatter):

    #thresholds
    ytit="$\\rm Median redshift$"
    xtit="$\\rm Flux\, density\, cut\, (mJy)$"

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.xscale('log')
    plt.ylim([0,7])
    plt.xlim([0.2,120]) # OTherwise the solid curves go funky at higher fluxes
    plt.xlabel('Flux density cut (mJy)',size=16)
    plt.ylabel('Median redshift',size=16)

    bands = (0, 1, 2, 4, 5, 6, 7)
    colors  = ('Black','purple','blue','aqua','green','orange','red')
    labels = ('100$\mu$m','250$\mu$m','450$\mu$m','850$\mu$m','1100$\mu$m','1400$\mu$m','2000$\mu$m')
    p = 0
    for j in bands:
        ind = np.where(zdist_flux_cuts[j,0,:] != 0)
        y = zdist_flux_cuts[j,0,ind]
        yerrdn  = zdist_flux_cuts[j,0,ind] - zdist_flux_cuts[j,1,ind]
        yerrup = zdist_flux_cuts[j,0,ind] + zdist_flux_cuts[j,1,ind]
        ax.fill_between(flux_threshs[ind],yerrdn[0],yerrup[0],facecolor=colors[p], alpha=0.2,interpolate=True)
        ax.plot(flux_threshs[ind],y[0],linestyle='solid',color=colors[p])
        p = p + 1

    plot_observations(ax)
    plt.text(16.0,6.5,'100$\mu$m',size=12,color='black')
    plt.text(16.0,6.25,'250$\mu$m',size=12,color='purple')
    plt.text(16.0,6.0,'450$\mu$m',size=12,color='blue')
    plt.text(16.0,5.75,'850$\mu$m',size=12,color='aqua')
    plt.text(40.0,6.5,'1100$\mu$m',size=12,color='green')
    plt.text(40.0,6.25,'1400$\mu$m',size=12,color='orange')
    plt.text(40.0,6.0,'2000$\mu$m',size=12,color='red')

    plt.text(0.22,6.7,'Lagos et al. 2019 model',size=12,color='black')
    ax.plot([0.25,0.5],[6.5,6.5],color='black',linewidth=2)
    plt.text(0.6,6.4,'Unlensed',size=12,color='black')

    plt.tight_layout()  # This makes sure labels dont go off page
    common.savefig(outdir, fig, "zdistribution-fluxcut-AllIRbands.pdf")
    
    #separating different bands to make it more readable
    #thresholds
    ytit="$\\rm Median redshift$"
    xtit="$\\rm Flux\, density\, cut\, (mJy)$"

    fig = plt.figure(figsize=(10,7))
    subplots = (231, 232, 233, 234, 235, 236)

    bands = (0, 1, 2, 4, 5, 7)
    colors  = ('Black','purple','blue','CadetBlue','green','red')
    labels = ('100$\mu$m','250$\mu$m','450$\mu$m','850$\mu$m','1100$\mu$m','2000$\mu$m')
    p = 0
    for j in bands:
        ax = fig.add_subplot(subplots[p])
        plt.xscale('log')
        plt.ylim([0,5])
        plt.xlim([1e-2,120]) # OTherwise the solid curves go funky at higher fluxes
        if(p == 0 or p == 3):
           plt.ylabel('Median redshift',size=12)
        if(p >= 3):
           plt.xlabel('Flux density cut (mJy)',size=12)
        
        ind = np.where(zdist_flux_cuts[j,0,:] != 0)
        y = zdist_flux_cuts[j,0,ind]
        yerrdn  = zdist_flux_cuts[j,0,ind] - zdist_flux_cuts[j,1,ind]
        yerrup = zdist_flux_cuts[j,0,ind] + zdist_flux_cuts[j,1,ind]
        ax.fill_between(flux_threshs[ind],yerrdn[0],yerrup[0],facecolor=colors[p], alpha=0.5,interpolate=True)
        yerrdn  = zdist_flux_cuts[j,0,ind] - zdist_flux_cuts_scatter[j,1,ind]
        yerrup = zdist_flux_cuts[j,0,ind] + zdist_flux_cuts_scatter[j,2,ind]
        ax.fill_between(flux_threshs[ind],yerrdn[0],yerrup[0],facecolor=colors[p], alpha=0.2,interpolate=True)
        ax.plot(flux_threshs[ind],y[0],linestyle='solid',color=colors[p],label=labels[p])

        if(p == 0):
           ax.plot(9.0, 0.52, 'o', mfc='black',markersize=9) #Berta+11
        elif (p == 1):
           ax.plot(20.0, 0.97, 'o', mfc='purple',markersize=9) #Bethermin+12b
        elif (p == 2):
            ax.plot(5.0, 1.4, 'o', mfc='blue',markersize=9) #Geach+13
            ax.plot(13.0, 1.95, 'o', mfc='blue',markersize=9) #Casey+13
        elif (p == 3):
            ax.plot(4.1, 2.5, 'o', mfc='CadetBlue',markersize=9) #Wardlow+11
            ax.plot(3.0, 2.2, 'o', mfc='CadetBlue',markersize=9) #Chapman+05
            ax.plot(3.6, 2.7, 'D', mec='CadetBlue',markersize=9,mew=2)  # da Cunha+15
            ax.plot(3.6, 2.3, 'D', mec='CadetBlue',markersize=9,mew=2)  # Simpson+14
            ax.plot(8.0, 2.65, 'D', mec='CadetBlue',markersize=9,mew=2) # Simpson+17
            ax.plot(2.25, 2.74, 'D', mec='CadetBlue',markersize=9, mew=2)    # Cowie+18
        elif (p == 4):
            ax.plot(4.2, 3.1, 'D', mec='green',markersize=9,mew=3) #Smolcic+12
            ax.plot(1.0, 2.2, 'o', mfc='green',markersize=9) #Michalowski+12
            ax.plot(2.0, 2.6, 'o', mfc='green',markersize=9) #Yun+12
            ax.plot(3.5, 2.48, 'D', mec='green',markersize=9,mew=3)  # Brisbin+17
            ax.plot(3.3, 3.1, 'D', mec='green',markersize=9,mew=3) # Miettinen15 as revised by Brisbin17
        elif (p == 5): 
            ax.plot(0.25, 2.91, 'o', mfc='red',markersize=9) #Staguhn+14

        common.prepare_legend(ax, 'k', loc='upper left')
        p = p + 1

  
    plt.tight_layout()  # This makes sure labels dont go off page
    common.savefig(outdir, fig, "zdistribution-fluxcut-AllIRbands-splitpanels.pdf")

def prepare_data(phot_data, ids_sed, hdf5_data, subvols, lightcone_dir, nbands, bands, zdist_flux_cuts, zdist_flux_cuts_scatter):

    (dec, ra, zobs, idgal) = hdf5_data
 
    bin_it = functools.partial(us.medians_cum_err, xbins=flux_threshs_log)
    bin_it_scatter = functools.partial(us.wmedians_cum, xbins=flux_threshs_log)

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
    SEDs_dust_bulge_d = phot_data[3]
    SEDs_dust_bulge_m = phot_data[4]

    #print SEDs_dust(0)
    indices = range(len(bands))
    for i, j in zip(bands, indices):
        #calculate number counts for total magnitude as well as no dust magnitudes
        ind = np.where((SEDs_dust[i,:] > 0) & (SEDs_dust[i,:] < 40))
        m = np.log10(10.0**(SEDs_dust[i,ind]/(-2.5))*3631.0*1e3) #in mJy
        zdist_flux_cuts[j] = bin_it(x=m[0,:],y=zobs[ind])
        zdist_flux_cuts_scatter[j] = bin_it_scatter(x=m[0,:],y=zobs[ind])

def main():

    lightcone_dir = '/group/pawsey0119/clagos/Stingray/output/medi-SURFS/Shark-TreeFixed-ReincPSO-kappa0p002/deep-optical/'
    outdir= '/group/pawsey0119/clagos/Stingray/output/medi-SURFS/Shark-TreeFixed-ReincPSO-kappa0p002/deep-optical/Plots/'
    #'/mnt/su3ctm/clagos/Stingray/output/medi-SURFS/Shark-Lagos18-final/deep-optical/'
    obsdir= '/home/clagos/shark/data/'

    Variable_Ext = True
    sed_file = "Sting-SED-eagle-rr14-testmmbands"
    subvols = (0,1) #range(20) #(40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63) 
    #0,1,2,3,4,5,6,7,8,9,10,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63) # #(0,10,11,12,13,14,15,16,17) #2,3,4) #range(64) 

    # Loop over redshift and subvolumes
    plt = common.load_matplotlib()

    totarea =  107.8890011908422 #286 #10.0 #deg2 107.8890011908422 #deg2

    areasub = totarea/64.0 * len(subvols)  #deg2

    #100, 250, 450, 850, band-7, band-6, band-5, band-4
    bands = (21, 23, 25, 26, 29, 30, 31, 32)

    fields_sed = {'SED/ap_dust': ('total', 'disk', 'bulge_t',  'bulge_d', 'bulge_m')}

    ids_sed, seds = common.read_photometry_data_hdf5(lightcone_dir, fields_sed, subvols, sed_file)
 
    fields = {'galaxies': ('dec', 'ra', 'zobs',
                           'id_galaxy_sky')}

    hdf5_data = common.read_lightcone(lightcone_dir, fields, subvols)

    nbands = len(seds[0])
    zdist_flux_cuts = np.zeros(shape = (len(bands), 2,len(flux_threshs)))
    zdist_flux_cuts_scatter = np.zeros(shape = (len(bands), 3,len(flux_threshs)))

    prepare_data(seds, ids_sed, hdf5_data, subvols, lightcone_dir, nbands, bands, zdist_flux_cuts, zdist_flux_cuts_scatter)

    if(Variable_Ext):
       outdir = os.path.join(outdir, 'eagle-rr14')

    plot_redshift(plt, outdir, obsdir, zdist_flux_cuts, zdist_flux_cuts_scatter)

if __name__ == '__main__':
    main()
