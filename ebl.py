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

# Constants
GyrToYr = 1e9
Zsun = 0.0127
XH = 0.72
PI = 3.141592654
MpcToKpc = 1e3
c_light = 299792458.0 #m/s

# Mass function initialization

mlow = 0
mupp = 30
dm = 0.5
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

file = 'Shark_SED_bands.dat'
lambda_bands = np.loadtxt(file,usecols=[0],unpack=True)
freq_bands   = c_light / (lambda_bands * 1e-10) #in Hz
lambda_bands = np.log10(lambda_bands)

def plot_ebl(plt, outdir, obsdir, ebl, ebl_nodust):

    xlf_obs  = xlf
 
    xtit="$\\rm log_{10}(\\lambda/\\mu m)$"
    ytit="$\\rm log_{10}(\\nu I_{\\nu}/ nW m^{-2} sr^{-1})$"

    xmin, xmax, ymin, ymax = -1, 3.5, -2, 2
    xleg = xmin + 0.2 * (xmax-xmin)
    yleg = ymax - 0.1 * (ymax-ymin)

    fig = plt.figure(figsize=(7,5))

    ax = fig.add_subplot(111)
    common.prepare_ax(ax, xmin, xmax, ymin, ymax, xtit, ytit, locators=(1, 1, 20, 20))


    ax.plot(lambda_bands-4.0, np.log10(ebl[0]), 'kD', alpha=0.5, label='Total')
    ax.plot(lambda_bands-4.0, np.log10(ebl_nodust[0]), 'kd', alpha=0.5, label='Intrinsic')
    ax.plot(lambda_bands-4.0, np.log10(ebl[1]), 'bs', alpha=0.5, label='Disks')
    ax.plot(lambda_bands-4.0, np.log10(ebl[2]), 'ro', alpha=0.5, label='Bulges')

    common.prepare_legend(ax, ['k','k','b','r'], loc="lower left")
    common.savefig(outdir, fig, "ebl-deep-lightcone-optical.pdf")

def prepare_data(phot_data, phot_data_nod, ids_sed, hdf5_data, subvols, lightcone_dir, ebl, nbands, ebl_nodust):

    (dec, ra, zobs, idgal) = hdf5_data
   
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

    SEDs_nodust   = phot_data_nod[0]

    #print SEDs_dust(0)
    for i in range(0,nbands):
        #calculate number counts for total magnitude as well as no dust magnitudes
        ind = np.where((SEDs_dust[i,:] > 0) & (SEDs_dust[i,:] < 40))
        totmag = sum(sum(10.0**(SEDs_dust[i,ind]/(-2.5))*3631.9*1e-26)) #in W Hz^-1 m^-2
        ebl[0,i] = totmag*freq_bands[i]

        ind = np.where((SEDs_nodust[i,:] > 0) & (SEDs_nodust[i,:] < 40))
        totmag = sum(sum(10.0**(SEDs_nodust[i,ind]/(-2.5))*3631.9*1e-26)) #in W Hz^-1 m^-2
        ebl_nodust[0,i] = totmag*freq_bands[i]

        #contribution from disks and bulges
        ind = np.where((SEDs_dust_disk[i,:] > 0) & (SEDs_dust_disk[i,:] < 40))
        totmag = sum(sum(10.0**(SEDs_dust_disk[i,ind]/(-2.5))*3631.9*1e-26)) #in W Hz^-1 m^-2
        ebl[1,i] = totmag*freq_bands[i]

        ind = np.where((SEDs_dust_bulge[i,:] > 0) & (SEDs_dust_bulge[i,:] < 40))
        totmag = sum(sum(10.0**(SEDs_dust_bulge[i,ind]/(-2.5))*3631.9*1e-26)) #in W Hz^-1 m^-2
        ebl[2,i] = totmag*freq_bands[i]

        ind = np.where((SEDs_dust_bulge_d[i,:] > 0) & (SEDs_dust_bulge_d[i,:] < 40))
        totmag = sum(sum(10.0**(SEDs_dust_bulge_d[i,ind]/(-2.5))*3631.9*1e-26)) #in W Hz^-1 m^-2
        ebl[3,i] = totmag*freq_bands[i]

        ind = np.where((SEDs_dust_bulge_m[i,:] > 0) & (SEDs_dust_bulge_m[i,:] < 40))
        totmag = sum(sum(10.0**(SEDs_dust_bulge_m[i,ind]/(-2.5))*3631.9*1e-26)) #in W Hz^-1 m^-2
        ebl[4,i] = totmag*freq_bands[i]

def main():

    lightcone_dir = '/group/pawsey0119/clagos/Stingray/output/medi-SURFS/Shark-TreeFixed-ReincPSO-kappa0p002/deep-optical/'
    outdir= '/group/pawsey0119/clagos/Stingray/output/medi-SURFS/Shark-TreeFixed-ReincPSO-kappa0p002/deep-optical/Plots/'
    #'/mnt/su3ctm/clagos/Stingray/output/medi-SURFS/Shark-Lagos18-final/deep-optical/'
    obsdir= '/home/clagos/shark/data/'

    Variable_Ext = True
    sed_file = "Sting-SED-eagle-rr14-testmmbands"
    subvols = (0,1) #,2,3,4,5,6,7,8,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35)
    #0,1,2,3,4,5,6,7,8,9,10,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63) # #(0,10,11,12,13,14,15,16,17) #2,3,4) #range(64) 
    # Loop over redshift and subvolumes
    plt = common.load_matplotlib()

    totarea =  107.8890011908422 #286 #10.0 #deg2 107.8890011908422 #deg2

    areasub = totarea/64.0 * len(subvols)  #deg2

    fields_sed = {'SED/ap_nodust': ('total', 'disk', 'bulge_t')}

    ids_sed_ab, seds_nod = common.read_photometry_data_hdf5(lightcone_dir, fields_sed, subvols, sed_file)

    fields_sed = {'SED/ap_dust': ('total', 'disk', 'bulge_t',  'bulge_d', 'bulge_m')}

    ids_sed, seds = common.read_photometry_data_hdf5(lightcone_dir, fields_sed, subvols, sed_file)
 
    fields = {'galaxies': ('dec', 'ra', 'zobs',
                           'id_galaxy_sky')}

    hdf5_data = common.read_lightcone(lightcone_dir, fields, subvols)

    nbands = len(seds[0])
    ebl = np.zeros(shape = (5, nbands))
    ebl_nodust = np.zeros(shape = (5, nbands))

    prepare_data(seds, seds_nod, ids_sed, hdf5_data, subvols, lightcone_dir, ebl, nbands, ebl_nodust)

    if(totarea > 0.):
        areast = areasub/3282.8 #to convert to steroradian
        ebl   = ebl/areast
        ebl_nodust = ebl_nodust/areast

    ebl = ebl * 1e9 #in nW m^-2 sr^-1
    ebl_nodust = ebl_nodust * 1e9  #in nW m^-2 sr^-1

    if(Variable_Ext):
       outdir = os.path.join(outdir, 'eagle-rr14')

    plot_ebl(plt, outdir, obsdir, ebl, ebl_nodust)
    #plot_redshift(plt, outdir, obsdir, zdist)

if __name__ == '__main__':
    main()
