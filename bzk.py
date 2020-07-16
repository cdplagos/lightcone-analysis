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

def plot_scos_bzk(plt, outdir, so_bzk, so_smg):

    x = [1,2,3,4,5,6,7,8,9,10]
    xtit="$\\rm J_{\\rm up}$"
    ytit="$\\rm S_{\\rm CO}/Jy km/s$"

    xmin, xmax, ymin, ymax = 0.5, 10.5, 0, 10
    xleg = xmin + 0.2 * (xmax-xmin)
    yleg = ymax - 0.1 * (ymax-ymin)

    fig = plt.figure(figsize=(5,4))
    ax = fig.add_subplot(111)
    common.prepare_ax(ax, xmin, xmax, ymin, ymax, xtit, ytit, locators=(1, 1, 0.5, 0.5))
    plt.subplots_adjust(left=0.2, bottom=0.15)

    print 'number of galaxies',len(so_bzk[:,0])
    for j in range(0,len(so_bzk[:,0])):
        scoplot = so_bzk[j,:]
        ax.plot(x, scoplot,linestyle='solid')

    common.savefig(outdir, fig, "SCO_BzKs.pdf")

    fig = plt.figure(figsize=(5,4))
    ax = fig.add_subplot(111)
    common.prepare_ax(ax, xmin, xmax, ymin, ymax, xtit, ytit, locators=(1, 1, 0.5, 0.5))
    plt.subplots_adjust(left=0.2, bottom=0.15)

    print 'number of galaxies',len(so_smg[:,0])
    for j in range(0,len(so_smg[:,0])):
        scoplot = so_smg[j,:]
        ax.plot(x, scoplot,linestyle='solid')

    common.savefig(outdir, fig, "SCO_SMGs.pdf")


def prepare_data(phot_data, ids_sed, hdf5_data, hdf5_co_data, subvols, lightcone_dir, ncounts, nbands, zdist, med_socled):

    def medians(y):
        result = np.zeros(shape = 3)
        result[0] = np.median(y)
        obj_bin = len(y)
        #sort array on 1/y because we want it to sort from the smallest to the largest item, and the default of argsort is to order from the largest to the smallest.
        IDs = np.argsort(y,kind='quicksort')
        ID16th = int(np.floor(obj_bin*0.16))+1   #take the lower edge.
        ID84th = int(np.floor(obj_bin*0.84))-1   #take the upper edge.
        result[1] = np.abs(result[0] - y[IDs[ID16th]])
        result[2] = np.abs(y[IDs[ID84th]] - result[0])
        return result 

    (dec, ra, zobs, idgal) = hdf5_data
    (SCO, SCO_peak) = hdf5_co_data
    #components of apparent magnitudes:
    #(len(my_data), 2, 2, 5, nbands)
    #0: disk instability bulge
    #1: galaxy merger bulge
    #2: total bulge
    #3: disk
    #4: total
    SEDs_dust   = phot_data[0]
    SEDs_dust_disk = phot_data[1]

    bandsg = 3
    bandsz = 6
    bandsk = 10
   
    print SCO.shape 
    colbzk = (SEDs_dust[bandsz,:]-SEDs_dust[bandsk,:]) - (SEDs_dust[bandsg,:]-SEDs_dust[bandsz,:])
    bzkselec = np.where((colbzk > -0.2) & (SEDs_dust[bandsk,:] < 20) & (SEDs_dust[bandsk,:] > 10))
    so_bzk = SCO[bzkselec]
    for i in range(10):
        meds = medians(so_bzk[:,i])
        med_socled[0,:,i] = meds

    #for a,b,c in zip( med_socled[0,0,:], med_socled[0,1,:],med_socled[0,2,:]):
    #    print a,b,c

    print 'number of selected galaxies',len(so_bzk[:,0]) 
    print 'median BzK redshift', np.median(zobs[bzkselec])

    ind = np.where((SEDs_dust[26,:] > -10) & (SEDs_dust[26,:] < 14.6525))
    so_smg = SCO[ind]

    print 'number of selected galaxies',len(zobs[ind])

    return (so_bzk,so_smg)

def main():

    lightcone_dir = '/group/pawsey0119/clagos/Stingray/output/medi-SURFS/Shark-TreeFixed-ReincPSO-kappa0p002/deep-optical-narrow/'
    outdir= '/group/pawsey0119/clagos/Stingray/output/medi-SURFS/Shark-TreeFixed-ReincPSO-kappa0p002/deep-optical-narrow/Plots/'
    #'/mnt/su3ctm/clagos/Stingray/output/medi-SURFS/Shark-Lagos18-final/deep-optical/'
    obsdir= '/home/clagos/shark/data/'

    Variable_Ext = True
    sed_file = "Sting-SED-eagle-rr14-steep"
    subvols = (0,1) #range(20) #(40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63) 
    #0,1,2,3,4,5,6,7,8,9,10,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63) # #(0,10,11,12,13,14,15,16,17) #2,3,4) #range(64) 

    # Loop over redshift and subvolumes
    plt = common.load_matplotlib()

    totarea =  10.0 #107.8890011908422 #286 #10.0 #deg2 107.8890011908422 #deg2

    areasub = totarea/64.0 * len(subvols)  #deg2

    fields_sed = {'SED/ap_dust': ('total', 'disk')}

    ids_sed, seds = common.read_photometry_data_hdf5(lightcone_dir, fields_sed, subvols, sed_file)
 
    fields = {'galaxies': ('dec', 'ra', 'zobs',
                           'id_galaxy_sky')}

    hdf5_data = common.read_lightcone(lightcone_dir, fields, subvols)

    fields = {'galaxies': ('SCO','SCO_peak')}
 
    hdf5_co_data = common.read_co_lightcone(lightcone_dir, fields, subvols)

    nbands = len(seds[0])
    ncounts = np.zeros(shape = (5, nbands, len(mbins)))
    zdist = np.zeros(shape = (len(zbins)))
    med_socled = np.zeros(shape = (2,3,10))
    (so_bzk,so_smg) = prepare_data(seds,  ids_sed, hdf5_data, hdf5_co_data, subvols, lightcone_dir, ncounts, nbands, zdist, med_socled)

    if(Variable_Ext):
       outdir = os.path.join(outdir, 'eagle-rr14-steep')

    #plot_numbercounts(plt, outdir, obsdir, ncounts)
  
    #plot_scos_bzk(plt, outdir, so_bzk, so_smg)

if __name__ == '__main__':
    main()
