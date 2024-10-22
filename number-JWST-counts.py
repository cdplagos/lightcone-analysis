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

mlow = 8
mupp = 30
dm = 0.25
mbins = np.arange(mlow,mupp,dm)
xlf   = mbins + dm/2.0

def plot_numbercounts(plt, outdir, obsdir, ncounts):

    xlf_obs  = xlf
 
    xtit="$\\rm app\, mag (AB)$"
    ytit="$\\rm log_{10}(N/{\\rm mag}^{-1}/deg^2)$"

    xmin, xmax, ymin, ymax = 10, 30, -2 , 5.5
    xleg = xmin + 0.2 * (xmax-xmin)
    yleg = ymax - 0.1 * (ymax-ymin)

    fig = plt.figure(figsize=(6,7))

    labels = ("F070W", "F090W", "F115W", "F150W", "F200W", "F277W", "F356W", "F444W", "F560W", "F770W", "F1000W", "F1130W", "F1280W", "F1500W", "F1800W", "F2100W", "F2550W")
    colors = ('Indigo','purple','Navy','DarkTurquoise', 'Aquamarine', 'Green','PaleGreen','GreenYellow','Gold','Yellow','Orange','OrangeRed','red','DarkRed','FireBrick','Crimson','IndianRed','LightCoral','Maroon','brown','Sienna','SaddleBrown','Chocolate','Peru','DarkGoldenrod','Goldenrod','SandyBrown')

    ax = fig.add_subplot(111)
    common.prepare_ax(ax, xmin, xmax, ymin, ymax, xtit, ytit, locators=(2, 2, 1, 1))
    for idx, b in enumerate(labels):

        #Predicted LF
        ind = np.where(ncounts[:,idx] != 0)
        y = ncounts[ind,idx]
        ax.plot(xlf_obs[ind],y[0],'k', linewidth=3, label=b, color=colors[idx])
 
    common.prepare_legend(ax, colors, loc='lower right')
    common.savefig(outdir, fig, "number-counts-JWST.pdf")

def prepare_data(phot_data, subvols, lightcone_dir, ncounts, nbands):

    #components of apparent magnitudes:
    #(len(my_data), 2, 2, 5, nbands)
    #1: disk
    #0: total
    SEDs_dust   = phot_data[0]
    SEDs_dust_disk = phot_data[1]

    #print SEDs_dust(0)
    for i in range(0,nbands):
        #calculate number counts for total magnitude as well as no dust magnitudes
        ind = np.where((SEDs_dust[i,:] > 0) & (SEDs_dust[i,:] < 40))
        H, bins_edges = np.histogram(SEDs_dust[i,ind],bins=np.append(mbins,mupp))
        ncounts[:,i] = ncounts[:,i] + H

def main():

    lightcone_dir = '/scratch/pawsey0119/clagos/Stingray/output/medi-SURFS/Shark-TreeFixed-ReincPSO-kappa0p002/deep-optical-final/'
    outdir= '/scratch/pawsey0119/clagos/Stingray/output/medi-SURFS/Shark-TreeFixed-ReincPSO-kappa0p002/deep-optical-final/Plots/'
    obsdir= '/software/projects/pawsey0119/clagos/shark/data/'

    Variable_Ext = True
    sed_file = "Sting-SED-JWST-eagle-rr14"
    #subvols = range(64) #[0] #(0,1,2,3,4,5,6,7,8,9,10) #,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35)
    #0,1,2,3,4,5,6,7,8,9,10,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63) # #(0,10,11,12,13,14,15,16,17) #2,3,4) #range(64) 
    #subvols = [9,34,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63]
    subvols = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58] 
    #range(64) #1,2,3,4,5,6,7,8,10,11,12,13,14,15,16,17,18,19,20,21,22,23,25,26,27,28,29,30,31,32,33,34]

    # Loop over redshift and subvolumes
    plt = common.load_matplotlib()

    totarea =  107.8890011908422 #286 #10.0 #deg2 107.8890011908422 #deg2

    areasub = totarea/64.0 * len(subvols)  #deg2

    fields_sed = {'SED/ap_dust': ('total', 'disk')}

    ids_sed, seds = common.read_photometry_data_hdf5(lightcone_dir, fields_sed, subvols, sed_file)
 
    nbands = len(seds[0])
    ncounts = np.zeros(shape = (len(mbins), nbands))

    prepare_data(seds, subvols, lightcone_dir, ncounts, nbands)

    if(totarea > 0.):
        ncounts   = ncounts/areasub/dm

    # Take logs
    ind = np.where(ncounts > 1e-5)
    ncounts[ind] = np.log10(ncounts[ind])

    nJWST=17
    nbands_to_save = np.zeros(shape = (len(mbins), nJWST+1))
    nbands_to_save[:,0] = xlf
    nbands_to_save[:,1:nJWST+1] = ncounts[:,0:nJWST]
    np.savetxt("NumberCountsJWST_Lagos18.txt", nbands_to_save)

    plot_numbercounts(plt, outdir, obsdir, ncounts)

if __name__ == '__main__':
    main()
