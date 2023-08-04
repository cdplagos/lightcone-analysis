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

zlow = 0
zupp = 6
dz = 0.1
zbins = np.arange(zlow,zupp,dz)
xz   = zbins + dz/2.0

btbins=[0, 0.1, 0.2, 0.4, 0.5, 0.8, 1.0]

def plot_bt_evo(plt, outdir, bt_evolution, z, rhostar, fit):

    fig = plt.figure(figsize=(5,4.5))
    ytit = "$\\rm log_{10} (\\rm rho_{\\star}/ M_{\\odot} cMpc^{-3})$"
    xtit = "redshift"
    xmin, xmax, ymin, ymax = 0, 5, 4, 10
    xleg = xmax - 0.2 * (xmax - xmin)
    yleg = ymax - 0.1 * (ymax - ymin)

    ax = fig.add_subplot(111)
    common.prepare_ax(ax, xmin, xmax, ymin, ymax, xtit, ytit, locators=(0.1, 1, 0.1, 1))

    #Predicted relation
    cols = ['DarkSlateBlue','LightSeaGreen', 'DarkSeaGreen', 'GreenYellow', 'Yellow', 'DarkOrange', 'Crimson', 'DarkRed']
    labels = ['$\\rm  B/T =0$', '$\\rm  0<B/T<0.1$', '$\\rm  0.1<B/T<0.2$', '$\\rm  0.2<B/T<0.4$', '$\\rm  0.4<B/T< 0.5$', '$\\rm  0.5<B/T<0.8$', '$\\rm  B/T>0.8$']

    ax.plot(z, rhostar, linestyle='solid', color='k')
    x = z
    ax.plot(z, (fit[3] + fit[2] * x + fit[1] * x**2.0 + fit[0]**2.0), linestyle='dashed', color='k')    

    for i in range(0,len(cols)-1):
        ax.plot(xz, np.log10(bt_evolution[i,:]), linestyle='solid',color=cols[i], label=labels[i])

    common.prepare_legend(ax, cols, loc=4)
    common.savefig(outdir, fig, 'BT_evolution.pdf')


def prepare_data(hdf5_data, hdf5_data_shark, subvols, lightcone_dir, bt_evolution):

    (dec, ra, zobs, mstarb, mstard) = hdf5_data
    print (mstarb, mstard)
    #(SCO, id_cos) = co_hdf5_data
    (h0, volh, z, mstartot_box) = hdf5_data_shark

    rhostar = np.log10(mstartot_box / volh * h0**2.0)
    ind = np.where( mstartot_box > 0)
    x = z[ind]
    fit = np.polyfit(x, rhostar[ind], 3)
    rhoresidual = rhostar[ind] / (fit[3] + fit[2] * x + fit[1] * x**2.0 + fit[0] * x**3.0)

    mstartot = ((mstarb+mstard)/h0)
    BT = mstarb / (mstarb+mstard)

    for i in range(0,len(xz)-1):
        ind_all = np.where((mstartot > 0) & (zobs >= zbins[i]) & (zobs < zbins[i+1]))
        print(xz[i],zbins[i],zbins[i+1])
        zmed = xz[i]
        BTin = BT[ind_all]
        mstarin = mstartot[ind_all]
        rho_z = fit[3] + fit[2] * zmed + fit[1] * zmed**2.0 + fit[0] * zmed**3.0
        mall = np.sum(mstarin)
        for b in range(0,len(btbins)):
            if b == 0:
               ind = np.where(BTin <= btbins[b])
            else:
               ind = np.where((BTin > btbins[b-1]) & (BTin <= btbins[b])) 
            bt_evolution[b,i] = np.sum(mstarin[ind])/mall * 10.0**rho_z
       
    return(fit, z,rhostar)

def main():

    lightcone_dir = '/group/pawsey0119/clagos/Stingray/output/medi-SURFS/Shark-TreeFixed-ReincPSO-kappa0p002/deep-optical/'
    modeldir = '/group/pawsey0119/clagos/SHARK_Out/medi-SURFS/Shark-TreeFixed-ReincPSO-kappa0p002/'
    outdir= '/group/pawsey0119/clagos/Stingray/output/medi-SURFS/Shark-TreeFixed-ReincPSO-kappa0p002/deep-optical/Plots/'
    obsdir= '/home/clagos/shark/data/'


    #initialize relevant matrices
    bt_evolution = np.zeros(shape = (len(btbins),len(zbins)))

    subvols = [0] #range(64) #(0,1,2,3,4,5,6,7,8,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35)
    #(9,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63) #0,1,2,3,4,5,6,7,8,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35) #range(64)
    sed_file = "Sting-SED-eagle-rr14"

    # Loop over redshift and subvolumes
    plt = common.load_matplotlib()
    totarea = 107.8890011908422 #deg2
    areasub = totarea/64.0 * len(subvols)  #deg2

    fields = {'galaxies': ('dec', 'ra', 'zobs',
                           'mstars_bulge','mstars_disk')}

    hdf5_data = common.read_lightcone(lightcone_dir, fields, subvols)

    fields_shark = {'global': ('redshifts', 'mstars')}
    # Read data from each subvolume at a time and add it up
    # rather than appending it all together
    for idx, subvol in enumerate(subvols):
        subvol_data = common.read_data(modeldir, 199, fields_shark, [subvol])
        if idx == 0:
            hdf5_data_shark        = subvol_data
        else:
            for subvol_datum, hdf5_datum in zip(subvol_data[3:], hdf5_data_shark[3:]):
                hdf5_datum += subvol_datum
                #select the most massive black hole from the last list item

    # Also make sure that the total volume takes into account the number of subvolumes read
    hdf5_data_shark[1] = hdf5_data_shark[1] * len(subvols)

    (fit, z, rhostar) = prepare_data(hdf5_data, hdf5_data_shark, subvols, lightcone_dir, bt_evolution)
    plot_bt_evo(plt, outdir, bt_evolution, z, rhostar, fit)


if __name__ == '__main__':
    main()
