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

from astropy.io import ascii
import numpy as np
import common
import os

##################################
h0 = 0.6751

mlow = -3.0
mupp = 3.0
dm = 0.2
mbins = np.arange(mlow,mupp,dm)
xlf   = mbins + dm/2.0


def prepare_data(photo_data_radio, ids_sed, hdf5_data, subvols, lightcone_dir,  nbands):

    (dec, ra, zobs, idgal, sfrb, sfrd, mstarb, mstard, rsb, rsd) = hdf5_data
   
    SEDs_dust_radio   = photo_data_radio[0]

    mstartot = np.log10((mstarb+mstard)/h0)

    band_selec = 12
    bands_radio = (9, 10, 11, 12, 13, 14, 15)
    bands_ALMA = (2, 3, 4, 5, 6, 7, 8)

   
    ncounts = np.zeros(shape = (len(bands_ALMA), len(mbins)))
    ncounts_cum = np.zeros(shape = (len(bands_ALMA), len(mbins)))

    for i, b in enumerate(bands_ALMA):
        ind = np.where((SEDs_dust_radio[b,:] > 0) & (SEDs_dust_radio[b,:] < 40))
        m = np.log10(10.0**(SEDs_dust_radio[b,ind]/(-2.5))*3631.0*1e3) #in mJy
        H, bins_edges = np.histogram(m,bins=np.append(mbins,mupp))
        ncounts[i,:] = ncounts[i,:] + H



   #(0): "r_SDSS", "Band_ionising_photons", "Band9_ALMA", "Band8_ALMA",
   #(4): "Band7_ALMA", "Band6_ALMA", "Band5_ALMA", "Band4_ALMA", "Band3_ALMA",
   #(9): "BandX_VLA", "BandC_VLA", "BandS_VLA", "BandL_VLA", "Band_610MHz",
   #(14): "Band_325MHz", "Band_150MHz"

    ind = np.where((zobs <= 6) & (SEDs_dust_radio[band_selec,:] <= 31.40006562228223) & (SEDs_dust_radio[band_selec,:] > 0))
    zin = zobs[ind]
    rain = ra[ind]
    decin = dec[ind]
    smin = mstartot[ind]
    seds_rbands = SEDs_dust_radio[band_selec,ind]
    sedsin = SEDs_dust_radio[:,ind]
    seds_rbands = seds_rbands[0,:]
    sedsin = sedsin[:,0,:]
    print(seds_rbands.shape)

    print(len(mstartot), len(zin))

    data_to_write = np.zeros(shape = (len(zin), len(bands_radio)+5))
    for i in range(0, len(zin)):
        data_to_write[i,0] = decin[i]
        data_to_write[i,1] = rain[i]
        data_to_write[i,2] = zin[i]
        data_to_write[i,3] = smin[i]
        data_to_write[i,3] = seds_rbands[i]
        for j, b in enumerate(bands_radio):
            data_to_write[i,j+5] = sedsin[b,i]

    writeon = False
    if(writeon == True):
       ascii.write(data_to_write, '/group/pawsey0119/clagos/Stingray/output/medi-SURFS/Shark-TreeFixed-ReincPSO-kappa0p002/deep-optical-final/Shark-Lightcone-radio-only-ultradeep.csv', names = ['dec', 'ra', 'redshift', 'log10(mstar)', 'app_z_SDSS', 'app_BandX_VLA', 'app_BandC_VLA', 'app_BandS_VLA', 'app_BandL_VLA', 'app_Band_610MHz', 'app_Band_325MHz', 'app_Band_150MHz'], overwrite = True, format='csv', fast_writer=False)
     
    return (ncounts, ncounts_cum, bands_ALMA) 
def main():

    lightcone_dir = '/group/pawsey0119/clagos/Stingray/output/medi-SURFS/Shark-TreeFixed-ReincPSO-kappa0p002/deep-optical-final/'
    outdir= '/group/pawsey0119/clagos/Stingray/output/medi-SURFS/Shark-TreeFixed-ReincPSO-kappa0p002/deep-optical-final/Plots/'
    obsdir= '/home/clagos/shark/data/'

    subvols = [0,1,2,3,4,5] #range(64) 
   #[0] #8,9,13,14,15,16,18,20,22,25,30,33,34,38,39,40,46,47,48,52,55,61]
    #0,1,2,3,4,5,7,10,11,12,17,19,21,24,26,27,28,29,31,32,36,37,42,43,44,45,49,50,53,54,56,57,58,59,60,62)
    #(6,9,13,14,15,16,20,22,25,30,33,34,38,39,46,47,48,52,55,61)
    #range(64) #(0,1,2,3,4,5,6,7,8,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35)
    #(9,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63) #0,1,2,3,4,5,6,7,8,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35) #range(64)
    sed_file = "Sting-SED-eagle-rr14"
    sed_file_radio = "Sting-SED-eagle-rr14-radio-only-v0"

    # Loop over redshift and subvolumes
    plt = common.load_matplotlib()
    totarea = 107.8890011908422 #deg2
    areasub = totarea/64.0 * len(subvols)  #deg2
    print("effective survey area,", areasub)
    #fields_sed = {'SED/ab_dust': ('total', 'disk', 'bulge_t')}

    #ids_sed_ab, seds_ab = common.read_photometry_data_hdf5(lightcone_dir, fields_sed, subvols)

    fields_sed = {'SED/ap_dust': ('total', 'bulge_t')}

    #ids_sed, seds = common.read_photometry_data_hdf5(lightcone_dir, fields_sed, subvols, sed_file)
    ids_sed, seds_radio = common.read_photometry_data_hdf5(lightcone_dir, fields_sed, subvols, sed_file_radio)

    fields = {'galaxies': ('dec', 'ra', 'zobs',
                           'id_galaxy_sky','sfr_burst','sfr_disk','mstars_bulge','mstars_disk','rstar_bulge_apparent','rstar_disk_apparent')}

    hdf5_data = common.read_lightcone(lightcone_dir, fields, subvols, "mock")

    nbands = len(seds_radio[0])
    (ncounts, ncounts_cum, bands_ALMA) =  prepare_data(seds_radio, ids_sed, hdf5_data, subvols, lightcone_dir, nbands)

    for b in range(0,len(bands_ALMA)):
            for j in range(0,len(mbins)):
                ncounts_cum[b,j] = np.sum(ncounts[b,j:len(mbins)])
            print('#Cumulative number counts at band', b)
            print('#log10(S_low/mJy) N_total[deg^-2]')
            for m,a in zip(xlf-dm*0.5,ncounts_cum[b,:]):
                print(m,a/areasub)
            #print('#S_low/mJy N_total[deg^-2] N_bulge[deg^-2] N_disk[deg^-2]')
            #for m,a,c,d in zip(xlf3-dm3*0.5,ncounts_cum_nat[0,b,:],ncounts_cum_nat[1,b,:],ncounts_cum_nat[2,b,:]):
            #    print( m,a/areasub,c/areasub,d/areasub)

            print('#Differential number counts at band', b)
            print('#log10(S_mid/mJy) N_total[deg^-2 dex^-1]')
            for d,e in zip(xlf[:],ncounts[b,:]):
                print(d,e/areasub/dm)
            #print('#S_mid/mJy N_total[deg^-2 mJy^-1] N_bulge[deg^-2 mJy^-1] N_disk[deg^-2 mJy^-1]')
            #for d,e,f,g in zip(xlf3[:],ncounts_nat[0,b,:],ncounts_nat[1,b,:],ncounts_nat[2,b,:]):
            #    print(d,e/areasub/dm3,f/areasub/dm3,g/areasub/dm3)

if __name__ == '__main__':
    main()
