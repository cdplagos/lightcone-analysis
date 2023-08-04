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
from astropy.cosmology import FlatLambdaCDM
import numpy as np
import common
import os
import h5py

##################################
h0 = 0.6751

h        = 6.6261e-27 #cm2 g s-1
PI       = 3.141592654
Lsunwatts = 3.846e26
c_speed = 299792458.0 #in m/s
c_speed_cm = c_speed * 1e2 #in cm/s
radio_model = "Bressan02"
mpc_to_cm = 3.086e24

cosmo = FlatLambdaCDM(H0=67.51, Om0=0.3121, Tcmb0=2.725)

d10pc = 3.086e+19 #10 parsecs in cm
dfac10pc = 4 * PI * d10pc**2

#define parameters associated to AGN luminosity
alpha_adaf = 0.1
delta = 0.0005
beta = 1 - alpha_adaf / 0.55
eta_edd = 4
mcrit_nu = 0.001 * (delta / 0.0005) * (1 - beta) / beta * alpha_adaf**2
spin = 0.1

A_adaf = 5e-4
A_td = A_adaf/100


def radio_luminosity_agn(mbh, macc):

    #input mbh has to be in Msun
    #input macc has to be in Msun/yr
    #output luminosity in erg/s
  
    Ljet_ADAF = np.zeros ( shape = len(mbh))
    Ljet_td = np.zeros ( shape = len(mbh))
    mdot_norm = np.zeros ( shape = len(mbh))
    Ledd = np.zeros ( shape = len(mbh))

    ind = np.where(mbh > 0)
    Ledd[ind] = 1.28e46 * (mbh[ind]/1e8) #in erg/s

    ind = np.where((mbh > 0) & (macc > 0))
    mdot_edd = Ledd[ind] / (0.1 * c_speed_cm**2) * 1.586606334841629e-26 #in Msun/yr
    mdot_norm[ind] = macc[ind] / mdot_edd

    Ljet_ADAF = 2e45 * (mbh/1e9) * (mdot_norm/0.01) * spin**2 #in erg/s
    Ljet_td = 2.5e43 * (mbh/1e9)**1.1 * (mdot_norm/0.01)**1.2 * spin**2 #in erg/s

    ind = np.where(mdot_norm > eta_edd)
    mdot_norm[ind] = eta_edd
 
    return (Ljet_ADAF, Ljet_td, mdot_norm)

def radio_luminosity_per_freq (Ljet_ADAF, Ljet_td, mdot_norm, mbh, nu):

    #here nu has to be rest-frame in GHz
    lum_1p4GHz_adaf = A_adaf * (mbh / 1e9 * mdot_norm / 0.01)**0.42 * Ljet_ADAF
    lum_1p4GHz_td = A_td * (mbh / 1e9)**0.32 * (mdot_norm / 0.01)**(-1.2) * Ljet_td

    freq_hz = nu * 1e9
    lum_nu = (lum_1p4GHz_adaf + lum_1p4GHz_td) * (nu / 1.4)**(-0.7) / freq_hz #in erg/s/Hz

    return (lum_nu, (lum_1p4GHz_adaf + lum_1p4GHz_td))


def prepare_data(hdf5_data, subvol, lightcone_dir):

    (dec, ra, zobs, idgal, sfrb, sfrd, mstarb, mstard, mbh, mbh_acc_hh, mbh_acc_sb, idgal_sam) = hdf5_data
    macc = (mbh_acc_hh + mbh_acc_sb)/h0/1e9
    mbh = mbh / h0

    (Ljet_ADAF, Ljet_td, mdot_norm) = radio_luminosity_agn(mbh, macc)

    obs_selection_freq = (8.4, 5.0, 3.0, 1.4, 0.61, 0.325, 0.15) #GHz

    #the range frequencies selected will depend on the redshift of the sources

    flux_radio_agn = np.zeros(shape = (len(obs_selection_freq), len(macc)))
    lum_radio_agn = np.zeros(shape = (len(obs_selection_freq), len(macc)))
    lum_radio_agn_total = np.zeros(shape = (len(macc)))

    dL = cosmo.comoving_distance(zobs) * (1.0 + zobs) #luminosity distance in Mpc
    dfacgals = 4 * PI * (dL * mpc_to_cm)**2.0 #in cm^2

    for i, nu in enumerate(obs_selection_freq):
        nu_gals = nu * (1.0 + zobs[:]) #frequency in the rest frame
        (lum_radio_agn[i,:], lum_radio_agn_total) = radio_luminosity_per_freq (Ljet_ADAF[:], Ljet_td[:], mdot_norm[:], mbh[:], nu_gals[:])
        flux_radio_agn[i,:] = lum_radio_agn[i,:] / dfacgals[:] * 1e26 #in mJy

    mstartot = np.log10((mstarb+mstard)/h0)

    ind = np.where((zobs <= 6) & (macc > 0) & (flux_radio_agn[3,:] > 1e-6))
    zin = zobs[ind]
    rain = ra[ind]
    decin = dec[ind]
    smin = mstartot[ind]
    sedsin_agn = flux_radio_agn[:,ind]

    sedsin_agn = sedsin_agn[:,0,:]

    data_to_write = np.zeros(shape = (len(zin), len(obs_selection_freq)+4))
 
    for i in range(0, len(zin)):
        data_to_write[i,0] = decin[i]
        data_to_write[i,1] = rain[i]
        data_to_write[i,2] = zin[i]
        data_to_write[i,3] = smin[i]
        for j, b in enumerate(obs_selection_freq):
            data_to_write[i,j+4] = sedsin_agn[j,i]

    print(len(macc), len(zin)) 
    writeon = True
    if(writeon == True):
       #ascii.write(data_to_write, '/group/pawsey0119/clagos/Stingray/output/medi-SURFS/Shark-TreeFixed-ReincPSO-kappa0p002/deep-optical-final/Shark-Lightcone-radio-only-ultradeep-AGN-only.csv', names = ['dec', 'ra', 'redshift', 'log10(mstar)', 'flux_8p4GHz', 'flux_5GHz', 'flux_3GHz', 'flux_1p4GHz', 'flux_610MHz', 'flux_325MHz', 'flux_150MHz'], overwrite = True, format='csv', fast_writer=False)

       # will write the hdf5 files with AGN luminosities
       # will only write galaxies with mstar>0 as those are the ones being written in SFH.hdf5
       ind = np.where( (mstard + mstarb) > 0)
       file_to_write = os.path.join(lightcone_dir, 'split', 'radio_agn_luminosities_%02d.hdf5' % subvol)
       print ('Will write AGN luminosities to %s' % file_to_write)
       hf = h5py.File(file_to_write, 'w')
   
       hf.create_dataset('galaxies/id_galaxy_sky', data=idgal[ind])
       hf.create_dataset('galaxies/id_galaxy_sam', data=idgal_sam[ind])
       hf.create_dataset('galaxies/lum_radio_agn_frequency', data=lum_radio_agn[:,ind])
       hf.create_dataset('galaxies/lum_radio_agn_total', data=lum_radio_agn_total[ind])
       hf.create_dataset('frequencies', data=obs_selection_freq)
       hf.close()
   
   
def main():

    lightcone_dir = '/scratch/pawsey0119/clagos/Stingray/output/medi-SURFS/Shark-TreeFixed-ReincPSO-kappa0p002/deep-optical-final/'
    outdir= '/scratch/pawsey0119/clagos/Stingray/output/medi-SURFS/Shark-TreeFixed-ReincPSO-kappa0p002/deep-optical-final/Plots/'
    obsdir= '/software/projects/pawsey0119/clagos/shark/data/'

    subvols = range(64) 

    # Loop over redshift and subvolumes
    plt = common.load_matplotlib()
    totarea = 107.8890011908422 #deg2
    areasub = totarea/64.0 * len(subvols)  #deg2
    print("effective survey area,", areasub)
    #fields_sed = {'SED/ab_dust': ('total', 'disk', 'bulge_t')}

    #ids_sed_ab, seds_ab = common.read_photometry_data_hdf5(lightcone_dir, fields_sed, subvols)

    fields = {'galaxies': ('dec', 'ra', 'zobs',
                           'id_galaxy_sky','sfr_burst','sfr_disk',
                           'mstars_bulge','mstars_disk','mbh',
                           'mbh_acc_hh', 'mbh_acc_sb', 'id_galaxy_sam')}

    name = 'mock'
    for subv in subvols:
        hdf5_data = common.read_lightcone(lightcone_dir, fields, [subv], name)
        prepare_data(hdf5_data, subv, lightcone_dir)


if __name__ == '__main__':
    main()
