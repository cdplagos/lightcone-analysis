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

def ionising_photons(m, wave):

    #m is a vector of AB absolute magnitudes in a band with central wavelength wave
    #wavelength input has to be in angstrom

    Q = np.zeros(shape = len(m))
    
    wave_m = wave * 1e-10 #wavelength in m
    wave_cm = wave_m  * 1e2 #wavelength in cm
    freq = c_speed / wave_m #Hz
    hc =  h * (c_speed * 1e2) #h*c in cgs

    ind = np.where(m != -999)
    lum = 10.0**((m[ind] + 48.6) / (-2.5)) * dfac10pc * freq * wave_cm #we want to convert from Luminosity to int(lambda*Lum_lambda*dlambda)
    Q[ind] = lum / hc #rate of ionising photons in s^-1.
  
    return Q

def freefree_lum(Q, nu):

    #Q is the rate of ionising photos in s^-1
    #nu is the frequency in GHz
    #output in erg/s/Hz

    T = 1e4 #temperature in K
    lum = Q/6.3e32 * (T/1e4)**0.45 * (nu)**(-0.1)

    return lum

def synchrotron_lum(SFR, nu):
   
    #SFR in Msun/yr
    #nu is the frequency in GHz
    #output in erg/s/Hz

    ENT = 1.44
    ESNR = 0.06 * ENT
    alpha = -0.8
  
    T = 1e4
    EM = 6e6 #pc * cm**-6
    tau = (T/1e4)**(-1.35) * (nu / 1.4)**(-2.1) * EM / 6e6

    comp1 = ESNR * (nu / 1.49)**(-0.5) + ENT * (nu / 1.49)**(alpha) * np.exp(-tau)
    nuSNCC = SFR * 0.011148
    lum = comp1 * 1e30 * nuSNCC

    return lum

def prepare_data(photo_data_radio, seds_nod, ids_sed, hdf5_data, subvols, lightcone_dir,  nbands):

    (dec, ra, zobs, idgal, sfrb, sfrd, mstarb, mstard) = hdf5_data
    sfr = (sfrd + sfrb)/h0/1e9

    SEDs_dust_radio   = photo_data_radio[0]
    total_mags_nod = seds_nod[1]

    ion_mag = total_mags_nod[1,:]
    q_ionis = ionising_photons(ion_mag, 912.0) #in s^-1
    obs_selection_freq = (8.4, 5.0, 3.0, 1.4, 0.61, 0.325, 0.15) #GHz

    #the range frequencies selected will depend on the redshift of the sources

    lum_radio = np.zeros(shape = (len(obs_selection_freq), len(q_ionis)))
    lum_ratio = np.zeros(shape = (len(obs_selection_freq), len(q_ionis)))
    flux_radio = np.zeros(shape = (len(obs_selection_freq), len(q_ionis)))

    dL = cosmo.comoving_distance(zobs) * (1.0 + zobs) #luminosity distance in Mpc
    dfacgals = 4 * PI * (dL * mpc_to_cm)**2.0 #in cm^2

    for i, nu in enumerate(obs_selection_freq):
        nu_gals = nu * (1.0 + zobs[:]) #frequency in the rest frame
        lum_radio[i,:] = freefree_lum(q_ionis[:], nu_gals[:]) + synchrotron_lum(sfr[:], nu_gals[:])
        lum_ratio[i,:] = freefree_lum(q_ionis[:], nu_gals[:]) / lum_radio[i,:]
        flux_radio[i,:] = lum_radio[i,:] / dfacgals[:] * 1e26 #in mJy

    mstartot = np.log10((mstarb+mstard)/h0)

    ind = np.where((zobs <= 6) & (flux_radio[3,:] > 1e-6))
    zin = zobs[ind]
    rain = ra[ind]
    decin = dec[ind]
    smin = mstartot[ind]
    sedsin = flux_radio[:,ind]
    sedsin = sedsin[:,0,:]

    data_to_write = np.zeros(shape = (len(zin), len(obs_selection_freq)+4))

    print(len(sfr), len(zin))
 
    for i in range(0, len(zin)):
        data_to_write[i,0] = decin[i]
        data_to_write[i,1] = rain[i]
        data_to_write[i,2] = zin[i]
        data_to_write[i,3] = smin[i]
        for j, b in enumerate(obs_selection_freq):
            data_to_write[i,j+4] = sedsin[j,i]


    writeon = False
    if(writeon == True):
       ascii.write(data_to_write, '/group/pawsey0119/clagos/Stingray/output/medi-SURFS/Shark-TreeFixed-ReincPSO-kappa0p002/deep-optical/Shark-Lightcone-radio-only-ultradeep-Bressan02.csv', names = ['dec', 'ra', 'redshift', 'log10(mstar)', 'flux_8p4GHz', 'flux_5GHz', 'flux_3GHz', 'flux_1p4GHz', 'flux_610MHz', 'flux_325MHz', 'flux_150MHz'], overwrite = True, format='csv', fast_writer=False)
    
def main():

    lightcone_dir = '/scratch/pawsey0119/clagos/Stingray/output/medi-SURFS/Shark-TreeFixed-ReincPSO-kappa0p002/deep-optical-final/'
    outdir= '/scratch/pawsey0119/clagos/Stingray/output/medi-SURFS/Shark-TreeFixed-ReincPSO-kappa0p002/deep-optical-final/Plots/'
    obsdir= '/software/projects/pawsey0119/clagos/shark/data/'

    subvols = [0] #range(64) 
    sed_file = "Sting-SED-eagle-rr14"
    sed_file_radio = "Sting-SED-eagle-rr14-radio-only"

    # Loop over redshift and subvolumes
    plt = common.load_matplotlib()
    totarea = 107.8890011908422 #deg2
    areasub = totarea/64.0 * len(subvols)  #deg2
    print("effective survey area,", areasub)
    #fields_sed = {'SED/ab_dust': ('total', 'disk', 'bulge_t')}

    #ids_sed_ab, seds_ab = common.read_photometry_data_hdf5(lightcone_dir, fields_sed, subvols)

    fields_sed = {'SED/ap_dust': ('total', 'bulge_t')}
    fields_sed_nod = {'SED/ab_nodust': ('total', 'bulge_t')}

    ids_sed, seds_radio = common.read_photometry_data_hdf5(lightcone_dir, fields_sed, subvols, sed_file_radio)
    ids_sed, seds_nod = common.read_photometry_data_hdf5(lightcone_dir, fields_sed_nod, subvols, sed_file_radio)

    fields = {'galaxies': ('dec', 'ra', 'zobs',
                           'id_galaxy_sky','sfr_burst','sfr_disk',
                           'mstars_bulge','mstars_disk')}

    hdf5_data = common.read_lightcone(lightcone_dir, fields, subvols, "mock")

    nbands = len(seds_radio[0])
    prepare_data(seds_radio, seds_nod, ids_sed, hdf5_data, subvols, lightcone_dir, nbands)


if __name__ == '__main__':
    main()
