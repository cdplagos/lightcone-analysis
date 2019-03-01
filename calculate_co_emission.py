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
from scipy import interpolate
import common
import os
import math 
import h5py

##################################

# Constants
GyrToYr = 1e9
Zsun = 0.0127
XH = 0.72
PI = 3.141592654
MpcToKpc = 1e3
c_light = 299792458.0 #m/s
Sigma0=1e-10 #yr^{-1}*zsun
Fx0=1e-2 #erg/s/cm^2


gammaSFR = 1.0
alphaFx  = 1.0 
Av = 4

h0 = 0.677
zsun = 0.0189

def prepare_data(hdf5_data, subvol, lightcone_dir):

    #read ascii table from Bayet et al. (2011)
    datanu = np.genfromtxt('emlines_carbonmonoxide_restnu.data', delimiter=' ', skip_header=11)
    nuco = datanu

    dataPDR  = np.genfromtxt('emlines_carbonmonoxide_Bayet11.data', delimiter=' ', skip_header=10)

    CRs = dataPDR[:,5]/5E-17 #normalize cosmic rays rate by 5e-17 s^{-1}.
    Xconv_Av3 = dataPDR[:,6:16]/1e20 #normalize conversion factors by 10^20 cm^(-2) (K km/s)^(-1)
    Xconv_Av8 = dataPDR[:,16:26]/1e20 #normalize conversion factors by 10^20 cm^(-2) (K km/s)^(-1)

    #Convert all the relevant quantities to logarithm as the 3D interpolation is done in the logarithmic space.
    Xconv_Av3 = np.log10(Xconv_Av3)
    Xconv_Av8 = np.log10(Xconv_Av8)
 
    Zmod = np.log10(dataPDR[:,4])
    GUV  = np.log10(dataPDR[:,0])
    CRs  = np.log10(CRs)

    nh   = dataPDR[:,3]
    ind = np.where(nh == 1e4)
    interpolator = interpolate.LinearNDInterpolator(zip(Zmod[ind], GUV[ind], CRs[ind]), np.squeeze(Xconv_Av3[ind]))
    interpolator_nn = interpolate.NearestNDInterpolator(zip(Zmod[ind], GUV[ind], CRs[ind]), np.squeeze(Xconv_Av3[ind]))

    MinZ = min(Zmod) #!define minimum metallicity probed by the models.
    MaxZ = max(Zmod) #!define maximum metallicity probed by the models.

    MinGUV = np.min(GUV)  #!define minimum GUV probed by the models.
    MaxGUV = np.max(GUV)  #!define maximum GUV probed by the models.

    MinCRs = np.min(CRs) #!define minimum CRs probed by the models.
    MaxCRs = np.max(CRs) #!define maximum CRs probed by the models.

    #read galaxy information in lightcone
    (dec, ra, zobs, idgal, mmol_b, mmol_d, rd, rb, zd, zb, sfr_d, sfr_b, shi, dc) = hdf5_data
   
    SFRtot = (sfr_d + sfr_b)/1e9/h0
    SFRburst = sfr_b/1e9/h0
    SFRdisk = sfr_d/1e9/h0

    r50_disk = rd * 1e3/ h0
    r50_bulge = rb * 1e3/ h0

    zcoldg_d = np.log10(zd/zsun)
    z_zero = np.where(zd <= 0)
    zcoldg_d[z_zero] = MinZ
    zcoldg_d = np.clip(zcoldg_d, MinZ, MaxZ)

    zcoldg_b = np.log10(zb/zsun)
    ind = np.where(zb <= 0)
    zcoldg_b[ind] = MinZ
    zcoldg_b = np.clip(zcoldg_b, MinZ, MaxZ)
   
    mHI = 2.356e5 / (1.0 + zobs) * pow(dc, 2.0) * shi * 1e+26 #HI mass in the disk in Msun
    Mgas_disk =  mHI /XH + mmol_d/h0
    Mgas_bulge = mmol_b/h0

    # calculation UV radiation field.
    def get_guv(sfr, mgas, z):
        guv = (sfr / mgas / (z/zsun)) / Sigma0
        guv = gammaSFR * np.log10(guv)    
        is_zero = np.where((mgas <= 0) | (sfr <= 0))
        guv[is_zero] = MinGUV
        return np.clip(guv, MinGUV, MaxGUV)

    guv_disk = get_guv(SFRdisk, Mgas_disk, zd)
    guv_bulge = get_guv(SFRburst, Mgas_bulge, zb)

    def get_co_emissions(mmol, zcoldg, guv):

        shape = len(mmol), 10
        CRRayFlux = np.full((len(guv)), 1.)

        ind = np.where(mmol > 0)
        mcold = np.zeros(shape)
        for i in range(0,10):
            mcold[ind, i] = mmol[ind]

	# Interpolate linearly first
        # If extrapolation is needed we use the nearest neighbour interpolator
        xco = np.zeros(shape)
        xco[ind, :] = 10.0 ** interpolator(zip(zcoldg[ind], guv[ind], CRRayFlux[ind]))
        isnan = np.where(np.isnan(xco[:, 0]))
        xco[isnan, :] = 10.0 ** interpolator_nn(zip(zcoldg[isnan], guv[isnan], CRRayFlux[isnan]))

        lco = np.zeros(shape)
        lco[ind, :] = mcold[ind] * XH / 313./ xco[ind]
        for i in range(0, 10):
            lco[ind, :] = lco[ind] / pow(i + 1.0, 2.0)
        return lco

    LCOb = get_co_emissions(mmol_b, zcoldg_b, guv_bulge)
    LCOd = get_co_emissions(mmol_d, zcoldg_d, guv_disk)

    #define CO luminosity in units of Jy km/s Mpc^2
    LCOtot = LCOd + LCOb

    #will calculate integrated line flux in Jy km/s
    SCOtot = np.full((len(mmol_d), 10),0.0)
    nuCO_obs = np.full((len(mmol_d), 10),0.0)

    for i in range(0,10):
        SCOtot[:,i] = LCOtot[:,i]/(4.0*PI)/pow((1.0+zobs) * dc/h0, 2.0)
        nuCO_obs[:,i] = nuco[i]/(1.0+zobs)

    file_to_write = os.path.join('CO_SLED_%02d.hdf5' % subvol)
    hf = h5py.File(file_to_write, 'w')

    hf.create_dataset('galaxies/id_galaxy_sam', data=idgal)
    hf.create_dataset('galaxies/zobs', data=zobs)
    hf.create_dataset('galaxies/ra', data=ra)
    hf.create_dataset('galaxies/dec', data=dec)
    hf.create_dataset('galaxies/dc', data=dc)
    hf.create_dataset('galaxies/SCO', data=SCOtot)
    hf.create_dataset('frequency_co_rest', data=nuco)
    hf.close()
    

def main():

    lightcone_dir = '/mnt/su3ctm/clagos/Stingray/output/medi-SURFS/Shark-Lagos18-final/deep-optical/'
    outdir= '/home/clagos/'
    #'/mnt/su3ctm/clagos/Stingray/output/medi-SURFS/Shark-Lagos18-final/deep-optical/'
    obsdir= '/home/clagos/git/shark/data/'

    subvols = range(64) 

    # Loop over redshift and subvolumes
    plt = common.load_matplotlib()
    totarea = 107.8890011908422 #deg2
    areasub = totarea/64.0 * len(subvols)  #deg2

    fields = {'galaxies': ('dec', 'ra', 'zobs',
                           'id_galaxy_sky', 'mmol_bulge', 'mmol_disk', 'rgas_disk', 
                           'rstar_bulge', 'zgas_disk', 'zgas_bulge', 'sfr_disk', 'sfr_burst', 
                           's_hi', 'dc')}

    for sv in subvols:
       hdf5_data = common.read_lightcone(lightcone_dir, fields, [sv])
       prepare_data(hdf5_data, sv, lightcone_dir)

if __name__ == '__main__':
    main()
