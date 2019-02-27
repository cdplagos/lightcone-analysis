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

def prepare_data(hdf5_data, subvols, lightcone_dir):

    #read ascii table from Bayet et al. (2011)
    datanu = np.genfromtxt('emlines_carbonmonoxide_restnu.data', delimiter=' ', skip_header=11)
    nuco = datanu

    dataPDR  = np.genfromtxt('emlines_carbonmonoxide_Bayet11.data', delimiter=' ', skip_header=10)

    CRs = dataPDR[:,5]/5E-17 #normalize cosmic rays rate by 5e-17 s^{-1}.
    Xconv_Av3 = np.log10(dataPDR[:,6:16]/1e20) #normalize conversion factors by 10^20 cm^(-2) (K km/s)^(-1)
    Xconv_Av8 = np.log10(dataPDR[:,16:26]/1e20) #normalize conversion factors by 10^20 cm^(-2) (K km/s)^(-1)

    #Convert all the relevant quantities to logarithm as the 3D interpolation is done in the logarithmic space.
    Xconv_Av3 = np.log10(Xconv_Av3)
    Xconv_Av8 = np.log10(Xconv_Av8)
 
    Zmod = np.log10(dataPDR[:,4])
    GUV  = np.log10(dataPDR[:,0])
    CRs  = np.log10(CRs)

    nh   = dataPDR[:,3]
    ind = np.where(nh == 1e4)
    interpolator = interpolate.LinearNDInterpolator(zip(Zmod[ind], GUV[ind], CRs[ind]), np.squeeze(Xconv_Av3[ind]))

    MinZ = min(Zmod) #!define minimum metallicity probed by the models.
    MaxZ = max(Zmod) #!define maximum metallicity probed by the models.

    MinGUV = min(GUV)  #!define minimum GUV probed by the models.
    MaxGUV = max(GUV)  #!define maximum GUV probed by the models.

    MinCRs = min(CRs) #!define minimum CRs probed by the models.
    MaxCRs = max(CRs) #!define maximum CRs probed by the models.

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
    ind = np.where(zcoldg_d < MinZ)
    zcoldg_d[ind] = MinZ
    ind = np.where(zcoldg_d > MaxZ)
    zcoldg_d[ind] = MaxZ

    zcoldg_b = np.log10(zb/zsun)
    ind = np.where(zb <= 0)
    zcoldg_b[ind] = MinZ
    ind = np.where(zcoldg_b < MinZ)
    zcoldg_b[ind] = MinZ
    ind = np.where(zcoldg_b > MaxZ)
    zcoldg_b[ind] = MaxZ
   
    mHI = 2.356e5 / (1.0 + zobs) * pow(dc, 2.0) * shi * 1e+26 #HI mass in the disk in Msun
    Mgas_disk =  mHI /XH + mmol_d
    Mgas_bulge = mmol_b

    #calculation UV radiation field.
    GUV_disk = (SFRdisk/Mgas_disk/(zd/zsun)) / Sigma0
    GUV_disk = gammaSFR * np.log10(GUV_disk)    
    guv_zero = np.where((Mgas_disk <= 0) | (SFRdisk <= 0))
    GUV_disk[guv_zero] = MinGUV
    ind = np.where(GUV_disk < MinGUV)
    GUV_disk[ind] = MinGUV
    ind = np.where(GUV_disk > MaxGUV)
    GUV_disk[ind] = MaxGUV

    GUV_bulge = (SFRburst/Mgas_bulge/(zb/zsun)) / Sigma0
    GUV_bulge = gammaSFR * np.log10(GUV_bulge)
    guv_zero  = np.where((Mgas_bulge <= 0) | (SFRburst <= 0))
    GUV_bulge[guv_zero] = MinGUV
    ind = np.where(GUV_bulge < MinGUV)
    GUV_bulge[ind] = MinGUV
    ind = np.where(GUV_bulge > MaxGUV)
    GUV_bulge[ind] = MaxGUV

    CRRayFlux = np.full((len(GUV_disk)),1.0)

    XCOd = np.full((len(mmol_d), 10),0.0)  
    XCOb = np.full((len(mmol_d), 10),0.0)  
    LCOd = np.full((len(mmol_d), 10),0.0)  
    LCOb = np.full((len(mmol_d), 10),0.0)  

    mcoldd = np.full((len(mmol_d), 10),0.0)  
    LCOb = np.full((len(mmol_d), 10),0.0)  

    jline = range(10)
    ind = np.where(mmol_d > 0)
    for i in range(0,10):
        mcoldd[ind,i] = mmol_d[ind]
    XCOd[ind,:] = pow(10.0, interpolator(zip(zcoldg_d[ind], GUV_disk[ind], CRRayFlux[ind])))
    LCOd[ind,:] = mcoldd[ind,:] * XH / 313./ XCOd[ind,:]
    LCO10 = mmol_d[ind] * XH / 313./ XCOd[ind,0]
    for i in range(0,10):
         LCOd[ind,:] = LCOd[ind,:] / pow(i+1.0, 2.0)
        
    print LCOd[0:10,0], LCO10[0:10]
    
    ind = np.where(mmol_b > 0)
    LCOb[ind] = interpolator(zip(zcoldg_b[ind], GUV_bulge[ind], CRRayFlux[ind]))


def main():

    lightcone_dir = '/mnt/su3ctm/clagos/Stingray/output/medi-SURFS/Shark-Lagos18-final/deep-optical/'
    outdir= '/home/clagos/'
    #'/mnt/su3ctm/clagos/Stingray/output/medi-SURFS/Shark-Lagos18-final/deep-optical/'
    obsdir= '/home/clagos/git/shark/data/'

    subvols = (0,1) #range(64) 

    # Loop over redshift and subvolumes
    plt = common.load_matplotlib()
    totarea = 107.8890011908422 #deg2
    areasub = totarea/64.0 * len(subvols)  #deg2

    fields = {'galaxies': ('dec', 'ra', 'zobs',
                           'id_galaxy_sky', 'mmol_bulge', 'mmol_disk', 'rgas_disk', 
                           'rstar_bulge', 'zgas_disk', 'zgas_bulge', 'sfr_disk', 'sfr_burst', 
                           's_hi', 'dc')}

    hdf5_data = common.read_lightcone(lightcone_dir, fields, subvols)
 
    prepare_data(hdf5_data, subvols, lightcone_dir)

if __name__ == '__main__':
    main()
