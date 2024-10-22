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

import h5py

##################################
h0 = 0.6751

def compute_z0_mvir(hdf5_data, sam_dir, subvol, output_fname):

    (dec, ra, zobs, idgal, sfrb, sfrd, mstarb, mstard, rsb, rsd, mvir, typeg,
     id_galaxy_sam, snapshot, subvolume, id_halo_sam) = hdf5_data

    #loop over snapshot, subvolume, find id_galaxy_sam, its hosthalo and find the Mvirz0.

    final_mvir = np.zeros(shape = len(zobs))
    snaps = np.unique(snapshot)
    #loop over snapshots to access final halo mass
    for s in snaps:
        input_dir = os.path.join(sam_dir, str(s), str(subvol)) + '/galaxies.hdf5'
        with h5py.File(input_dir,'r') as f:
             final_z0_mvir  = f['/halo/final_z0_mvir'][()]
             galaxy_galo_id = f['/galaxies/id_halo'][()]
        gals = np.where(snapshot == s)
        ids_gals = gals[0]
        id_halo_sam_gals = id_halo_sam[gals]
        for id_gal, ids in enumerate(id_halo_sam_gals):
            #print(id_gal, ids)
            final_mvir[ids_gals[id_gal]] = final_z0_mvir[ids-1]

    hf = h5py.File(output_fname,'w')
    print("will write in", output_fname)
    hf.create_dataset('galaxies/mvir_z0', data=final_mvir)
    hf.create_dataset('galaxies/id_galaxy_sam', data=id_galaxy_sam)
    hf.create_dataset('galaxies/snapshot', data=snapshot)
    hf.create_dataset('galaxies/subvolume', data=subvolume)
    hf.close()

def main():

    lightcone_dir = '/scratch/pawsey0119/clagos/Stingray/output/medi-SURFS/Shark-TreeFixed-ReincPSO-kappa0p002/deep-optical-final/'
    sam_dir = '/scratch/pawsey0119/clagos/SHARK_Out/medi-SURFS/Shark-TreeFixed-ReincPSO-kappa0p002/'
    output_dir = lightcone_dir + 'split/'
    bname = 'final_mvir'
    subvols = range(64) #(0,1,2,3,4,5,6,7,8,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35)
    #(9,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63) #0,1,2,3,4,5,6,7,8,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35) #range(64)

    fields = {'galaxies': ('dec', 'ra', 'zobs',
                           'id_galaxy_sky','sfr_burst','sfr_disk','mstars_bulge',
                           'mstars_disk','rstar_bulge_apparent','rstar_disk_apparent',
                           'mvir_hosthalo','type','id_galaxy_sam','snapshot','subvolume','id_halo_sam')}

    for subv in subvols:
        hdf5_data = common.read_lightcone(lightcone_dir, 'split/', fields, [subv], "mock")
        output_fname = os.path.join(output_dir, bname + "_%02d.hdf5" % subv)
        compute_z0_mvir(hdf5_data, sam_dir, subv, output_fname)
    

if __name__ == '__main__':
    main()
