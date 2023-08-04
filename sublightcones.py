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

import argparse
import logging
import os

import h5py
import numpy as np

logger = logging.getLogger(__name__)

def _copy_subvolume(f_in, output_fname, subvol):

    galaxies = f_in['galaxies']

    # Skip empty selections
    selection = np.where((galaxies['subvolume'][()] == subvol) & (galaxies['zobs'][()] < 2) & (galaxies['mstars_bulge'][()] + galaxies['mstars_disk'][()] > 5e8))[0].tolist()
    if not selection:
        logger.warning('No data found for subvolume %d', subvol)
        return
    logger.info('Selection for subvolume %d: %d elements (%.2f %% of total)',
                subvol, len(selection), 100. * len(selection) / len(galaxies['subvolume'][()]))

    with h5py.File(output_fname, 'w') as f_out:
        logger.info('Processing subvol %d, writing data to %s', subvol, output_fname)
        # Copy all data from these groups
        for group in ('run_info', 'parameters'):
            f_out.create_group(group)
            original_group = f_in[group]
            for dataset in original_group:
                logger.info('Copying %s/%s', group, dataset)
                f_out[group][dataset] = original_group[dataset][()]
        # Copy subset of all datasets in galaxies
        f_out.create_group('galaxies')
        for dataset in galaxies:
            logger.info('Copying galaxies/%s', dataset)
            f_out['galaxies'][dataset] = galaxies[dataset][()][selection] #.value[selection][:]

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='The file to split')
    parser.add_argument('-o', '--output-dir', help='The directory where the split files will be written to', default='.')

    opts = parser.parse_args()
    if not opts.input:
        parser.error('An input is needed')

    logging.basicConfig(level=logging.INFO)

    bname, _ = os.path.splitext(os.path.basename(opts.input))
    output_fname_pattern = os.path.join(opts.output_dir, bname + "_subselection_%02d.hdf5")

    with h5py.File(opts.input) as f_in:
        subvols = f_in['galaxies']['subvolume']
        logger.info('Processing %d subvolumes', len(subvols))
        uni_subvols = np.unique(subvols)
        logger.info('%d unique subvolumes found: %r', len(uni_subvols), uni_subvols)
        for subvol in uni_subvols:
            _copy_subvolume(f_in, output_fname_pattern % subvol, subvol)

if __name__ == '__main__':
    main()
