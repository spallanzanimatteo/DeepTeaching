"""
BSD 2-Clause License

Copyright (c) 2018, Matteo Spallanzani
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""


import os
import sys
from urllib.request import urlretrieve
import zipfile
import numpy as np


################
# DATA RETRIEVAL
################
def reporthook(count, block_size, total_size):
    """See documentation of :obj:urllib.request.urlretrieve."""
    progress_size = int(count*block_size)
    progress_percent = (progress_size / total_size) * 100
    sys.stdout.write("\r...{:4.2f}%%, %d MiB".format(progress_percent, progress_size/(1024*1024)))
    sys.stdout.flush()


def maybe_download(source, local_archive='\\'.join([os.getcwd(), 'data', 'dataset.zip'])):
    """Download the specified zip archive to the specified location.

    Args:
        source (:obj:`str`): the URL of the source zip archive.
        local_archive (:obj:`str`): the full path to the downloaded archive,
            including its name.

    """
    if not os.path.isfile(local_archive):
        urlretrieve(source, filename=local_archive, reporthook=reporthook)
        print('Download complete.')
    else:
        print('Archive already exists!')


def maybe_unzip(local_archive='\\'.join([os.getcwd(), 'data', 'dataset.zip']), unzip_path='\\'.join([os.getcwd(), 'data', 'unzipped_dataset'])):
    """Unzip the specified local zip archive to the specified folder.

    Args:
        local_archive (:obj:`str`): the full path to the zip archive,
            including its name.
        unzip_path (:obj:`str`): the full path to the unzipped folder.

    """
    if not os.path.exists(unzip_path):
        os.mkdir(unzip_path)
        with zipfile.ZipFile(local_archive, 'r') as archive:
            archive.extractall(unzip_path)
        print('Unzip complete.')
    else:
        print('Archive already uncompressed!')


###############
# DATA BATCHING
###############
def get_batches(x, y, bs=1):
    """Partition the set of I/O pairs (x, y) into batches of given size.

    x and y are supposed to be n-dimensional ndarrays, with n>=2. The first
    dimension is the global number of available I/O pairs; x and y should
    have the same number of items (i.e. x.shape[0]==y.shape[0] should return
    True).

    Args:
        x (:obj:`numpy.ndarray`): the vectors of observable variables.
        y (:obj:`numpy.ndarray`): the vectors of response variables.
        bs (:obj:`int`): the number of pairs in each batch.

    Returns:
        batches (:obj:`list` of :obj:`list`): list of pairs of ndarrays (X, Y);
            X's rows are the observables, Y's rows are the corresponding
            response variables.

    """
    batches = list()
    num_examples = x.shape[0]
    for start in range(0, num_examples, bs):
        end = min(start+bs, num_examples)
        batch = [x[start:end], y[start:end]]
        batches.append(batch)
    return batches


def get_batches_bptt(x, y, ts=1, bs=1):
    """Partition global sequences x and y into smaller sequences, then group
    these sequences into batches.

    x and y are supposed to be n-dimensional ndarrays, with n>=2. The first
    dimension is the global number of available frames; x and y should
    have the same number of items (i.e. x.shape[0]==y.shape[0] should return
    True).

    Args:
        x (:obj:`numpy.ndarray`): the vectors of observable variables.
        y (:obj:`numpy.ndarray`): the vectors of response variables.
        ts (:obj:`int`): the number of time steps that represent an event, i.e.
            the sequence length on which the model will be trained.
        bs (:obj:`int`): the batch size, i.e. the number of sequences in each
            batch.

    Returns:
        batches (:obj:`list` of :obj:`list` of :obj:`list`): list of batches;
            every batch is a list whose elements are two lists:
            $[X[t]]_{t=0,...ts-1}$ and $[Y[t]]_{t=0,...ts-1}$; both these lists
            are ndarrays containing the same number bs of frames (one for each
            batch example).

    """
    batches = list()
    num_examples = x.shape[0]
    frames_per_batch = ts * bs
    num_batches = num_examples // frames_per_batch
    for start in range(0, num_batches*frames_per_batch, frames_per_batch):
        end = start + frames_per_batch
        next_x = x[start:end]
        next_y = y[start:end]
        batch_x = list()
        batch_y = list()
        for t in range(ts):
            x_t = np.vstack(next_x[t:frames_per_batch:ts, None, :])
            batch_x.append(x_t)
            y_t = np.vstack(next_y[t:frames_per_batch:ts, None, :])
            batch_y.append(y_t)
        batches.append([batch_x, batch_y])

    return batches
