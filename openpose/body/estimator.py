# -*- coding: utf-8 -*-
"""
OpenPose body pose estimator.
Created on Wed Sep 11 19:00:00 2019
Author: Prasun Roy | CVPRU-ISICAL (http://www.isical.ac.in/~cvpr)
GitHub: https://github.com/prasunroy/openpose-pytorch

Original author: Zhizhong Huang
Original source: https://github.com/Hzzone/pytorch-openpose

"""


import cv2
import numpy as np
import os
import shutil
import tempfile
import torch
from scipy.ndimage.filters import gaussian_filter
from tqdm import tqdm
from urllib.parse import urlparse
from urllib.request import urlopen
from .model import BodyPoseModel


model_url = 'https://www.dropbox.com/s/mun9eh2509pw32n/openpose_body_coco_pose_iter_440000.pth?dl=1'
model_dir = os.path.join(os.path.expanduser('~'), '.cache/torch/checkpoints/')


def _download_url_to_file(url, path, progress=True):
    link = urlopen(url)
    meta = link.info()
    content_length = meta.get_all('Content-Length')
    if content_length is not None and len(content_length) > 0:
        file_size = int(content_length[0])
    else:
        file_size = None
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    try:
        with tqdm(total=file_size, disable=not progress,
                  unit='B', unit_scale=True, unit_divisor=1024) as pbar:
            while True:
                buffer = link.read(8192)
                if len(buffer) == 0:
                    break
                temp_file.write(buffer)
                pbar.update(len(buffer))
        temp_file.close()
        shutil.move(temp_file.name, path)
    finally:
        temp_file.close()
        if os.path.exists(temp_file.name):
            os.remove(temp_file.name)


def _load_state_dict_from_url(model_url, model_dir, progress=True):
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    urlparts = urlparse(model_url)
    filename = os.path.basename(urlparts.path.split('/')[-1])
    cached_file = os.path.join(model_dir, filename)
    if not os.path.isfile(cached_file):
        print(f'Downloading: "{model_url}" to {cached_file}')
        _download_url_to_file(model_url, cached_file, progress)
    return torch.load(cached_file)


def _load_state_dict(model, state_dict):
    model_state_dict = {}
    for name in model.state_dict().keys():
        model_state_dict[name] = state_dict['.'.join(name.split('.')[1:])]
    model.load_state_dict(model_state_dict)
    return model


def _pad_image(image, stride=1, padvalue=0):
    assert len(image.shape) == 2 or len(image.shape) == 3
    h, w = image.shape[:2]
    pads = [None] * 4
    pads[0] = 0 # left
    pads[1] = 0 # top
    pads[2] = 0 if (w % stride == 0) else stride - (w % stride) # right
    pads[3] = 0 if (h % stride == 0) else stride - (h % stride) # bottom
    num_channels = 1 if len(image.shape) == 2 else image.shape[2]
    image_padded = np.ones((h+pads[3], w+pads[2], num_channels), dtype=np.uint8) * padvalue
    image_padded = np.squeeze(image_padded)
    image_padded[:h, :w] = image
    return image_padded, pads


def _get_keypoints(candidates, subsets):
    k = subsets.shape[0]
    keypoints = np.zeros((k, 18, 3), dtype=np.int32)
    for i in range(k):
        for j in range(18):
            index = np.int32(subsets[i][j])
            if index != -1:
                x, y = np.int32(candidates[index][:2])
                keypoints[i][j] = (x, y, 1)
    return keypoints


class BodyPoseEstimator(object):
    
    def __init__(self, pretrained=False):
        self._model = BodyPoseModel()
        if torch.cuda.is_available():
            self._model = self._model.cuda()
        if pretrained:
            state_dict = _load_state_dict_from_url(model_url, model_dir)
            self._model = _load_state_dict(self._model, state_dict)
        self._model.eval()
    
    def __call__(self, image):
        scales = [0.5]
        stride = 8
        bboxsize = 368
        padvalue = 128
        thresh_1 = 0.1
        thresh_2 = 0.05
        
        multipliers = [scale * bboxsize / image.shape[0] for scale in scales]
        heatmap_avg = np.zeros((image.shape[0], image.shape[1], 19))
        paf_avg = np.zeros((image.shape[0], image.shape[1], 38))
        
        for scale in multipliers:
            image_scaled = cv2.resize(image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            image_padded, pads = _pad_image(image_scaled, stride, padvalue)
            image_tensor = np.expand_dims(np.transpose(image_padded, (2, 0, 1)), 0)
            image_tensor = np.float32(image_tensor) / 255.0 - 0.5
            image_tensor = np.ascontiguousarray(image_tensor)
            image_tensor = torch.from_numpy(image_tensor).float()
            if torch.cuda.is_available():
                image_tensor = image_tensor.cuda()
            
            with torch.no_grad():
                Mconv7_stage6_L1, Mconv7_stage6_L2 = self._model(image_tensor)
            
            Mconv7_stage6_L1 = Mconv7_stage6_L1.cpu().numpy() # paf
            Mconv7_stage6_L2 = Mconv7_stage6_L2.cpu().numpy() # heatmap
            
            heatmap = np.transpose(np.squeeze(Mconv7_stage6_L2), (1, 2, 0))
            heatmap = cv2.resize(heatmap, (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
            heatmap = heatmap[:image_padded.shape[0] - pads[3], :image_padded.shape[1] - pads[2], :]
            heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)
            
            paf = np.transpose(np.squeeze(Mconv7_stage6_L1), (1, 2, 0))
            paf = cv2.resize(paf, (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
            paf = paf[:image_padded.shape[0] - pads[3], :image_padded.shape[1] - pads[2], :]
            paf = cv2.resize(paf, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)
            
            heatmap_avg += (heatmap / len(multipliers))
            paf_avg += (paf / len(multipliers))
        
        all_peaks = []
        num_peaks = 0
        
        for part in range(18):
            map_orig = heatmap_avg[:, :, part]
            map_filt = gaussian_filter(map_orig, sigma=3)
            
            map_L = np.zeros_like(map_filt)
            map_T = np.zeros_like(map_filt)
            map_R = np.zeros_like(map_filt)
            map_B = np.zeros_like(map_filt)
            map_L[1:, :] = map_filt[:-1, :]
            map_T[:, 1:] = map_filt[:, :-1]
            map_R[:-1, :] = map_filt[1:, :]
            map_B[:, :-1] = map_filt[:, 1:]
            
            peaks_binary = np.logical_and.reduce(
                (map_filt >= map_L, map_filt >= map_T,
                 map_filt >= map_R, map_filt >= map_B,
                 map_filt > thresh_1)
            )
            peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]))
            peaks_ids = range(num_peaks, num_peaks + len(peaks))
            peaks_with_scores = [peak + (map_orig[peak[1], peak[0]],) for peak in peaks]
            peaks_with_scores_and_ids = [peaks_with_scores[i] + (peaks_ids[i],) \
                                         for i in range(len(peaks_ids))]
            all_peaks.append(peaks_with_scores_and_ids)
            num_peaks += len(peaks)
        
        map_idx = [[31, 32], [39, 40], [33, 34], [35, 36], [41, 42], [43, 44],
                   [19, 20], [21, 22], [23, 24], [25, 26], [27, 28], [29, 30],
                   [47, 48], [49, 50], [53, 54], [51, 52], [55, 56], [37, 38],
                   [45, 46]]
        limbseq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10],
                   [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17],
                   [1, 16], [16, 18], [3, 17], [6, 18]]
        
        all_connections = []
        spl_k = []
        mid_n = 10
        
        for k in range(len(map_idx)):
            score_mid = paf_avg[:, :, [x - 19 for x in map_idx[k]]]
            candidate_A = all_peaks[limbseq[k][0] - 1]
            candidate_B = all_peaks[limbseq[k][1] - 1]
            n_A = len(candidate_A)
            n_B = len(candidate_B)
            index_A, index_B = limbseq[k]
            if n_A != 0 and n_B != 0:
                connection_candidates = []
                for i in range(n_A):
                    for j in range(n_B):
                        v = np.subtract(candidate_B[j][:2], candidate_A[i][:2])
                        n = np.sqrt(v[0] * v[0] + v[1] * v[1])
                        v = np.divide(v, n)
                        
                        ab = list(zip(np.linspace(candidate_A[i][0], candidate_B[j][0], num=mid_n),
                                      np.linspace(candidate_A[i][1], candidate_B[j][1], num=mid_n)))
                        vx = np.array([score_mid[int(round(ab[x][1])), int(round(ab[x][0])), 0] for x in range(len(ab))])
                        vy = np.array([score_mid[int(round(ab[x][1])), int(round(ab[x][0])), 1] for x in range(len(ab))])
                        score_midpoints = np.multiply(vx, v[0]) + np.multiply(vy, v[1])
                        score_with_dist_prior = sum(score_midpoints) / len(score_midpoints) + min(0.5 * image.shape[0] / n - 1, 0)
                        criterion_1 = len(np.nonzero(score_midpoints > thresh_2)[0]) > 0.8 * len(score_midpoints)
                        criterion_2 = score_with_dist_prior > 0
                        if criterion_1 and criterion_2:
                            connection_candidate = [i, j, score_with_dist_prior, score_with_dist_prior + candidate_A[i][2] + candidate_B[j][2]]
                            connection_candidates.append(connection_candidate)
                connection_candidates = sorted(connection_candidates, key=lambda x: x[2], reverse=True)
                connection = np.zeros((0, 5))
                for candidate in connection_candidates:
                    i, j, s = candidate[0:3]
                    if i not in connection[:, 3] and j not in connection[:, 4]:
                        connection = np.vstack([connection, [candidate_A[i][3], candidate_B[j][3], s, i, j]])
                        if len(connection) >= min(n_A, n_B):
                            break
                all_connections.append(connection)
            else:
                spl_k.append(k)
                all_connections.append([])
        
        candidate = np.array([item for sublist in all_peaks for item in sublist])
        subset = np.ones((0, 20)) * -1
        
        for k in range(len(map_idx)):
            if k not in spl_k:
                part_As = all_connections[k][:, 0]
                part_Bs = all_connections[k][:, 1]
                index_A, index_B = np.array(limbseq[k]) - 1
                for i in range(len(all_connections[k])):
                    found = 0
                    subset_idx = [-1, -1]
                    for j in range(len(subset)):
                        if subset[j][index_A] == part_As[i] or subset[j][index_B] == part_Bs[i]:
                            subset_idx[found] = j
                            found += 1
                    if found == 1:
                        j = subset_idx[0]
                        if subset[j][index_B] != part_Bs[i]:
                            subset[j][index_B] = part_Bs[i]
                            subset[j][-1] += 1
                            subset[j][-2] += candidate[part_Bs[i].astype(int), 2] + all_connections[k][i][2]
                    elif found == 2:
                        j1, j2 = subset_idx
                        membership = ((subset[j1] >= 0).astype(int) + (subset[j2] >= 0).astype(int))[:-2]
                        if len(np.nonzero(membership == 2)[0]) == 0:
                            subset[j1][:-2] += (subset[j2][:-2] + 1)
                            subset[j1][-2:] += subset[j2][-2:]
                            subset[j1][-2] += all_connections[k][i][2]
                            subset = np.delete(subset, j2, 0)
                        else:
                            subset[j1][index_B] = part_Bs[i]
                            subset[j1][-1] += 1
                            subset[j1][-2] += candidate[part_Bs[i].astype(int), 2] + all_connections[k][i][2]
                    elif not found and k < 17:
                        row = np.ones(20) * -1
                        row[index_A] = part_As[i]
                        row[index_B] = part_Bs[i]
                        row[-1] = 2
                        row[-2] = sum(candidate[all_connections[k][i, :2].astype(int), 2]) + all_connections[k][i][2]
                        subset = np.vstack([subset, row])
        
        del_idx = []
        
        for i in range(len(subset)):
            if subset[i][-1] < 4 or subset[i][-2] / subset[i][-1] < 0.4:
                del_idx.append(i)
        subset = np.delete(subset, del_idx, axis=0)
        
        return _get_keypoints(candidate, subset)
