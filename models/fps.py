from torch.autograd import Function
import warnings
import torch
import numpy as np
import numexpr as ne


try:
    import models._ext as _ext
except ImportError:
    from torch.utils.cpp_extension import load
    import glob
    import os.path as osp
    import os

    warnings.warn("Unable to load pointnet2_ops cpp extension. JIT Compiling.")

    _ext_src_root = osp.join(osp.dirname(__file__), "_ext-src")
    _ext_sources = glob.glob(osp.join(_ext_src_root, "src", "*.cpp")) + glob.glob(
        osp.join(_ext_src_root, "src", "*.cu")
    )
    _ext_headers = glob.glob(osp.join(_ext_src_root, "include", "*"))

    os.environ["TORCH_CUDA_ARCH_LIST"] = "3.7+PTX;5.0;6.0;6.1;6.2;7.0;7.5"
    _ext = load(
        "_ext",
        sources=_ext_sources,
        extra_include_paths=[osp.join(_ext_src_root, "include")],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-Xfatbin", "-compress-all"],
        with_cuda=True,
    )

class FurthestPointSampling(Function):
    @staticmethod
    def forward(ctx, xyz, npoint):
        # type: (Any, torch.Tensor, int) -> torch.Tensor
        r"""
        Uses iterative furthest point sampling to select a set of npoint features that have the largest
        minimum distance
        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor where N > npoint
        npoint : int32
            number of features in the sampled set
        Returns
        -------
        torch.Tensor
            (B, npoint) tensor containing the set
        """
        fps_inds = _ext.furthest_point_sampling(xyz, npoint)
        ctx.mark_non_differentiable(fps_inds)
        return fps_inds

    @staticmethod
    def backward(xyz, a=None):
        return None, None


farthest_point_sample_cuda = FurthestPointSampling.apply


def fps_cuda(points, npoint):
    # returns farthers point distance sampling
    # shuffle each sequence individually but keep correspondance throughout the sequence
    # to use this CUDA based version you must insert multiprocessing.set_start_method('spawn') in the main function
    """
    Input:
        points: pointcloud data, [N, 3]
    Return:
        points: farthest sampled pointcloud, [npoint]
    """

    fps_idx = farthest_point_sample_cuda(points, npoint).to(torch.int64)
    out_points = index_points(points, fps_idx)
    return out_points


def fps_torch(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device)* 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def fps_np(points, npoint):
    # returns farthers point distance sampling
    # shuffle each sequence individually but keep correspondance throughout the sequence
    """
    Input:
        points: pointcloud data, [N, 3]
    Return:
        points: farthest sampled pointcloud, [npoint]
    """
    xyz = points[0] #use the first frame for sampling
    N, C = xyz.shape
    centroids = np.zeros(npoint)
    distance = np.ones(N) * 1e10
    farthest = np.random.randint(0, N)
    idxs = np.array(farthest)[None]

    for i in range(npoint-1):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.array(np.argmax(distance, -1))[None]
        idxs = np.concatenate([idxs, farthest])


    return points[:, idxs]

def fps_ne(points, npoint):
    # returns farthers point distance sampling
    # shuffle each sequence individually but keep correspondance throughout the sequence
    """
    Input:
        points: pointcloud data, [N, 3]
    Return:
        points: farthest sampled pointcloud, [npoint]
    """
    xyz = points[0] #use the first frame for sampling
    N, C = xyz.shape
    centroids = np.zeros(npoint)
    distance = np.ones(N) * 1e10
    farthest = np.random.randint(0, N)
    idxs = np.array(farthest)[None]

    for i in range(npoint-1):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = ne.evaluate('sum((xyz - centroid) ** 2, 1)')
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.array(np.argmax(distance, -1))[None]
        idxs = np.concatenate([idxs, farthest])


    return points[:, idxs]

def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points