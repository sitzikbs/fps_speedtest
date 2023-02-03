# cimport numpy as np
# import numpy as np
#
# def fps_cython(np.ndarray[np.float64_t, ndim=3] points, int  n_points):
#
#     cdef np.ndarray[np.float64_t, ndim = 2] xyz = points[0]
#     cdef int N = xyz.shape[0]
#     cdef int C = xyz.shape[1]
#     cdef np.ndarray[np.float64_t, ndim = 1] centroids = np.empty(N)
#     cdef np.ndarray[np.float64_t, ndim = 1] distance = np.ones(N) * 1e10
#     cdef int farthest = np.random.default_rng().integers(0, N)
#     cdef np.ndarray[np.int64_t, ndim = 1] idxs = np.array(farthest)[None]
#     cdef int i
#     cdef np.ndarray[np.float64_t, ndim = 1] centroid
#     cdef np.ndarray[np.float64_t, ndim = 1] dist
#     cdef np.ndarray[np.uint8_t, ndim = 1, cast=True] mask
#     for i in range(n_points - 1):
#         centroids[i] = farthest
#         centroid = xyz[farthest, :]
#         dist = np.sum((xyz - centroid) ** 2, -1)
#         mask = dist < distance
#         distance[mask] = dist[mask]
#         farthest = int(np.argmax(distance, -1).item())
#         idxs = np.concatenate([idxs, np.array(farthest)[None]])
#
#     return points[:, idxs]

cimport numpy as np
import numpy as np

cdef extern from "math.h":
    double sqrt(double x)

cdef int N, C, i, j, farthest
cdef double centroid, dist, max_distance
cdef double[:] centroids, distance
cdef int[:] idxs

def fps_cython(double[:,:,:] points, int  n_points):

    xyz = points[0]
    N = xyz.shape[0]
    C = xyz.shape[1]
    centroids = np.empty(N)
    distance = [1e10 for i in range(N)]
    farthest = np.random.randint(0, N)
    idxs = [farthest]
    for i in range(n_points - 1):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        for j in range(N):
            dist = 0
            for k in range(C):
                dist += (xyz[j,k] - centroid[k]) ** 2
            dist = sqrt(dist)
            if dist < distance[j]:
                distance[j] = dist
            if dist > max_distance:
                max_distance = dist
                farthest = j
        idxs.append(farthest)
    return points[:, idxs]