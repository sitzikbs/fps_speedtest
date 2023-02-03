import numpy as np
import torch
import models.fps as fps
import fps_cython
import time

batch_size, n_points = 16, 1024
points = np.random.rand(batch_size, n_points, 3)

sample_npoints = 128

st = time.time()
out = fps_cython.fps_cython(points, int(sample_npoints))
et = time.time()
elapsed_time = et - st
print('Execution time cython:', elapsed_time, 'seconds')

st = time.time()
out = fps.fps_np(points, sample_npoints)
et = time.time()
elapsed_time = et - st
print('Execution time np:', elapsed_time, 'seconds')

st = time.time()
out = fps.fps_ne(points, sample_npoints)
et = time.time()
elapsed_time = et - st
print('Execution time numexpr:', elapsed_time, 'seconds')


points = torch.randn(batch_size, n_points, 3).cuda().contiguous()
st = time.time()
out = fps.fps_torch(points, sample_npoints)
et = time.time()
elapsed_time = et - st
print('Execution time torch:', elapsed_time, 'seconds')

points = torch.randn(batch_size, n_points, 3).cuda().contiguous()
st = time.time()
out = fps.fps_cuda(points, sample_npoints)
et = time.time()
elapsed_time = et - st
print('Execution time cuda:', elapsed_time, 'seconds')