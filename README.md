# Speed test for FPS

 This repos provides multiple farthst point sampling for 3D point clouds. 
 To compile the cuda version simply run the relevant setyp file using:
 
`python setup.py install`

Run `main.py` to print out the timing results. 

Conclusions: CUDA implementation is fastest and torch is slowest. 
The Cython implementation is not optimized and could probably do better. 

If you want to avoid compilation issues you can simply use the np or numexpr versions.
