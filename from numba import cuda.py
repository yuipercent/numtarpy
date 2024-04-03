from pickletools import uint2
from numba import cuda
import numba
import numpy
import time

@cuda.jit()
def ntarFx000(kernelmodes,protocolCopyMemory,):
    running= True
    while running==True:
        cudaGridIndex = cuda.grid(1)
        for commandIndex in range(128):
            protocolCopyMemory[commandIndex]= kernelmodes[commandIndex]
        if protocolCopyMemory[0] == True:
            running= 0

hm2 = cuda.mapped_array(256,dtype=numpy.uint8,)
hm2[0] = 0
hm4 = cuda.device_array(256,dtype=numpy.uint8)

ntarFx000[1,1](hm2,hm4,)
a = 0
for p in range(0,999999):
    a += 1
hm2[0] = 1
