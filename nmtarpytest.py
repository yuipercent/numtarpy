import numtarpy as nmtar
import numpy
import sys
import os
import numba

class Chunk():
    tarConverter = nmtar.tarBankConverter()
    def __init__(self,px,py,pz,ab):
        self.__x = px
        self.__y = py
        self.__z = pz
        self.__models = ab
    
    def getpos(self):
        return (self.__x,self.__y,self.__z)

    def getmodels(self):
        return self.__models

    def __tarredformat__(self):
        return (nmtar.tarred.int32,nmtar.tarred.int32,nmtar.tarred.int32,nmtar.tarred.nparray(nmtar.tarred.uint8))

class WRLD():
    """A class used to represent the game environnement. """
    def __init__(self,rworld_,aworld_,a2world_,indexrepo_,isactive_):
        self.rworld = nmtar.tarBank(Chunk)
        self.__isactive = isactive_

def emptyWRLD():
    return WRLD(numpy.ndarray(24576,dtype=numpy.int16),numpy.full(8192,False,dtype=bool),numpy.full(256,False,dtype=bool),numpy.ndarray(1,numpy.uint8),False)

nmtory = nmtar.refinery()

with nmtory.config as param:
    param.useStandardModel()
    param.ProtocolAmount = 6
    param.ProtocolMemoryAmount = 2048

trpipe = nmtory.pipe()
with trpipe.config as param:
    param

#@nmtory.archCommand()
#def stop():
#    nmtory.locals.running = False
