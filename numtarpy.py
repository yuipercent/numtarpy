"""
numTar - User interaction module
================================
Module used to create GPU executable functions on the nmtory Workboard model\n\n
Provides:\n
- 1 : A module for simplifying gpu code writing
- 2 : A way of vizualizing and controlling in real time data treatment\n
GET STARTED : Refinery class :
------------------------------
instantiate a refinery type object to get started. The refinery type object is a representation of the GPU.\n
numtarpy.archCommand() decorator:
---------------------------------

"""
from __future__ import annotations
from typing import Any
import numba
from numba import cuda
import numpy
from math import *
import numtarpipe as tarp
import os
import math
from typing import Callable,Concatenate
import ntarskript as ntarsk

class tarred():
    nparray = ntarsk.tarred.nparray
    func = ntarsk.tarred.func
    int32 = ntarsk.tarred.int32
    int64 = ntarsk.tarred.int64
    int16 = ntarsk.tarred.int16
    int8 = ntarsk.tarred.int8
    uint64 = ntarsk.tarred.uint64
    uint32 = ntarsk.tarred.uint32
    uint16 = ntarsk.tarred.uint16
    uint8 = ntarsk.tarred.uint8
    float32 = ntarsk.tarred.float32
    float64 = ntarsk.tarred.float64
    complex64 = ntarsk.tarred.complex64
    complex128 = ntarsk.tarred.complex128
    boolean = ntarsk.tarred.boolean

class compiled():
    nparray = ntarsk.compiled.nparray
    func = ntarsk.compiled.func
    int32 = ntarsk.compiled.int32
    int64 = ntarsk.compiled.int64
    int16 = ntarsk.compiled.int16
    int8 = ntarsk.compiled.int8
    uint64 = ntarsk.compiled.uint64
    uint32 = ntarsk.compiled.uint32
    uint16 = ntarsk.compiled.uint16
    uint8 = ntarsk.compiled.uint8
    float32 = ntarsk.compiled.float32
    float64 = ntarsk.compiled.float64
    complex64 = ntarsk.compiled.complex64
    complex128 = ntarsk.compiled.complex128
    boolean = ntarsk.compiled.boolean

class arbitraryARCHrep():
    pass

class pipeWrapper():
    """numTar wrapper for cpu-cached classes used for organizing, transfering and handling datas before gpu processing"""
    isDefd = dict()          # Used to know wether class has been defined before
    
    def wrapper(self,*args,**kwargs):
        Specs = tarp.transfer[0]
        classType = tarp.transfer[1]

        #---------------------------------------------------------------------
        if not classType in pipeWrapper.isDefd:         # Class def algo
            tarp.pipeClassWrapped.initPipe(classType,Specs)
        #----------------------------------------------------------------------

        #print(args,kwargs,classType)

    def __call__(self,*args):
        specs = tarp.transfer
        tarp.transfer = (specs,args[0])
        return self.wrapper

class tarBankConverter():
    pass

class compTypes():
    class hierarchicIndex():
        pass

class speTypes():
    ids = 0
    def __init__(self,dof):
        self.id = speTypes.ids
        speTypes.ids += 1

class refMemStorage():
    def __init__(self : refinery):
        self.aCtM = cuda.mapped_array(self.aCtMamount,dtype=numpy.uint16)
        self.aCtMc = cuda.device_array(self.aCtMamount,dtype=numpy.uint16)
        self.aCf = cuda.mapped_array(self.aCfamount,dtype=numpy.bool_)
        self.aCfc = cuda.device_array(self.aCfamount,dtype=numpy.bool_)

class refinery(refMemStorage):
    cc_cores_per_SM_dict = {(2,0) : 32,(2,1) : 48,(3,0) : 192,(3,5) : 192,(3,7) : 192,(5,0) : 128,(5,2) : 128,(6,0) : 64,(6,1) : 128,(7,0) : 64,(7,5) : 64,(8,0) : 64,(8,6) : 128,(8,9) : 128,(9,0) : 128}
    device = cuda.get_current_device()
    my_sms = getattr(device, 'MULTIPROCESSOR_COUNT')
    my_cc = device.compute_capability
    cores_per_sm = cc_cores_per_SM_dict.get(my_cc)
    total_cores = cores_per_sm*my_sms
    locals = tarp.ntarlocals()
    refamount : int = 0

    def __init__(self) -> refinery:
        self.MainStream = cuda.stream()
        self.CommandStream = cuda.stream()
        self.Commands = cuda.mapped_array
        self.CCopyMemory = cuda.device_array
        self.isLinked = False
        #self.Blocks = refinery.total_cores
        self.PackerGenerator = tarp.pipeGenerator(self)
        self.Generator = tarp.pipeGenerator(self)
        self.Generator.root = self
        self.ObjectMode : bool
        self.aCtMamount : int
        self.aCfamount : int
        self.pipe = lambda *args,**kwargs : refinery.pipe(self,*args,**kwargs)
        self.archCommand = lambda *args, **kwargs : tarp.protocolwrapper(self,*args,**kwargs)
        self.blocks = refinery.blocks(self,)
        self.divisions = tarp.nmtCManager(self)
        # LLvmlite stuff
        self.name = "ntaref_"+str(refinery.refamount)
        refinery.refamount += 1
        self.Module = tarp.ir.Module(name=self.name,)
    
    @property
    def config(self):
        """Context manager attached to the root refinery. Allows for controlled modifying of the refinery parameters. Usage : \n
        with object.config as variable:\n
        \tobject.aCtMamount = 64000
        \twith object.config"""
        self.__ntarconfigprotocols.currentparag = self
        return self.__ntarconfigprotocols(self)    
    
    def launchCurrentConfig(self):
        self.cudaMemStorage = refinery.refMemStorage(self)
    
    def link(self,obj):
        self.__Link = obj
        self.__isLinked = True
        
    def triggerInheritance(self):   
        super().__init__()

    class pipe():
        """all those efforts to have my variable colored in green instead of yellow wtf im so dumb"""
        def __init__(self,root : refinery,*args,**kwargs):
            self.root = root
        
    class blocks(tarp.ntarsk.execenv):
        def __init__(self,root : refinery):
            super().__init__(root)
        def __getitem__(self,*args,**kwargs):
            raise refinery.handlingException("The refinery has not been initializated")

    class protocol(): # Placeholder essentially
        def __call__(*args,**kwargs):
            raise tarp.PipeWrappingError("The refinery parameters have to be defined before functions or classes can be created. Usage:\n0\twith %yourobject%.param as %variable%:\n1\t\t%variable%.aCtM = 8...")

    class taskBoard(): # Placeholder essentially
        def __init__(self,ref : refinery):
            self.tasks = numpy.zeros(())

    class blockGroup():
        """Call it as a function if used outside of a gpu compiled code. Call as method if used inside gpu context"""
        Divisions = list()
        def addDivision(divisionof):
            a = speTypes(divisionof)
            refinery.blockGroup.Divisions.append(a)
        
        def __init__(self,refin):
            refinery.blockGroup.addDivision(refin)
    
    class __ntarconfigprotocols():
        currentparag = None
        def __enter__(self):
            return self
        def __init__(self,gpu : refinery):
            self.__ntarrefinery = gpu
            self.IsObjectMode : bool
            self.ProtocolMemoryAmount : int
            self.BlockAmount : int
            self.AdminBlockAmount : int
            self.TransferMemoryAmount : int
            self.aCtMamount: int
            self.UseDoubleTMemory : bool
            self.PrintQueryMemory : int
            self.__setbuiltins()
        
        def useStandardModel(self):
            self.IsObjectMode = True
            self.ProtocolAmount = 64
            self.BlockAmount = refinery.total_cores//2
            self.AdminBlockAmount = 1
            self.TransferMemoryAmount = 64_000
            self.aCtMamount = 64_000
            self.UseDoubleTMemory = True
            self.PrintQueryMemory = 64_000
            self.GMemorySize = 4_000_000
            self.aCfamount = 64
        
        def __enter__(self):
            return self
        def __exit__(self,a,b,c):
            transfer = self.__ntarrefinery
            transfer.Commands = transfer.Commands(self.ProtocolAmount+1,dtype=numpy.bool_,wc=True)
            transfer.CCopyMemory = transfer.CCopyMemory(self.ProtocolAmount+1,dtype=numpy.bool_)
            transfer.ObjectMode = self.IsObjectMode
            transfer.aCfamount = self.ProtocolAmount
            transfer.aCtMamount = self.ProtocolMemoryAmount
            transfer.triggerInheritance()
            transfer.blocks = tarp.nmtCManager(self)
        
        class blockSpe():
            pass

            class Tasks():
                pass
        
        def __setbuiltins(self):
            @tarp.protocolwrapper(self)
            def __ntarstop():
                refinery.locals.running = 0
            print(__ntarstop)
            quit()

        def __add_default_scripts(self):
            a = tarp.pipeGenerator(self)
    
    class handlingException(BaseException):
        pass
            
class tarBank():
    """A way of storing a large quantities of same-type class objects. This way of storing datas is much faster than regular methods"""
    def __init__(self,dtype):
        self.Shape = numpy.full(1,0)
        self.dType = dtype
        try:
            dtype.__tarredformat__
        except AttributeError:
            raise tarp.PipeWrappingError("The given tar type has no __tarredformat__ attribute. If you are using a builtin python type, you might consider using regular arrays instead. tarBanks are for custom classes only")
        self.argN = len(dtype.__tarredformat__(dtype))
