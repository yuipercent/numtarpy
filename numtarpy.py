"""
numTar
======
Module used to create GPU executable functions on the nmtory Workboard model\n\n
Provides:\n
- 1 : A module for simplifying gpu code writing
- 2 : A way of vizualizing and controlling in real time data treatment\n
GET STARTED : Refinery class :
------------------------------
instantiate a refinery type object to get started. The refinery type object is a representation of the GPU.\n
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

class tarred():
    nparray = tarp.tarred.nparray
    func = tarp.tarred.func
    int32 = tarp.tarred.int32
    int64 = tarp.tarred.int64
    int16 = tarp.tarred.int16
    int8 = tarp.tarred.int8
    uint64 = tarp.tarred.uint64
    uint32 = tarp.tarred.uint32
    uint16 = tarp.tarred.uint16
    uint8 = tarp.tarred.uint8
    float32 = tarp.tarred.float32
    float64 = tarp.tarred.float64
    complex64 = tarp.tarred.complex64
    complex128 = tarp.tarred.complex128
    boolean = tarp.tarred.boolean

class compiled():
    nparray = tarp.compiled.nparray
    func = tarp.compiled.func
    int32 = tarp.compiled.int32
    int64 = tarp.compiled.int64
    int16 = tarp.compiled.int16
    int8 = tarp.compiled.int8
    uint64 = tarp.compiled.uint64
    uint32 = tarp.compiled.uint32
    uint16 = tarp.compiled.uint16
    uint8 = tarp.compiled.uint8
    float32 = tarp.compiled.float32
    float64 = tarp.compiled.float64
    complex64 = tarp.compiled.complex64
    complex128 = tarp.compiled.complex128
    boolean = tarp.compiled.boolean

class ntutil():
    methodclasses = dict()
    class methodclass_supporter_parent():
        def __init__(self,*args,**kwargs):
            ntutil.methodclasses[self] = frozenset(args)

        def __call__(self,wrapped : type | None = None,*args,**kwargs):
            if wrapped == None:
                return self
            wrapped = wrapped(*args,**kwargs)
            self.__module__ = wrapped.__module__
            self.__doc__ = wrapped.__doc__
            self.__dict__ = wrapped.__dict__
            self.wrapped = wrapped
            return self
        
        def __getattr__(self,attrstr,):
            attr = getattr(self.wrapped,attrstr)
            if isinstance(attr,type) and attrstr and attrstr in ntutil.methodclasses[self]:
                return lambda *args,**kwargs : attr(self.wrapped,*args,**kwargs)
            return attr
    def methodclass_addSupport(wrappedclass : type,*args,**kwargs):
        return wrappedclass
setattr(ntutil,"methodclass_addSupport",ntutil.methodclass_supporter_parent)

class arbitraryARCHrep():
    pass

class pipeWrapper(tarp.pipeSpecs):
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

class nmtCManager():
    def __init__(self,root):
        self.root = root

class refMemStorage():
    def __init__(self : refinery):
        self.aCtM = cuda.mapped_array(self.aCtMamount,dtype=numpy.uint16)
        self.aCtMc = cuda.device_array(self.aCtMamount,dtype=numpy.uint16)
        self.aCf = cuda.mapped_array(self.aCfamount,dtype=numpy.bool_)
        self.aCfc = cuda.device_array(self.aCfamount,dtype=numpy.bool_)

class ntarlocals():
    blockAmount : int
    threadAmount : int
    threadId : int
    blockId : int
    gridId : int
    
    def __init__(self):
        return
    
    @property
    def running(self):
        return
    
    @running.setter
    def running(self):
        print("g")
        yield "YOURMOM"

@ntutil.methodclass_addSupport("pipe","blocks")
class refinery(refMemStorage):
    cc_cores_per_SM_dict = {(2,0) : 32,(2,1) : 48,(3,0) : 192,(3,5) : 192,(3,7) : 192,(5,0) : 128,(5,2) : 128,(6,0) : 64,(6,1) : 128,(7,0) : 64,(7,5) : 64,(8,0) : 64,(8,6) : 128,(8,9) : 128,(9,0) : 128}
    device = cuda.get_current_device()
    my_sms = getattr(device, 'MULTIPROCESSOR_COUNT')
    my_cc = device.compute_capability
    cores_per_sm = cc_cores_per_SM_dict.get(my_cc)
    total_cores = cores_per_sm*my_sms
    locals = ntarlocals()

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
        def __init__(self,root : refinery):
            self.root = root
    
    class archCommand(tarp.pipeSpecs):
        """Wrapper for writing kernel commands."""
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

    class blocks():
        pass

    class archCommand():
        def __call__(*args,**kwargs):
            print(args,kwargs)
            raise tarp.PipeWrappingError("The refinery parameters have to be defined before functions or classes can be created. Usage:\n0\twith %yourobject%.param as %variable%:\n1\t\t%variable%.aCtM = 8...")

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
        
        class blockSpe():
            pass

            class Tasks():
                pass
        
        class _script_behavior_analyser():
            def __init__(self,compilationmode : str,*args : list[tarp.compiledTypes],):
                self.Catcher = list()
                self.Space = 0
                if compilationmode == "protocol":
                    for arg in args:
                        if not isinstance(arg,tarp.compiledTypes):
                            raise TypeError("Argument of an a protocol must be numtarpy compiled types")
                        self.Space += arg.space
            def __call__(self,decor : function,):
                self.Decor = decor
                with self as param:
                    self.Decor()
                    quit()
                raise TypeError("UWULAND")
            
            def __enter__(self):
                pass

            def __exit__(self,a,b,c):
                print(a,b,c)
        
        def __add_default_scripts(self):
            a = tarp.pipeGenerator(self)
        
        def __setbuiltins(self):
            @self._script_behavior_analyser("protocol",compiled.boolean)
            def __ntarstop():
                refinery.locals.running = False

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



