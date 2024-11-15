"""Module for tarp pipe class compilation/interpretation"""
from __future__ import annotations
import numba
from numba import cuda
import numpy
from math import *
import traceback
import inspect
import sys
import os
import numtarcomp as ntarcp
import ntarskript as ntarsk

class pipeLog():
    def __init__(self):
        self.logs = list()
    
    def log(self,strr):
        self.logs.append(strr)
    
    def print(self):
        for log in self.logs:
            print(log)

logs = pipeLog()

#=============== v classes utilitaires v ================

class PipeWrappingError(BaseException):
    def raiseErr(reason):
        raise PipeWrappingError(reason)

    def __init__(self,traceback):
        print(traceback)

    def __call__(self,traceback):
        print(traceback,"called")

@ntarcp.tarredtypederivative_cprwtr
class ntarlocals():
    blockAmount : int
    threadAmount : int
    threadId : int
    blockId : int
    gridId : int
    
    def __init__(self):
        return

    running = ntarsk.cpdvar(ntarsk.tarsgn(ntarsk.isp.Parameter("running",ntarsk.isp._ParameterKind(0),annotation=bool)))

#=============== ^ classes utilitaires ^ ================

#============== v classes de compilation v ==============

class nmtCManager():
    def __init__(self,root):
        self.root = root
    def __getitem__(self,key : int | slice):
        if not isinstance(key,(int,slice)):
            raise TypeError("__getitem__ only supports slices or ints.")
        return ntarsk.cntxt_gntr(ntarsk.execenv(self.root))

class protocolwrapper():
    def astarp(root,function):
        return protocolwrapper(root)(function)
    """Wrapper for writing kernel commands."""
    def __init__(self,root):
        self.root = root
    def __call__(self,func : function):
        self.wfunc = func
        self.cfunc = ntarsk.fctncp(ntarsk.compiled.boolean)  # C'est un décorateur à la base
        with ntarsk.skinter_error_manager(self.cfunc,self.wfunc):                                   # Ce manageur de contexte sert à customiser l'erreur pour l'utilisateur
            self.cfunc(self.wfunc)                                                                  # Call manuellement le call qui se fait à la décoration de la fonction
                                                                                                    # La fonction n'est pas encore compilée au moment du call
        self.__call__ = self.__rcall__  # Remplace le callinit par le vrai call de la fonction
        raise NotImplementedError("to do")
        return self
    def __rcall__(self,*args,**kwargs):
        return self.cfunc(*args,**kwargs)

class pipeGenerator():
    GeneratedF = 0
    def __init__(self,root):
        self.name = "ntarFx"+(3-len(str(pipeGenerator.GeneratedF)))*"0"+str(pipeGenerator.GeneratedF)
        self.root = root
        pipeGenerator.GeneratedF += 1
        self._adminblock = ntarcp.generator()
        self._code = ntarcp.generator()
        self._packer = ntarcp.generator()
        self.ProcessRefs = ntarcp.pinBoard()
        self.protocolAmounts : int = 0
    
    def addprotocol(self,protocol : ntarcp.generator):
        self._adminblock.newLine().construct((ntarcp.statement("if"),ntarcp.struct("protocolCopyMemory[%protocolid%] == True".replace("%protocolid%",str(self.protocolAmounts))),ntarcp.struct(":")))
        self._adminblock.paste(protocol,isindentstatic=True)
        self.protocolAmounts += 1
    
    def generateMainKernel(self):
        self.__generateadmin()
        self.__generatemain()
        self._code.paste(self._adminblock)
        self.__generatepacker()
        return self._code.__asscript__()

    def __generateadmin(self):
        self._adminblock.newLine().construct((ntarcp.struct("if cuda.blockIdx == "),ntarcp.expression("root.BlockAmount",{"root":self.root}),ntarcp.struct(":")))
        self._adminblock.newLine().construct((ntarcp.statement("for"),ntarcp.var("commandIndex"),ntarcp.statement("in"),ntarcp.var("range"),ntarcp.struct("("),ntarcp.expression("root.aCfamount",{"root":self.root}),ntarcp.struct(")"),ntarcp.struct(":")))
        self._adminblock.newLine().construct((ntarcp.var("protocolCopyMemory"),ntarcp.struct("[commandIndex]"),ntarcp.statement("="),ntarcp.var("kernelmodes"),ntarcp.struct("[commandIndex]")))
        self._adminblock.decreaseIndent()
        self._adminblock.newLine().construct((ntarcp.statement("if"),ntarcp.struct("protocolCopyMemory[0] == True"),ntarcp.struct(":")))
        self._adminblock.newLine().construct((ntarcp.var("running"),ntarcp.statement("="),ntarcp.expression("0")))

    def __generatemain(self):
        pref = self.ProcessRefs
        toedit = self._code.newLine()
        toedit.construct(
            (ntarcp.struct("@cuda.jit"),ntarcp.struct("("),ntarcp.pinnedIndex("cudajitarg",pref),ntarcp.struct(")"))  )
        toedit = self._code.newLine()
        toedit.construct(
            (ntarcp.statement("def"),ntarcp.struct(self.name),ntarcp.struct("("),ntarcp.var("kernelmodes"),ntarcp.struct(","),ntarcp.var("protocolCopyMemory"),ntarcp.struct(","),ntarcp.var("actmp"),ntarcp.struct(","),ntarcp.var("actm"),ntarcp.struct(","),ntarcp.pinnedIndex("defargs",pref),ntarcp.struct(")"),ntarcp.struct(":"))
        )
        if self.root == True:
            pass
        toedit = self._code.newLine()
        toedit.construct((ntarcp.var("running"),ntarcp.statement("="),ntarcp.expression("True")))
        toedit = self._code.newLine()
        toedit.construct((ntarcp.statement("while"),ntarcp.var("running"),ntarcp.struct("=="),ntarcp.expression("True"),ntarcp.struct(":")))
        self._code.newLine().construct((ntarcp.statement("cudaGridIndex = cuda.grid(1)"),))
    
    def __generatepacker(self):
        self._packer.newLine().construct((ntarcp.statement("def"),ntarcp.var("launch"),ntarcp.struct("("),ntarcp.var("self"),ntarcp.struct(","),ntarcp.struct(")"),ntarcp.struct(":"))) 

class compilation_protocol():
    """\nDefault methods :\n
    self@compilation_protocol.pin : creates a pin from a variable's name
    self@compilation_protocol.isDefd : returns a boolean of if the varname has already been defined
    self@compilation_protocol.asVaref : converts a varname into a ntarcomp var object and references it
    """
    def __init__(self,mode : str = "classic"):
        self._generator = ntarcp.generator()
        self.varep = ntarcp.varepository()
        self.cntxtdata = None
        self.__pinb = ntarcp.pinBoard()
        self.mode = mode
        
    def isDefd(self,varname : str):
        return self.varep.hasItem(varname)
    
    def asVaref(self,varname : str):
        if self.isDefd(varname):
            return self.varep[varname]
        else:
            return self.varep.addvaref(varname)
        
    def pin(self,varname : str):
        a = ntarcp.pinnedIndex(varname,self.__pinb)
        self._generator.content.pin(a,self._generator.content.len)
        self.__pinb.pin(a)

protocolCompiler = compilation_protocol(mode="protocol")