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

global transfer

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

class numtarTypes():
    def __init__(self,t):
        self.type = t
    
    def __str__(self):
        return "numtarpipe.numtarTypes "+self.type+" type object"

class compiledTypes(numtarTypes):
    def __init__(self,t,space : int):
        self.space = space
        super().__init__(t)
    class nparray(numtarTypes):
        def __init__(self,filltype,fillvalue = None):
            super().__init__("nparray")
            self.ArrayType = filltype
            self.fillValue = fillvalue
            self.__name__ = "nparray"
    class func(numtarTypes):
        def __init__(self,kwargs):
            self.attrdict = {"parallel":True,"nogil":True,"fastmath":False}
            super().__init__("func")
            for kword in kwargs.keys():
                self.attrdict[kword] = kwargs[kword]

class tarredTypes(numtarTypes):
    def __init__(self,t):
        super().__init__(t)
    class nparray(numtarTypes):
        def __init__(self,filltype,fillvalue = None):
            self.ArrayType = filltype
            self.fillValue = fillvalue
            self.__name__ = "nparray"

    class func(numtarTypes):
        def __init__(self,kwargs):
            self.attrdict = {"parallel":True,"nogil":True,"fastmath":False}
            super().__init__("func")
            for kword in kwargs.keys():
                self.attrdict[kword] = kwargs[kword]

class compiled(numtarTypes):
    def nparray(type_,):
        return compiledTypes.nparray(type_)
    def func(**kwargs):
        """Arguments can be passed to modify the compilation behavior. Acceptable arguments are arguments accepted by the numba njit compilator :\n'_dbg_extend_lifetimes', '_dbg_optnone', '_nrt', 'boundscheck', 'debug', 'error_model', 'fastmath', 'forceinline', 'forceobj', 'inline', 'looplift', 'no_cfunc_wrapper', 'no_cpython_wrapper', 'no_rewrites', 'nogil', 'nopython', 'parallel', 'target_backend' \n\nand nmtar specific arguments such as 'pdegree',"""
        return compiledTypes.func(kwargs)
    
    boolean = compiledTypes("boolean",space=1)
    int32 = compiledTypes("int32",space=32)
    int64 = compiledTypes("int64",space=64)
    int16 = compiledTypes("int16",space=16)
    int8 = compiledTypes("int8",space=8)
    uint64 = compiledTypes("uint64",space=64)
    uint32 = compiledTypes("uint32",space=32)
    uint16 = compiledTypes("uint16",space=16)
    uint8 = compiledTypes("uint8",space=8)
    float32 = compiledTypes("float32",space=32)
    float64 = compiledTypes("float64",space=64)
    complex64 = compiledTypes("complex64",space=64)
    complex128 = compiledTypes("complex128",space=128)

class tarred(numtarTypes):
    def nparray(type_):
        return tarredTypes.nparray(type_)

    func = tarredTypes("compiledFunc")
    boolean = tarredTypes("boolean")
    int32 = tarredTypes("int32")
    int64 = tarredTypes("int64")
    int16 = tarredTypes("int16")
    int8 = tarredTypes("int8")
    uint64 = tarredTypes("uint64")
    uint32 = tarredTypes("uint32")
    uint16 = tarredTypes("uint16")
    uint8 = tarredTypes("uint8")
    float32 = tarredTypes("float32")
    float64 = tarredTypes("float64")
    complex64 = tarredTypes("complex64")
    complex128 = tarredTypes("complex128")

class PipeWrappingError(BaseException):
    def raiseErr(reason):
        raise PipeWrappingError(reason)

    def __init__(self,traceback):
        print(traceback)

    def __call__(self,traceback):
        print(traceback,"called")

class PipeCompilationError(BaseException):
    pass

#=============== ^ classes utilitaires ^ ================

#============== v classes de compilation v ==============

class pipeSpecs():
    def __init__(self,specs,nowarn : bool = False):
        global transfer
        self.nowarn = nowarn
        self.specs = self.__interpret(specs)
        self.Attributes = self.specs[0]
        self.AttrTypes = self.specs[1]
        transfer = self
        self.toremove = list()
    
    def warn(self,warn):
        if self.nowarn == False:
            print(warn) 
    
    def __iter__(self):
        toreturn = list()
        for p in range(0,len(self.Attributes)):
            yield (self.Attributes[p],self.AttrTypes[p],p)
    
    def whatis(self,string):
        return self.AttrTypes[self.Attributes.index(string)]

    def isin(self,string):
        return string in self.Attributes

    def pop(self,i):
        self.Attributes.pop(i)
        return self.AttrTypes.pop(i)

    def currentstate(self,funcname_):
        """returns state of compilation"""
        funcname_ = funcname_.replace("self.","")
        if self.isin(funcname_):
            if isinstance(self.whatis(funcname_),(compiledTypes.func,compiledTypes.nparray)):
                currentstate = 1
            elif isinstance(self.whatis(funcname_),(tarredTypes.func,tarredTypes.nparray)):
                currentstate = 2
            else:
                raise PipeCompilationError("Unrecognized spec encountered during latter part of compilation :\n    with "+funcname_+" of type "+str(self.whatis(funcname_))+"\nThis error is likely not your fault, please report this to the developers")
        else:
            currentstate = 0
        return currentstate
    
    def __interpret(self,specs):
        toconstruct0 = list()
        toconstruct1 = list()
        if not isinstance(specs,(list,tuple)):  # Check if decorator was called
            raise PipeWrappingError("The pipeWrapper decorator has to be called")
        
        indexx = 0
        try:

            for index,spec in enumerate(specs):
                indexx = index
                if isinstance(spec[1],type(compiled.func)):
                    spec[1] = spec[1]()         # Corrects the code if func was not called
                if not isinstance(spec[1],(numtarTypes)):
                    PipeWrappingError.raiseErr("Error with spec interpretation at "+str(spec)+", spec line "+str(index)+"\n   Attribute type setter is of type "+spec[1].__name__+" while a numtar native type was expected")
                toconstruct0.append(spec[0])    # Construct spec list
                toconstruct1.append(spec[1])    # Construct spec list
        
        except Exception as error:
            raise PipeWrappingError(str(error)+", malformed spec in line "+str(indexx)+' of spec, here would be a correct example :\n@nmtar.pipeWrapper([\n\t("__rworld",nmtar.compiled.nparray(nmtar.compiled.int16)),\n\t("__aworld",nmtar.compiled.nparray(nmtar.compiled.boolean)),\n\t("__a2world",nmtar.compiled.nparray(nmtar.compiled.boolean)),\n\t("indexFreeChunk",nmtar.compiled.func(parallel=True,pdegree = 1)),\n],nowarn=True)')
        return toconstruct0, toconstruct1

class pipeCode():
    varchar = frozenset(("a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z","A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z","_"))
    varcharn = frozenset(("1","2","3","4","5","6","7","8","9","0"))
    statements = frozenset(("class","def","return","pass","yield","raise","for","in","and","or","while","continue","break","try","except","finally","if","else","elif"))
    operators = frozenset(("=","*","%","+","-","/"))
    openingbrack = frozenset(("(","[","{"))
    closingbrack = frozenset((")","]","}"))
    brackequ = {"(":")","[":"]","{":"}"}
    brackequ2 = {"(":0,"[":1,"{":2}
    tarstatements = frozenset(("each","among"))
    exceptions = frozenset(("False","True"))     
    __authorized = {"pi":pi,"e":e,"tarredTypes":tarredTypes,"compiledTypes":compiledTypes,"tarred":tarred,"compiled":compiled,"cos":cos,"sin":sin,"tan":tan,"atan":atan,"acos":acos,"asin":asin}
    
    def __init__(self,code : any = list()):
        self.code = code
        self.functions = dict()
        self.indent = list()
        self.pinnedIndex = list()
        self.doIndentDecrease = 0
        a = 0

    def newLine(self,index : int = -1):
        a = ntarcp.pline(self)
        if not len(self.code) == 0:
            cindent = self.indent[index] + self.code[index].isIndentIncreaser() - self.doIndentDecrease
            self.doIndentDecrease = 0
            self.code.insert(index,a)
            self.indent.insert(index,cindent)
        else:
            self.code.append(a)
            self.indent.append(0)
        return a
    
    def decreaseIndent(self):
        self.doIndentDecrease = 1
    
    def convert(self):
        """Converts a pipeCode object composed of str list into formatted pipeCode"""
        for index,line in enumerate(self.code):             # Get l'index où le code de la classe commece
            if line.strip()[:5] == "class":
                a = index
                break
            
        self.code = self.code[a:]                           # Cut le decorateur
        self.indent = [0 for k in range(0,len(self.code))]  # Créé la liste des indents
        d = pipeCode.countindent(self.code[1])              # Get le nombre de indent de la classe
        # ========== Decoupe la str pour chaque ligne ====================================
        for index,line in enumerate(self.code):
            self.code[index], self.indent[index] = self.codetolist(line,d)
            
        for index,line in enumerate(self.code):
            self.convertl(line,index)

    def convertl(self,line : list[str],index : int = -1,recurp : list[int] = [0,0,0]) -> ntarcp.pline:
        """Args correspond to: Line to be converted, Index of line in full code, Recursion levels for each bracket type"""
        recurpcopy = recurp[:]
        # Gère le indent si la fonction est executée récursivement
        if index < 0:
            cline = ntarcp.pline(0)
        else:
            cline = ntarcp.pline(self.indent[index])
        cstate = 0
        cline.setscope(self.env)                                # Met en commun les 2 scopes

        # Commence le processing de la liste
        for i,part in enumerate(line):
            if part in pipeCode.openingbrack:                       # Cas où c'est un bracket
                a = self.select(index,part,recurpcopy)                  # Recupère la sous partie du code
                recurpcopy[ pipeCode.brackequ2[ part ] ] += 1           # Modifie le niveau de récursion en préparation
                cline.construct( self.convertl(a,-1,recurpcopy) )       # Convertie la subpart
            elif part in pipeCode.statements:                       # Cas où c'est un statement
                a = ntarcp.statement(part)
            elif part in pipeCode.tarstatements:                    # Cas où c'est un statement custom
                a = ntarcp.statement(part,True)
            else:                                                   # Cas où c'est une var
                pass
    
    def select(self,lineindex : int,btype : str = '(',recurp_ : list[int] = [0,0,0]):
        b = list()
        itd = -1
        itp = False
        for l in range(lineindex,len(self.code)):
            for part in self.code[l]:
                if part == btype:
                    itd += 1
                if itd >= recurp_[pipeCode.brackequ2[btype]]:
                    itp = True
                    b.append(part)
                if part == pipeCode.brackequ[btype]:
                    itd -= 1
                    if itp == True and itd+1 == recurp_[pipeCode.brackequ2[btype]]:
                        return b[1:-1]
        return b[1:-1]


    @property
    def countindent(line : str,) -> int:
        """Returns indentation value of raw code line"""
        line = line.replace("\t","    ")
        for p in range(0,len(line)):
            if not line[p] == " ":
                return p
        return 0

    def __isvar(self,str_,charstate):
        if charstate == 1:
            if not str_ in pipeCode.statements and not str_ == "":
                self.env.vars.add(str_)

    def codetolist(self,line,d):
        a = list()
        b = 0
        charstate = 0
        previouscharstate = 99
        prei = 0
        pd = False
        for i in range(0,len(line)):
            actl = line[i]
            if charstate >= 0:
                if actl == "#":
                    break
                elif actl in pipeCode.openingbrack:
                    charstate += 50
                elif actl in pipeCode.closingbrack:
                    charstate += 50
                elif actl in pipeCode.varchar:
                    charstate =1
                elif actl in (":",";"," "):
                    charstate = 3
                elif actl in pipeCode.operators:
                    charstate = 4
                elif charstate == 1 and actl in pipeCode.varcharn:
                    charstate = 1
                elif actl in ("\"","'"):
                    charstate = -1
                elif actl in pipeCode.varcharn:
                    charstate = 5
                else:
                    charstate = 2
            else:
                if actl in ("\"","'"):
                    charstate = 0
            if not charstate == previouscharstate:
                if previouscharstate == -1:
                    pd = line[prei:i]
                else:
                    pd = line[prei:i].strip()
                if not pd == str():
                    a.append(pd)
                    self.__isvar(pd,previouscharstate)
                previouscharstate = charstate
                prei = i
        a.append(line[prei:])
        self.__isvar(pd,previouscharstate)
        b = pipeCode.countindent(line)//d
        return a,b
        
    def authorize(a):
        if a in (os,sys):
            raise PipeCompilationError("os and sys modules cannot be added to global scope")
        elif a in (numpy,):
            raise PipeCompilationError("The numpy module cannot be modified")
        pipeCode.__authorized[a.__name__] = a
    
    def leval(line : str,localvars : dict = {},istesting : bool = False):
        try:
            a = eval(line,pipeCode.__authorized,localvars)
            return (None,a)
        except BaseException as err:
            if istesting == False:
                raise PipeCompilationError(str(err)+" :\n   with '"+line.strip()+"'\n   To add a variable to the global scope of compilation, you can use the nmtar.tarp.pipeCode.authorize function")
            err0 = str(err)
            if "'" in err0:
                a = err0.index("'")+1
                return (type(err),err0[a:err0.index("'",a)])
            else:
                return type(err),err0
    
    def getitem(self,i):
        return self.code[i]
    
    def __iter__(self,):
        for lindex,line in enumerate(self.code):
            yield (line,lindex)

class pipeClassWrapped():
        
    def __init_subclass__(cls) -> None:
        pass
    
    def decompose(source,specs):
        """Decompose source code into a list of all function's code"""
        toreturn = list()
        tarredvars = list()
        compiledvars = list()
        csource = source.replace("\t","   ")
        
        while "\n" in csource:      # Decoupe le code en liste de lignes
            if not csource.strip() == "":
                toreturn.append(csource[:csource.index("\n")])
                csource = csource[csource.index("\n")+1:]
            else:
                csource = csource[csource.index("\n")+1:]
        toreturn.append(csource)

        for AttrName,Spec,i in specs:               # Separe dans des listes à part les trucs tarred ou compilés
            if isinstance(Spec,tarredTypes):
                tarredvars.append((AttrName,Spec))
            elif isinstance(Spec,compiledTypes):
                compiledvars.append((AttrName,Spec))
        return pipeCode(toreturn).convert(), pipeSpecs(tarredvars,True), pipeSpecs(compiledvars,True)

    def initPipe(name,specs_ : pipeSpecs):
        classObj = inspect.getsource(name)
        t = pipeClassWrapped.decompose(classObj,specs_)
        Codes = t[0]
        Tarvar = t[1]
        Compvar = t[2]
        if Tarvar == list() and specs_.nowarn == False:    # Warning for beginners :ddd
            print("numtarpipe.pipeWrapperWarning: wrapped class does not contain any tarred value and will therefore not be able to send any data to refinery")
        rDfdcodes = list()
        cctor = "@numba.njit()\n\tclass "+name.__name__+"(pipeClassWrapped):\n"
        
        ctor = "\tclass "+name.__name__+"():\n"
        print(Codes)

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