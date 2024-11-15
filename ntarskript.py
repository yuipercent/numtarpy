"""Compilation utils for the interpretation of code"""

from __future__ import annotations
from bdb import set_trace
import collections
import types
from numtarcomp import *
import inspect as isp
import traceback

class skinter_errors():
    class interpretation_error(BaseException):
        def __init__(self,root : skinter_error_manager,):
            self.root = root
            cntxt : fctncp = self.root.rtcntxt
            self.add_note("   Function "+str(cntxt.Decor)+" of signature "+str(cntxt.ispSign))
            self.add_note("   of tarsign "+str(cntxt.decorSignature)+" using the following dvars :")
            dummyvars = cntxt.dummyvars
            for item in dummyvars:
                item : cpdvar
                self.add_note("      var "+item.tarsign.parameter.name+": "+str(item.tarsign.parameter.annotation))
            super().__init__("An exception occured during the handling of the script interpretation of :")
    
    class context_management_error(BaseException):
        def __init__(self,root : context_repo):
            self.root = root
            self.add_note("   Error when trying to update current context "+str(self.root))
            self.add_note("   The context has already been closed and attached to "+str(self.root.ccntxt))
            super().__init__()

class skinter_error_manager():
    def __init__(self,rootcontext : fctncp,rootfunc : function):
        self.rtcntxt = rootcontext
        self.rtfunc = rootfunc
    def __enter__(self):
        return None
    def init_skinter_error(self,skerror : type):
        return skerror(self)
    def __exit__(self,errort,err,tracebck):
        if not err == None:
            raise self.init_skinter_error(skinter_errors.interpretation_error) from err

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

# =======================================

class bhlog():
    """template class for events"""
    pass

class event():
    """Section that contains all the event that the skript interpreters can yield to the context manager"""
    class assignement(bhlog):
        def __init__(self,varef : structuralclass,expression : structuralclass):
            self.varef = varef
            self.expression = expression
    class call(bhlog):
        def __init__(self,func : function,args : tuple = (),kwargs : dict = {}):
            self.args = args
            self.kwargs = kwargs
            self.calledfunc = func
        def __str__(self):
            return "<call of function "+str(self.calledfunc)+" with args : "+str(self.args)+", "+str(self.kwargs)+">"
    class invoque(bhlog): 
        def __init__(self,obj):
            self.varef = obj
    class using_start(bhlog):
        def __init__(self,eenv : execenv):
            pass
    class using_stop():
        def __init__(self,eenv : execenv):
            pass

class tarsgn():
    generictypes = {
        bool : tarredtypederivative_cprwtr(bool),
        int : tarredtypederivative_cprwtr(int)
    }
    """A class for storing the presumed signature of dummy vars"""
    def __init__(self,parameter : isp.Parameter):
        self.parameter = parameter
    def from_parameter(parameter : isp.Parameter):
        return tarsgn(parameter)

class cpdvar():
    """Compilation dummyvar : dummy variable used for the interpretation of script"""
    def __init__(self,signature : tarsgn,tags : list[str] = list()):
        self.tarsign = signature
        self.logs : list[event] = list()
        self.tags : list[str] = tags
        fctncp.skcntxt.refndvar(self)
    def tag(self,atag : str):
        self.tags.append(atag)
        return self
    def remove_tag(self,atag : str):
        if atag in self.tags:
            self.tags.pop(self.tags.index(atag))
            return True
        else:
            return False
    def __add__(self,to):
        fctncp.skcntxt.yieldsk((event.call(self.__add__,to),))
        return self
    def __div__(self,to):
        print("nope2")
    def __mul__(self,to):
        print("on m'a multiplié par",to,"gros fdp")

class classm_extractor():
    """A class that stores information about the signatures of a function"""
    def __init__(self,signs : isp.Signature,rootcntxt : fctncp):
        self.signs = classm_extractor.__extract__(signs)
        self.returntype = tarsgn.from_parameter(signs.return_annotation)
        self.rtcntxt = rootcntxt
    
    def __extract__(signs : isp.Signature) -> collections.OrderedDict:
        """internal processing, converts isp signatures into ntar compatibles classess"""
        tararg = collections.OrderedDict()
        for item in signs.parameters.keys():
            argdict : collections.OrderedDict = signs.parameters
            parameter : isp.Parameter = argdict[item]
            assert isinstance(parameter.annotation,(types.GenericAlias,type))
            tararg[parameter.name] = tarsgn.from_parameter(parameter)
        return tararg
    
    @property
    def dummy(self):
        """Returns arguments as dummy vars to pass to functions"""
        toreturn = list()
        for arg in self.signs.values():
            toreturn.append(cpdvar(arg,["argvar"]))
        return tuple(toreturn)

class execenv():
    """Template class for gpu execution environments (blocks, threads)"""
    def __init__(self,root):
        self.root = root

class context_layer():
    """A context layer"""
    def __init__(self, root : None | context_layer = None):
        self.Layer = list()
        self.root = root
        self.dvars : list[cpdvar] = list()
        
    def yieldsk(self,obj):
        self.Layer.append(obj)
    def __iter__(self):
        for item in self.Layer:
            yield item

class context_repo():
    """A list of the context layers"""
    def __init__(self,ccntxt : context_layer = context_layer()):
        self.ccntxt = ccntxt
        self.savedcntxt : dict[function,context_layer] = dict()
        self.isClosed : bool = False

    def newcntxt(self):
        a = context_layer(self)    # Variable problématique
        # Potentielle issue dans le futur, problème a été résolu mais jsp si ça causera des problèmes dans le futur donc voilà
        self.ccntxt = a
        self.isClosed = False

    def yieldsk(self,obj : tuple):
        if self.isClosed:
            raise skinter_errors.context_management_error(self)
        for item in obj:
            self.ccntxt.yieldsk(item)

    def closeccntxt(self):
        a = self.ccntxt
        self.ccntxt = self.ccntxt.root
        self.isClosed = True
        return context_repo(a)

    def saveccntxt(self,key : function):
        self.savedcntxt[key] = self.ccntxt

    def refndvar(self,ndvar : cpdvar):
        self.ccntxt.dvars.append(ndvar)

    def __iter__(self):
        for item in self.ccntxt:
            if isinstance(item,context_layer):
                yield (item,1)
            else:
                yield (item,0)

class fctncp():
    """A decorator to organize the function compilation. Use fctncp.skcntxt.yieldsk to report events."""
    skcntxt = context_repo()
    def __init__(self,*args : list[compiledTypes],):
        self.CatchedDecor : None | context_repo = None
        self.Decor : None | function = None
        self.CompiledDecor : generator | None = None

    def __call__(self,decor : function,):
        fctncp.skcntxt.newcntxt()
        
        self.Decor : function = decor
        self.ispSign = isp.signature(decor)
        sign = classm_extractor(self.ispSign,self)
        self.decorSignature = sign
        execarg = sign.dummy
        self.dummyvars = execarg
        decor(*execarg)
        
        fctncp.skcntxt.saveccntxt(self.Decor)
        self.CatchedDecor = fctncp.skcntxt.closeccntxt()
        self.__call__ = self.__compcall__
        return self.CatchedDecor

    def __compcall__(self,decor : function,*args,**kwargs):
        fctncp.skcntxt.yieldsk((event.invoque(decor),event.call(*args,**kwargs)))

class cntxt_gntr():
    """A class for generating contexts during ntar compiled functions"""
    def __init__(self,eenv : execenv):
        self.eenv = eenv
    
    def __enter__(self,):
        self.cevent = event.using_start(self)
        fctncp.skcntxt.yieldsk((self.cevent,))
        return cpdvar(tarsgn(int))
    def __exit__(self,a : BaseException | None,b : str | None,c):
        if not a == None:
            raise b
        fctncp.skcntxt.yieldsk((event.using_stop(self.cevent),))
# A finir