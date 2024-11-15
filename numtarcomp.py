"""utilitary classes for compiling code."""

from __future__ import annotations

class tar_attribute_error(BaseException):
    def __init__(self,tartype : tarredtypederivative_cprwtr,attrstr : str):
        self.root = tartype
        self.add_note("Using tarredtypederivative object : "+str(tarredtypederivative_cprwtr)+" of class "+str(self.root.ctype))
        self.add_note("When accessing attribute of name : "+attrstr+" which could not be accessed.")
        self.add_note("Known attributes of "+str(tartype)+" are the following:")
        for item in tartype.cfs.keys():
            self.add_note(" Function"+str(self.cfs[item])+" of attrname "+item)
        super().__init__()

class tarredtypederivative_cprwtr():
    def __init__(self,classtype : type,*args,**kwargs):
        print(classtype,args,kwargs)
        quit()
        self.cfs : dict[str,function] = dict()
        self.ctype = classtype
        self.cobj = classtype(args,kwargs)
    
    def __getattr__(self,attrstr : str):
        if not attrstr in self.cfs:
            raise tar_attribute_error

class rwtr_protocol():
    def __init__(self):
        pass

class structuralclass():
    def __init__(self):
        self.prefix = str()
        self.suffix = str()
        self.isIndentIncreaser = 0

class pinnedIndex():
    def __init__(self,name : str,board : pinBoard = None):
        self.root : None | pinList = None
        self.varef = name
        self.root = None
        if not board == None:
            board.pin(self)
    
    @property
    def pIndex(self):
        return self.root[self]

class pinList():
    """A list that supports index pinning"""
    def __init__(self):
        self.main = list()
        self.pinkeys = list()
        self.pinvalues = list()
    
    def __updatepins(self,ind : int,addvalue : int):
        """Internal processing : used to update the indexes of the variables. ind argument is not included in the interval"""
        for indexindex,index in enumerate(self.pinvalues):
            if index > ind:     # <=> placer le pin après l'object car l'object lui même est fixe
                self.pinvalues[indexindex] += addvalue

    def __updatepinsinclusive(self,ind : int,addvalue : int):
        """Internal processing : used to update the indexes of the variables. ind argument is included in the interval"""
        for indexindex,index in enumerate(self.pinvalues):
            if index >= ind:    # <=> placer le pin avant l'objet car l'objet lui même est déplacé
                self.pinvalues[indexindex] += addvalue
    
    def pin(self,pin : pinnedIndex,index : int):
        """Adds a pin at the end of the list"""
        self.pinkeys.append(pin)        # Aucune update nécéssaire car le pin est à la fin
        self.pinvalues.append(index)
        pin.root = self
    
    @property
    def len(self):
        return len(self.main)

    def append(self,obj):
        """Appends object to the list. If said object is of type pinnedIndex, said pin will be assigned to the current length of the list"""
        if isinstance(obj,pinnedIndex):
            self.pin(obj,len(self.main))
        else:
            self.main.append(obj)

    def insert(self,i : int,obj):
        """Inserts a pin after the object of index i"""
        self.main.insert(i,obj)
        self.__updatepins(i,1)  # Update les indexs des objects

    def insertinclusive(self,i : int,obj):
        """Inserts a pin before the object of index i"""
        self.main.insert(i,obj)
        self.__updatepins(i,1)  # Update les indexs des objects

    def __iter__(self):
        for item in self.main:
            yield item
    
    def __getitem__(self,obj):
        if isinstance(obj,pinnedIndex):
            return self.pinvalues[self.pinkeys.index(obj)]
        elif isinstance(obj,int):
            return self.main[obj]
        else:
            raise TypeError("__getitem__ only supports pinnedIndex and int, got: "+str(obj))

class expression(structuralclass):
    def __init__(self,exp : str,locals_ : dict[str] = dict()):
        super().__init__()
        self.exp = str(eval(exp,locals_,{}))
    def __asscript__(self) -> str:
        return self.prefix+self.exp+self.suffix

class statement(structuralclass):
    def __init__(self,state : str,istarred : bool = False):
        super().__init__()
        self.state = state
        self.tarred = istarred
        self.suffix = " "
        if state in ("in",):
            self.prefix = " "
        if state in ("if","elif","with","class","def","try","except","finally","else","for","while"):
            self.isIndentIncreaser = True
        else:
            self.isIndentIncreaser = False
    def __asscript__(self) -> str:
        return self.prefix+self.state+self.suffix

class struct(structuralclass):
    def __init__(self,state : str):
        super().__init__()
        self.state = state
        self.suffix = ""
        if self.state == ":":
            self.isIndentIncreaser = 1
    def __asscript__(self) -> str:
        return self.state
    
class var(structuralclass):
    def __init__(self,name : str):
        super().__init__()
        self.access = name
        self.suffix = ""
    def __asscript__(self,):
        return self.access

class pline():
    def __init__(self):
        self.line = pinList()
    def len(self):
        return len(self.line.main)
    def construct(self,objet : tuple):
        actind = 0
        for item in objet:
            self.line.append(item)
        actind += 1
    def __asscript__(self):
        a = str()
        for item in self.line:
            a += item.__asscript__()
        return a
    def __getitem__(self,ind : int):
        return self.line.main[ind]

class generator():
    def __init__(self):
        self.content = pinList()
        self.indents = pinList()
        self.cindent = 0
    def newLine(self,noautoindent : bool = False) -> pline:
        a = pline()
        if noautoindent == False and self.content.len() >= 1 and self.content[-1].len() > 0:    # Evite l'erreur dans le cas où c'est la première ligne
            self.cindent += self.content[-1][-1].isIndentIncreaser                              # Modifie l'indent si besoin
        self.content.append(a)
        self.indents.append(self.cindent)
        return a
    def decreaseIndent(self):
        self.cindent -= 1

    def __asscript__(self) -> str:
        a = str()
        for ind,item in enumerate(self.content):
            a = a + "    "*self.indents[ind] + item.__asscript__()+"\n"
        return a
    def iterlines(self):
        for ind,item in enumerate(self.content):
            yield (self.indents[ind], self.content[ind])
    
    def paste(self,topaste : generator,isindentstatic : bool = False) -> generator:
        """Paste a generator's code on to this one. isindentstatic arguments is to modify if you want the code's indent to be the same as the pasted code's end or to be reset afterwards"""
        ogindent = self.cindent
        for strindent,strline in topaste.iterlines():
            if isindentstatic == False:
                self.cindent = ogindent + strindent                 # Modifie l'indent manuellement dans le cas où la fonction veut l'indent original après.
            self.newLine(noautoindent = True).construct(strline)    # Utilise sans l'autoindent car le code a déjà ses propres indents
        return self

class varepository():
    def __init__(self):
        self.repo : dict[str,var] = dict()
        self.varwrappers : dict[str,tarredtypederivative_cprwtr] = dict()
    def addvaref(self,varname : str,**carac):
        a = var(carac)
        self.repo[varname] = a
        return a
    def __getitem__(self,varname : str):
        return self.repo[varname]
    def hasItem(self,varname):
        return (varname in(self.repo))

class pinBoard():
    class pinRep():
        def __init__(self,varef : pinnedIndex):
            self.varef = varef
        
        def relativeinsert(self,offset : int,object):
            self.varef.root.insert(self.varef.pIndex+offset,object)
        
        def prepend(self,object):
            self.varef.root.insertinclusive(self.varef.pIndex,object)

    def __init__(self):
        self.pins = dict()
    def pin(self,obj : pinnedIndex):
        self.pins[obj.varef] = obj
    def __getitem__(self,key : str):
        return pinBoard.pinRep(self.pins[key])

