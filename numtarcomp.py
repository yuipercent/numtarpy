from __future__ import annotations
from typing import Callable,Concatenate

class structuralclass():
    def __init__(self):
        self.prefix = str()
        self.suffix = str()
        self.isIndentIncreaser = 0

class pinnedIndex():
    def __init__(self,name : str,board : pinBoard = None):
        self.root : None | pinList
        self.varef = name
        self.root = None
        if not board == None:
            board.pin(self)
    
    @property
    def pIndex(self):
        return self.root[self]

class pinList():
    def __init__(self):
        self.main = list()
        self.pinkeys = list()
        self.pinvalues = list()
    
    def __updatepins(self,ind : int,addvalue : int):
        for indexindex,index in enumerate(self.pinvalues):
            if index > ind:
                self.pinvalues[indexindex] += addvalue

    def __updatepinsinclusive(self,ind : int,addvalue : int):
        for indexindex,index in enumerate(self.pinvalues):
            if index >= ind:
                self.pinvalues[indexindex] += addvalue
    
    def pin(self,pin : pinnedIndex,index : int):
        self.pinkeys.append(pin)
        self.pinvalues.append(index)
        pin.root = self
    
    def len(self):
        return len(self.main)

    def append(self,obj):
        """Appends object to the list. If said object is of type pinnedIndex, said pin will be assigned to the current length of the list"""
        if isinstance(obj,pinnedIndex):
            self.pin(obj,len(self.main))
        else:
            self.main.append(obj)

    def insert(self,i : int,obj):
        self.main.insert(i,obj)
        self.__updatepins(i,1)

    def insertinclusive(self,i : int,obj):
        self.main.insert(i,obj)
        self.__updatepins(i,1)

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
        self.indents = list()
        self.cindent = 0
    def newLine(self,noautoindent : bool = False) -> pline:
        a = pline()
        if noautoindent == False and self.content.len() >= 1 and self.content[-1].len() > 0:
            self.cindent += self.content[-1][-1].isIndentIncreaser
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
    
    def paste(self,topaste : generator) -> generator:
        ogindent = self.cindent
        for strindent,strline in topaste.iterlines():
            self.cindent = ogindent + strindent
            self.newLine(noautoindent = True).construct(strline)
        return self

class varepository():
    def __init__(self):
        pass

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