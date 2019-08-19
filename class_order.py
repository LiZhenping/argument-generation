# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 15:24:54 2019

@author: lizhenping
"""







class A():
    def __init__(self, name):
        print('A的class' ,self.__class__, name)
    print('this is A class')
    #print(name)
    def test(self,a):
        print(a)
    
        
class B(A):
    def __init__(self, name):
        self._name = name
        #A.__init__(self, name)
        print('B的class', self.__class__, name)
    print('this is B class')
        
class C(B):
    def __init__(self, name):
        self.name = name
        print('C的class')
        B.__init__(self, name)
    print("this is C class")
    def pr(self):
        print(self.name)
        
if __name__ == '__main__':
 
    c = C('lee')
    
    
class A():
    def __init__(self, name):
        print('A的class' ,self.__class__, name)
        
class B(A):
    def __init__(self, name):
        self._name = name
        A.__init__(self, name)
        print('B的class', self.__class__, name)
    print('this is B class')
        
class C(B):
    def __init__(self, name):
        B.__init__(self, name)
        print('C的class')
        
if __name__ == '__main__':
 
    c = C('lee')