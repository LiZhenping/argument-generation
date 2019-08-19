# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 09:51:01 2019

@author: lizhenping
"""

class Animal(object):
    def __init__(self, name):
        print('Animal的class' ,self.__class__, name)
  
    print('this is Animal class')
    def __call__(self, words):
        print ("Animal: ", words)
        


class Bnimal(Animal):
    def __init__(self, name):
        self._name = name
        #A.__init__(self, name)
        print('Bnimal的class', self.__class__, name)
    print('this is B class')
    def __call__(self, words):
        print ("Bnimal: ", words)
class C(Bnimal):
    def __init__(self, name):
        self.name = name
        print('C的class')
        Bnimal.__init__(self, name)
    print("this is C class")
    def __call__(self, words):
        print ("Cnimal: ", words)   
    
if __name__ == '__main__':
 
    c = C('lee')
    c("this")