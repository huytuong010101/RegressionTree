import pandas as pd

class Node:
    def __init__(self, parent=None, childs: tuple=(None, None), dataset: pd.DataFrame=None):
        self.__parent = parent
        self.__childs = childs
        self.__dataset = dataset
        self.__attr = None
        self.__threshold = None
        self.__is_category = False
        self.__model = None
        self.__model_variable = []
        
    def set_condition(self, attr: str, threshold, is_category=False):
        self.__attr = attr
        self.__threshold = threshold
        self.__is_category = is_category
        
    def get_condition(self):
        return self.__attr, self.__threshold, self.__is_category
    
    def is_leaf(self):
        return self.__childs == (None, None)
            
    @property
    def model(self):
        return self.__model
    
    @property
    def model_variable(self):
        return self.__model_variable
    
    @property
    def parent(self):
        return self.__parent
    
    @property
    def childs(self):
        return self.__childs
    
    @property
    def dataset(self):
        return self.__dataset
    
    @model.setter
    def model(self, value):
        self.__model = value
    
    @parent.setter
    def parent(self, value):
        self.__parent = value
        
    @childs.setter
    def childs(self, value):
        self.__childs = value
        
    @dataset.setter
    def dataset(self, value):
        self.__dataset = value
        
    @model_variable.setter
    def model_variable(self, value):
        self.__model_variable = value
        

