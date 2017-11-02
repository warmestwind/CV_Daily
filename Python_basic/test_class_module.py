import class_module
import sys
import os
a=sys_modules.a1(1)
print(a.c)

print('module=',class_module.a1.__module__)
print('module=',a.__module__)

print(sys.modules[a.__module__].__file__)
path1=os.path.pardir
print(path1)
path2=sys.modules[a.__module__].__file__
print(os.path.abspath(os.path.join(path2,path1)))
