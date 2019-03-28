# -*- coding: utf-8 -*-
# @Author: Marte
# @Date:   2019-03-20 09:01:43
# @Last Modified by:   Marte
# @Last Modified time: 2019-03-27 22:21:01
from numpy import *

a = range(10)
print(a)
print(len(a))
for i in range(10):
    index = int(random.uniform(0,len(a)))
    del(a[index])

print(a)
print(len(a))