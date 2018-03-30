#https://www.cnblogs.com/cmt110/p/7464944.html
#!/usr/bin/python
# -*- coding: UTF-8 -*-
 
from xml.dom.minidom import parse
import xml.dom.minidom
 
# 使用minidom解析器打开 XML 文档
DOMTree = xml.dom.minidom.parse(r'E:\Dataset\VOC2012\Annotations\2007_000027.xml')
collection = DOMTree.documentElement

# 得到文档元素对象
print(collection.nodeName)
print(collection.nodeValue)
print(collection.nodeType)
print(collection.ELEMENT_NODE)

bb = collection.getElementsByTagName('width')
b= bb[0]
print(b.nodeName)

# 得到标签间的数值
print(b.firstChild.data)
