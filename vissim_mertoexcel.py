import xml.etree.ElementTree as ET

# 替换为你的mer文件路径
mer_file_path = 'path_to_your_mer_file.mer'

# 由于Mer文件实际上是一个XML文件，我们可以使用ElementTree来解析它
tree = ET.parse(mer_file_path)
root = tree.getroot()

# 打印根元素
print(root.tag)

# 遍历并打印所有元素
for child in root:
    print('-' * 10)
    print(child.tag, child.attrib)
    for i in child:
        print(i.tag, i.text)

