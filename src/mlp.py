import math
import numpy as np

from operator import itemgetter

with open(r'D:\ETS\ELE778\GitHub\ele778_lab2\data\raw\train\man\ae\1a.txt', 'r') as content_file:
    content = content_file.read().split('\n')
content_file.close()

for index, line in enumerate(content):
    content[index] = line.split(' ')
    content[index].pop()    # supprime le dernier élément vide de chaque ligne

content.pop()   # supprime la dernière ligne qui est vide

for index1, inner in enumerate(content):
    for index2, string in enumerate(inner):
        content[index1][index2] = float(string)

content.sort(key=itemgetter(12), reverse=True)

print(content[0][0])
