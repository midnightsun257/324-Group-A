f = open('datalist.txt','r')
lines = f.readlines()

nndatalist = []
with open('ndatalist', 'w') as j:
    for i in lines:
        j.write('python cutevent.py -i'+i )
print(j)
#print(lines)