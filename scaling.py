f = open('datalist.txt','r')
lines = f.readlines()

nndatalist = []
counter=0
with open('ndatalist', 'w') as j:
    for i in lines:
        i.replace('\n','')
        print(i)
        counter = +1
        j.write('python cutevent.py -i'+i+'-o postcutevents' +str(counter)+'.root\n')
print(j)
