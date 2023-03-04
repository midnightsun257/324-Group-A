with open('datalist.txt', "r") as f:
    data= f.read().split()
counter=0
j= open('ndatalist.txt', "w")
for i in data:
    counter = counter +1
    j.write('python cutevent.py -i'+i+'-o postcutevents' +str(counter)+'.root\n')
#print(data)
