import argparse
import os
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='Input')
parser.add_argument('-o', '--output', help='Output')
args = parser.parse_args()
files = os.listdir(args.input)

with open('datalist.txt', "r") as f:
    data= f.read().split()
    counter=0
    j= open('ndatalist.txt', "w")

    for i in files:
        counter = counter +1
        j.write('python letsmakecuts.py -i'+ args.input + i + ' -o ' + args.output + 'postcutevents' + str(counter) +'.root\n')

    f.close()
    j.close()
