#Python v.2.7.6
#Runs on Linux Mint 17.3
#Requires install of Pandas

import os, csv, sys, re, subprocess
import pandas as pd


#make list of files and sort
infiles=[]
for file in os.listdir("./infiles"):
    if file.endswith(".paths"):
        infiles.append(os.path.join( file))
infiles.sort()

#make a list without file extensions
filenames=[k.replace(".paths",'') for k in infiles]

#make directory if needed
directory='./outfiles'
if not os.path.exists(directory):
    os.mkdir(directory)

#extract pathways from files to new file
for p in infiles:
    n = './infiles/' + p
    m = './outfiles/' + p

#split at tab and keep second half
    with open(n) as f:
        with open(m, 'w') as f1:
            for line in f:
                line2 = line.split("\t", 1)[-1]
                line3 = re.sub("\t", '', line2)
                f1.write(line3)
#remove first line
    with open(m, 'r') as fin:
        data = fin.read().splitlines(True)
    with open(m, 'w') as fout:
        fout.writelines(data[1:])

#sort via bash
    bashCommand = 'sort -u ' + m + ' > ./outfiles/paths.tmp'
    subprocess.call(bashCommand, shell=True)
    bashCommand = 'mv ./outfiles/paths.tmp ' + m
    subprocess.call(bashCommand, shell=True)



# PAIRWISE COMPARISONS

#write pairs from filenames to a file
with open("./outfiles/pairs.csv", 'wb') as f:
    w=csv.writer(f)
    for x, y in [(x,y) for x in filenames for y in filenames]:
        z=[x,y]
        w.writerow(z)

#make lists
combined = []
common = []
distances = []

with open('./outfiles/pairs.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:

#combine via sort via bash
        bashCommand = 'sort -u ./outfiles/' + row[0] + '.paths ./outfiles/' + row[1] + '.paths > ./outfiles/combined_'  + row[0] + '_' + row[1] + '.paths'
        subprocess.call(bashCommand, shell=True)

#comm via bash
        bashCommand = 'comm ./outfiles/' + row[0] + '.paths ./outfiles/' + row[1] + '.paths -1 -2 > ./outfiles/common_' + row[0] + '_' + row[1] + '.paths'
        subprocess.call(bashCommand, shell=True)

#count number of lines
        file_comb = './outfiles/combined_' + row[0] + '_' + row[1] + '.paths'
        flen_comb = sum(1 for line in open(file_comb))
        combined.append(flen_comb)

        file_comm = './outfiles/common_' + row[0] + '_' + row[1] + '.paths'
        flen_comm = sum(1 for line in open(file_comm))
        common.append(flen_comm)

#do math
        dist = 1 - (flen_comm / float(flen_comb))
        distances.append(dist)

#make a bamf file
df = pd.read_csv('./outfiles/pairs.csv', names=['Bug 1', 'Bug 2', 'Common', 'Combined', 'Distances'])

#data dump
df['Common'] = common
df['Combined'] = combined
df['Distances'] = distances
 
#write to file
df.to_csv('./outfiles/pairwise.csv', index=False)
print df 
print ''


#Pairwise to distance matrix
df = pd.DataFrame(filenames)

num=len(filenames)
distx = distances[:]
for p in filenames:
    disty = distx[:]
#delete up to first 3
    del disty[num:]
    df[p] = disty
#delete fist three
    del distx[:num]

df.to_csv('./outfiles/matrix.csv', index=False)

print df

