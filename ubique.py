#Python v.2.7.6
#Runs on Linux Mint 17.3
#Requires install of Pandas

import os, csv, sys, re, subprocess
import pandas as pd
from pandas.io.parsers import count_empty_vals



indir = './infiles/'

#make directories if needed
outdir = './outfiles/'
if not os.path.exists(outdir):
    os.mkdir(outdir)
    
tmpdir = outdir + 'tmp/'
if not os.path.exists(tmpdir):
    os.mkdir(tmpdir)



#make list of bugs from infiles
bugs=[]
for file in os.listdir(indir):
    if file.endswith('.paths'):
        bug = file.replace('.paths','')
        if bug not in bugs:
            bugs.append(os.path.join(bug))
bugs.sort()



entries=[]
print 'Preparing infiles...'
for bug in bugs:
    inbug = indir + bug + '.paths'
    tmpbug = tmpdir + bug + '.tmp.paths'
    outbug = outdir + bug + '.sorted.paths'

#split at tab and keep second half
    with open(inbug) as f:
        with open(tmpbug, 'w') as f1:
            for line in f:
                line2 = line.split("\t", 1)[-1]
                line3 = re.sub("\t", '', line2)
                f1.write(line3)
#remove first line
    with open(tmpbug, 'r') as fin:
        data = fin.read().splitlines(True)
    with open(tmpbug, 'w') as fout:
        fout.writelines(data[1:])

#sort via bash
    bashCommand = 'sort -u ' + tmpbug + ' > ' + outbug
    subprocess.call(bashCommand, shell=True)
 
for bug in bugs: 
    bugy = outdir + bug + '.sorted.paths'
 
#find unique entries    
    with open(bugy, 'r') as b:
        for entry in b:
            entrz = entry[:-1]
            if entrz not in entries:
                entries.append(entrz)
entries.sort
print entries
print ''


print 'Counting...'
returns = []
for entry in entries:
    county=0
    for bug in bugs:
        bugy = outdir + bug + '.sorted.paths'
        with open(bugy, 'r') as b:
            for line in b:
                liney = line[:-1]
                if entry == liney:
                    county += 1
    returns.append(county)
print returns
print ''

df = pd.DataFrame(columns=['Entry', 'Count'])
df['Entry']=entries
df['Count']=returns

df.to_csv(outdir+'ubique.csv', index=False)

