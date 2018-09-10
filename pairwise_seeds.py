#Python v.2.7.6
#Runs on Linux Mint 17.3
#Requires install of Pandas

import os, csv, sys, re, subprocess
import pandas as pd


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
    if file.endswith('_seeds.txt'):
        bug = file.replace('_seeds.txt','')
        if bug not in bugs:
            bugs.append(os.path.join(bug))
    elif file.endswith('_nonseeds.txt'):
        bug = file.replace('_nonseeds.txt','')
        if bug not in bugs:
            bugs.append(os.path.join(bug))
bugs.sort()

#write pairs from filenames to a file
with open(tmpdir + 'pairs.csv', 'wb') as f:
    w=csv.writer(f)
    for x, y in [(x,y) for x in bugs for y in bugs]:
        z=[x,y]
        w.writerow(z)

print 'Preparing infiles...'
for bug in bugs:
    inseed = indir + bug + '_seeds.txt'
    innonseed = indir + bug + '_nonseeds.txt'
    outseed = tmpdir + bug + '_seeds.txt'
    outnonseed = tmpdir + bug + '_nonseeds.txt'
    tmpseed = tmpdir + bug + '_seeds_tmp'
#verify complementary infiles exist
    if os.path.exists(inseed) and os.path.exists(innonseed):
        pass
    else:
        print 'Missing a complementary infile for' , bug
        sys.exit()
#sort via BASH
    bashCommand = 'sort -u ' + inseed +  ' > ' +  tmpseed
    subprocess.call(bashCommand, shell=True)
    bashCommand = 'sort -u ' + innonseed + ' > ' + outnonseed
    subprocess.call(bashCommand, shell=True)    
#trim confidence scores from seed list
    with open(tmpseed) as f:
        with open(outseed, 'w') as f1:
            for line in f:
#split at tab and first half + a return
                line2 = line[:-3] + "\n"
                f1.write(line2)
    os.remove(tmpseed)
    print bug                
                           
print ''
print 'Computing competition...'
#make lists
combined = []
common = []
competition = []
#define pariwise files
with open(tmpdir + 'pairs.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        inseed1 = tmpdir + row[0] + '_seeds.txt'
        inseed2 = tmpdir + row[1] + '_seeds.txt'
        outcomb = tmpdir + 'combined_'  + row[0] + '_' + row[1] + '_seeds.txt'
        outcomm = tmpdir + 'common_'  + row[0] + '_' + row[1] + '_seeds.txt'
#combine via sort via bash
        bashCommand = 'sort -u ' + inseed1 + ' ' + inseed2 + ' > ' + outcomb
        subprocess.call(bashCommand, shell=True)
#comm via bash
        bashCommand = 'comm -1 -2 ' + inseed1 + ' ' + inseed2 + ' > ' + outcomm
        subprocess.call(bashCommand, shell=True)
#count number of lines
        flen_comb = sum(1 for line in open(outcomb))
        combined.append(flen_comb)
        flen_comm = sum(1 for line in open(outcomm))
        common.append(flen_comm)
#do math
        comp = (flen_comm / float(flen_comb))
        competition.append(comp)
#make a dataframe
df = pd.read_csv(tmpdir+'pairs.csv', names=['Bug 1', 'Bug 2', 'Common', 'Combined', 'Competition'])
#data dump
df['Common'] = common
df['Combined'] = combined
df['Competition'] = competition
#write to file
df.to_csv(outdir+'competition_pairs.csv', index=False)
#print df                
#Pairwise competition matrix
df = pd.DataFrame(bugs)
num=len(bugs)
compx = competition[:]
for b in bugs:
    compy = compx[:]
    del compy[num:]
    df[b] = compy
    del compx[:num]
df.to_csv(outdir+'competition_matrix.csv', index=False)
print 'Done.'

#compare seeds and non-seeds
print ''
print 'Computing complementarity...'
#make lists
seedz = []
commonsns = []
complementarity = []
#define pairwise files
with open(tmpdir + 'pairs.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        inseed = tmpdir + row[0] + '_seeds.txt'
        innonseed = tmpdir + row[1] + '_nonseeds.txt'
        outcomm = tmpdir + 'common_'  + row[0] + '_' + row[1] + '_sns.txt'
#comm via bash
        bashCommand = 'comm -1 -2 ' + inseed + ' ' + innonseed + ' > ' + outcomm
        subprocess.call(bashCommand, shell=True)
#count number of lines
        flen_seeds = sum(1 for line in open(inseed))
        seedz.append(flen_seeds)
        flen_comm = sum(1 for line in open(outcomm))
        commonsns.append(flen_comm)
#do math
        comp = (flen_comm / float(flen_seeds))
        complementarity.append(comp)
#make a dataframe
df = pd.read_csv(tmpdir+'pairs.csv', names=['Bug 1', 'Bug 2', 'Seeds', 'Common_SNS', 'Asymmetric'])
#data dump
df['Seeds'] = seedz
df['Common_SNS'] = commonsns
df['Asymmetric'] = complementarity
#write to file
df.to_csv(outdir+'complementarity_pairs.csv', index=False)

#Append complementary complementarity scores
pw = outdir + 'complementarity_pairs.csv'
ss = []
cs = []
sc = []
with open(pw, 'r') as f1:
    reader = csv.reader(f1)
    next(f1)
    for row in reader:
        a1 = row[0]
        a2 = row[1]
        s1 = float(row[2])
        c1 = float(row[3])
        with open(pw, 'r') as f2:
            reader = csv.reader(f2)
            next(f2)
            for row in reader:
                b1 = row[0]
                b2 = row[1]
                s2 = float(row[2])
                c2 = float(row[3])                    
                if a1==b2 and b1==a2:
                    ss.append(s2)
                    cs.append(c2)
                    symcom =  (c1 + c2) / (s1 + s2)
                    sc.append(symcom)
#make a dataframe
r = outdir + 'complementarity_pairs.csv'
df = pd.read_csv(r)
#dump data
df['Seeds S2'] = ss
df['SNS 2'] = cs
df['Symmetric'] = sc
#rewrite pairwise file
df.to_csv(r, index=False)

#Pairwise asymmetric matrix
df = pd.DataFrame(bugs)
num=len(bugs)
compx = complementarity[:]
for b in bugs:
    compy = compx[:]
    del compy[num:]
    df[b] = compy
    del compx[:num]
df.to_csv(outdir+'complementary_asym_matrix.csv', index=False)

#Pairwise symmetric matrix
df = pd.DataFrame(bugs)
num=len(bugs)
compx = sc[:]
for b in bugs:
    compy = compx[:]
    del compy[num:]
    df[b] = compy
    del compx[:num]
df.to_csv(outdir+'complementary_sym_matrix.csv', index=False)

print 'Done.'
print ''
