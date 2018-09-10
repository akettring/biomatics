#Python v.2.7.6
#Runs on Linux Mint 17.3

#input is faa fasta protein file
#wgs broser -> bioproject -> protein # (bottom) -> send to file

import os, csv, sys, re, subprocess
import pandas as pd
import numpy as np
from Bio import SeqIO
from Bio.Blast.Applications import NcbiblastpCommandline
from Bio.Blast import NCBIXML
from multiprocessing import Pool 
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import shutil

threadz = 4 

#make list of infiles and sort
infiles = []
indir = './infiles/'
for file in os.listdir(indir):
    if file.endswith('.faa'):
        infiles.append(os.path.join( file))
infiles.sort()
#make a list without file extensions
filenames = [k.replace('.faa','') for k in infiles]
    
    
#make outifiles directory
outdir = './outfiles/'
if not os.path.exists(outdir):
    os.mkdir(outdir)
#make fasta directory
fasdir = outdir + 'fasta/'
if not os.path.exists(fasdir):
    os.mkdir(fasdir)
#make blast db directory
dbdir = outdir + 'blastdb/'
if not os.path.exists(dbdir):
    os.mkdir(dbdir)
#make pairwise blast directory
blastdir = outdir + 'blast/'
if not os.path.exists(blastdir):
    os.mkdir(blastdir)    
    

print 'Trimming input files...' 
def trim(fasname):
    og = indir + fasname + '.faa'
    ng = fasdir + fasname + '_new.faa'
    tg = fasdir + fasname + '_tmp.faa'
    shutil.copy(og, ng)
    n = 1
    n_tot = 0
    while n > 0:    
        with open(ng, "rU") as input_handle, open(tg, "w") as output_handle:
            n=0
            for seq_record in SeqIO.parse(input_handle , 'fasta'):
                length = len(seq_record.seq)
                last = seq_record.seq[-1:]
                if 'X' in last:
                    new_seq = str(seq_record.seq[:-1])
                    old_id = seq_record.id
                    old_name = seq_record.name
                    old_desc = seq_record.description
                    rec = SeqRecord(Seq(new_seq), id=old_id, name=old_name, description=old_desc)
                    SeqIO.write(rec, output_handle, 'fasta')
                    n += 1
                else:
                    #print seq_record.seq            
                    SeqIO.write(seq_record, output_handle, 'fasta')
            n_tot += n
        shutil.copy(tg, ng)    
    os.remove(tg)
    print 'Trimmed', n_tot, 'terminal Xs from', fasname
for file in filenames:
    trim(file)
print ''


print 'Making BLAST databases...'
cmds = []
for file in filenames:
    bashCommand = 'makeblastdb -in ' + fasdir + file + '_new.faa -dbtype prot -out ' + dbdir + file
    cmds.append(bashCommand)
FNULL = open(os.devnull, 'w')
def dater(cmd):
    print cmd
    p = subprocess.Popen(cmd, shell=True, stdout=FNULL, stderr=subprocess.STDOUT)
    p.wait()
pool = Pool(threadz)
for cmd in cmds:
    pool.apply_async(dater, [cmd])
pool.close()
pool.join()  
print ''


print 'Pairwise BLAST...'
#make csv for pairs
o = outdir + 'pairs.csv'
with open(o, 'w') as f:
    writer = csv.writer(f)
    for x, y in [(x,y) for x in filenames for y in filenames]:
        z = [x,y]
        writer.writerow(z)
#appropriate blast commands
cmds = []
p1 = outdir + 'pairs.csv'
with open(p1, 'r') as f1:
    reader = csv.reader(f1)
    for row in reader:
        i = fasdir + row[0] + '_new.faa'
        d = dbdir + row[1] 
        o = blastdir + row[0] + '_' + row[1] + '.xml'
        blasty = NcbiblastpCommandline(query=i, db=d, out=o, outfmt=5, max_hsps_per_subject=1, num_alignments=3)
        cmds.append(str(blasty))
#run them in parallel      
def blaster(cmd):
    print cmd
    p = subprocess.Popen(cmd, shell=True)
    p.wait()
pool = Pool(threadz)
for cmd in cmds:
    pool.apply_async(blaster, [cmd])
pool.close()
pool.join()


#make csv for pairs
#o = outdir + 'pairs.csv'
#with open(o, 'w') as f:
#    writer = csv.writer(f)
#    for x, y in [(x,y) for x in filenames for y in filenames]:
#        z = [x,y]
#        writer.writerow(z)

print ''
print 'Analyzing BLAST results...'
q_list = []
p_list = []
with open('./outfiles/pairs.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        pf = blastdir + row[0] + '_' + row[1] + '.xml'
        print pf
        h = open(pf)
        que_tot = 0
        pos_tot = 0
        for blast_records in NCBIXML.parse(h):            
            blast_records.alignments.sort(key = lambda align: max(hsp.positives for hsp in align.hsps), reverse=True)            
            que = blast_records.query_letters
            que_tot += que
            alignz = iter(blast_records.alignments)
            if blast_records.alignments != []:
                aligny = next(alignz)
                hspz = iter(aligny.hsps)
                hsp = next(hspz)
                escore = hsp.expect
                if escore < 0.01:
                    pos = hsp.positives 
                    pos_tot += pos        
        q_list.append(que_tot)
        p_list.append(pos_tot)
#make a dataframe
r = outdir + 'pairs.csv'
df = pd.read_csv(r, names=['Bug 1', 'Bug 2', 'AAs Queried', 'Positives'])    
#dump data
df['AAs Queried'] = q_list
df['Positives'] = p_list
#write to file
o = outdir + 'pairwise.csv'
df.to_csv(o, index=False)

# Find and add inverse pairs
pw = outdir + 'pairwise.csv'
ps = []
qs = []
ds = []
with open(pw, 'r') as f1:
    reader = csv.reader(f1)
    next(f1)
    for row in reader:
        a1 = row[0]
        a2 = row[1]
        q1 = float(row[2])
        p1 = float(row[3])
        with open(pw, 'r') as f2:
            reader = csv.reader(f2)
            next(f2)
            for row in reader:
                b1 = row[0]
                b2 = row[1]
                q2 = float(row[2])
                p2 = float(row[3])                    
                if a1==b2 and b1==a2:
                    qs.append(q2)
                    ps.append(p2)
                    dd = 1- (p1 + p2) / (q1 + q2)
                    ds.append(dd)
#make a dataframe
r = outdir + 'pairwise.csv'
df = pd.read_csv(r)
#dump data
df['AAs 2'] = qs
df['Pos 2'] = ps
df['Distance'] = ds
#rewrite pairwise file
o = outdir + 'pairwise.csv'
df.to_csv(o, index=False)

#pairwise to distance matrix
df = pd.DataFrame(filenames)
num=len(filenames)
distx = ds[:]
for p in filenames:
    disty = distx[:]
#delete up to first 3
    del disty[num:]
    df[p] = disty
#delete fist three
    del distx[:num]
o = outdir + 'matrix.csv'
df.to_csv(o, index=False)

print ''
print 'All done!'
