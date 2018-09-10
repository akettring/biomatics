import pandas as pd
import numpy as np
import scipy
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import itertools
import ast
import csv
import math
pd.options.display.float_format = '{:.3f}'.format

cond0=['1040','IBU42','1PSI','15PSI']
with open('abrv0.txt') as f:
    abrv0 = f.read().splitlines()







#READ OBSERVATION AND CONFIG DATA

od_reads = pd.read_csv('od600.csv')
od_reads = od_reads.set_index(['condition', 'group', 'config', 'row'])
od_reads = od_reads.stack()
od_reads = od_reads.reset_index()
od_reads = od_reads.rename(columns={'level_4':'col', 0:'abs'})
od_reads['g1'] = od_reads.group.str[:1]
od_reads['g2'] = od_reads.group.str[-1:]


ph_reads = pd.read_csv('sup_bpb.csv')
ph_reads = ph_reads.set_index(['condition', 'type', 'wavelength',
                                'group', 'config', 'row'])
ph_reads = ph_reads.stack()
ph_reads = ph_reads.reset_index()
ph_reads = ph_reads.rename(columns={'level_6':'col', 0:'abs'})
ph_reads['g1'] = ph_reads.group.str[:1]
ph_reads['g2'] = ph_reads.group.str[-1:]


config = pd.read_csv('config.csv')
config = config.set_index(['config', 'row'])
config = config.stack()
config = config.reset_index()
config = config.rename(columns={'level_2':'col', 0:'pair'})
config['p1'] = config.pair.str[:1]
config['p2'] = config.pair.str[-1:]








#ASSIGN CONFIG TO OBSERVATIONS

list1=[]
for index,row in od_reads.iterrows():
	cfg=row['config']
	ruw=row['row']
	cul=row['col']
	list1.append(config[(config.config==cfg)&(config.row==ruw)&
                        (config.col==cul)]['p1'].values[0])

list2=[]
for index,row in od_reads.iterrows():
	cfg=row['config']
	ruw=row['row']
	cul=row['col']
	list2.append(config[(config.config==cfg)&(config.row==ruw)&
                        (config.col==cul)]['p2'].values[0])

od_reads['p1']=list1
od_reads['p2']=list2



list1=[]
for index,row in ph_reads.iterrows():
	cfg=row['config']
	ruw=row['row']
	cul=row['col']
	list1.append(config[(config.config==cfg)&(config.row==ruw)&
                        (config.col==cul)]['p1'].values[0])

list2=[]
for index,row in ph_reads.iterrows():
	cfg=row['config']
	ruw=row['row']
	cul=row['col']
	list2.append(config[(config.config==cfg)&(config.row==ruw)&
                        (config.col==cul)]['p2'].values[0])

ph_reads['p1']=list1
ph_reads['p2']=list2











# CHECK YASELF (might not work)

ph_test=[]
for index,row in ph_reads.iterrows():
	idc=row['group']+' '+row['config']+' '+row['row']+' '+row['col']
	ph_test.append(idc)

ph_test.sort()

od_test=[]
for index,row in od_reads.iterrows():
	idk=row['group']+' '+row['config']+' '+row['row']+' '+row['col']
	od_test.append(idods.groupby('abrv1')['abs'].mean()k)

od_test.sort()

in_ph_only=[]
for item in ph_test:
	if item not in od_test:
		in_ph_only.append(item)
		


in_od_only=[]
for item in od_test:
	if item not in ph_test:
		in_od_only.append(item)












#ASSIGN STRAIN ABBREVIATIONS BASED ON CONFIG

strains0 = pd.read_csv('strains0.csv')
strains1 = pd.read_csv('strains1.csv')

abrv1_list=[]
abrv2_list=[]
for index,row in od_reads.iterrows():
	g1=row['g1']
	p1=row['p1']
	g2=row['g2']
	p2=row['p2']
	if row['config']!='delta':
		if row['p1'] is 'm' :
			abrv1 = 'media'
			abrv2 = 'media'
		elif row['p1'] is 'i':
			#for dubs: make sure left isos are from g1
			if g1 != g2 and int(row['col']) < 7:
				abrv2 = strains0[(strains0.group==g1)&
                        (strains0.number==int(p2))]['abrv'].values[0]
				abrv1 = abrv2
			else:
				abrv2 = strains0[(strains0.group==g2)&
                        (strains0.number==int(p2))]['abrv'].values[0]
				abrv1 = abrv2
		else:
			abrv1 = strains0[(strains0.group==g1)&(strains0.number==int(p1))
                            ]['abrv'].values[0]
			abrv2 = strains0[(strains0.group==g2)&(strains0.number==int(p2))
                            ]['abrv'].values[0]
	else:
		if row['p1'] is 'm' :
			abrv1 = 'media'
			abrv2 = 'media'
		elif row['p1'] is 'i':
			#for dubs: make sure left isos are from g1
			if g1 != g2 and int(row['col']) < 7:
				abrv2 = strains1[(strains1.group==g1)&
                        (strains1.number==int(p2))]['abrv'].values[0]
				abrv1 = abrv2
			else:
				abrv2 = strains1[(strains1.group==g2)&
                        (strains1.number==int(p2))]['abrv'].values[0]
				abrv1 = abrv2
		else:
			abrv1 = strains1[(strains1.group==g1)&
                        (strains1.number==int(p1))]['abrv'].values[0]
			abrv2 = strains1[(strains1.group==g2)&
                        (strains0.number==int(p2))]['abrv'].values[0]
	abrv1_list.append(abrv1)
	abrv2_list.append(abrv2)


od_reads['abrv1']=abrv1_list
od_reads['abrv2']=abrv2_list


abrv1_list=[]
abrv2_list=[]
for index,row in ph_reads.iterrows():
	g1=row['g1']
	p1=row['p1']
	g2=row['g2']
	p2=row['p2']
	if row['config']!='delta':
		if row['p1'] is 'm' :
			abrv1 = 'media'
			abrv2 = 'media'
		elif row['p1'] is 'i':
			#for dubs: make sure left isos are from g1
			if g1 != g2 and int(row['col']) < 7:
				abrv2 = strains0[(strains0.group==g1)&
                        (strains0.number==int(p2))]['abrv'].values[0]
				abrv1 = abrv2
			else:
				abrv2 = strains0[(strains0.group==g2)&
                        (strains0.number==int(p2))]['abrv'].values[0]
				abrv1 = abrv2
		else:
			abrv1 = strains0[(strains0.group==g1)&
                    (strains0.number==int(p1))]['abrv'].values[0]
			abrv2 = strains0[(strains0.group==g2)&
                    (strains0.number==int(p2))]['abrv'].values[0]
	else:
		if row['p1'] is 'm' :
			abrv1 = 'media'
			abrv2 = 'media'
		elif row['p1'] is 'i':
			#for dubs: make sure left isos are from g1
			if g1 != g2 and int(row['col']) < 7:
				abrv2 = strains1[(strains1.group==g1)&
                        (strains1.number==int(p2))]['abrv'].values[0]
				abrv1 = abrv2
			else:
				abrv2 = strains1[(strains1.group==g2)&
                        (strains1.number==int(p2))]['abrv'].values[0]
				abrv1 = abrv2
		else:
			abrv1 = strains1[(strains1.group==g1)&
                    (strains1.number==int(p1))]['abrv'].values[0]
			abrv2 = strains1[(strains1.group==g2)&
                    (strains0.number==int(p2))]['abrv'].values[0]
	abrv1_list.append(abrv1)
	abrv2_list.append(abrv2)


ph_reads['abrv1']=abrv1_list
ph_reads['abrv2']=abrv2_list



od_reads = od_reads.reset_index(drop=True)
ph_reads = ph_reads.reset_index(drop=True)









# MANUAL CORRECTIONS 

ods=od_reads.copy()

ods.loc[(ods['condition']=='1040')&(ods['group']=='AxA')
        &(ods['config']=='alpha')&(ods['row']=='B')
        &(ods['col']=='7'),'abs']=np.NaN

ods.loc[(ods['condition']=='1PSI')&(ods['group']=='BxC')
        &(ods['config']=='beta')&(ods['row']=='B')
        &(ods['col']=='7'),'abs']=np.NaN

ods.loc[(ods['condition']=='15PSI')&(ods['group']=='CxB')
        &(ods['config']=='beta')&(ods['row']=='B')
        &(ods['col']=='7'),'abs']=np.NaN


numz=['1','2','3','4','5','6']
for n in numz:
    ods.loc[(ods['condition']=='1PSI')&(ods['group']=='CxA')
            &(ods['config']=='alpha')&(ods['row']=='H')
            &(ods['col'].astype(int)>6)
            &(ods['p2']==n),'abs'] = ods.loc[(ods['condition']=='1PSI')
            &(ods['group']=='CxA')&(ods['config']=='alpha')
            &(ods['row']=='H')&(ods['col'].astype(int)<7)
            &(ods['p2']==n),'abs'].values

ods.loc[(ods['condition']=='1PSI')&(ods['group']=='CxA')
        &(ods['config']=='alpha')&(ods['row']=='H')
        &(ods['col'].astype(int)<7),'abs']=np.NaN




for n in numz:
    ods.loc[(ods['condition']=='IBU42')&(ods['group']=='CxB')
            &(ods['col']=='11')
            &(ods['p1']==n),'abs'] = ods.loc[(ods['condition']=='IBU42')
            &(ods['group']=='CxB')&(ods['row']=='B')
            &(ods['col'].astype(int)>6)&(ods['p2']==n),'abs'].values

ods.loc[(ods['condition']=='IBU42')&(ods['group']=='CxB')
        &(ods['row']=='B')&(ods['col'].astype(int)>6),'abs']=np.NaN




ods.to_csv('od_neat.csv', float_format='%.3f')







phs=ph_reads.copy()

phs.loc[(phs['condition']=='1040')&(phs['group']=='AxA')
        &(phs['config']=='alpha')&(phs['row']=='B')
        &(phs['col']=='7'),'abs']=np.NaN

phs.loc[(phs['condition']=='1PSI')&(phs['group']=='BxC')
        &(phs['config']=='beta')&(phs['row']=='B')
        &(phs['col']=='7'),'abs']=np.NaN

phs.loc[(phs['condition']=='15PSI')&(phs['group']=='CxB')
        &(phs['config']=='beta')&(phs['row']=='B')
        &(phs['col']=='7'),'abs']=np.NaN

tps=['sup','bpb']
wls=[449,590]
numz=['1','2','3','4','5','6']
for tp in tps:
    for wl in wls:
        for n in numz:
            phs.loc[(phs['condition']=='1PSI')&(phs['group']=='CxA')
                    &(phs['config']=='alpha')&(phs['row']=='H')
                    &(phs['col'].astype(int)>6)
                    &(phs['type']==tp)&(phs['wavelength']==wl)
                    &(phs['p2']==n),'abs'] = phs.loc[
                            (phs['condition']=='1PSI')
                    &(phs['group']=='CxA')&(phs['config']=='alpha')
                    &(phs['row']=='H')&(phs['col'].astype(int)<7)
                    &(phs['type']==tp)&(phs['wavelength']==wl)
                    &(phs['p2']==n),'abs'].values





phs.loc[(phs['condition']=='1PSI')&(phs['group']=='CxA')
        &(phs['config']=='alpha')&(phs['row']=='H')
        &(phs['col'].astype(int)<7),'abs']=np.NaN



for tp in tps:
    for wl in wls:
        for n in numz:
            phs.loc[(phs['condition']=='IBU42')&(phs['group']=='CxB')
                    &(phs['type']==tp)&(phs['wavelength']==wl)
                    &(phs['col']=='11')
                    &(phs['p1']==n),'abs'] = phs.loc[(phs['condition']=='IBU42')
                    &(phs['group']=='CxB')&(phs['row']=='B')
                    &(phs['type']==tp)&(phs['wavelength']==wl)
                    &(phs['col'].astype(int)>6)&(phs['p2']==n),'abs'].values





phs.loc[(phs['condition']=='IBU42')&(phs['group']=='CxB')
        &(phs['row']=='B')&(phs['col'].astype(int)>6),'abs']=np.NaN



phs.to_csv('ph_neat.csv', float_format='%.3f')


















# COMBINE ALL DATA INTO ONE FRAME

ods = pd.read_csv('od_neat.csv', index_col=0)
phs = pd.read_csv('ph_neat.csv', index_col=0)


comb0=list(itertools.combinations_with_replacement(abrv0,2))
comb0.sort()
comb0.append(('media','media'))

od_list=[]
sup440_list=[]
sup590_list=[]
bpb440_list=[]
bpb590_list=[]
cond_list=[]
comb_list=[]
for cond in cond0:
    for comb in comb0:
        c0=comb[0]
        c1=comb[1]
        cond_list.append(cond)
        comb_list.append(comb)
        # OD600
        odz=ods[(ods.condition==cond)&
        (((ods.abrv1==c0)&(ods.abrv2==c1))|
        ((ods.abrv1==c1)&(ods.abrv2==c0)))]['abs'].values.tolist()
        odz = ["%.3f"%item for item in odz]
        od_list.append(odz)
        # SUP 440
        sup440z=phs[(phs.condition==cond)&
        (phs.type=='sup')&(phs.wavelength==440)&
        (((phs.abrv1==c0)&(phs.abrv2==c1))|
        ((phs.abrv1==c1)&(phs.abrv2==c0)))]['abs'].values.tolist()
        sup440z = ["%.3f"%item for item in sup440z]
        sup440_list.append(sup440z)
        # SUP 590
        sup590z=phs[(phs.condition==cond)&
        (phs.type=='sup')&(phs.wavelength==590)&
        (((phs.abrv1==c0)&(phs.abrv2==c1))|
        ((phs.abrv1==c1)&(phs.abrv2==c0)))]['abs'].values.tolist()
        sup590z = ["%.3f"%item for item in sup590z]        
        sup590_list.append(sup590z)
        # BPB 440
        bpb440z=phs[(phs.condition==cond)&
        (phs.type=='bpb')&(phs.wavelength==440)&
        (((phs.abrv1==c0)&(phs.abrv2==c1))|
        ((phs.abrv1==c1)&(phs.abrv2==c0)))]['abs'].values.tolist()
        bpb440z = ["%.3f"%item for item in bpb440z]       
        bpb440_list.append(bpb440z)
        # BPB 590
        bpb590z=phs[(phs.condition==cond)&
        (phs.type=='bpb')&(phs.wavelength==590)&
        (((phs.abrv1==c0)&(phs.abrv2==c1))|
        ((phs.abrv1==c1)&(phs.abrv2==c0)))]['abs'].values.tolist()
        bpb590z = ["%.3f"%item for item in bpb590z] 
        bpb590_list.append(bpb590z)




combined=pd.DataFrame({'cond':cond_list,
                       'pair':comb_list,
                       'od600':od_list,
                       'sup440':sup440_list,
                       'sup590':sup590_list,
                       'bpb440':bpb440_list,
                       'bpb590':bpb590_list},
                        columns=['cond','pair','od600',
                          'sup440','sup590','bpb440','bpb590'])







combined.to_csv('combined.tsv', float_format='%.3f',
                quoting=csv.QUOTE_NONE,  sep='\t')




combined = pd.read_csv('combined.tsv', index_col=0, sep='\t',
                         converters={'pair':ast.literal_eval,
                                    'od600':ast.literal_eval,
                                    'sup440':ast.literal_eval,
                                    'sup590':ast.literal_eval,
                                    'bpb440':ast.literal_eval,
                                    'bpb590':ast.literal_eval})






# CACLCULATE AVERAGES, STD, N


od600avg_list=[]
sup440avg_list=[]
sup590avg_list=[]
bpb440avg_list=[]
bpb590avg_list=[]
odstd_list=[]
sup440std_list=[]
sup590std_list=[]
bpb440std_list=[]
bpb590std_list=[]
odcv_list=[]
odn_list=[]
phn_list=[]
for index, row in combined.iterrows():
    od600s=[float(i) for i in row['od600'] if i != 'nan']
    od600avg_list.append(round(np.mean(od600s),3))
    odstd_list.append(round(np.std(od600s),3))
    odcv_list.append(round( np.std(od600s)/np.mean(od600s) ,3))
    odn_list.append(len(od600s))
    # find bad reads
    sup590s=[float(i) for i in row['sup590'] if i != 'nan']
    bads=[i for i,x in enumerate(sup590s) if x > 0.29]
    try:
        bads.reverse()
    except:
        pass
    # s440
    s440=[float(i) for i in row['sup440'] if i != 'nan']
    for b in bads:
        del s440[b]
    sup440avg_list.append(round(np.mean(s440),3))
    sup440std_list.append(round(np.std(s440),3))
    # s590
    s590=[float(i) for i in row['sup590'] if i != 'nan']
    for b in bads:
        del s590[b]
    sup590avg_list.append(round(np.mean(s590),3))
    sup590std_list.append(round(np.std(s590),3))
    # b440
    b440=[float(i) for i in row['bpb440'] if i != 'nan']
    for b in bads:
        del b440[b]
    bpb440avg_list.append(round(np.mean(b440),3))
    bpb440std_list.append(round(np.std(b440),3))
    # b590
    b590=[float(i) for i in row['bpb590'] if i != 'nan']
    for b in bads:
        del b590[b]
    bpb590avg_list.append(round(np.mean(b590),3))
    bpb590std_list.append(round(np.std(b590),3))
    phn_list.append(len(b590))


combined['od600avg'] = od600avg_list
combined['sup440avg'] = sup440avg_list
combined['sup590avg'] = sup590avg_list
combined['bpb440avg'] = bpb440avg_list
combined['bpb590avg'] = bpb590avg_list
combined['od600std'] = odstd_list
combined['sup440std'] = sup440std_list
combined['sup590std'] = sup590std_list
combined['bpb440std'] = bpb440std_list
combined['bpb590std'] = bpb590std_list
combined['od600cv'] = odcv_list
combined['od_n'] = odn_list
combined['ph_n'] = phn_list









# SUBTRACT BLANK VALUES

od600bs_list=[]
od440bs_list=[]
od590bs_list=[]
m=('media','media')
for index, row in combined.iterrows():
    c=row['cond']
    b=combined[(combined.cond==c)&
            (combined.pair==m)]['od600avg'].values[0]
    od600bs_list.append(round(row['od600avg']-b,3)+0.001)
    od440bs_list.append(round(row['bpb440avg']-row['sup440avg'],3))
    od590bs_list.append(round(row['bpb590avg']-row['sup590avg'],3))


combined['od600bs']=od600bs_list
combined['od440bs']=od440bs_list
combined['od590bs']=od590bs_list







# CALCULATE PH FROM RATIO

ratio_list=[]
ph_list=[]
for index, row in combined.iterrows():
    r=row['od590bs']/row['od440bs']
    ratio_list.append(r)
    ph_list.append(round((0.7549992537*np.log(r))+3.3426,3))

combined['r590_440']=ratio_list
combined['ph']=ph_list





# PH CV ERROR PROPOGATION

phcv_list=[]
for index, row in combined.iterrows():    
    std590=math.sqrt( (row['bpb590std']**2) + (row['sup590std']**2) )
    std440=math.sqrt( (row['bpb440std']**2) + (row['sup440std']**2) )
    cv590=std590/row['od590bs']
    cv440=std440/row['od440bs']
    phcv=math.sqrt( (cv590**2) + (cv440**2) )
    phcv_list.append(round(phcv,3))


combined['ph_cv'] = phcv_list















combined.to_csv('combined1.tsv', float_format='%.3f',
                quoting=csv.QUOTE_NONE,  sep='\t')




combined = pd.read_csv('combined1.tsv', index_col=0, sep='\t',
                         converters={'pair':ast.literal_eval,
                                    'od600':ast.literal_eval,
                                    'sup440':ast.literal_eval,
                                    'sup590':ast.literal_eval,
                                    'bpb440':ast.literal_eval,
                                    'bpb590':ast.literal_eval})



















# COCULTURE TO ISO COMPARISONS FOR OD

od_iav_list=[]
od_iav_ci_list=[]
od_iav_std_list=[]
od_iav_n_list=[]
od_iav_p_list=[]
od_iav_sig_list=[]
od_max_list=[]
od_max_ci_list=[]
od_max_std_list=[]
od_max_n_list=[]
od_max_p_list=[]
od_max_sig_list=[]

med=('media','media')
for index, row in combined.iterrows():
    p0=row['pair'][0]
    p1=row['pair'][1]
    od_iav=np.nan
    od_iav_ci=np.nan
    od_iav_std=np.nan
    od_iav_n=np.nan
    od_iav_p=np.nan
    od_iav_sig=np.nan
    od_max=np.nan
    od_max_ci=np.nan
    od_max_std=np.nan
    od_max_n=np.nan
    od_max_p=np.nan
    od_max_sig=np.nan
    if p0=='media':
        pass
    elif p0==p1:
        pass
    else:
        d0=(p0,p0)
        d1=(p1,p1)
        b0=combined[(combined.cond==row['cond'])&
                    (combined.pair==d0)]['od600bs'].values[0]
        b1=combined[(combined.cond==row['cond'])&
                    (combined.pair==d1)]['od600bs'].values[0]
        od_iav=np.mean([b0,b1])
        od_iav_ci=row['od600bs']/od_iav
        s0=combined[(combined.cond==row['cond'])&
                    (combined.pair==d0)]['od600std'].values[0]
        s1=combined[(combined.cond==row['cond'])&
                    (combined.pair==d1)]['od600std'].values[0]
        od_iav_std=math.sqrt((s0**2)+(s1**2))
        od_iav_n=combined[(combined.cond==row['cond'])&
                    (combined.pair==d0)]['od_n'].values[0]
        tt=scipy.stats.ttest_ind_from_stats(
            row['od600bs'], row['od600std'], row['od_n'],
            od_iav, od_iav_std, od_iav_n, equal_var=False)
        try:
            od_iav_p=tt[1]
        except:
            pass
        if ( (od_iav_p<0.05) and 
             ( abs(row['od600bs']-od_iav) >0.1) ):
            od_iav_sig=True
        else:
            od_iav_sig=False
        od_max=max([b0,b1])
        od_max_ci=row['od600bs']/od_max
        od_max_std=combined[(combined.cond==row['cond'])&
                        (combined.od600bs==od_max)&
                        ((combined.pair==d0)|(combined.pair==d1))]['od600std'].values[0]
        od_max_n=combined[(combined.cond==row['cond'])&
                        (combined.od600bs==od_max)&
                        ((combined.pair==d0)|(combined.pair==d1))]['od_n'].values[0]
        tt=scipy.stats.ttest_ind_from_stats(
            row['od600bs'], row['od600std'], row['od_n'],
            od_max, od_max_std, od_max_n, equal_var=False)
        try:
            od_max_p=tt[1]
        except:
            pass
        if ( (od_max_p<0.05) and 
             ( abs(row['od600bs']-od_max) >0.1) ):
            od_max_sig=True
        else:
            od_max_sig=False
    od_iav_list.append(od_iav)
    od_iav_ci_list.append(od_iav_ci)
    od_iav_std_list.append(od_iav_std)
    od_iav_n_list.append(od_iav_n)
    od_iav_p_list.append(od_iav_p)
    od_iav_sig_list.append(od_iav_sig)
    od_max_list.append(od_max)
    od_max_ci_list.append(od_max_ci)
    od_max_std_list.append(od_max_std)
    od_max_n_list.append(od_max_n)
    od_max_p_list.append(od_max_p)
    od_max_sig_list.append(od_max_sig)



combined['od_iav']=od_iav_list
combined['od_iav_ci']=od_iav_ci_list
combined['od_iav_std']=od_iav_std_list
combined['od_iav_n']=od_iav_n_list
combined['od_iav_p']=od_iav_p_list
combined['od_iav_sig']=od_iav_sig_list
combined['od_max']=od_max_list
combined['od_max_ci']=od_max_ci_list
combined['od_max_std']=od_max_std_list
combined['od_max_n']=od_max_n_list
combined['od_max_p']=od_max_p_list
combined['od_max_sig']=od_max_sig_list













# COCULTURE TO ISO COMPARISONS FOR PH

ph_iav_list=[]
ph_min_list=[]
ph_min_p_list=[]
ph_min_sig_list=[]
ph_min_diff1_list=[]
ph_min_diff2_list=[]

for index, row in combined.iterrows():
    p0=row['pair'][0]
    p1=row['pair'][1]
    ph_iav=np.nan
    ph_min=np.nan
    ph_min_p=np.nan
    ph_min_sig=np.nan
    ph_min_diff1=np.nan
    ph_min_diff2=np.nan
    if p0=='media':
        pass
    elif p0==p1:
        pass
    else:
        d0=(p0,p0)
        d1=(p1,p1)
        h0=combined[(combined.cond==row['cond'])&
                    (combined.pair==d0)]['ph'].values[0]
        h1=combined[(combined.cond==row['cond'])&
                    (combined.pair==d1)]['ph'].values[0]
        ph_iav=np.log10(np.mean([10**h0,10**h1]))
#        ph_iav_n=combined[(combined.cond==row['cond'])&
#                        (combined.ph==ph_min)&
#                        ((combined.pair==d0)|(combined.pair==d1))]['ph_n'].values[0]
#        ph_iav_cv=
        ph_min=min([h0,h1])
        ph_min_diff1=row['ph']-ph_min
        H=10**-row['ph']
        H_min=10**-ph_min
        ph_min_n=combined[(combined.cond==row['cond'])&
                        (combined.ph==ph_min)&
                        ((combined.pair==d0)|(combined.pair==d1))]['ph_n'].values[0]
        ph_min_cv=combined[(combined.cond==row['cond'])&
                        (combined.ph==ph_min)&
                        ((combined.pair==d0)|(combined.pair==d1))]['ph_cv'].values[0]
        ph_min_std=H_min*ph_min_cv
        ph_std=row['ph_cv']*H
        H_delta=H-H_min
        ph_media=3.05
        H_media=10**-ph_media
        if (H_media+H_delta) < 0:
            print row
        phz = -np.log10(H_media+H_delta)
        ph_min_diff2 = phz-ph_media
        try:
            tt=scipy.stats.ttest_ind_from_stats(
                H, ph_std, row['ph_n'],
                H_min, ph_min_std, ph_min_n, equal_var=False)
            ph_min_p=tt[1]
            if ( (ph_min_p<0.05) and 
             ( abs(ph_min_diff2) >0.1) ):
                ph_min_sig=True
            else:
                ph_min_sig=False
        except:
            ph_min_sig=False
    ph_iav_list.append(ph_iav)
    ph_min_list.append(ph_min)
    ph_min_p_list.append(ph_min_p)
    ph_min_sig_list.append(ph_min_sig)
    ph_min_diff1_list.append(ph_min_diff1)
    ph_min_diff2_list.append(ph_min_diff2)


max([incom for incom in ph_min_diff2_list if str(incom) != 'nan'])
min([incom for incom in ph_min_diff2_list if str(incom) != 'nan'])


combined['ph_iav']=ph_iav_list
combined['ph_min']=ph_min_list
combined['ph_min_p']=ph_min_p_list
combined['ph_min_sig']=ph_min_sig_list
combined['ph_min_diff1']=ph_min_diff1_list
combined['ph_min_diff2']=ph_min_diff2_list





x=combined['ph_min_diff1']
y=combined['ph_min_diff2']
z=combined['ph']

#cmap = sns.cubehelix_palette(as_cmap=True)
cmap= sns.dark_palette('red', as_cmap=True, reverse=True)
f, ax = plt.subplots()
points = ax.scatter(x, y, c=z, s=50, cmap=cmap)
f.colorbar(points)
plt.show()














combined.to_csv('combined2.tsv', float_format='%.3f',
                quoting=csv.QUOTE_NONE,  sep='\t')






combined = pd.read_csv('combined2.tsv', index_col=0, sep='\t',
                         converters={'pair':ast.literal_eval,
                                    'od600':ast.literal_eval,
                                    'sup440':ast.literal_eval,
                                    'sup590':ast.literal_eval,
                                    'bpb440':ast.literal_eval,
                                    'bpb590':ast.literal_eval})













# ADD GENETIC DISTANCE DATA



# PROT

prot=pd.read_csv('dist_prot.csv')


prot_list=[]
for index,row in combined.iterrows():
    p0=row['pair'][0]
    p1=row['pair'][1]
    dist=np.nan
    if p0!=p1:
        try:
            dist=round(prot[(prot.Bug_1==p0)&(prot.Bug_2==p1)]['Distance'].values[0],3)
        except:
            pass
    prot_list.append(dist)

combined['dist_prot']=prot_list




# PATH (CONVERT NAMES) 

path=pd.read_csv('dist_path.csv')
abrv2=pd.read_csv('abrv2.csv')
b1=[]
b2=[]
for index, row in path.iterrows():
    b1.append(abrv2[abrv2.name==row['Bug_1']]['ab'].values[0])
    b2.append(abrv2[abrv2.name==row['Bug_2']]['ab'].values[0])


path['Bug_1']=b1
path['Bug_2']=b2

path_list=[]
for index,row in combined.iterrows():
    p0=row['pair'][0]
    p1=row['pair'][1]
    dist=np.nan
    if p0!=p1:
        try:
            dist=round(path[(path.Bug_1==p0)&(path.Bug_2==p1)]['Distance'].values[0],3)
        except:
            pass
    path_list.append(dist)

combined['dist_path']=path_list





# SEED (CONVERT NAMES) 

seed=pd.read_csv('dist_seed.csv')
b1=[]
b2=[]
for index, row in seed.iterrows():
    b1.append(abrv2[abrv2.name==row['Bug_1']]['ab'].values[0])
    b2.append(abrv2[abrv2.name==row['Bug_2']]['ab'].values[0])

seed['Bug_1']=b1
seed['Bug_2']=b2


competition_list=[]
complementarity_list=[]
ratio_list=[]
for index,row in combined.iterrows():
    p0=row['pair'][0]
    p1=row['pair'][1]
    dist1=np.nan
    dist2=np.nan
    dist3=np.nan
    if p0!=p1:
        try:
            dist1=round(seed[(seed.Bug_1==p0)&(seed.Bug_2==p1)]['Competition'].values[0],3)
            dist2=round(seed[(seed.Bug_1==p0)&(seed.Bug_2==p1)]['Complementarity'].values[0],3)
            dist3=round((dist2/dist1),3)
        except:
            pass
    competition_list.append(dist1)
    complementarity_list.append(dist2)
    ratio_list.append(dist3)

combined['seed_compete']=competition_list
combined['seed_complem']=complementarity_list
combined['seed_ratio']=ratio_list





# GGDC (CONVERT NAMES) 

b1=[]
b2=[]
abrv2=pd.read_csv('abrv2.csv')
ggdc=pd.read_csv('dist_ggdc.csv', index_col=0)
ggdc=ggdc.stack().reset_index()
ggdc.columns=['Bug_1','Bug_2','ggdc']
for index, row in ggdc.iterrows():
    b1.append(abrv2[abrv2.name==row['Bug_1']]['ab'].values[0])
    b2.append(abrv2[abrv2.name==row['Bug_2']]['ab'].values[0])

ggdc['Bug_1']=b1
ggdc['Bug_2']=b2


ggdc_list=[]
for index,row in combined.iterrows():
    p0=row['pair'][0]
    p1=row['pair'][1]
    dist=np.nan
    if p0!=p1:
        try:
            dist=round(ggdc[(ggdc.Bug_1==p0)&(ggdc.Bug_2==p1)]['ggdc'].values[0],3)
        except:
            pass
    ggdc_list.append(dist)

combined['dist_ggdc']=ggdc_list


# Need some ribosomal and ANI distances!!!



combined.to_csv('combined3.tsv', float_format='%.3f',
                quoting=csv.QUOTE_NONE,  sep='\t')






combined = pd.read_csv('combined3.tsv', index_col=0, sep='\t',
                         converters={'pair':ast.literal_eval,
                                    'od600':ast.literal_eval,
                                    'sup440':ast.literal_eval,
                                    'sup590':ast.literal_eval,
                                    'bpb440':ast.literal_eval,
                                    'bpb590':ast.literal_eval})



