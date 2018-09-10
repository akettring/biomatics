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

measures=['od600bs', 'od_max', 'od_iav', 'od_iav_ci', 'od_max_ci', 'ph', 'ph_min', 'ph_iav','ph_min_diff1','ph_min_diff2']
metrics=['dist_ggdc', 'dist_prot', 'dist_path','seed_compete', 'seed_complem', 'seed_ratio']




combined = pd.read_csv('combined3.tsv', index_col=0, sep='\t',
                         converters={'pair':ast.literal_eval,
                                    'od600':ast.literal_eval,
                                    'sup440':ast.literal_eval,
                                    'sup590':ast.literal_eval,
                                    'bpb440':ast.literal_eval,
                                    'bpb590':ast.literal_eval})







#pH normalization scatterplot

x=combined['ph_min_diff1']
y=combined['ph_min_diff2']
z=combined['ph']

sns.set(style='white', font='Arial', font_scale=2)
sns.set_style("ticks")

cmap= sns.dark_palette('red', as_cmap=True, reverse=True)
f, ax = plt.subplots()
points = ax.scatter(x, y, c=z, s=25, cmap=cmap)
f.colorbar(points)
plt.xlabel('Delta pH Raw')
plt.ylabel('Delta pH Scaled')

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.show()












#Bromophenol blue 

sns.set(style='white', font='Arial', font_scale=2)
sns.set_style("ticks")


col_list = ["dandelion", "yellowgreen", "green", "bluegreen", "blue", "burple"]
col_list.reverse()
col_list_palette = sns.xkcd_palette(col_list)


bpb1=pd.read_csv('bpb1.csv',index_col=0)
bpb1=bpb1.stack().reset_index().rename(columns={'level_1':'pH', 0:'Absorbance'})
bpb1.pH = bpb1.pH.astype(float)
bpb1.sort_values(by=['pH','Wavelength']).reindex(columns=['pH','Wavelength','Absorbance']).reset_index(drop=True)


fig, (ax1,ax2) = plt.subplots(1,2)

phlist=bpb1['pH'].unique().tolist()
for p,c in zip(phlist, col_list_palette):
    xlist=bpb1[bpb1.pH==p]['Wavelength'].values
    ylist=bpb1[bpb1.pH==p]['Absorbance'].values
    print xlist,ylist,c
    ax1.plot(xlist,ylist,color=c,label=p,lw=3)

legend=ax1.legend()



bpb2=pd.read_csv('bpb2.csv')
ps=bpb2['pH'].values.tolist()
rs=bpb2['ratio'].values.tolist()
ax2.scatter(ps,rs,color=col_list_palette,s=200)


def fun(r):
    return 0.7549992537*np.log(r)+3.3426

y=np.arange(0.66, 22.20, 0.1)
ax2.plot(fun(y),y)


ax1.set_title('A')
ax1.set_xlabel('Wavelength (nm)')
ax1.set_ylabel('Absorbance')
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)

ax2.set_title('B')
ax2.set_xlabel('Growth Media pH')
ax2.set_ylabel('OD 590/440 nm')
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)



plt.show()








measures=['od600bs', 'od_max', 'od_iav', 'od_iav_ci', 'od_max_ci', 'ph', 'ph_min', 'ph_iav','ph_min_diff1','ph_min_diff2']
metrics=['dist_ggdc', 'dist_prot', 'dist_path','seed_compete', 'seed_complem', 'seed_ratio']










# NULL MODEL SELECTION

m0_list=[]
m1_list=[]
slope_list=[]
p_list=[]
r2_list=[]
for m0 in measures:
    for m1 in measures:
        x=combined[m0].values
        y=combined[m1].values
        mask = ~np.isnan(x) & ~np.isnan(y)
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x[mask],y[mask])
        m0_list.append(m0)
        m1_list.append(m1)
        slope_list.append(slope)
        p_list.append(p_value)
        r2_list.append((r_value**2))

mes_df=pd.DataFrame({'m0':m0_list,
                    'm1':m1_list,
                    'slope':slope_list,
                    'p':p_list,
                    'r2':r2_list})


mes_df.to_csv('mes_df.csv')

# Do measures correlate? 
    # By p value < 0.05, all except (od_iav_ci, od600bs), (od_iav_ci,ph_min)
    # By r squared > 0.1, only (ph, ph_min 0.82) & (od_iav_ci, od_max_ci 0.59)

# Which is better null model (avg or minmax)?
    #            m0         m1     p    r2  slope
    # 1     od600bs     od_max 0.000 0.840  0.931
    # 2     od600bs     od_iav 0.000 0.786  0.701
    # 46         ph     ph_min 0.000 0.823  0.911
    # 47         ph     ph_iav 0.000 0.401  0.586

# Visual comparison of linear regression. Also V-plots.







#Old conditional null graphs
g=sns.lmplot(y='od600bs', x='od_max', hue='cond', data=combined, palette='Blues_d')
plt.show()
g=sns.lmplot(y='od600bs', x='od_iav', hue='cond', data=combined, palette='Blues_d')
plt.show()
g=sns.lmplot(y='ph', x='ph_min', hue='cond', data=combined, palette='Reds_d')
plt.show()
g=sns.lmplot(y='ph', x='ph_iav', hue='cond', data=combined, palette='Reds_d')
plt.show()









# New null model graphs
sns.set(style='white', font='Arial', font_scale=2)
sns.set_style("ticks")


fig, (ax1,ax3) = plt.subplots(1,2)
fig, (ax2,ax4) = plt.subplots(1,2)

g1=sns.regplot(x='od_max', y='od600bs', data=combined, color='b', ax=ax1)
g2=sns.regplot(x='ph_min', y='ph', data=combined, color='r', ax=ax2)
g3=sns.regplot(x='od_iav', y='od600bs', data=combined, color='b', ax=ax3)
g4=sns.regplot(x='ph_iav', y='ph', data=combined, color='r', ax=ax4)

ax1.set_xlim(0, 2)
ax1.set_ylim(0, 2)
ticks=np.arange(0.5,2,0.5)
ax1.set_xticks(ticks)
ax1.set_yticks(ticks)
ax1.set_xlabel('OD600 MAX')
ax1.set_ylabel('Co-culture OD600')
ax1.set_title('A')

ax2.set_xlim(2.5, 5.5)
ax2.set_ylim(2.5, 5.5)
ticks2=np.arange(3,6,1)
ax2.set_xticks(ticks2)
ax2.set_yticks(ticks2)
ax2.set_xlabel('pH MIN')
ax2.set_ylabel('Co-culture pH')
ax2.set_title('C')

ax3.set_xlim(0, 2)
ax3.set_ylim(0, 2)
ax3.set_xticks(ticks)
ax3.set_yticks(ticks)
ax3.set_yticklabels('')
ax3.set_xlabel('OD600 AVG')
ax3.set_ylabel('')
ax3.set_title('B')

ax4.set_xlim(2.5, 5.5)
ax4.set_ylim(2.5, 5.5)
ax4.set_xticks(ticks2)
ax4.set_yticks(ticks2)
ax4.set_yticklabels('')
ax4.set_xlabel('pH AVG')
ax4.set_ylabel('')
ax4.set_title('D')

ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
ax4.spines['top'].set_visible(False)

plt.show()









# SELECTING A BIOMATIC METRIC

m0_list=[]
m1_list=[]
slope_list=[]
p_list=[]
r2_list=[]
for m0 in metrics:
    for m1 in metrics:
        x=combined[m0].values
        y=combined[m1].values
        mask = ~np.isnan(x) & ~np.isnan(y)
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x[mask],y[mask])
        m0_list.append(m0)
        m1_list.append(m1)
        slope_list.append(slope)
        p_list.append(p_value)
        r2_list.append((r_value**2))


met_df=pd.DataFrame({'m0':m0_list,
                    'm1':m1_list,
                    'slope':slope_list,
                    'p':p_list,
                    'r2':r2_list})

met_df.sort_values(by=['r2'], ascending=False)








# Do metrics correlate? (Yes, All)

sns.set(style='white', font='Arial', font_scale=2)
sns.set_style("ticks")

fig, (ax1,ax2,ax3) = plt.subplots(1,3)

g1=sns.regplot(y='dist_prot', x='seed_compete', data=combined, color='Black', marker='v', ax=ax1, scatter_kws={'s':50})
g2=sns.regplot(y='dist_prot', x='seed_complem', data=combined, color='Black', marker='+', ax=ax2, scatter_kws={'s':75})
g3=sns.regplot(y='dist_prot', x='seed_ratio', data=combined, color='Black', marker='*', ax=ax3, logx=True, scatter_kws={'s':75})

ax1.set_ylim(0, 1)
ax2.set_ylim(0, 1)
ax3.set_ylim(0, 1)

ax1.set_xlim(0, 0.8)
ax2.set_xlim(0, 0.2)
ax3.set_xlim(0, 1)

ax1.set_ylabel('Proteome BLAST Distance')
ax2.set_ylabel('')
ax3.set_ylabel('')

ax2.set_yticklabels('')
ax3.set_yticklabels('')

ax1.set_xlabel('Seed Competition')
ax2.set_xlabel('Seed Complementarity')
ax3.set_xlabel('Seed Ratio')

ax1.set_title('A')
ax2.set_title('B')
ax3.set_title('C')

sns.despine()

plt.show()











# Which metrics and measures correlate? 
#       od600bs and pH well explained by prot, path, and seeds
#       od_iav_ci only explained by seed complem

mx_list=[]
ms_list=[]
slope_list=[]
p_list=[]
r2_list=[]
for mx in metrics:
    for ms in measures:
        x=combined[mx].values
        y=combined[ms].values
        mask = ~np.isnan(x) & ~np.isnan(y)
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x[mask],y[mask])
        mx_list.append(mx)
        ms_list.append(ms)
        slope_list.append(slope)
        p_list.append(p_value)
        r2_list.append((r_value**2))

mm_df=pd.DataFrame({'mx':mx_list,
                    'ms':ms_list,
                    'slope':slope_list,
                    'p':p_list,
                    'r2':r2_list})

mm_df.to_csv('mm_df.csv')

mm_df.sort_values(by=['ms'], ascending=False)



plt.show(sns.lmplot(x='dist_prot', y='ph', hue='cond', data=combined))
plt.show(sns.lmplot(x='dist_prot', y='od600bs', hue='cond', data=combined))
plt.show(sns.lmplot(x='dist_prot', y='od_max_ci', hue='cond', data=combined))
plt.show(sns.lmplot(x='dist_prot', y='ph_min_diff2', hue='cond', data=combined))









# Under which conditions do metrics, and measures correlate?
# 1040, ibu42, 15psi best explained

sig_list=[]
for c in cond0:
    for ms in measures:
        for mx in metrics:
            x=combined[(combined.cond==c)][mx].values
            y=combined[(combined.cond==c)][ms].values
            mask = ~np.isnan(x) & ~np.isnan(y)
            slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x[mask],y[mask])
            if p_value < 0.05:
                s= c, ms, mx, p_value
                sig_list.append(s)

for s in sig_list:
    print s
    print








# GENETIC DISTANCE PLOTS

g=sns.lmplot(x='dist_prot', y='od600bs', hue='cond', data=combined, palette='Blues_d')
plt.show()

g=sns.lmplot(x='dist_prot', y='ph', hue='cond', data=combined, palette='Blues_d')
plt.show()

g=sns.lmplot(x='dist_prot', y='ph_min_diff1', hue='cond', data=combined, palette='Reds_d')
plt.show()

g=sns.lmplot(x='seed_ratio', y='ph_min_diff2', hue='cond', data=combined, palette='Reds_d')

plt.show()








# GENETIC DISTANCE PLOTS TU


fig, (ax1,ax3,ax2,ax4) = plt.subplots(1,4)

g1=sns.regplot(y='seed_ratio', x='od600bs', data=combined, color='b', ax=ax1)
g2=sns.regplot(y='seed_ratio', x='ph', data=combined, color='r', ax=ax2)

g3=sns.regplot(y='seed_ratio', x='od_max_ci', data=combined, color='b', ax=ax3)
g4=sns.regplot(y='seed_ratio', x='ph_min_diff2', data=combined, color='r', ax=ax4)



ax1.set_ylabel('Metabolic Seed Ratio')
ax2.set_ylabel('')
ax3.set_ylabel('')
ax4.set_ylabel('')

ax2.set_yticklabels('')
ax3.set_yticklabels('')
ax4.set_yticklabels('')

ax1.set_xlabel('OD600')
ax2.set_xlabel('pH')
ax3.set_xlabel('EXP/MAX')
ax4.set_xlabel('pH MIN delta')

ax1.set_title('A')
ax2.set_title('C')
ax3.set_title('B')
ax4.set_title('D')

sns.despine()

plt.show()













# ISOLATE HEATMAPS


strains0 = pd.read_csv('strains0.csv')
strains1=[]
for ab in abrv0:
    name=(strains0[strains0.abrv==ab]['genus'].values[0]+' '+
            strains0[strains0.abrv==ab]['species'].values[0])
    strains1.append(name)
    


c_list=[]
n_list=[]
o_list=[]
p_list=[]
for c in cond0:
    for ab in abrv0:
        c_list.append(c)
        name=(strains0[strains0.abrv==ab]['genus'].values[0]+' '+
            strains0[strains0.abrv==ab]['species'].values[0])
        n_list.append(name)
        aa=(ab,ab)
        od=combined[(combined.cond==c)&
                    (combined.pair==aa)]['od600bs'].values[0]
        o_list.append(od)
        p=combined[(combined.cond==c)&
                    (combined.pair==aa)]['ph'].values[0]
        p_list.append(p)


idf=pd.DataFrame({'cond':c_list,
                    'name':n_list,
                    'od':o_list,
                    'ph':p_list})


od = idf.pivot('name', 'cond', 'od')
ph = idf.pivot('name', 'cond', 'ph')


od = od.reindex(strains1)
ph = ph.reindex(strains1)


od = od.reindex(['1040', 'IBU42', '1PSI', '15PSI'], axis=1)
ph = ph.reindex(['1040', 'IBU42', '1PSI', '15PSI'], axis=1)



sns.set(style='whitegrid', font='Arial', font_scale=1)

fig, (ax1, ax2) = plt.subplots(1,2)
sns.heatmap(od, annot=True, fmt='.1f',
                    cmap='Blues', linewidths=.5, ax=ax1,
                    cbar=False)

sns.heatmap(ph, annot=True, fmt='.1f',
                    cmap='Reds_r', linewidths=.5, ax=ax2,
                    cbar=False, yticklabels=False)

ax1.set_ylabel('')    
ax1.set_xlabel('')
ax1.set_title('A')
ax2.set_ylabel('')    
ax2.set_xlabel('')
ax2.set_title('B')

ax1.set_xticklabels(ax1.get_xticklabels(),rotation=315)
ax2.set_xticklabels(ax2.get_xticklabels(),rotation=315)

ax1.set_yticklabels(ax1.get_yticklabels(),style='oblique')

plt.show()













# COMBINED V-PLOTS

comb1=list(itertools.combinations(abrv0,2))
comb1.sort()

cond_list=[]
isococ_list=[]
od600_list=[]
pH_list=[]
for c0 in cond0:
    for c1 in comb1:
        c2=(c1[1],c1[0])
        c=combined[(combined.cond==c0)&
                    ((combined.pair==c1)|(combined.pair==c2))
                  ]['od600bs'].values[0]
        i=combined[(combined.cond==c0)&
                    ((combined.pair==c1)|(combined.pair==c2))
                  ]['od_max'].values[0]
        pc=combined[(combined.cond==c0)&
                    ((combined.pair==c1)|(combined.pair==c2))
                  ]['ph'].values[0]
        pi=combined[(combined.cond==c0)&
                    ((combined.pair==c1)|(combined.pair==c2))
                  ]['ph_min'].values[0]
        cond_list.append(c0)
        isococ_list.append('MAX')
        od600_list.append(i)
        pH_list.append(pi)
        cond_list.append(c0)
        isococ_list.append('EXP')
        od600_list.append(c)
        pH_list.append(pc)
 


v_dat=pd.DataFrame({'cond':cond_list,
                    'isococ':isococ_list,
                    'od600':od600_list,
                    'ph':pH_list})



comb1=list(itertools.combinations(abrv0,2))
comb1.sort()

cond_list=[]
isococ_list=[]
od600_list=[]
pH_list=[]
for c0 in cond0:
    for c1 in comb1:
        c2=(c1[1],c1[0])
        c=combined[(combined.cond==c0)&
                    ((combined.pair==c1)|(combined.pair==c2))
                  ]['od600bs'].values[0]
        i=combined[(combined.cond==c0)&
                    ((combined.pair==c1)|(combined.pair==c2))
                  ]['od_max'].values[0]
        pc=combined[(combined.cond==c0)&
                    ((combined.pair==c1)|(combined.pair==c2))
                  ]['ph'].values[0]
        pi=combined[(combined.cond==c0)&
                    ((combined.pair==c1)|(combined.pair==c2))
                  ]['ph_min'].values[0]
        cond_list.append(c0)
        isococ_list.append('MIN')
        od600_list.append(i)
        pH_list.append(pi)
        cond_list.append(c0)
        isococ_list.append('EXP')
        od600_list.append(c)
        pH_list.append(pc)
 


v_dat0=pd.DataFrame({'cond':cond_list,
                    'isococ':isococ_list,
                    'od600':od600_list,
                    'ph':pH_list})





cond_list=[]
isococ_list=[]
od600_list=[]
pH_list=[]
for c0 in cond0:
    for c1 in comb1:
        c2=(c1[1],c1[0])
        c=combined[(combined.cond==c0)&
                    ((combined.pair==c1)|(combined.pair==c2))
                  ]['od600bs'].values[0]
        i=combined[(combined.cond==c0)&
                    ((combined.pair==c1)|(combined.pair==c2))
                  ]['od_iav'].values[0]
        pc=combined[(combined.cond==c0)&
                    ((combined.pair==c1)|(combined.pair==c2))
                  ]['ph'].values[0]
        pi=combined[(combined.cond==c0)&
                    ((combined.pair==c1)|(combined.pair==c2))
                  ]['ph_iav'].values[0]
        cond_list.append(c0)
        isococ_list.append('AVG')
        od600_list.append(i)
        pH_list.append(pi)
        cond_list.append(c0)
        isococ_list.append('EXP')
        od600_list.append(c)
        pH_list.append(pc)
 



v_dat2=pd.DataFrame({'cond':cond_list,
                    'isococ':isococ_list,
                    'od600':od600_list,
                    'ph':pH_list})




sns.set(style='whitegrid', font='Arial', font_scale=2)


fig, (ax1, ax3) = plt.subplots(1,2)

fig, (ax2, ax4) = plt.subplots(1,2)

vod=sns.violinplot(x='cond', y='od600', hue='isococ', data=v_dat,
                 split=True, inner='quartile', linewidth=2,
                 cut=-0.25, palette='Blues', scale='count', bw=.25,
                 ax=ax1, legend_out = True)

vph=sns.violinplot(x='cond', y='ph', hue='isococ', data=v_dat0,
                 split=True, inner='quartile', linewidth=2,
                 cut=-0.25, palette='Reds', scale='count', bw=.25,
                 ax=ax2, legend_out = True)


ax1.set_ylim(0, 2)
ticks=np.arange(0,2,0.5)
ax1.set_yticks(ticks)
ax1.set_xlabel('')
ax1.set_ylabel('OD600')
ax1.legend_.set_title('')
ax1.legend(loc='upper right')
ax1.set_title('A')

ax2.set_ylim(2.5, 6)
ticks2=np.arange(2.5,6,0.5)
ax2.set_yticks(ticks2)
ax2.set_xlabel('')
ax2.set_ylabel('pH')
ax2.legend_.set_title('')
ax2.legend(loc='upper center')
ax2.set_title('C')

vod2=sns.violinplot(x='cond', y='od600', hue='isococ', data=v_dat2,
                 split=True, inner='quartile', linewidth=2,
                 cut=-0.25, palette='Blues', scale='count', bw=.25,
                 ax=ax3, legend_out = True)

vph2=sns.violinplot(x='cond', y='ph', hue='isococ', data=v_dat2,
                 split=True, inner='quartile', linewidth=2,
                 cut=-0.25, palette='Reds', scale='count', bw=.25,
                 ax=ax4, legend_out = True)


ax3.set_ylim(0, 2)
ticks=np.arange(0,2,0.5)
ax3.set_yticks(ticks)
ax3.set_yticklabels('')
ax3.set_xlabel('')
ax3.set_ylabel('')
ax3.legend_.set_title('')
ax3.legend(loc='upper right')
ax3.set_title('B')

ax4.set_ylim(2.5, 6)
ticks2=np.arange(2.5,6,0.5)
ax4.set_yticks(ticks2)
ax4.set_yticklabels('')
ax4.set_xlabel('')
ax4.set_ylabel('')
ax4.legend_.set_title('')
ax4.legend(loc='upper center')
ax4.set_title('D')


plt.show()



#for c in cond0:
#    iso=v_dat[(v_dat.cond==c)&(v_dat.isococ=='Null Model')]['od600'].values
#    coc=v_dat[(v_dat.cond==c)&(v_dat.isococ=='Co-culture')]['od600'].values
#    tt=scipy.stats.ttest_ind(iso, coc, equal_var=False, nan_policy='omit')
#    print c,tt[1]

#for c in cond0:
#    iso=v_dat[(v_dat.cond==c)&(v_dat.isococ=='Null Model')]['ph'].values
#    coc=v_dat[(v_dat.cond==c)&(v_dat.isococ=='Co-culture')]['ph'].values
#    tt=scipy.stats.ttest_ind(iso, coc, equal_var=False, nan_policy='omit')
#    print c,tt[1]













# DOUBLE HEATMAPAS - Raw OD and pH



c_list=[]
p0_list=[]
p1_list=[]
od_list=[]
sig_list=[]

for index, row in combined.iterrows():
    if row['pair'][0]=='media':
        pass
    elif row['pair'][0]==row['pair'][1]: 
        pass
    else:
        c_list.append(row['cond'])
        p0_list.append(row['pair'][0])
        p1_list.append(row['pair'][1])
        od_list.append(row['od600bs'])
        sig_list.append(row['od_max_sig'])
        if row['pair'][0]==row['pair'][1]:
            pass
        else:
            c_list.append(row['cond'])
            p0_list.append(row['pair'][1])
            p1_list.append(row['pair'][0])
            od_list.append(row['od600bs'])
            sig_list.append(row['od_max_sig'])


df_od=pd.DataFrame({'cond':c_list,'p0':p0_list,'p1':p1_list,
                     'od':od_list,'sig':sig_list},
                columns=['cond','p0','p1','od','sig'])
df_od=df_od.sort_values(by=['cond','p0','p1','sig'])
df_od=df_od.reset_index(drop=True)



c_list=[]
p0_list=[]
p1_list=[]
md_list=[]
sig2_list=[]

for index, row in combined.iterrows():
    if row['pair'][0]=='media':
        pass
    elif row['pair'][0]==row['pair'][1]: 
        pass
    else:
        c_list.append(row['cond'])
        p0_list.append(row['pair'][0])
        p1_list.append(row['pair'][1])
        md_list.append(row['ph'])
        sig2_list.append(row['ph_min_sig'])
        if row['pair'][0]==row['pair'][1]:
            pass
        else:
            c_list.append(row['cond'])
            p0_list.append(row['pair'][1])
            p1_list.append(row['pair'][0])
            md_list.append(row['ph'])
            sig2_list.append(row['ph_min_sig'])  


df_ph=pd.DataFrame({'cond':c_list,'p0':p0_list,
                'p1':p1_list, 'ph':md_list,'sig2':sig2_list},
                columns=['cond','p0','p1','ph','sig2'])

df_ph=df_ph.sort_values(by=['cond','p0','p1'])
df_ph=df_ph.reset_index(drop=True)



sns.set(style='whitegrid', font='Arial', font_scale=0.8)
#redsd=sns.color_palette("Reds_d")
titlefont={'size':14, 'weight':'bold'}

alist=['A','B','C','D']
for c,a in zip(cond0,alist):
    dfo1=df_od[df_od.cond==c]
    dfo2=dfo1.pivot(index='p0', columns='p1', values='od')
    dfo2=dfo2.reindex(index=abrv0, columns=abrv0)
    #
    masko = np.zeros_like(dfo2)
    masko[np.triu_indices_from(masko)] = True
    masko[np.tril_indices_from(masko)] = False
    #
    dfp1=df_ph[df_ph.cond==c]
    dfp2=dfp1.pivot(index='p0', columns='p1', values='ph')
    dfp2=dfp2.reindex(index=abrv0, columns=abrv0)
    #
    maskp = np.zeros_like(dfp2)
    maskp[np.tril_indices_from(maskp)] = True
    maskp[np.triu_indices_from(maskp)] = False
    with sns.axes_style('white'):
        sns.heatmap(dfo2, cmap='Blues', annot=True,# annot_kws={"weight": "bold"},
                    fmt='.1f', linewidths=.1, mask=masko, cbar=False,
                    vmax=1.5, vmin=0, square=True)
        sns.heatmap(dfp2, cmap='Reds_r', annot=True,# annot_kws={"weight": "bold"},
                    fmt='.1f', linewidths=.1, mask=maskp, cbar=False,
                    vmax=5.3, vmin=3)
        plt.title(a+' - '+c,fontdict=titlefont)
        plt.xlabel('')
        plt.ylabel('')
        plt.xticks(fontsize=12)
        plt.yticks(rotation=0, fontsize=12)
        plt.show()










# compute average pH based on condition

for c in cond0:
    print c
    phs=combined[combined.cond==c]['ph'].values
    phz=[10**float(i) for i in phs if ~np.isnan(i)]
    print ' avg_ph=', round(np.log10(np.mean(phz)),3)
    print



















# DOUBLE HEATMAPAS - Null model comparisons


c_list=[]
p0_list=[]
p1_list=[]
ci_list=[]
sig_list=[]

for index, row in combined.iterrows():
    if row['pair'][0]=='media':
        pass
    elif row['pair'][0]==row['pair'][1]: 
        pass
    else:
        c_list.append(row['cond'])
        p0_list.append(row['pair'][0])
        p1_list.append(row['pair'][1])
        ci_list.append(row['od_max_ci'])
        sig_list.append(row['od_max_sig'])
        if row['pair'][0]==row['pair'][1]:
            pass
        else:
            c_list.append(row['cond'])
            p0_list.append(row['pair'][1])
            p1_list.append(row['pair'][0])
            ci_list.append(row['od_max_ci'])
            sig_list.append(row['od_max_sig'])


df_od=pd.DataFrame({'cond':c_list,'p0':p0_list,'p1':p1_list,
                     'ci':ci_list,'sig':sig_list},
                columns=['cond','p0','p1','ci','sig'])
df_od=df_od.sort_values(by=['cond','p0','p1','sig'])
df_od=df_od.reset_index(drop=True)



c_list=[]
p0_list=[]
p1_list=[]
md_list=[]
sig2_list=[]

for index, row in combined.iterrows():
    if row['pair'][0]=='media':
        pass
    elif row['pair'][0]==row['pair'][1]: 
        pass
    else:
        c_list.append(row['cond'])
        p0_list.append(row['pair'][0])
        p1_list.append(row['pair'][1])
        md_list.append(row['ph_min_diff2'])
        sig2_list.append(row['ph_min_sig'])
        if row['pair'][0]==row['pair'][1]:
            pass
        else:
            c_list.append(row['cond'])
            p0_list.append(row['pair'][1])
            p1_list.append(row['pair'][0])
            md_list.append(row['ph_min_diff2'])
            sig2_list.append(row['ph_min_sig'])  


df_ph=pd.DataFrame({'cond':c_list,'p0':p0_list,
                'p1':p1_list, 'ph':md_list,'sig2':sig2_list},
                columns=['cond','p0','p1','ph','sig2'])

df_ph=df_ph.sort_values(by=['cond','p0','p1'])
df_ph=df_ph.reset_index(drop=True)




sns.set(style='whitegrid', font='Arial', font_scale=0.8)
#edsd=sns.color_palette("Reds_d")
titlefont={'size':14, 'weight':'bold'}

elist=['E','F','G','H']
for c,e in zip(cond0,elist):
    dfo1=df_od[df_od.cond==c]
    dfo2=dfo1.pivot(index='p0', columns='p1', values='ci')
    dfo2=dfo2.reindex(index=abrv0, columns=abrv0)
    #
    dfo3=dfo1.pivot(index='p0', columns='p1', values='sig')
    dfo3=dfo3.reindex(index=abrv0, columns=abrv0)*-1
    dfo4=dfo3.replace(to_replace=[0,-1],value=[0,1]).fillna(value=1).values
    dfo3=dfo3.replace(to_replace=[0,-1],value=[1,0]).fillna(value=1).values
    #
    masko = np.zeros_like(dfo2)
    masko[np.triu_indices_from(masko)] = True
    masko[np.tril_indices_from(masko)] = False
    #
    masko2=masko+dfo3
    masko2=((masko2!=0)*1).astype(float)
    #
    masko3=masko+dfo4
    masko3=((masko3!=0)*1).astype(float)
    #
    dfp1=df_ph[df_ph.cond==c]
    dfp2=dfp1.pivot(index='p0', columns='p1', values='ph')
    dfp2=dfp2.reindex(index=abrv0, columns=abrv0)
    #
    dfp3=dfp1.pivot(index='p0', columns='p1', values='sig2')
    dfp3=dfp3.reindex(index=abrv0, columns=abrv0)*-1
    dfp4=dfp3.replace(to_replace=[0,-1],value=[0,1]).fillna(value=1).values
    dfp3=dfp3.replace(to_replace=[0,-1],value=[1,0]).fillna(value=1).values
    #
    maskp = np.zeros_like(dfp2)
    maskp[np.tril_indices_from(maskp)] = True
    maskp[np.triu_indices_from(maskp)] = False
    #
    maskp2=maskp+dfp3
    maskp2=((maskp2!=0)*1).astype(float)
    #
    maskp3=maskp+dfp4
    maskp3=((maskp3!=0)*1).astype(float)
    #
    with sns.axes_style('white'):
        sns.heatmap(dfo2, cmap='Blues', annot=False,
                    fmt='.1f', linewidths=0, mask=masko3, cbar=False,
                    vmax=2, vmin=0, square=True)
        sns.heatmap(dfo2, cmap='Blues', annot=True,# annot_kws={"weight": "bold"},
                    fmt='.1f', linewidths=0, mask=masko2, cbar=False,
                    vmax=2, vmin=0, square=True)
        sns.heatmap(dfp2, cmap='Reds_r', annot=False,
                    fmt='.1f', linewidths=0, mask=maskp3, cbar=False,
                    vmax=.9, vmin=-0.2)
        sns.heatmap(dfp2, cmap='Reds_r', annot=True,# annot_kws={"weight": "bold"},
                    fmt='.1f', linewidths=0.1, mask=maskp2, cbar=False,
                    vmax=.9, vmin=-0.2)
        plt.title(e+' - '+c,fontdict=titlefont)
        plt.xlabel('')
        plt.ylabel('')
        plt.xticks(fontsize=12)
        plt.yticks(rotation=0, fontsize=12)
        plt.show()

















# SIGNIFICANT RELATIONSHIPS

df=combined[['cond','od_max_ci','od_max_sig','ph_min_diff2','ph_min_sig','dist_prot','seed_ratio']]
osp_list=[]
osn_list=[]
psp_list=[]
psn_list=[]

for index, row in df.iterrows():
    if row['od_max_ci']>1:
        op=True
        on=False
    else:
        op=False
        on=True
    osp=row['od_max_sig']*op
    osn=row['od_max_sig']*on
    if row['ph_min_diff2']<0:
        pn=True
        pp=False
    else:
        pn=False
        pp=True
    psn=row['ph_min_sig']*pn
    psp=row['ph_min_sig']*pp
    osp_list.append(osp)
    osn_list.append(osn)
    psp_list.append(psp)
    psn_list.append(psn)

df['osp']=osp_list
df['osn']=osn_list
df['psp']=psp_list
df['psn']=psn_list

g=sns.lmplot(x='dist_prot', y='osp', hue='cond', data=df, palette='Blues_d')
plt.show()
g=sns.lmplot(x='dist_prot', y='osn', hue='cond', data=df, palette='Blues_d')
plt.show()
g=sns.lmplot(x='dist_prot', y='psp', hue='cond', data=df, palette='Reds_d')
plt.show()
g=sns.lmplot(x='dist_prot', y='psn', hue='cond', data=df, palette='Reds_d')
plt.show()






