from matplotlib import rcParams
from matplotlib import cm
import numpy as np
import matplotlib.pyplot as pp

pp.style.use('ggplot')
rcParams["font.size"] = 20
rcParams['axes.edgecolor']='#b0b0b0'
rcParams['axes.labelcolor']='black'
rcParams['ytick.color']='black'
rcParams['xtick.color']='black'

hatches = ['.','//','\\\\',None]

#Fig 2A
#file is 2a_human_actions.csv
data=np.loadtxt('2a_human_actions.csv',delimiter=',')

cum0=np.zeros(10)
fig,ax = pp.subplots(1,1)
for o in [0,2,1,3]:#Order is to get the types in [I,S,C,Other] ordering
    pp.bar(range(0,10),data[o],bottom=cum0,width=0.5,hatch=hatches[o],
              edgecolor='k',linewidth=2)
    cum0+=n[o]

    
ax.set_facecolor('w')
ax.grid(visible='both',color='xkcd:gray')
ax.set_ylabel('Frequency')
ax.set_xlabel('Action')

ax.set_xticks(np.arange(0.5,10.5,1),range(0,10));
pp.savefig('actions_tsne.pdf',bbox_inches='tight')

#Fig 2d
#file is 2d_rl_actions.csv
data=np.loadtxt('2d_rl_actions.csv',delimiter=',')
hatches = ['.','//','\\\\',None]
cum0=np.zeros(10)
fig,ax = pp.subplots(1,1)
i=0
for o in [2,1,0,3]: #note order is different because DBSCAN types for RL are different than humans
    pp.bar(bins[1:],data[o],bottom=cum0,width=0.5,hatch=hatches[i],
           # facecolor=ggcolors[i],
              edgecolor='k',linewidth=2)
    cum0+=n[o]
    i+=1

    
ax.set_facecolor('w')
ax.grid(visible='both',color='xkcd:gray')

ax.set_ylabel('Frequency')
ax.set_xlabel('Action')
ax.set_xticks(np.arange(0.5,10.5,1),range(0,10));
pp.savefig('actions_tsne_rl_simplified.pdf',bbox_inches='tight')

#Fig 2b
#file is 2b_specialization.csv
toplot = pandas.loadtxt('2b_specialization.csv')#using pandas because this array is ragged

fig,ax = pp.subplots(1,1,figsize=(5,8))

parts=ax.violinplot(toplot,points=500,showmeans=False,showmedians=False,showextrema=False);
ax.boxplot(toplot,medianprops={"color":'k',"linewidth":2},widths=0.25,boxprops={"linewidth":2},showfliers=False)
ax.set_xticks([1,2,3,4],labels=["Independent","Cooperative\nServer","Cooperative\nChef","Other"],rotation=90)

colors = ['#E24A33','#348ABD','#988ED5','#777777','#FBC15E','#8EBA42']
i=0
for pc in parts['bodies']:
    pc.set_facecolor(colors[i])
    pc.set_hatch(hatches[i])
    pc.set_edgecolor('black')
    pc.set_linewidth(0)
    pc.set_alpha(0.5)
    i+=1
ax.scatter([1,2,3,4],[np.mean(toplot[0]),np.mean(toplot[1]),np.mean(toplot[2]),np.mean(toplot[3])],
           marker='d',edgecolor='k',linewidths=2,s=200,c=colors[:4])

ax.set_ylabel('Specialization Index')
ax.set_facecolor('w')
ax.grid(visible='both',color='xkcd:gray')
ax.set_aspect(4/100);
pp.savefig('tsne_cluster_collaboration_human.pdf',bbox_inches='tight')

#Fig 2e
#file is 2e_rl_specialization.csv
toplot = pandas.loadtxt('2e_rl_specialization.csv')#using pandas because this array is ragged

fig,ax = pp.subplots(1,1,figsize=(5,8))
parts=ax.violinplot(toplot,points=500,showmeans=False,showmedians=False,showextrema=False);
ax.boxplot(toplot,medianprops={"color":'k',"linewidth":2},widths=0.25,boxprops={"linewidth":2},showfliers=False)
ax.set_xticks([1,2,3,4],labels=["Independent","Cooperative\nServer","Cooperative\nChef",
                                    "Other"],rotation=90)

colors = ['#E24A33','#348ABD','#988ED5','#777777','#FBC15E','#8EBA42']
i=0
for pc in parts['bodies']:
    pc.set_facecolor(colors[i])
    pc.set_hatch(hatches[i])
    pc.set_edgecolor('black')
    pc.set_linewidth(0)
    pc.set_alpha(0.5)
    i+=1
ax.scatter([1,2,3,4],[np.mean(i) for i in toplot],
           marker='d',edgecolor='k',linewidths=2,s=200,c=colors[:4])
    
ax.set_ylabel('Specialization Index')
ax.set_facecolor('w')
ax.grid(visible='both',color='xkcd:gray')
ax.set_aspect(4/100);
pp.savefig('tsne_cluster_collaboration_simplified_rl.pdf',bbox_inches='tight')

#Fig 2c
#file is 2c_clusters_by_map.csv
count_of_types = np.loadtxt('2c_clusters_by_map.csv',delimiter=',')
fig,axs=pp.subplots(nrows=1,ncols=1,figsize=(8,5))
xs=[0,1,
   2.2,3.2]

cum0=np.zeros(4)

for o in [0,2,1,3]:#again ordering is to adjust for both map order and dbscan type order
    axs.bar(xs,count_of_types[[1,0,3,2],o],bottom=cum0,width=0.5,hatch=hatches[o],
              edgecolor='k',linewidth=2)
    cum0+=count_of_types[[1,0,3,2],o]

# ax.set_aspect(4.0/160)
axs.set_xticks([0.5,2.7],labels=['\nOpen','\nPartially Blocked'])
axs.set_xticks(xs,labels=['HA','MA','HA','MA'],minor=True)
axs.set_ylabel('Frequency',labelpad=-0.1)

# axs.set_xlim(0,4.5)
axs.set_facecolor('w')
axs.grid(visible='both',color='xkcd:gray')
# axs.set_ylim(0,1.05)
axs.set_ylim(0,145)
legend=pp.legend(["Independent","Cooperative Server","Cooperative Chef","Other"],
         bbox_to_anchor=[1,1])

pp.savefig('clusters_tsne.pdf',bbox_inches='tight',bbox_extra_artists = [legend]);

#Fig 2f
#file is 2f_rl_clusters_by_map.csv
count_of_types = np.loadtxt('2f_rl_clusters_by_map.csv',delimiter=',')

fig,axs=pp.subplots(nrows=1,ncols=1,figsize=(8,5))
xs=[0,1,
   2.2,3.2]

cum0=np.zeros(4)

i=0
for o in [2,1,0,3]: #ordering for DBSCAN
    axs.bar(xs,count_of_types[:,o],bottom=cum0,width=0.5,hatch=hatches[i],
              edgecolor='k',linewidth=2)
    cum0+=count_of_types[:,o]
    i+=1


axs.set_xticks([0.5,2.7],labels=['\nOpen','\nPartially Blocked'])
axs.set_xticks(xs,labels=['HA','MA','HA','MA'],minor=True)
axs.set_ylabel('Frequency',labelpad=-0.1)

# axs.set_xlim(0,4.5)
axs.set_facecolor('w')
axs.grid(visible='both',color='xkcd:gray')
axs.set_ylim(0,1.05)
legend=pp.legend(["Independent","Cooperative Server","Cooperative Chef","Other"],
         bbox_to_anchor=[1,1])
pp.savefig('clusters_tsne_rl_simplified.pdf',bbox_inches='tight',bbox_extra_artists = [legend]);






