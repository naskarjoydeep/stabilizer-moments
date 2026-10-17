"""Figures 1-4 of the paper (files fig0_growing.pdf, fig1_powerlaw.pdf, fig2_exponent.pdf,
fig3_singleinterval.pdf).

    python3 make_figures.py          writes figures/*.pdf

The arrays below are the rounded values printed in results/ising.txt (length-two ratios,
single-interval slopes) and results/clock.txt (clock fits), and Blume-Capel powers from
results/records.json; Figure 1 reads the coordinates printed in results/pfaffian.txt and
Figure 2(b) reads the sectors record in results/records.json.  Palette: dataviz reference categorical slots 1,2,3,7
(validated: CVD adjacent dE 9.2, normal-vision 27.6).  The Blume-Capel negative
control is deliberately NOT a categorical slot -- it is neutral ink + open markers,
because it is a control rather than a peer series (and orange+red FAILS validation)."""
import numpy as np, matplotlib
from pathlib import Path
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent; OUT=HERE/'figures'; OUT.mkdir(exist_ok=True)

S1,S2,S3,S7="#2a78d6","#eb6834","#1baf7a","#4a3aa7"
INK,INK2,MUTED,GRID,AXIS="#0b0b0b","#52514e","#898781","#e1e0d9","#c3c2b7"
plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],
  'mathtext.fontset':'cm','font.size':9,'axes.labelsize':10,'axes.linewidth':0.7,
  'xtick.labelsize':9,'ytick.labelsize':9,'legend.fontsize':8.5,
  'axes.edgecolor':AXIS,'xtick.color':MUTED,'ytick.color':MUTED,
  'axes.labelcolor':INK,'text.color':INK,'figure.dpi':150,
  'xtick.direction':'in','ytick.direction':'in','xtick.top':True,'ytick.right':True})

eta=np.array([2.500e-01,1.111e-01,6.250e-02,2.778e-02,1.562e-02,6.944e-03,3.906e-03,
              1.736e-03,9.766e-04,4.340e-04,2.441e-04,1.085e-04,6.104e-05])
I2 =np.array([1.9038e-01,1.5414e-01,1.3368e-01,1.0974e-01,9.5481e-02,7.8470e-02,6.8251e-02,
              5.6032e-02,4.8692e-02,3.9923e-02,3.4663e-02,2.8389e-02,2.4631e-02])
Im={2:np.array([8.0828e-02,5.2912e-02,3.9439e-02,2.6178e-02,1.9604e-02,1.3055e-02,9.7876e-03,
                6.5233e-03,4.8920e-03,3.2611e-03,2.4458e-03,1.6305e-03,1.2229e-03]),
    3:np.array([2.6237e-02,1.3821e-02,8.8777e-03,4.7947e-03,3.1058e-03,1.6873e-03,1.0952e-03,
                5.9585e-04,3.8695e-04,2.1061e-04,1.3679e-04,7.4455e-05,4.8359e-05]),
    4:np.array([6.9829e-03,2.9472e-03,1.6290e-03,7.1513e-04,4.0054e-04,1.7748e-04,9.9724e-05,
                4.4288e-05,2.4906e-05,1.1067e-05,6.2248e-06,2.7664e-06,1.5561e-06]),
    5:np.array([1.6654e-03,5.6035e-04,2.6607e-04,9.4828e-05,4.5907e-05,1.6585e-05,8.0667e-06,
                2.9240e-06,1.4239e-06,5.1656e-07,2.5161e-07,9.1300e-08,4.4475e-08])}
COL={2:S1,3:S2,4:S3,5:S7}; MK={2:'o',3:'s',4:'^',5:'D'}

# ---------------- Figure 2 (fig1_powerlaw.pdf): charged power laws + the crossover ----------------
import json
sec=sorted(json.load(open(HERE/'results'/'records.json'))['sectors'],key=lambda r:-r['eta'])
se=np.array([float(r['eta']) for r in sec])
getc=lambda t,a: np.array([float(r[f'{t}_a{a}']) for r in sec])
def below(ax,txt):
    ax.text(0.5,-0.22,txt,transform=ax.transAxes,ha='center',va='top',fontsize=9,color=INK)
fig,(axA,axB)=plt.subplots(1,2,figsize=(6.9,3.6))

axA.grid(True,which='major',color=GRID,lw=0.5,zorder=0)
axA.plot(eta,I2,ls='--',lw=1.3,color=INK2,marker='v',ms=3.4,mfc='white',mew=0.9,
         zorder=3,label=r'full $I_2$ ($\alpha=1$)')
for a in (2,3,4,5):
    axA.plot(eta,Im[a],lw=1.3,color=COL[a],marker=MK[a],ms=3.4,mfc='white',mew=0.9,
             zorder=4,label=rf'$\alpha={a}$')
axA.set_xscale('log'); axA.set_yscale('log')
axA.set_xlim(4e-5,0.5); axA.set_ylim(1e-8,0.5)
axA.set_xlabel(r'cross-ratio $\eta$')
axA.set_ylabel(r'$Z^{\rm ch}_\alpha/[Z_\alpha(A)Z_\alpha(B)]$ or $I_2$')
axA.legend(frameon=False,loc='lower right',handlelength=1.5,borderpad=0.2,
           labelspacing=0.22,fontsize=8)
below(axA,'(a)')

axB.grid(True,color=GRID,lw=0.5,zorder=0)
aa=np.array([2,3,4,5,6]); xx=np.linspace(0.7,6.3,200)
a1=np.polyfit(np.log(eta[-4:]),np.log(I2[-4:]),1)[0]
axB.plot(xx,xx/4,ls=':',lw=0.9,color=S1,zorder=2)
axB.axhline(1.0,ls=':',lw=0.9,color=S2,zorder=2)
axB.plot(xx,np.minimum(xx/4,1.0),lw=1.0,color=MUTED,zorder=3)
fitex=lambda t:[np.polyfit(np.log(se[-4:]),np.log(np.abs(getc(t,a)[-4:])),1)[0] for a in aa]
axB.plot(aa,fitex('chg'),color=S1,marker='o',ms=4.4,lw=0,mfc='white',mew=1.1,label='charged',zorder=5)
axB.plot(aa,fitex('neut'),color=S2,marker='s',ms=4.4,lw=0,mfc='white',mew=1.1,label='neutral',zorder=5)
axB.plot(aa,fitex('tot'),color=INK,marker='D',ms=4.6,lw=0,mfc=INK,mew=0,label='total',zorder=6)
axB.plot([1],[a1],color=INK2,marker='v',ms=4.4,lw=0,mfc='white',mew=1.1,
         label=r'$I_2$ ($\alpha=1$)',zorder=5)
axB.set_xlim(0.6,6.4); axB.set_ylim(0.15,1.68); axB.set_xticks([1,2,3,4,5,6])
axB.set_xlabel('R\u00e9nyi index $\\alpha$')
axB.set_ylabel(r'fitted exponent in $\eta$')
axB.legend(frameon=False,loc='upper left',handlelength=1.2,borderpad=0.2,
           labelspacing=0.22,fontsize=8)
below(axB,'(b)')
fig.tight_layout(); fig.savefig(OUT/'fig1_powerlaw.pdf',bbox_inches='tight'); plt.close(fig)

# ---------------- Figure 3 (fig2_exponent.pdf): fixed-support exponents ----------------
mods=[('Ising (charged)',      [2,3,4,5],[0.50006,0.75010,1.00016,1.25024],1/8,S1,'o'),
      ('3-state Potts (charged)',[2,3,4],  [0.5283,0.7918,1.0557],           2/15,S2,'s'),
      (r'$\mathbb{Z}_4$ FZ (charged)',[2,3,4],[0.5037,0.7526,1.0031],         1/8,S3,'^'),
      (r'$\mathbb{Z}_5$ FZ (charged)',[2,3,4],[0.4740,0.7033,0.9366],         4/35,S7,'D')]
bc=([2,3,4],[0.7834,1.0666,1.1302])
fig,axL=plt.subplots(figsize=(5.5,3.5))
axL.grid(True,color=GRID,lw=0.5,zorder=0)
for nm,al,p,D,col,mk in mods:
    axL.plot(al,p,color=col,marker=mk,ms=4.2,lw=1.3,mfc='white',mew=0.9,label=nm,zorder=4)
    x=np.array([0,5.6]); axL.plot(x,2*D*x,ls=':',lw=0.7,color=col,zorder=2)
axL.plot(bc[0],bc[1],color=MUTED,marker='x',ms=5,lw=1.1,ls='--',
         label='Blume-Capel (full ratio)',zorder=3)
axL.set_xlim(0,5.6); axL.set_ylim(0,1.45); axL.set_xticks([0,1,2,3,4,5])
axL.set_xlabel('R\u00e9nyi index $\\alpha$'); axL.set_ylabel(r'fitted exponent in $\eta$')
axL.legend(frameon=False,loc='upper left',handlelength=1.6,borderpad=0.2,labelspacing=0.22)
fig.tight_layout(); fig.savefig(OUT/'fig2_exponent.pdf',bbox_inches='tight'); plt.close(fig)

# ---------------- Figure 4 (fig3_singleinterval.pdf): the single-interval logarithm ----------------
lm=np.array([2.5,3.5,4.5,5.5]); inv=1/lm
ent=np.array([0.1391,0.1358,0.1337,0.1322]); mom=np.array([0.1187,0.1211,0.1222,0.1228])
fig,ax=plt.subplots(figsize=(5.4,3.5))
ax.grid(True,color=GRID,lw=0.5,zorder=0)
for y,col,mk,lab,ex in ((ent,INK2,'v',r'local slope of $J_{S_2}$',0.1267),
                        (mom,S1,'o',r'local slope of $J_{\widehat{\mathcal{M}}_2}$',0.1264)):
    ax.plot(inv,y,color=col,marker=mk,ms=4.2,lw=1.3,mfc='white',mew=0.9,label=lab,zorder=4)
    m,b=np.polyfit(inv,y,1)
    xx=np.array([0,inv.max()*1.08]); ax.plot(xx,b+m*xx,ls=':',lw=0.8,color=col,zorder=2)
    ax.plot([0],[ex],color=col,marker=mk,ms=5.6,mfc=col,mew=0,zorder=5)
ax.axhline(0.125,color=MUTED,lw=0.7,ls='--',zorder=1)
ax.annotate(r'$c/4=0.125$',(0.43,0.125),xytext=(0,5),textcoords='offset points',
            fontsize=8,color=INK2,ha='right')
ax.set_xlim(-0.02,0.46); ax.set_ylim(0.113,0.145)
ax.set_xlabel(r'$1/(\ell+1/2)$'); ax.set_ylabel('local logarithmic slope')
ax.legend(frameon=False,loc='lower right',handlelength=1.6,borderpad=0.2)

fig.tight_layout(); fig.savefig(OUT/'fig3_singleinterval.pdf',bbox_inches='tight'); plt.close(fig)
# ---------------- Figure 1 (fig0_growing.pdf): Pfaffian coefficients for growing intervals ----------------
import re
pf=open(HERE/'results'/'pfaffian.txt').read()
pts=lambda key: np.array([[float('0'+a),float('0'+b)] for a,b in
                          re.findall(r'\((\.\d+),(\.\d+)\)', re.search(rf'^  {key}: (.*)$',pf,re.M).group(1))])
c2,la1=pts('C2'),pts('lA1')
fig,(axA,axB)=plt.subplots(1,2,figsize=(6.9,3.2))
for ax in (axA,axB):
    ax.grid(True,color=GRID,lw=0.5,zorder=0); ax.set_xlim(0,0.52); ax.set_xticks([0,.1,.2,.3,.4,.5])
    ax.set_xlabel(r'$1/\sqrt{\ell}$')
axA.axhline(0.25,ls='--',lw=0.9,color=INK2,zorder=2)
axA.annotate(r'$1/4$ (conditional)',(0.51,0.25),xytext=(0,-4),textcoords='offset points',
             fontsize=8,color=INK2,ha='right',va='top')
axA.plot(c2[:,0],c2[:,1],color=S1,marker='o',ms=4.2,lw=1.3,mfc='white',mew=0.9,zorder=4)
axA.set_ylim(0.17,0.26); axA.set_yticks([.18,.20,.22,.24,.25]); axA.set_ylabel(r'$\mathcal{C}_2(\ell)$')
axB.plot(la1[:,0],la1[:,1],color=S2,marker='s',ms=4.0,lw=1.3,mfc='white',mew=0.9,zorder=4)
axB.set_ylim(0.0248,0.028); axB.set_yticks([.025,.026,.027,.028]); axB.set_ylabel(r'$\ell A_1(\ell)$ at $\alpha=2$')
below(axA,'(a)'); below(axB,'(b)')
fig.tight_layout(); fig.savefig(OUT/'fig0_growing.pdf',bbox_inches='tight'); plt.close(fig)
print("wrote figures/fig0_growing.pdf, figures/fig1_powerlaw.pdf, figures/fig2_exponent.pdf, figures/fig3_singleinterval.pdf")
