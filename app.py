# -*- coding: utf-8 -*-
# StructureLab - application autonome.

# MOTEUR DE CALCUL
"""Elastic beam model; SI internally, independent positive uncertainties."""
from dataclasses import asdict, dataclass, field
from functools import lru_cache
import json
import math

import numpy as np
from scipy import stats
import sympy as sp
from sympy.physics.continuum_mechanics.beam import Beam

VARIABLES = ('load_kn','strength_mpa','young_gpa','width_mm','height_mm','length_m')
LABELS = {'load_kn':'Charge P','strength_mpa':'Limite élastique fy','young_gpa':'Module E',
          'width_mm':'Largeur b','height_mm':'Hauteur h','length_m':'Portée L'}
LIMITS = {'length_m':(.5,12.),'width_mm':(20.,500.),'height_mm':(40.,1000.),
          'load_kn':(.1,500.),'young_gpa':(100.,250.),'strength_mpa':(100.,700.)}


@dataclass(frozen=True)
class Design:
    length_m: float = 4.
    width_mm: float = 100.
    height_mm: float = 160.
    load_kn: float = 65.
    young_gpa: float = 210.
    strength_mpa: float = 235.
    limit_ratio: int = 300
    self_weight: bool = True
    distribution: str = 'Lognormale'
    cov: dict = field(default_factory=lambda:dict(load_kn=.2,strength_mpa=.1,young_gpa=.05,width_mm=.02,height_mm=.02,length_m=0.))

    def validate(self):
        for key,(low,high) in LIMITS.items():
            value=getattr(self,key)
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not low<=value<=high:
                raise ValueError(f'Valeur hors limites : {key} ({low} à {high}).')
        if not isinstance(self.self_weight,bool): raise ValueError('Poids propre : booléen attendu.')
        if isinstance(self.limit_ratio,bool) or not isinstance(self.limit_ratio,int) or not 100<=self.limit_ratio<=1000:
            raise ValueError('Dénominateur de flèche : entier de 100 à 1000.')
        if self.distribution not in ('Lognormale','Uniforme'): raise ValueError('Distribution non reconnue.')
        if not isinstance(self.cov,dict) or set(self.cov)!=set(VARIABLES): raise ValueError('Coefficients de variation incomplets.')
        for value in self.cov.values():
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not 0<=value<=.4:
                raise ValueError('CV attendu entre 0 et 40 %.')
        return self

    def dumps(self): return json.dumps(asdict(self),sort_keys=True,ensure_ascii=False)

    @classmethod
    def loads(cls,value):
        try:
            data=json.loads(value)
            if not isinstance(data,dict): raise ValueError('Objet JSON attendu.')
            return cls(**data).validate()
        except (TypeError,json.JSONDecodeError) as exc:
            raise ValueError('Fichier de paramètres invalide.') from exc


@lru_cache(maxsize=1)
def influence():
    """Derive unit-load solutions with SymPy's established beam solver."""
    solutions=[]
    x=sp.Symbol('x')
    for uniform in (False,True):
        beam=Beam(1,1,1,variable=x)
        ra=beam.apply_support(0,'pin')
        rb=beam.apply_support(1,'roller')
        if uniform: beam.apply_load(-1,0,0,end=1)
        else: beam.apply_load(-1,sp.Rational(1,2),-1)
        beam.solve_for_reaction_loads(ra,rb)
        moment=beam.bending_moment()
        deflection=beam.deflection()
        solutions.append(dict(moment=float(abs(moment.subs(x,sp.Rational(1,2)))),
                              deflection=float(abs(deflection.subs(x,sp.Rational(1,2)))),
                              curve=sp.lambdify(x,deflection.rewrite(sp.Piecewise),'numpy')))
    return solutions


def evaluate(design,values=None):
    design.validate()
    v=values if values is not None else {key:getattr(design,key) for key in VARIABLES}
    length=v['length_m']; width=v['width_mm']/1000; height=v['height_mm']/1000
    load=v['load_kn']*1000; young=v['young_gpa']*1e9; strength=v['strength_mpa']*1e6
    inertia=width*height**3/12
    modulus=width*height**2/6
    weight=7850*9.81*width*height if design.self_weight else 0.*load
    point,uniform=influence()
    moment=point['moment']*load*length+uniform['moment']*weight*length**2
    stress=moment/modulus
    deflection=(point['deflection']*load*length**3+uniform['deflection']*weight*length**4)/(young*inertia)
    bending=stress/strength
    service=deflection/(length/design.limit_ratio)
    return dict(moment_knm=moment/1000,stress_mpa=stress/1e6,deflection_mm=deflection*1000,
                bending=bending,service=service,utilization=np.maximum(bending,service),
                mass_kg=7850*width*height*length,inertia_m4=inertia,
                weight_kn_m=weight/1000,limit_mm=length/design.limit_ratio*1000)


def beam_curve(design,points=161):
    result=evaluate(design)
    point,uniform=influence()
    u=np.linspace(0,1,points)
    length=design.length_m
    y=(design.load_kn*1000*length**3*point['curve'](u)+result['weight_kn_m']*1000*length**4*uniform['curve'](u))/(design.young_gpa*1e9*result['inertia_m4'])
    return u*length,-np.abs(y)*1000


def probability(mask):
    count=int(np.count_nonzero(mask)); total=len(mask)
    interval=stats.binomtest(count,total).proportion_ci(confidence_level=.95,method='exact')
    return dict(count=count,total=total,p=count/total,low=float(interval.low),high=float(interval.high))


def simulate(design,n=10000,seed=42):
    design.validate()
    if isinstance(n,bool) or not isinstance(n,int) or not 1000<=n<=100000: raise ValueError('1000 à 100000 tirages attendus.')
    if isinstance(seed,bool) or not isinstance(seed,int) or not 0<=seed<=2147483647: raise ValueError('Graine invalide.')
    rng=np.random.default_rng(seed)
    values={}
    for key in VARIABLES:
        mean=getattr(design,key); cov=design.cov[key]
        # Draw standard normals even for fixed inputs: paired variants share draws.
        z=rng.standard_normal(n)
        if not cov: values[key]=np.full(n,mean)
        elif design.distribution=='Lognormale':
            s=np.sqrt(np.log1p(cov**2))
            values[key]=mean*np.exp(s*z-s*s/2)
        else:
            values[key]=mean*(1+np.sqrt(3)*cov*(2*stats.norm.cdf(z)-1))
    results=evaluate(design,values)
    risk={key:probability(results[key]>1) for key in ('bending','service','utilization')}
    sensitivity=[]
    for key in VARIABLES:
        if design.cov[key] and np.ptp(results['utilization'])>0:
            rho=float(stats.spearmanr(values[key],results['utilization']).statistic)
            sensitivity.append(dict(variable=LABELS[key],rho=rho))
    checkpoints=np.unique(np.linspace(100,n,60,dtype=int))
    masks=results['utilization']>1
    cumulative=np.cumsum(masks)
    convergence=[dict(n=int(k),p=float(cumulative[k-1]/k)) for k in checkpoints]
    return dict(values=values,results=results,risk=risk,sensitivity=sensitivity,convergence=convergence,seed=seed,n=n)


# VISUALISATIONS
import numpy as np
import plotly.graph_objects as go


GREEN='#12805C'
RED='#C84148'
INK='#252A32'
GRAY='#9098A2'
GOLD='#BA861A'


def finish(fig,height=380):
    fig.update_layout(template='plotly_white',height=height,margin=dict(l=16,r=16,t=35,b=100),
                      font=dict(family='Arial',color=INK,size=12),paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)',legend=dict(orientation='h',y=-.42,yanchor='top',x=0,font=dict(size=11)),
                      hovermode='closest')
    fig.update_xaxes(automargin=True,title_standoff=12)
    fig.update_yaxes(automargin=True)
    return fig


def beam(design,amplification=20):
    x,y=beam_curve(design)
    length=design.length_m
    result=evaluate(design)
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=[0,length],y=[0,0],mode='lines',line=dict(color=INK,width=10),name='Poutre initiale',hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=x,y=y/1000*amplification,mode='lines',line=dict(color=GREEN,width=4),name=f'Déformée × {amplification}',customdata=y,
                             hovertemplate='x : %{x:.2f} m<br>Flèche réelle : %{customdata:.2f} mm<extra></extra>'))
    fig.add_trace(go.Scatter(x=[0,length],y=[-.04*length]*2,mode='markers',marker=dict(symbol=['triangle-up','triangle-up'],size=20,color=INK),showlegend=False,hoverinfo='skip'))
    fig.add_annotation(x=length/2,y=.01*length,ax=length/2,ay=.28*length,axref='x',ayref='y',showarrow=True,arrowhead=2,arrowwidth=3,arrowcolor=RED,
                       text=f'P = {design.load_kn:g} kN',font=dict(color=RED,size=15))
    if design.self_weight:
        for pos in np.linspace(.08*length,.92*length,9):
            if abs(pos-length/2)<length*.05: continue
            fig.add_annotation(x=pos,y=.02*length,ax=pos,ay=.12*length,axref='x',ayref='y',showarrow=True,arrowhead=2,arrowwidth=1.2,arrowcolor=GOLD,text='')
        fig.add_annotation(x=length*.95,y=.17*length,text=f"Poids propre : {result['weight_kn_m']:.2f} kN/m",showarrow=False,xanchor='right',font=dict(color=GOLD,size=11))
    low=min(-.18*length,float(np.min(y))/1000*amplification-length*.05)
    fig.update_xaxes(title='Position sur la poutre (m)',range=[-.08*length,1.08*length],showgrid=True,gridcolor='#E2EBE7',zeroline=False)
    fig.update_yaxes(range=[low,.36*length],visible=False,fixedrange=True)
    fig=finish(fig,380)
    fig.update_layout(plot_bgcolor='#F5F9F7')
    return fig


def histogram(values,label):
    counts,edges=np.histogram(values,bins=45)
    centers=(edges[:-1]+edges[1:])/2
    fig=go.Figure(go.Bar(x=centers,y=counts,marker_color=[RED if x>1 else GREEN for x in centers],width=np.diff(edges)*.9,hovertemplate='Ratio : %{x:.3f}<br>Tirages : %{y}<extra></extra>'))
    fig.add_vline(x=1,line_color=RED,line_dash='dash',annotation_text='Seuil = 1',annotation_position='top right')
    fig.update_xaxes(title=label)
    fig.update_yaxes(title='Nombre de tirages')
    return finish(fig)


def sensitivity(rows):
    rows=sorted(rows,key=lambda row:abs(row['rho']))
    fig=go.Figure(go.Bar(y=[r['variable'] for r in rows],x=[r['rho'] for r in rows],orientation='h',marker_color=[RED if r['rho']>0 else GREEN for r in rows],hovertemplate='%{y}<br>ρ : %{x:.3f}<extra></extra>'))
    fig.update_xaxes(range=[-1,1],title='Corrélation de Spearman avec le ratio maximal')
    return finish(fig)


def convergence(rows):
    fig=go.Figure(go.Scatter(x=[r['n'] for r in rows],y=[100*r['p'] for r in rows],line=dict(color=GREEN,width=3),fill='tozeroy',fillcolor='rgba(18,128,92,.09)'))
    fig.update_xaxes(title='Nombre de tirages')
    fig.update_yaxes(title='Dépassement observé (%)',rangemode='tozero')
    return finish(fig,300)


# RAPPORT PDF
import datetime as dt
import io
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak



def create_report(design,simulation,name='Variante active',comparison=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    result=evaluate(design)
    styles=getSampleStyleSheet()
    story=[]
    def p(value,style='BodyText'): return Paragraph(escape(str(value)),styles[style])
    def table(rows):
        t=Table([[p(c) for c in row] for row in rows],colWidths=[230,269],repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E7F3EE')),('GRID',(0,0),(-1,-1),.4,colors.HexColor('#D3DBD8')),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),8),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]))
        story.extend([t,Spacer(1,12)])
    story += [p('StructureLab','Title'),p('Fiabilité des structures - Rapport pédagogique','Heading2'),p(name),p(dt.datetime.now().strftime('%d/%m/%Y %H:%M')),Spacer(1,12)]
    story.append(p('Hypothèses','Heading2'))
    story.append(p('Poutre pleine rectangulaire en acier homogène, simplement appuyée, charge ponctuelle centrale, petites déformations et élasticité linéaire. Poids propre optionnel. Aucune vérification normative, de déversement, de cisaillement ou des assemblages. Les dépassements simulés ne sont pas des probabilités de rupture réelle.'))
    table([['Paramètre','Valeur'],['Portée L',f'{design.length_m:g} m'],['Section b x h',f'{design.width_mm:g} x {design.height_mm:g} mm'],['Charge centrale P',f'{design.load_kn:g} kN'],['E / fy',f'{design.young_gpa:g} GPa / {design.strength_mpa:g} MPa'],['Flèche limite',f'L/{design.limit_ratio} (choix pédagogique)'],['Poids propre','Inclus (7850 kg/m3)' if design.self_weight else 'Exclu']])
    table([['Résultat nominal','Valeur'],['Contrainte maximale',f"{result['stress_mpa']:.3f} MPa"],['Flèche maximale',f"{result['deflection_mm']:.3f} mm"],['Ratio contrainte / fy',f"{result['bending']:.4f}"],['Ratio flèche / limite',f"{result['service']:.4f}"]])
    fig,ax=plt.subplots(figsize=(7,2.3))
    x,y=beam_curve(design)
    ax.plot(x,y,color='#12805C',linewidth=2)
    ax.axhline(0,color='#9098A2',linewidth=1)
    ax.set(xlabel='Position (m)',ylabel='Flèche réelle (mm)',title='Déformée élastique théorique')
    fig.tight_layout()
    img=io.BytesIO(); fig.savefig(img,format='png',dpi=150); plt.close(fig)
    img.seek(0); story.append(Image(img,width=470,height=155))
    story.append(PageBreak())
    story.append(p('Simulation de Monte-Carlo','Heading1'))
    story.append(p(f"{simulation['n']} tirages ; graine {simulation['seed']} ; loi {design.distribution} ; variables indépendantes. Paramètres des lois exprimés en moyenne et coefficient de variation (CV)."))
    table([['Variable','CV'],*[[LABELS[k],f'{design.cov[k]*100:g} %'] for k in VARIABLES]])
    rows=[['Critère','Estimation et intervalle exact à 95 %']]
    for key,label in [('bending','Contrainte > fy'),('service','Flèche > limite'),('utilization','Au moins un dépassement')]:
        risk=simulation['risk'][key]
        rows.append([label,f"{risk['p']*100:.3f} % ; [{risk['low']*100:.3f} ; {risk['high']*100:.3f}] % ({risk['count']}/{risk['total']})"])
    table(rows)
    story.append(p('Zéro dépassement observé ne signifie pas un risque nul. L’intervalle traduit l’erreur d’échantillonnage conditionnelle aux hypothèses choisies, pas l’incertitude du modèle. Au-delà de fy, la flèche élastique reste une extrapolation.'))
    fig,ax=plt.subplots(figsize=(7,2.8))
    ax.hist(simulation['results']['utilization'],bins=40,color='#12805C',alpha=.85)
    ax.axvline(1,color='#C84148',linestyle='--',label='Seuil = 1')
    ax.set(xlabel='Ratio maximal',ylabel='Tirages',title='Distribution du ratio maximal')
    ax.legend(); fig.tight_layout()
    image=io.BytesIO(); fig.savefig(image,format='png',dpi=140); plt.close(fig)
    image.seek(0); story.append(Image(image,width=470,height=188))
    if simulation['sensitivity']:
        story.append(p('Sensibilité exploratoire','Heading2'))
        table([['Variable','Corrélation de Spearman avec le ratio maximal'],*[[r['variable'],f"{r['rho']:.3f}"] for r in sorted(simulation['sensitivity'],key=lambda r:-abs(r['rho']))]])
        story.append(p('Une corrélation de rang ne représente pas une part de contribution ni une preuve de causalité.'))
    if comparison:
        story.append(p('Variante mémorisée pour comparaison','Heading2'))
        other,sim=comparison
        risk=sim['risk']['utilization']
        table([['Paramètre','Variante B'],['Section',f'{other.width_mm:g} x {other.height_mm:g} mm'],['Portée / charge',f'{other.length_m:g} m / {other.load_kn:g} kN'],['E / fy',f'{other.young_gpa:g} GPa / {other.strength_mpa:g} MPa'],['Flèche limite',f'L/{other.limit_ratio}'],['Poids propre','Inclus' if other.self_weight else 'Exclu'],['Loi',other.distribution],['CV des variables',' ; '.join(f'{LABELS[k]} : {other.cov[k]*100:g} %' for k in VARIABLES)],['Dépassement combiné',f"{risk['p']*100:.3f} % ; IC 95 % [{risk['low']*100:.3f} ; {risk['high']*100:.3f}] %"]])
    story.append(p('Références et formules','Heading2'))
    story.append(p('SymPy Beam (Euler-Bernoulli) : docs.sympy.org/latest/modules/physics/continuum_mechanics/beam.html. SciPy binomtest et spearmanr : docs.scipy.org/doc/scipy/reference/stats.html.'))
    story.append(p('I = bh³/12 ; W = bh²/6 ; Mmax = PL/4 + qL²/8 ; flèche = PL³/(48EI) + 5qL⁴/(384EI). q = rho g bh si le poids propre est inclus.'))
    def footer(canvas,doc):
        canvas.setFont('Helvetica',8)
        canvas.drawString(42,24,'StructureLab - Étude pédagogique, hors dimensionnement réglementaire')
        canvas.drawRightString(A4[0]-42,24,str(doc.page))
    out=io.BytesIO()
    SimpleDocTemplate(out,pagesize=A4,leftMargin=42,rightMargin=42,topMargin=36,bottomMargin=42).build(story,onFirstPage=footer,onLaterPages=footer)
    return out.getvalue()


STYLE = """<style>
:root { color-scheme: light; }
[data-testid="stAppViewContainer"] { background: #fff; color: #263630; }
[data-testid="stHeader"] { background: rgba(255,255,255,.96); }
[data-testid="stMainBlockContainer"] { max-width: 1320px; padding-top: 4.2rem; }
[data-testid="stSidebar"] { background:#f3f7f5; border-right:1px solid #dbe5df; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 { font-size:1.25rem; }
[data-testid="stMarkdownContainer"] { overflow-wrap:anywhere; letter-spacing:0; }
[data-testid="stMarkdownContainer"] h1 { font-size:2.4rem; font-weight:750; letter-spacing:0; color:#1a352b; }
[data-testid="stMarkdownContainer"] h2 { font-size:1.4rem; font-weight:650; letter-spacing:0; color:#263d33; }
[data-testid="stMarkdownContainer"] h3 { font-size:1.15rem; letter-spacing:0; }
[data-testid="stCaptionContainer"] { color:#65766c; }
[data-testid="stMetric"] { padding:16px 0; border-top:2px solid #dce8e1; }
[data-testid="stMetricLabel"] { color:#617268; }
[data-testid="stMetricValue"] > div { font-size:1.8rem; font-weight:650; color:#1a4c39; white-space:normal; overflow-wrap:anywhere; }
[data-testid="stExpander"] { border-color:#d8e3dc; border-radius:6px; }
[data-testid="stSidebar"] [data-testid="stExpander"] { background:#fff; }
[data-testid="stNumberInputContainer"], [data-testid="stTextInputRootElement"] { border-radius:6px; }
[data-testid="stSegmentedControl"] button, [data-testid="stButtonGroup"] button { min-height:42px; }
button { border-radius:6px !important; }
button p { white-space:normal; }
button[aria-pressed="true"], button[data-variant="segmented_control"][aria-checked="true"] { background:#e4f2e9 !important; border-color:#19845c !important; color:#12643f !important; }
[data-testid="stSlider"] [role="group"] > div > div:first-child { filter:hue-rotate(125deg) saturate(.65); }
[data-testid="stSlider"] [data-rac][style*="transform: translate"] { background:#16865c !important; box-shadow:0 0 0 3px #e0eee6; }
[data-testid="stSliderThumbValue"] { color:#12643f; }
.sl-kicker { color:#668172; font-size:11px; font-weight:700; margin:0 0 6px; }
.sl-meta { display:flex; flex-wrap:wrap; gap:10px 22px; padding:12px 0 18px; border-bottom:1px solid #dfe8e2; margin-bottom:12px; color:#52695c; font-size:13px; }
.sl-meta strong { color:#214c37; font-weight:650; }
.sl-sidebar-brand { display:flex; align-items:center; gap:10px; margin:0 0 20px; color:#204e39; font-size:18px; font-weight:750; }
.sl-mark { display:grid; grid-template-columns:repeat(3,5px); align-items:end; gap:3px; height:25px; }
.sl-mark i { display:block; background:#16865c; width:5px; border-radius:1px; }
.sl-mark i:nth-child(1) { height:12px; } .sl-mark i:nth-child(2) { height:23px; } .sl-mark i:nth-child(3) { height:17px; }
[data-testid="stPlotlyChart"] { margin:6px 0 12px; }
@media(max-width:640px) {
 [data-testid="stMainBlockContainer"] { padding:4.2rem 1rem 1.3rem; }
 [data-testid="stMarkdownContainer"] h1 { font-size:1.95rem; }
 [data-testid="stMarkdownContainer"] h2 { font-size:1.25rem; }
 [data-testid="stColumn"] { min-width:0; width:100%; flex:1 1 100%; }
 [data-testid="stMetric"] { padding:10px 0; }
 button { min-height:44px; }
 input, textarea { font-size:16px !important; }
 .sl-meta { gap:8px 16px; font-size:12px; }
}
</style>"""

STYLE = STYLE.replace('</style>', '''
[data-testid="stAppViewContainer"] { background:#eef3f5; }
[data-testid="stSidebar"] { background:#173f3c; border:0; }
.sl-sidebar-brand { color:#fff; }
.sl-mark i { background:#53d1ac; }
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color:#cde8de !important; }
[data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stCaptionContainer"] { color:#52695c !important; }
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] > div > div > div > div > [data-testid="stCaptionContainer"],
[data-testid="stSidebar"] [data-testid="stSlider"] label,
[data-testid="stSidebar"] [data-testid="stNumberInput"] label { color:#e9f6f0 !important; }
[data-testid="stSidebar"] [data-testid="stExpander"] { background:#f2f8f5; border-color:#51776c; }
[data-testid="stSidebar"] [data-testid="stExpander"] label { color:#263d33 !important; }
[data-testid="stSidebar"] [data-testid="stSliderThumbValue"] { color:#70e6be; }
.sl-banner { padding:24px 28px; background:#173f3c; border-bottom:6px solid #2bc49a; margin-bottom:20px; }
.sl-banner h1 { color:#fff; font-size:2rem; line-height:1.25; margin:0; padding:0; }
.sl-banner .sl-kicker { color:#8de9ce; margin-bottom:8px; }
.sl-banner p { color:#d5ece4; margin:10px 0 0; }
.sl-meta { background:#e0ebe8; padding:14px 18px; border:0; color:#274c42; }
[class*="st-key-sl_kpi_"] { border-radius:6px; padding:6px 16px; margin:4px 0 10px; }
[class*="st-key-sl_kpi_"] [data-testid="stMetric"] { border:0; }
[class*="st-key-sl_kpi_"] [data-testid="stMetricLabel"],
[class*="st-key-sl_kpi_"] [data-testid="stMetricValue"] > div { color:white !important; }
.st-key-sl_kpi_0 { background:#2665b2; }
.st-key-sl_kpi_1 { background:#14745c; }
.st-key-sl_kpi_2 { background:#a65b18; }
.st-key-sl_kpi_3 { background:#a33e63; }
[data-testid="stBaseButton-secondary"] { border-color:#b8d0c9; background:#e1ede8; color:#174c3d; }
[data-testid="stBaseButton-primary"] { background:#14745c; border-color:#14745c; color:#fff; }
[data-testid="stButtonGroup"] button[aria-checked="true"],
[data-testid="stButtonGroup"] button[aria-pressed="true"] { background:#14745c !important; color:#fff !important; }
[data-testid="stPlotlyChart"] { border-top:3px solid #36a888; }
@media(max-width:640px) {
 .sl-banner { padding:22px 18px; }
 .sl-banner h1 { font-size:1.8rem; }
 .sl-banner p { font-size:14px; }
}
</style>''')

# STREAMLIT APPLICATION
from dataclasses import asdict
from pathlib import Path
import io
import json

import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(page_title='StructureLab · Fiabilité des structures',page_icon=':material/architecture:',layout='wide',initial_sidebar_state='auto')
st.html(STYLE)


@st.cache_data(ttl=1800,max_entries=4,show_spinner=False)
def simulation(data,n,seed):
    return simulate(Design.loads(data),n,seed)


def import_parameters():
    upload=st.session_state.get('import_file')
    if upload is None: return
    try:
        if upload.size>20000: raise ValueError('Fichier limité à 20 Ko.')
        config=Design.loads(upload.getvalue())
        for key in VARIABLES:
            st.session_state[key]=float(getattr(config,key))
            st.session_state['cv_'+key]=float(config.cov[key]*100)
        for key in ('self_weight','distribution','limit_ratio'):
            st.session_state[key]=getattr(config,key)
        st.session_state.import_message='Paramètres importés.'
    except (ValueError,UnicodeError) as exc:
        st.session_state.import_error=str(exc)


def reroll():
    st.session_state.seed=(int(st.session_state.seed)+1)%2147483648


def reset_parameters():
    defaults=Design()
    for key in VARIABLES:
        st.session_state[key]=float(getattr(defaults,key))
        st.session_state['cv_'+key]=float(defaults.cov[key]*100)
    for key in ('self_weight','distribution','limit_ratio'):
        st.session_state[key]=getattr(defaults,key)


def format_number(value,digits=1):
    return f'{value:,.{digits}f}'.replace(',',' ').replace('.',',')


def kpis(items):
    for start in range(0,len(items),2):
        for offset,(column,(label,value,help_text)) in enumerate(zip(st.columns(2),items[start:start+2])):
            with column, st.container(key=f'sl_kpi_{start + offset}'):
                st.metric(label,value,help=help_text)


def show_chart(fig,key):
    st.plotly_chart(fig,width='stretch',config={'displayModeBar':False,'scrollZoom':False,'responsive':True},key=key)


def risk_line(risk,label):
    st.markdown(f'**{label} : {format_number(risk["p"]*100,2)} %**')
    st.caption(f"{risk['count']} / {risk['total']} tirages · Intervalle exact à 95 % : [{format_number(risk['low']*100,3)} ; {format_number(risk['high']*100,3)}] %")


defaults=Design()
with st.sidebar:
    st.html('<div class="sl-sidebar-brand"><span class="sl-mark" aria-hidden="true"><i></i><i></i><i></i></span>StructureLab</div>')
    for destination, icon in [('Modèle','architecture'), ('Simulation','query_stats'), ('Comparaison','compare_arrows'), ('Méthode','functions')]:
        st.button(destination, icon=f':material/{icon}:', width='stretch', key=f'nav_{destination}',
                  on_click=lambda target=destination: st.session_state.update(view=target))
    st.markdown('### Paramètres')
    st.caption('Acier · section rectangulaire pleine')
    length=st.number_input('Portée L (m)',.5,12.,defaults.length_m,.1,key='length_m')
    width=st.slider('Largeur b (mm)',20.,500.,defaults.width_mm,5.,key='width_mm')
    height=st.slider('Hauteur h (mm)',40.,1000.,defaults.height_mm,5.,key='height_mm')
    load=st.slider('Charge centrale P (kN)',.1,500.,defaults.load_kn,.1,key='load_kn')
    with st.expander('Matériau et critère de flèche'):
        young=st.number_input('Module E (GPa)',100.,250.,defaults.young_gpa,1.,key='young_gpa')
        strength=st.number_input('Limite élastique fy (MPa)',100.,700.,defaults.strength_mpa,5.,key='strength_mpa')
        ratio=st.number_input('Flèche limite : L /',100,1000,300,10,key='limit_ratio',help='Critère pédagogique modifiable, sans valeur normative imposée.')
        weight=st.toggle('Inclure le poids propre',True,key='self_weight')
    with st.expander('Incertitudes'):
        law=st.selectbox('Loi des variables', ['Lognormale','Uniforme'],key='distribution')
        cov={key:st.number_input(f'CV · {LABELS[key]} (%)',0.,40.,defaults.cov[key]*100,1.,key='cv_'+key)/100 for key in VARIABLES}
        st.caption('Variables indépendantes. CV = écart-type / moyenne. CV nul : valeur fixe.')
    with st.expander('Tirages et reproductibilité'):
        n=st.select_slider('Nombre de tirages',[1000,5000,10000,25000,50000,100000],value=10000,key='n')
        seed=st.number_input('Graine aléatoire',0,2147483647,42,1,key='seed')
        st.button('Nouveau tirage',icon=':material/refresh:',on_click=reroll,width='stretch')
    with st.expander('Fichiers de paramètres'):
        st.file_uploader('Importer un projet JSON',type=['json'],key='import_file',on_change=import_parameters)
        st.button('Réinitialiser',icon=':material/restart_alt:',on_click=reset_parameters,width='stretch')

design=Design(length,width,height,load,young,strength,int(ratio),weight,law,cov).validate()
data=design.dumps()
nominal=evaluate(design)
st.html('''<header class="sl-banner"><div class="sl-kicker">MÉCANIQUE &amp; PROBABILITÉS</div>
<h1>StructureLab</h1><p>Modélisation des poutres / Simulation / Comparaison de variantes</p></header>''')
st.html(f'<div class="sl-meta"><span><strong>Acier</strong> · section pleine</span><span>Portée <strong>{length:g} m</strong></span><span>Section <strong>{width:g} × {height:g} mm</strong></span><span>Étude pédagogique</span></div>')
if 'import_message' in st.session_state: st.success(st.session_state.pop('import_message'))
if 'import_error' in st.session_state: st.error(st.session_state.pop('import_error'))
view=st.segmented_control('Vue',['Modèle','Simulation','Comparaison','Méthode'],default='Modèle',key='view',selection_mode='single',width='stretch') or 'Modèle'

if view=='Modèle':
    st.subheader('Modèle de poutre')
    amplification=st.select_slider('Amplification visuelle de la déformée',[1,5,10,20,50],value=20,key='amplification')
    show_chart(beam(design,amplification),'beam')
    kpis([
        ('Contrainte maximale',f"{format_number(nominal['stress_mpa'])} MPa",'Contrainte normale de flexion à la fibre extrême.'),
        ('Flèche maximale',f"{format_number(nominal['deflection_mm'],2)} mm",'Déformée élastique théorique au milieu.'),
        ('Ratio de flexion',format_number(nominal['bending'],3),'Contrainte / fy ; dépassement si le ratio est strictement supérieur à 1.'),
        ('Ratio de flèche',format_number(nominal['service'],3),'Flèche / (L / dénominateur choisi) ; dépassement si supérieur à 1.'),
    ])
    if nominal['bending']>1:
        st.warning('Contrainte nominale au-delà de fy : la déformée élastique affichée est une extrapolation.')
    elif nominal['service']>1:
        st.warning('La flèche nominale dépasse le critère pédagogique choisi.')
    else:
        st.success('Les deux ratios nominaux sont inférieurs ou égaux à 1 dans ce modèle.')
    st.caption(f"Section : {width:g} × {height:g} mm · Masse : {format_number(nominal['mass_kg'])} kg · Limite de flèche : {format_number(nominal['limit_mm'],2)} mm")

elif view=='Simulation':
    st.subheader('Mesurer l’effet des incertitudes')
    with st.spinner('Simulation en cours…'):
        sim=simulation(data,int(n),int(seed))
    risk=sim['risk']['utilization']
    kpis([
        ('Au moins un dépassement',f"{format_number(risk['p']*100,2)} %",'Union des dépassements de contrainte et de flèche.'),
        ('Tirages réalisés',f'{n:,}'.replace(',',' '),'Même configuration et même graine : mêmes résultats.'),
    ])
    risk_line(sim['risk']['bending'],'Contrainte supérieure à fy')
    risk_line(sim['risk']['service'],'Flèche supérieure à la limite')
    risk_line(risk,'Au moins un critère dépassé')
    if risk['count']==0: st.info('Aucun dépassement observé dans cet échantillon. Cela ne démontre pas une probabilité nulle.')
    if sim['risk']['bending']['count']:
        st.caption('Pour les tirages au-delà de fy, le modèle de flèche est une extrapolation élastique.')
    quantity=st.selectbox('Distribution affichée',['Ratio maximal','Ratio de flexion','Ratio de flèche'],key='hist_quantity')
    key={'Ratio maximal':'utilization','Ratio de flexion':'bending','Ratio de flèche':'service'}[quantity]
    show_chart(histogram(sim['results'][key],quantity),'histogram')
    st.subheader('Quels paramètres influencent le résultat ?')
    if sim['sensitivity']:
        show_chart(sensitivity(sim['sensitivity']),'sensitivity')
        st.caption('Corrélations de rang : signe positif, ratio généralement plus élevé ; signe négatif, ratio généralement plus faible. Ni pourcentage de contribution, ni causalité démontrée.')
    else: st.info('Tous les paramètres sont fixes : aucune corrélation à estimer.')
    st.subheader('Stabilisation de l’estimation')
    show_chart(convergence(sim['convergence']),'convergence')
    quantiles=pd.DataFrame({
        'Résultat':['Contrainte (MPa)','Flèche (mm)','Ratio maximal'],
        **{label:[float(np.quantile(sim['results'][k],q)) for k in ('stress_mpa','deflection_mm','utilization')] for label,q in [('P05',.05),('Médiane',.5),('P95',.95)]},
    })
    st.dataframe(quantiles,hide_index=True,width='stretch',alt='Quantiles des résultats simulés')

elif view=='Comparaison':
    st.subheader('Comparer deux variantes')
    if st.button('Mémoriser la variante actuelle comme B',icon=':material/bookmark_add:',width='stretch'):
        st.session_state.variant_b=data
    if 'variant_b' not in st.session_state:
        st.info('Mémorise une première variante, puis modifie les paramètres pour comparer la variante active A à la variante B.')
    else:
        other=Design.loads(st.session_state.variant_b)
        sim=simulation(data,int(n),int(seed))
        sim_b=simulation(other.dumps(),int(n),int(seed))
        nominal_b=evaluate(other)
        st.caption('A = paramètres actuels ; B = paramètres mémorisés. Tirages appariés : même graine et mêmes variables aléatoires de base.')
        for label,a,b in [
            ('Section',f'{width:g} × {height:g} mm',f'{other.width_mm:g} × {other.height_mm:g} mm'),
            ('Portée',f'{length:g} m',f'{other.length_m:g} m'),
            ('Charge centrale',f'{load:g} kN',f'{other.load_kn:g} kN'),
            ('Flèche nominale',f"{format_number(nominal['deflection_mm'],2)} mm",f"{format_number(nominal_b['deflection_mm'],2)} mm"),
            ('Masse',f"{format_number(nominal['mass_kg'])} kg",f"{format_number(nominal_b['mass_kg'])} kg"),
        ]:
            st.markdown(f'**{label}**')
            st.text(f'A : {a}   |   B : {b}',wrap=True)
        risk_line(sim['risk']['utilization'],'A · dépassement combiné')
        risk_line(sim_b['risk']['utilization'],'B · dépassement combiné')
        import plotly.graph_objects as go

        fig=go.Figure()
        for d,label,color in [(design,'Variante A',GREEN),(other,'Variante B',GOLD)]:
            x,y=beam_curve(d)
            fig.add_trace(go.Scatter(x=x,y=y,name=label,line=dict(color=color,width=3)))
        fig.update_xaxes(title='Position (m)'); fig.update_yaxes(title='Flèche réelle (mm)')
        show_chart(finish(fig),'comparison')
        with st.expander('Tous les paramètres des variantes'):
            st.json({'A':asdict(design),'B':asdict(other)})
        if st.button('Effacer la variante B',icon=':material/delete:'):
            del st.session_state.variant_b
            st.rerun()

else:
    st.subheader('Un modèle lisible et reproductible')
    st.markdown('**Poutre pleine rectangulaire en acier, simplement appuyée.** Charge ponctuelle au milieu et, si activé, poids propre uniformément réparti. Calcul élastique d’Euler-Bernoulli, petites déformations et matériau homogène.')
    st.latex(r'I=\frac{bh^3}{12},\quad W=\frac{bh^2}{6},\quad M_{max}=\frac{PL}{4}+\frac{qL^2}{8}')
    st.latex(r'\sigma_{max}=\frac{M_{max}}{W},\quad \delta_{max}=\frac{PL^3}{48EI}+\frac{5qL^4}{384EI}')
    st.markdown('La masse volumique du poids propre est fixée à **7 850 kg/m³**, avec g = 9,81 m/s². Le module SymPy Beam résout les cas unitaires ; les unités sont converties en SI avant calcul.')
    st.markdown('**Simulation.** Les entrées ont les moyennes saisies et les CV choisis. La lognormale est paramétrée pour respecter cette moyenne et ce CV ; l’uniforme utilise les bornes moyenne × (1 ± √3 × CV). Les variables sont indépendantes. Une géométrie simulée peut sortir des limites du panneau : ces limites portent sur les moyennes saisies.')
    st.markdown('**Critères.** Dépassement de flexion si σ > fy ; dépassement de flèche si δ > L / dénominateur. Le critère combiné est leur union, et non la somme des probabilités. Le dénominateur est un choix pédagogique, pas une prescription réglementaire.')
    st.markdown('**Précision.** Les intervalles à 95 % sont les intervalles binomiaux exacts de Clopper-Pearson fournis par SciPy. Ils décrivent l’erreur d’échantillonnage conditionnelle aux hypothèses. Les corrélations de Spearman sont des indices exploratoires de sensibilité.')
    st.warning('Ce laboratoire ne vérifie pas le déversement, le cisaillement, les assemblages, les instabilités, les combinaisons normatives ou les coefficients de sécurité. Ce n’est pas un logiciel de dimensionnement réglementaire. Un dépassement simulé n’est pas une probabilité de rupture réelle.')
    st.markdown('[Moteur de poutre SymPy](https://docs.sympy.org/latest/modules/physics/continuum_mechanics/beam.html) · [Statistiques SciPy](https://docs.scipy.org/doc/scipy/reference/stats.html)')

with st.expander('Exporter l’étude'):
    st.download_button('Paramètres JSON',data,file_name='structurelab-parametres.json',mime='application/json',icon=':material/download:',width='stretch')
    if st.button('Préparer le rapport PDF et les résultats CSV',icon=':material/picture_as_pdf:',width='stretch'):

        with st.spinner('Préparation des fichiers…'):
            sim=simulation(data,int(n),int(seed))
            other=Design.loads(st.session_state.variant_b) if 'variant_b' in st.session_state else None
            pair=(other,simulation(other.dumps(),int(n),int(seed))) if other else None
            pdf=create_report(design,sim,comparison=pair)
            frame=pd.DataFrame(sim['values'])
            for key in ('stress_mpa','deflection_mm','bending','service','utilization'): frame[key]=sim['results'][key]
            frame['depassement_combine']=sim['results']['utilization']>1
            st.session_state.exports=dict(fingerprint=(data,n,seed,st.session_state.get('variant_b')),pdf=pdf,csv=frame.to_csv(index=False).encode('utf-8-sig'))
    exports=st.session_state.get('exports')
    if exports and exports['fingerprint']==(data,n,seed,st.session_state.get('variant_b')):
        st.download_button('Rapport PDF',exports['pdf'],file_name='structurelab-rapport.pdf',mime='application/pdf',icon=':material/download:',width='stretch')
        st.download_button('Résultats CSV',exports['csv'],file_name='structurelab-simulation.csv',mime='text/csv',icon=':material/download:',width='stretch')
st.caption('Modèle pédagogique · Résultats conditionnels aux hypothèses · Aucun compte requis')
