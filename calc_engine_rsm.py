# calc_engine_rsm.py
# Motor de Superficie de Respuesta (CCD) para el flujo tipo Minitab
# Compatible con la arquitectura de calc_engine_plus.py (devuelve dicts R)

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

VIA_CCD = 'ccd'
VIA_INFO_CCD = {'nombre': 'Superficie de Respuesta (CCD)', 'emoji': 'D'}

# ------------------------------------------------------------------
# 1. GENERADOR DE MATRIZ CCD
# ------------------------------------------------------------------
def generar_matriz_ccd(k, alpha='face', n_centros=3, seed=None):
    """Genera matriz CCD codificada (-1/0/+1). alpha: 'face' (=1, como el articulo)
    o 'rot' (rotacional = 2^(k/4)). Incluye n_centros puntos centrales."""
    from itertools import product
    if alpha == 'face':
        a = 1.0
    elif alpha == 'rot':
        a = 2 ** (k / 4)
    else:
        a = float(alpha)
    esquinas = list(product([-1, 1], repeat=k))
    axiales = []
    for i in range(k):
        fila_mas = [0]*k; fila_mas[i] = a
        fila_menos = [0]*k; fila_menos[i] = -a
        axiales += [fila_mas, fila_menos]
    datos = esquinas + axiales + [[0]*k]*n_centros
    cols = [chr(65+i) for i in range(k)]
    df = pd.DataFrame(datos, columns=cols)
    df.insert(0, 'Corrida', range(1, len(df)+1))
    if seed is not None:
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
        df['Corrida'] = range(1, len(df)+1)
    return df

# ------------------------------------------------------------------
# 2. CODIFICACION (unidades reales <-> codificadas)
# ------------------------------------------------------------------
def codificar(df, factores):
    """Anade columnas '_c' codificadas (-1/0/+1) y devuelve el diccionario
    de codificacion {f: {'lo','c','hi'}}. Asume diseno simetrico."""
    coding = {}
    out = df.copy()
    for f in factores:
        lo, hi = float(df[f].min()), float(df[f].max())
        c = (lo + hi) / 2
        h = (hi - lo) / 2
        coding[f] = {'lo': lo, 'c': c, 'hi': hi, 'half': h}
        out[f + '_c'] = (df[f] - c) / h
    return out, coding

# ------------------------------------------------------------------
# 3. TERMINOS CUADRATICOS Y FORMULAS
# ------------------------------------------------------------------
def terminos_completos(factores):
    """Lineales + cuadrados + interacciones 2 a 2. Notacion 'A', 'A*A', 'A*B'."""
    t = list(factores)
    t += [f'{a}*{a}' for a in factores]
    t += [f'{a}*{b}' for i, a in enumerate(factores) for b in factores[i+1:]]
    return t

def _param(term, factores):
    """Nombre del parametro patsy/statsmodels para un termino."""
    partes = term.split('*')
    if len(partes) == 1:
        return f'Q("{partes[0]}_c")'
    if partes[0] == partes[1]:
        return f'I(Q("{partes[0]}_c") ** 2)'
    return ':'.join(f'Q("{p}_c")' for p in partes)

def _formula(ycol, factores, terms):
    lados = [_param(t, factores) for t in terms]
    return f'Q("{ycol}") ~ ' + (' + '.join(lados) if lados else '1')

# ------------------------------------------------------------------
# 4. ELIMINACION HACIA ATRAS (estilo Minitab, con jerarquia)
# ------------------------------------------------------------------
def _backward_elimination(df, ycol, factores, alpha=0.05):
    terms = terminos_completos(factores)
    historial = []
    while True:
        fml = _formula(ycol, factores, terms)
        m = smf.ols(fml, data=df).fit()
        pmap = {}
        for t in terms:
            pn = _param(t, factores)
            pmap[t] = m.pvalues.get(pn, np.nan)
        candidatos = [(t, p) for t, p in pmap.items() if pd.notna(p) and p > alpha]
        if not candidatos:
            break
        candidatos.sort(key=lambda x: -x[1])
        eliminado = None
        for t, p in candidatos:
            # Jerarquia: un efecto principal NO se quita si participa en una
            # interaccion que permanece en el modelo. Las interacciones y
            # cuadrados si se pueden quitar libremente.
            partes = t.split('*')
            if len(partes) == 1:  # efecto principal: respetar jerarquia con interacciones y cuadrados
                usado = any(('*' in ot) and (partes[0] in ot.split('*')) for ot in terms if ot != t)
                if usado:
                    continue
            # cuadrados e interacciones: libres
            eliminado = t
            break
        if eliminado is None:
            break
        terms.remove(eliminado)
        historial.append((eliminado, pmap[eliminado]))
    return terms, historial

# ------------------------------------------------------------------
# 5. AJUSTE CCD
# ------------------------------------------------------------------
def ajustar_ccd(df_raw, factores, ycol, alpha_elim=0.05, eliminar=True, terms=None):
    """Ajusta modelo cuadratico por OLS; si eliminar=True aplica backward elimination.
    Si terms se pasa explicitamente, ajusta exactamente esos terminos."""
    df, coding = codificar(df_raw, factores)

    terms_full = terminos_completos(factores)
    fml_full = _formula(ycol, factores, terms_full)
    m_full = smf.ols(fml_full, data=df).fit()

    if terms is not None:
        terms = list(terms)
        historial = []
    elif eliminar:
        terms, historial = _backward_elimination(df, ycol, factores, alpha_elim)
    else:
        terms, historial = list(terms_full), []

    fml = _formula(ycol, factores, terms)
    m = smf.ols(fml, data=df).fit()

    an = anova_lm(m, typ=3).reset_index().rename(
        columns={'index': 'Fuente', 'sum_sq': 'SS', 'df': 'GL', 'F': 'F', 'PR(>F)': 'p'})
    an = an[an['Fuente'] != 'Intercept'].copy()
    an['MS'] = an['SS'] / an['GL']
    an = an[['Fuente', 'SS', 'GL', 'MS', 'F', 'p']]
    an['Sig'] = ['OK' if (pd.notna(p) and p < 0.05) else ('-' if pd.isna(p) else 'NO') for p in an['p']]

    grupos = df_raw.groupby(factores)[ycol]
    sizes = grupos.size()
    ss_pe = sum((n - 1) * grupos.get_group(g).var(ddof=1)
                for g, n in sizes.items() if n >= 2 and not np.isnan(grupos.get_group(g).var(ddof=1)))
    df_pe = int((sizes[sizes >= 2] - 1).sum())
    lof = None
    sse = float((m.resid ** 2).sum())
    df_resid = int(m.df_resid)
    if df_pe >= 1 and df_resid > df_pe and sse > ss_pe:
        ss_lof = sse - ss_pe
        df_lof = df_resid - df_pe
        F_lof = (ss_lof / df_lof) / (ss_pe / df_pe)
        lof = {'SS_pe': ss_pe, 'df_pe': df_pe, 'SS_lof': ss_lof, 'df_lof': df_lof,
               'F': F_lof, 'p': stats.f.sf(F_lof, df_lof, df_pe)}

    effects = {}
    for t in terms:
        pn = _param(t, factores)
        effects[t] = {'coef': float(m.params.get(pn, np.nan)),
                      'p': float(m.pvalues.get(pn, np.nan)),
                      'efecto': float(2 * m.params.get(pn, np.nan))}

    eq_coded = _eq_str(m, terms, factores, coding=None, usar_letras=False)
    eq_uncoded = _eq_str(m, terms, factores, coding=coding, usar_letras=False)
    eq_coded_letras = _eq_str(m, terms, factores, coding=None, usar_letras=True)
    eq_uncoded_letras = _eq_str(m, terms, factores, coding=coding, usar_letras=True)

    R = dict(via=VIA_CCD, k=len(factores), factores=list(factores), respuesta=ycol,
             N=len(df), coding=coding, terms_full=terms_full, terms=terms,
             historial=historial, model=m, model_full=m_full,
             anova=an, R2=float(m.rsquared), R2_adj=float(m.rsquared_adj),
             S=float(np.sqrt(m.mse_resid)),
             lof=lof, effects=effects, y_bar=float(df[ycol].mean()),
             eq_coded=eq_coded, eq_uncoded=eq_uncoded,
             eq_coded_letras=eq_coded_letras, eq_uncoded_letras=eq_uncoded_letras,
             fitted=np.asarray(m.fittedvalues), resid=np.asarray(m.resid),
             tiene_bloques=False, tiene_centro=True)
    return R

# ------------------------------------------------------------------
# 6. ECUACIONES
# ------------------------------------------------------------------
def _eq_str(m, terms, factores, coding=None, usar_letras=False):
    """Ecuacion legible sin superindice cuadrado (usa AA, CC, AC o Factor*Factor)."""
    idx = {f: i for i, f in enumerate(factores)}
    n = len(factores)
    letras = [chr(65 + i) if i < 26 else f"X{i + 1}" for i in range(n)]

    b = {}
    b0 = float(m.params['Intercept'])
    for t in terms:
        pn = _param(t, factores)
        if pn not in m.params:
            continue
        coef = float(m.params[pn])
        p = t.split('*')
        if len(p) == 1:
            b[(idx[p[0]],)] = b.get((idx[p[0]],), 0.0) + coef
        else:
            i, j = sorted((idx[p[0]], idx[p[1]]))
            b[(i, j)] = b.get((i, j), 0.0) + coef

    if coding is None:
        B0 = b0
        lin = [b.get((i,), 0.0) for i in range(n)]
        cuad = [b.get((i, i), 0.0) for i in range(n)]
        cruz = {k: v for k, v in b.items() if len(k) == 2 and k[0] != k[1]}
    else:
        c = [coding[f]['c'] for f in factores]
        h = [coding[f]['half'] for f in factores]
        B0 = b0
        lin = [0.0] * n
        cuad = [0.0] * n
        cruz = {}
        for k, v in b.items():
            if len(k) == 1:
                i = k[0]
                B0 += v * (-c[i] / h[i])
                lin[i] += v / h[i]
        for k, v in b.items():
            if len(k) == 1:
                continue
            i, j = k
            if i == j:
                B0 += v * (c[i] ** 2 / h[i] ** 2)
                lin[i] += v * (-2 * c[i] / h[i] ** 2)
                cuad[i] += v / h[i] ** 2
            else:
                B0 += v * (c[i] * c[j] / (h[i] * h[j]))
                lin[i] += v * (-c[j] / (h[i] * h[j]))
                lin[j] += v * (-c[i] / (h[i] * h[j]))
                cruz[(i, j)] = v / (h[i] * h[j])

    def _sgn(val):
        return f"+ {val:.4f}" if val >= 0 else f"- {abs(val):.4f}"

    fs = list(factores)
    s = f"Y = {B0:.4f}"
    if usar_letras:
        for i, L in enumerate(letras):
            if abs(lin[i]) > 1e-12:
                s += f" {_sgn(lin[i])}{L}"
        for i, L in enumerate(letras):
            if abs(cuad[i]) > 1e-12:
                s += f" {_sgn(cuad[i])}{L}{L}"
        for (i, j), v in sorted(cruz.items()):
            if abs(v) > 1e-12:
                s += f" {_sgn(v)}{letras[i]}{letras[j]}"
    else:
        for i, f in enumerate(fs):
            if abs(lin[i]) > 1e-12:
                s += f" {_sgn(lin[i])}*{f}"
        for i, f in enumerate(fs):
            if abs(cuad[i]) > 1e-12:
                s += f" {_sgn(cuad[i])}*{f}*{f}"
        for (i, j), v in sorted(cruz.items()):
            if abs(v) > 1e-12:
                s += f" {_sgn(v)}*{fs[i]}*{fs[j]}"
    return s.strip()


# ------------------------------------------------------------------
# 7. PREDICCION
# ------------------------------------------------------------------
def predecir_ccd(R, punto):
    """punto: dict {factor: valor_real}. Devuelve Y predicho."""
    fila = {}
    for f, v in punto.items():
        c = R['coding'][f]
        fila[f + '_c'] = (v - c['c']) / c['half']
    X = pd.DataFrame([fila])
    fml = _formula(R['respuesta'], R['factores'], R['terms'])
    import re
    # statsmodels necesita tambien la columna de respuesta; usamos predict con filtro
    dfaux = X.copy()
    p = R['model'].params
    pred = float(p['Intercept'])
    for t in R['terms']:
        pn = _param(t, R['factores'])
        partes = t.split('*')
        prod = 1.0
        for pp in partes:
            prod *= fila[pp + '_c']
        pred += float(p[pn]) * prod
    return pred

# ------------------------------------------------------------------
# 8. GRAFICOS
# ------------------------------------------------------------------
def _zgrid(R, f1, f2, XX, YY, fijos=None):
    """Evaluacion VECTORIZADA (numpy) del modelo sobre una malla 2D.
    Los factores no graficados se fijan en su centro, o en el valor
    indicado en 'fijos' ({factor: valor_real})."""
    cod = R['coding']
    fs = R['factores']
    xc = {}
    for f in fs:
        if f == f1:
            xc[f] = (XX - cod[f]['c']) / cod[f]['half']
        elif f == f2:
            xc[f] = (YY - cod[f]['c']) / cod[f]['half']
        elif fijos and f in fijos:
            xc[f] = np.full_like(XX, (fijos[f] - cod[f]['c']) / cod[f]['half'])
        else:
            xc[f] = np.zeros_like(XX)
    Z = np.full_like(XX, float(R['model'].params['Intercept']))
    p = R['model'].params
    for t in R['terms']:
        b = float(p[_param(t, fs)])
        prod = np.ones_like(XX)
        for pp in t.split('*'):
            prod = prod * xc[pp]
        Z = Z + b * prod
    return Z

def graf_contorno_ccd(R, f1, f2, ax=None, titulo=None, niveles=None,
                      estilo='moderno', cmap=None, xlabel=None, ylabel=None, punto=None,
                      fijos=None):
    """Contorno 2D de una respuesta sobre el plano (f1, f2), otros factores al centro."""
    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=(7, 5.5))
    c1, c2 = R['coding'][f1], R['coding'][f2]
    x = np.linspace(c1['lo'], c1['hi'], 200)
    y = np.linspace(c2['lo'], c2['hi'], 200)
    XX, YY = np.meshgrid(x, y)
    Z = _zgrid(R, f1, f2, XX, YY, fijos=fijos)
    if niveles is None:
        niveles = 12
    if estilo == 'clasico':
        cf = ax.contourf(XX, YY, Z, levels=niveles, cmap=(cmap or 'terrain'))
        # Las lineas van EXACTAMENTE en las fronteras de las bandas de color
        line_lv = list(cf.levels)
        if len(line_lv) > 8:
            line_lv = line_lv[::2]   # subsampleo, pero siempre sobre fronteras reales
        cl = ax.contour(XX, YY, Z, levels=line_lv, colors='k', linewidths=0.7)
        ax.clabel(cl, fmt='%g', fontsize=8)
        ax.grid(color='white', alpha=0.3, linewidth=0.6)
    else:
        cf = ax.contourf(XX, YY, Z, levels=niveles, cmap='viridis')
        cl = ax.contour(XX, YY, Z, levels=cf.levels, colors='white', linewidths=0.8)
        ax.clabel(cl, fmt='%.2g', fontsize=8)
    if punto:
        ax.plot(punto[f1], punto[f2], 'r*', ms=14, mec='k')
    ax.set_xlabel(xlabel or f1)
    ax.set_ylabel(ylabel or f2)
    ax.set_title(titulo or f"Contorno: {R['respuesta']}")
    cb = plt.colorbar(cf, ax=ax, label=R['respuesta'])
    if isinstance(niveles, (list, tuple, np.ndarray)) and len(niveles) <= 15:
        cb.set_ticks(list(niveles))
    if created:
        fn = f"contorno_{R['respuesta']}_{f1}_{f2}.png"
        plt.savefig(fn, bbox_inches='tight'); plt.close()
        return fn

def graf_superficie_3d_ccd(R, f1, f2, titulo=None, estilo='moderno', cmap=None,
                           xlabel=None, ylabel=None, fijos=None):
    """Superficie 3D de una respuesta sobre el plano (f1, f2), otros factores fijos."""
    fig = plt.figure(figsize=(7.5, 5.8))
    ax = fig.add_subplot(111, projection='3d')
    c1, c2 = R['coding'][f1], R['coding'][f2]
    x = np.linspace(c1['lo'], c1['hi'], 60)
    y = np.linspace(c2['lo'], c2['hi'], 60)
    XX, YY = np.meshgrid(x, y)
    Z = _zgrid(R, f1, f2, XX, YY, fijos=fijos)

    if estilo == 'clasico':
        surf = ax.plot_surface(XX, YY, Z, cmap=(cmap or 'terrain'),
                               edgecolor='#333333', linewidth=0.25, alpha=0.92)
    else:
        surf = ax.plot_surface(XX, YY, Z, cmap='viridis',
                               edgecolor='none', alpha=0.90)

    ax.set_xlabel(xlabel or f1, labelpad=8)
    ax.set_ylabel(ylabel or f2, labelpad=8)
    ax.set_zlabel(R['respuesta'], labelpad=10)
    ax.set_title(titulo or f"Superficie 3D: {R['respuesta']}")
    fig.colorbar(surf, ax=ax, shrink=0.6, aspect=10, pad=0.1, label=R['respuesta'])
    plt.tight_layout()
    fn = f"superficie3d_{R['respuesta']}_{f1}_{f2}_{estilo}.png"
    plt.savefig(fn, bbox_inches='tight', dpi=150)
    plt.close()
    return fn

def graf_superposicion_ccd(Rs, rangos, f1, f2, punto=None, ax=None, xlabel=None, ylabel=None,
                           fijos=None):
    """Rs: lista de dicts R (uno por respuesta). rangos: {nombre_resp: (lo, hi)}.
    Dibuja region blanca donde TODAS las respuestas estan en rango (estilo fig. 4 del articulo)."""
    created = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.5, 6))
        created = True
    cod = Rs[0]['coding']
    x = np.linspace(cod[f1]['lo'], cod[f1]['hi'], 300)
    y = np.linspace(cod[f2]['lo'], cod[f2]['hi'], 300)
    XX, YY = np.meshgrid(x, y)
    base = {f: cod[f]['c'] for f in Rs[0]['factores']}
    ok = np.ones_like(XX, dtype=bool)
    cols = ['#1f5fb4', '#b41f1f', '#1f9b4f', '#8a2be2']
    for idx, R in enumerate(Rs):
        Z = _zgrid(R, f1, f2, XX, YY, fijos=fijos)
        lo, hi = rangos[R['respuesta']]
        ok &= (Z >= lo) & (Z <= hi)
        ax.contour(XX, YY, Z, levels=[lo, hi], colors=[cols[idx % 4]]*2,
                   linestyles=['-', '--'], linewidths=1.8)
        ax.plot([], [], color=cols[idx % 4],
                label=f"{R['respuesta']}: {lo} (—) / {hi} (- -)")
    ax.contourf(XX, YY, ok.astype(float), levels=[-0.5, 0.5, 1.5],
                colors=['lightgrey', 'white'], alpha=0.9)
    if punto:
        ax.plot(punto[f1], punto[f2], 'g*', ms=18, mec='k')
        ax.annotate('validación', (punto[f1], punto[f2]),
                    textcoords='offset points', xytext=(8, 8), color='darkgreen')
    ax.set_xlabel(xlabel or f1)
    ax.set_ylabel(ylabel or f2)
    ax.set_title('Contorno combinado (región blanca = todas las metas)')
    ax.legend(loc='upper right', fontsize=9)
    if created:
        fn = f"superposicion_{f1}_{f2}.png"
        plt.savefig(fn, bbox_inches='tight'); plt.close()
        return fn

def graf_residuales_ccd(R):
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.2))
    stats.probplot(R['resid'], dist='norm', plot=axes[0])
    axes[0].set_title('(a) Normalidad')
    axes[1].scatter(R['fitted'], R['resid'], alpha=0.75, edgecolor='k')
    axes[1].axhline(0, color='red', ls='--'); axes[1].set_title('(b) Resid. vs Ajustados')
    axes[1].set_xlabel('Y ajustado'); axes[1].set_ylabel('Residual')
    axes[2].scatter(range(1, len(R['resid'])+1), R['resid'], alpha=0.75, edgecolor='k')
    axes[2].axhline(0, color='red', ls='--'); axes[2].set_title('(c) Resid. vs Orden')
    plt.tight_layout()
    fn = 'residuales_ccd.png'
    plt.savefig(fn, bbox_inches='tight'); plt.close()
    return fn

# ------------------------------------------------------------------
# 9. OPTIMIZACION (deseabilidad de Derringer-Suich)
# ------------------------------------------------------------------
def optimizar_ccd(Rs, objetivos, n=21):
    """objetivos: {resp: {'tipo':'max'|'min'|'target', 'lo':..., 'hi':..., 'target':...}}
    Devuelve mejor punto (reales), predicciones y deseabilidad global."""
    cod = Rs[0]['coding']
    factores = Rs[0]['factores']
    fidx = {f: i for i, f in enumerate(factores)}
    grids = [np.linspace(cod[f]['lo'], cod[f]['hi'], n) for f in factores]
    mesh = np.meshgrid(*grids, indexing='ij')
    P = np.stack([m.ravel() for m in mesh], axis=1)     # (n^k, k) — vectorizado
    C = np.array([cod[f]['c'] for f in factores])
    H = np.array([cod[f]['half'] for f in factores])
    Xc = (P - C) / H
    npts = P.shape[0]
    d_glob = np.ones(npts)
    valido = np.ones(npts, dtype=bool)
    for R in Rs:
        o = objetivos[R['respuesta']]
        yv = np.full(npts, float(R['model'].params['Intercept']))
        p = R['model'].params
        for t in R['terms']:
            b = float(p[_param(t, factores)])
            prod = np.ones(npts)
            for pp in t.split('*'):
                prod = prod * Xc[:, fidx[pp]]
            yv = yv + b * prod
        lo, hi = o['lo'], o['hi']
        valido &= (yv >= lo) & (yv <= hi)
        if o['tipo'] == 'max':
            d = (yv - lo) / (hi - lo)
        elif o['tipo'] == 'min':
            d = (hi - yv) / (hi - lo)
        else:
            t0 = o['target']
            d = np.where(yv <= t0, (yv - lo) / (t0 - lo), (hi - yv) / (hi - t0))
        d_glob *= np.clip(d, 0.0, 1.0)
    d_glob[~valido] = -1.0
    i = int(np.argmax(d_glob))
    if d_glob[i] < 0:
        return None
    mejor = {f: float(P[i, j]) for j, f in enumerate(factores)}
    preds = {R['respuesta']: predecir_ccd(R, mejor) for R in Rs}
    return {'punto': mejor, 'predicciones': preds, 'deseabilidad': float(d_glob[i])}

# ------------------------------------------------------------------
# 10. DETECCION DE CCD (para el parche de app_plus.py)
# ------------------------------------------------------------------
def es_diseno_ccd(raw, factores):
    """True solo si el diseno tiene puntos AXIALES (exactamente un factor != 0).
    Asi no se confunde con un 2^k con puntos centrales (Via 2 de la app)."""
    try:
        num = raw[list(factores)].apply(pd.to_numeric, errors='coerce')
    except Exception:
        return False
    if num.isna().any().any() or len(factores) < 2:
        return False
    X = num.to_numpy(dtype=float)
    lo = X.min(axis=0); hi = X.max(axis=0)
    if np.any(hi == lo):
        return False
    Xc = 2 * (X - lo) / (hi - lo) - 1          # normalizar a [-1, 0, 1]
    axiales = ((np.abs(Xc) > 1e-5).sum(axis=1) == 1).any()
    return bool(axiales) and len(raw) > 2 ** len(factores)

# ------------------------------------------------------------------
# 11. TABLA DE COEFICIENTES Y GRAFICOS FACTORIALES PARA CCD
# ------------------------------------------------------------------
import re


def _safe_fname_ccd(texto):
    return re.sub(r'[^A-Za-z0-9_\-]+', '_', str(texto)).strip('_') or 'resp'


def _nombre_termino_ccd(t):
    partes = str(t).split('*')
    if len(partes) == 2 and partes[0] == partes[1]:
        return f"{partes[0]}²"
    return str(t)


def limpiar_anova_ccd(df_anova):
    """Convierte nombres internos de statsmodels (Q('...')) a nombres legibles."""
    an = df_anova.copy()
    if 'Fuente' in an.columns:
        def _clean(fuente):
            s = str(fuente).strip()
            m_sq = re.match(r'^I\(Q\("(.+?)_c"\)\s*\*\*\s*2\)$', s)
            if m_sq:
                return f"{m_sq.group(1)}²"
            s = re.sub(r'Q\("(.+?)_c"\)', r'\1', s)
            s = s.replace(':', '*')
            return s
        an['Fuente'] = an['Fuente'].apply(_clean)
    return an


def generar_tabla_coeficientes_ccd(R):
    """Devuelve DataFrame con Coef, SE Coef, t, p y significancia del modelo CCD (en escala codificada)."""
    m = R['model']
    factores = R['factores']
    filas = []
    if 'Intercept' in m.params:
        filas.append({
            'Termino': 'Constante (Intercepto)',
            'Coef': float(m.params['Intercept']),
            'SE Coef': float(m.bse['Intercept']),
            't': float(m.tvalues['Intercept']),
            'p': float(m.pvalues['Intercept']),
            'Sig': '-'
        })
    for t in R['terms']:
        pn = _param(t, factores)
        if pn in m.params:
            p_val = float(m.pvalues.get(pn, np.nan))
            filas.append({
                'Termino': _nombre_termino_ccd(t),
                'Coef': float(m.params[pn]),
                'SE Coef': float(m.bse[pn]),
                't': float(m.tvalues[pn]),
                'p': p_val,
                'Sig': 'OK' if (pd.notna(p_val) and p_val < 0.05) else ('-' if pd.isna(p_val) else 'NO')
            })
    return pd.DataFrame(filas)


def graf_pareto_ccd(R, usar_completo=False, alpha=0.05):
    """Diagrama de Pareto de efectos estandarizados |t| para el modelo CCD."""
    m = R.get('model_full') if (usar_completo and R.get('model_full') is not None) else R['model']
    terms = list(R.get('terms_full', R['terms'])) if (usar_completo and R.get('model_full') is not None) else list(R['terms'])
    factores = R['factores']

    labels, tvals, pvals = [], [], []
    usar_t = True
    for t in terms:
        pn = _param(t, factores)
        if pn not in m.params:
            continue
        tv = float(abs(m.tvalues.get(pn, np.nan)))
        if not np.isfinite(tv):
            usar_t = False
            break

    for t in terms:
        pn = _param(t, factores)
        if pn not in m.params:
            continue
        labels.append(_nombre_termino_ccd(t))
        if usar_t:
            tvals.append(float(abs(m.tvalues[pn])))
        else:
            tvals.append(float(abs(m.params[pn])))
        pvals.append(float(m.pvalues.get(pn, np.nan)))

    if not labels:
        return None

    df_res = int(getattr(m, 'df_resid', 0))
    t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df_res)) if (usar_t and df_res >= 1) else np.nan

    order = np.argsort(tvals)
    fig, ax = plt.subplots(figsize=(10, max(4, 0.38 * len(labels) + 2)))
    if pd.notna(t_crit):
        colors = ['#d9534f' if tvals[i] >= t_crit else '#7fb3d5' for i in order]
    else:
        colors = ['#7fb3d5' for _ in order]

    ax.barh([labels[i] for i in order], [tvals[i] for i in order], color=colors, edgecolor='black')

    if pd.notna(t_crit):
        lab = f'|t| critico = {t_crit:.2f} (alpha={alpha})'
        ax.axvline(t_crit, color='red', linestyle='--', label=lab)
        ylo, yhi = ax.get_ylim()
        ax.set_ylim(ylo, yhi + 0.8)
        ax.text(
            t_crit, yhi + 0.10, f'  {lab}  ', color='red', fontsize=10, fontweight='bold',
            ha='center', va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='red', alpha=0.95)
        )
        ax.legend(loc='lower right')

    ax.set_xlabel('|t| estandarizado' if usar_t else '|Coeficiente|')
    tipo_txt = 'Modelo completo' if usar_completo else 'Modelo actual'
    ax.set_title(f"Diagrama de Pareto CCD - {R['respuesta']} ({tipo_txt})")
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    fn = f"pareto_ccd_{_safe_fname_ccd(R['respuesta'])}.png"
    plt.savefig(fn, bbox_inches='tight', dpi=150)
    plt.close()
    return fn


def graf_seminormal_ccd(R, usar_completo=True):
    """Grafico Semi-normal de Daniel (efectos estandarizados |t|) para el modelo CCD."""
    m = R.get('model_full') if (usar_completo and R.get('model_full') is not None) else R['model']
    terms = list(R.get('terms_full', R['terms'])) if (usar_completo and R.get('model_full') is not None) else list(R['terms'])
    factores = R['factores']
    activos = set(R.get('terms', []))

    labels, vals, es_activo = [], [], []
    usar_t = all(
        np.isfinite(abs(m.tvalues.get(_param(t, factores), np.nan)))
        for t in terms if _param(t, factores) in m.params
    )

    for t in terms:
        pn = _param(t, factores)
        if pn not in m.params:
            continue
        labels.append(_nombre_termino_ccd(t))
        vals.append(float(abs(m.tvalues[pn])) if usar_t else float(abs(m.params[pn])))
        p_val = m.pvalues.get(pn, np.nan)
        sig = (pd.notna(p_val) and p_val < 0.05) or (not usar_completo and t in activos)
        es_activo.append(bool(sig))

    if len(vals) < 2:
        return None

    ae = np.array(vals, dtype=float)
    cnt = len(ae)
    o = np.argsort(ae)
    a = ae[o]
    ns = [labels[i] for i in o]
    acts_ord = [es_activo[i] for i in o]

    pp = (np.arange(1, cnt + 1) - 0.5) / cnt
    q = stats.norm.ppf((1.0 + pp) / 2.0)

    # PSE y ME de Lenth
    s0 = 1.5 * np.median(a)
    sub = a[a <= 2.5 * s0] if s0 > 0 else a
    pse = 1.5 * np.median(sub) if len(sub) > 0 else s0
    df_lenth = max(1.0, cnt / 3.0)
    me = float(stats.t.ppf(0.975, df_lenth) * pse) if pse > 0 else np.nan

    fig, ax = plt.subplots(figsize=(8.5, 5))
    cols = ['#d9534f' if act else 'steelblue' for act in acts_ord]
    ax.scatter(q, a, c=cols, s=90, edgecolor='black', zorder=3)
    for qi, yi, ni in zip(q, a, ns):
        ax.annotate(ni, (qi, yi), xytext=(5, 3), textcoords='offset points', fontsize=8)

    if pd.notna(me) and me > 0:
        nm = a < me
        sl, ic = np.polyfit(q[nm] if nm.sum() >= 2 else q, a[nm] if nm.sum() >= 2 else a, 1)
        lx = np.array([q.min() - 0.1, q.max() + 0.1])
        ax.plot(lx, sl * lx + ic, 'k--', label='Referencia')
        ax.axhline(me, color='red', linestyle='--', label=f'ME (Lenth) = {me:.3f}')
        ax.set_title(f"Semi-normal (Daniel) - {R['respuesta']} | PSE={pse:.3f}")
    else:
        ax.set_title(f"Semi-normal (Daniel) - {R['respuesta']}")

    ax.set_xlabel('Cuantiles half-normal')
    ax.set_ylabel('|t| estandarizado' if usar_t else '|Coeficiente|')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fn = f"seminormal_ccd_{_safe_fname_ccd(R['respuesta'])}.png"
    plt.savefig(fn, bbox_inches='tight', dpi=150)
    plt.close()
    return fn


def _predecir_con_modelo_ccd(m, terms, coding, factores, punto):
    fila = {f + '_c': (punto[f] - coding[f]['c']) / coding[f]['half'] for f in factores}
    p = m.params
    pred = float(p['Intercept'])
    for t in terms:
        pn = _param(t, factores)
        if pn not in p:
            continue
        prod = 1.0
        for pp in t.split('*'):
            prod *= fila[pp + '_c']
        pred += float(p[pn]) * prod
    return pred


def graf_interacciones_ccd(R, unidades=None, fijos=None, eje_x_por_interaccion=None, usar_completo=False):
    """Graficos de interaccion 2 a 2 sobre la superficie ajustada CCD (manteniendo otros factores fijos)."""
    factores = R['factores']
    coding = R['coding']
    m = R.get('model_full') if (usar_completo and R.get('model_full') is not None) else R['model']
    terms = list(R.get('terms_full', R['terms'])) if (usar_completo and R.get('model_full') is not None) else list(R['terms'])

    sig = [t for t in terms if '*' in t and t.split('*')[0] != t.split('*')[1]]
    if not sig:
        return []

    eje_x_map = eje_x_por_interaccion or {}
    base = {f: (fijos[f] if (fijos and f in fijos) else coding[f]['c']) for f in factores}
    files = []
    chunk_size = 2

    for i in range(0, len(sig), chunk_size):
        chunk = sig[i:i + chunk_size]
        fig, axes = plt.subplots(1, len(chunk), figsize=(6 * len(chunk), 4.5))
        if len(chunk) == 1:
            axes = [axes]

        for ax, name in zip(axes, chunk):
            fa, fb = name.split('*')
            x_fac = eje_x_map.get(name, fa)
            leg_fac = fb if x_fac == fa else fa

            cx, cl = coding[x_fac], coding[leg_fac]
            xs = np.linspace(cx['lo'], cx['hi'], 50)
            xs_pts = [cx['lo'], cx['hi']]

            def _eval_linea(x_vals, leg_val):
                ys = []
                for xv in x_vals:
                    pt = dict(base)
                    pt[x_fac] = float(xv)
                    pt[leg_fac] = float(leg_val)
                    ys.append(_predecir_con_modelo_ccd(m, terms, coding, factores, pt))
                return ys

            y_low_curve = _eval_linea(xs, cl['lo'])
            y_high_curve = _eval_linea(xs, cl['hi'])
            y_low_pts = _eval_linea(xs_pts, cl['lo'])
            y_high_pts = _eval_linea(xs_pts, cl['hi'])

            u_x = (unidades or {}).get(x_fac, '').strip()
            u_leg = (unidades or {}).get(leg_fac, '').strip()
            lbl_x = f"{x_fac} ({u_x})" if u_x else x_fac
            suf_u_leg = f" {u_leg}" if u_leg else ""

            ax.plot(xs, y_low_curve, '-', color='#90CAF9', linewidth=2.2)
            ax.plot(xs_pts, y_low_pts, 'o', color='#1976D2', markersize=8,
                    label=f"{leg_fac} = {cl['lo']:.4g}{suf_u_leg}")

            ax.plot(xs, y_high_curve, '-', color='#EF9A9A', linewidth=2.2)
            ax.plot(xs_pts, y_high_pts, 's', color='#D32F2F', markersize=8,
                    label=f"{leg_fac} = {cl['hi']:.4g}{suf_u_leg}")

            ax.set_xticks([cx['lo'], cx['c'], cx['hi']])
            ax.set_xlabel(lbl_x)
            ax.set_ylabel(f"Prediccion {R['respuesta']}")
            ax.set_title(f"Interaccion {x_fac} * {leg_fac}")
            ax.legend(loc='best', framealpha=0.9)
            ax.grid(alpha=0.3)

        plt.tight_layout()
        fn = f"interacciones_ccd_{_safe_fname_ccd(R['respuesta'])}_{i // chunk_size + 1}.png"
        plt.savefig(fn, bbox_inches='tight', dpi=150)
        plt.close()
        files.append(fn)

    return files

# ------------------------------------------------------------------
# 11. TABLA DE COEFICIENTES Y GRAFICOS FACTORIALES PARA CCD
# ------------------------------------------------------------------
import re


def _safe_fname_ccd(texto):
    return re.sub(r'[^A-Za-z0-9_\-]+', '_', str(texto)).strip('_') or 'resp'


def _nombre_termino_ccd(t):
    return str(t)


def limpiar_anova_ccd(df_anova):
    """Convierte nombres internos de statsmodels (Q('...')) a nombres legibles (Factor*Factor, sin superindice)."""
    an = df_anova.copy()
    if 'Fuente' in an.columns:
        def _clean(fuente):
            s = str(fuente).strip()
            m_sq = re.match(r'^I\(Q\("(.+?)_c"\)\s*\*\*\s*2\)$', s)
            if m_sq:
                f_nom = m_sq.group(1)
                return f"{f_nom}*{f_nom}"
            s = re.sub(r'Q\("(.+?)_c"\)', r'\1', s)
            s = s.replace(':', '*')
            return s
        an['Fuente'] = an['Fuente'].apply(_clean)
    return an


def generar_tabla_coeficientes_ccd(R):
    """Devuelve DataFrame con Coef, SE Coef, t, p y significancia del modelo CCD (en escala codificada)."""
    m = R['model']
    factores = R['factores']
    filas = []
    if 'Intercept' in m.params:
        filas.append({
            'Termino': 'Constante (Intercepto)',
            'Coef': float(m.params['Intercept']),
            'SE Coef': float(m.bse['Intercept']),
            't': float(m.tvalues['Intercept']),
            'p': float(m.pvalues['Intercept']),
            'Sig': '-'
        })
    for t in R['terms']:
        pn = _param(t, factores)
        if pn in m.params:
            p_val = float(m.pvalues.get(pn, np.nan))
            filas.append({
                'Termino': _nombre_termino_ccd(t),
                'Coef': float(m.params[pn]),
                'SE Coef': float(m.bse[pn]),
                't': float(m.tvalues[pn]),
                'p': p_val,
                'Sig': 'OK' if (pd.notna(p_val) and p_val < 0.05) else ('-' if pd.isna(p_val) else 'NO')
            })
    return pd.DataFrame(filas)


def graf_pareto_ccd(R, usar_completo=False, alpha=0.05):
    """Diagrama de Pareto de efectos estandarizados |t| para el modelo CCD."""
    m = R.get('model_full') if (usar_completo and R.get('model_full') is not None) else R['model']
    terms = list(R.get('terms_full', R['terms'])) if (usar_completo and R.get('model_full') is not None) else list(R['terms'])
    factores = R['factores']

    labels, tvals = [], []
    usar_t = True
    for t in terms:
        pn = _param(t, factores)
        if pn not in m.params:
            continue
        tv = float(abs(m.tvalues.get(pn, np.nan)))
        if not np.isfinite(tv):
            usar_t = False
            break

    for t in terms:
        pn = _param(t, factores)
        if pn not in m.params:
            continue
        labels.append(_nombre_termino_ccd(t))
        tvals.append(float(abs(m.tvalues[pn])) if usar_t else float(abs(m.params[pn])))

    if not labels:
        return None

    df_res = int(getattr(m, 'df_resid', 0))
    t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df_res)) if (usar_t and df_res >= 1) else np.nan

    order = np.argsort(tvals)
    fig, ax = plt.subplots(figsize=(10, max(4, 0.38 * len(labels) + 2)))
    if pd.notna(t_crit):
        colors = ['#d9534f' if tvals[i] >= t_crit else '#7fb3d5' for i in order]
    else:
        colors = ['#7fb3d5' for _ in order]

    ax.barh([labels[i] for i in order], [tvals[i] for i in order], color=colors, edgecolor='black')

    if pd.notna(t_crit):
        lab = f'|t| critico = {t_crit:.2f} (alpha={alpha})'
        ax.axvline(t_crit, color='red', linestyle='--', label=lab)
        ylo, yhi = ax.get_ylim()
        ax.set_ylim(ylo, yhi + 0.8)
        ax.text(
            t_crit, yhi + 0.10, f'  {lab}  ', color='red', fontsize=10, fontweight='bold',
            ha='center', va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='red', alpha=0.95)
        )
        ax.legend(loc='lower right')

    ax.set_xlabel('|t| estandarizado' if usar_t else '|Coeficiente|')
    tipo_txt = 'Modelo completo' if usar_completo else 'Modelo actual'
    ax.set_title(f"Diagrama de Pareto CCD - {R['respuesta']} ({tipo_txt})")
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    fn = f"pareto_ccd_{_safe_fname_ccd(R['respuesta'])}.png"
    plt.savefig(fn, bbox_inches='tight', dpi=150)
    plt.close()
    return fn


def graf_seminormal_ccd(R, usar_completo=True):
    """Grafico Semi-normal de Daniel (efectos estandarizados |t|) para el modelo CCD."""
    m = R.get('model_full') if (usar_completo and R.get('model_full') is not None) else R['model']
    terms = list(R.get('terms_full', R['terms'])) if (usar_completo and R.get('model_full') is not None) else list(R['terms'])
    factores = R['factores']
    activos = set(R.get('terms', []))

    labels, vals, es_activo = [], [], []
    usar_t = all(
        np.isfinite(abs(m.tvalues.get(_param(t, factores), np.nan)))
        for t in terms if _param(t, factores) in m.params
    )

    for t in terms:
        pn = _param(t, factores)
        if pn not in m.params:
            continue
        labels.append(_nombre_termino_ccd(t))
        vals.append(float(abs(m.tvalues[pn])) if usar_t else float(abs(m.params[pn])))
        p_val = m.pvalues.get(pn, np.nan)
        sig = (pd.notna(p_val) and p_val < 0.05) or (not usar_completo and t in activos)
        es_activo.append(bool(sig))

    if len(vals) < 2:
        return None

    ae = np.array(vals, dtype=float)
    cnt = len(ae)
    o = np.argsort(ae)
    a = ae[o]
    ns = [labels[i] for i in o]
    acts_ord = [es_activo[i] for i in o]

    pp = (np.arange(1, cnt + 1) - 0.5) / cnt
    q = stats.norm.ppf((1.0 + pp) / 2.0)

    s0 = 1.5 * np.median(a)
    sub = a[a <= 2.5 * s0] if s0 > 0 else a
    pse = 1.5 * np.median(sub) if len(sub) > 0 else s0
    df_lenth = max(1.0, cnt / 3.0)
    me = float(stats.t.ppf(0.975, df_lenth) * pse) if pse > 0 else np.nan

    fig, ax = plt.subplots(figsize=(8.5, 5))
    cols = ['#d9534f' if act else 'steelblue' for act in acts_ord]
    ax.scatter(q, a, c=cols, s=90, edgecolor='black', zorder=3)
    for qi, yi, ni in zip(q, a, ns):
        ax.annotate(ni, (qi, yi), xytext=(5, 3), textcoords='offset points', fontsize=8)

    if pd.notna(me) and me > 0:
        nm = a < me
        sl, ic = np.polyfit(q[nm] if nm.sum() >= 2 else q, a[nm] if nm.sum() >= 2 else a, 1)
        lx = np.array([q.min() - 0.1, q.max() + 0.1])
        ax.plot(lx, sl * lx + ic, 'k--', label='Referencia')
        ax.axhline(me, color='red', linestyle='--', label=f'ME (Lenth) = {me:.3f}')
        ax.set_title(f"Semi-normal (Daniel) - {R['respuesta']} | PSE={pse:.3f}")
    else:
        ax.set_title(f"Semi-normal (Daniel) - {R['respuesta']}")

    ax.set_xlabel('Cuantiles half-normal')
    ax.set_ylabel('|t| estandarizado' if usar_t else '|Coeficiente|')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fn = f"seminormal_ccd_{_safe_fname_ccd(R['respuesta'])}.png"
    plt.savefig(fn, bbox_inches='tight', dpi=150)
    plt.close()
    return fn


def _predecir_con_modelo_ccd(m, terms, coding, factores, punto):
    fila = {f + '_c': (punto[f] - coding[f]['c']) / coding[f]['half'] for f in factores}
    p = m.params
    pred = float(p['Intercept'])
    for t in terms:
        pn = _param(t, factores)
        if pn not in p:
            continue
        prod = 1.0
        for pp in t.split('*'):
            prod *= fila[pp + '_c']
        pred += float(p[pn]) * prod
    return pred


def graf_interacciones_ccd(R, unidades=None, fijos=None, eje_x_por_interaccion=None, usar_completo=False):
    """Graficos de interaccion 2 a 2 sobre la superficie ajustada CCD."""
    factores = R['factores']
    coding = R['coding']
    m = R.get('model_full') if (usar_completo and R.get('model_full') is not None) else R['model']
    terms = list(R.get('terms_full', R['terms'])) if (usar_completo and R.get('model_full') is not None) else list(R['terms'])

    sig = [t for t in terms if '*' in t and t.split('*')[0] != t.split('*')[1]]
    if not sig:
        return []

    eje_x_map = eje_x_por_interaccion or {}
    base = {f: (fijos[f] if (fijos and f in fijos) else coding[f]['c']) for f in factores}
    files = []
    chunk_size = 2

    for i in range(0, len(sig), chunk_size):
        chunk = sig[i:i + chunk_size]
        fig, axes = plt.subplots(1, len(chunk), figsize=(6 * len(chunk), 4.5))
        if len(chunk) == 1:
            axes = [axes]

        for ax, name in zip(axes, chunk):
            fa, fb = name.split('*')
            x_fac = eje_x_map.get(name, fa)
            leg_fac = fb if x_fac == fa else fa

            cx, cl = coding[x_fac], coding[leg_fac]
            xs = np.linspace(cx['lo'], cx['hi'], 50)
            xs_pts = [cx['lo'], cx['hi']]

            def _eval_linea(x_vals, leg_val):
                ys = []
                for xv in x_vals:
                    pt = dict(base)
                    pt[x_fac] = float(xv)
                    pt[leg_fac] = float(leg_val)
                    ys.append(_predecir_con_modelo_ccd(m, terms, coding, factores, pt))
                return ys

            y_low_curve = _eval_linea(xs, cl['lo'])
            y_high_curve = _eval_linea(xs, cl['hi'])
            y_low_pts = _eval_linea(xs_pts, cl['lo'])
            y_high_pts = _eval_linea(xs_pts, cl['hi'])

            u_x = (unidades or {}).get(x_fac, '').strip()
            u_leg = (unidades or {}).get(leg_fac, '').strip()
            lbl_x = f"{x_fac} ({u_x})" if u_x else x_fac
            suf_u_leg = f" {u_leg}" if u_leg else ""

            ax.plot(xs, y_low_curve, '-', color='#90CAF9', linewidth=2.2)
            ax.plot(xs_pts, y_low_pts, 'o', color='#1976D2', markersize=8,
                    label=f"{leg_fac} = {cl['lo']:.4g}{suf_u_leg}")

            ax.plot(xs, y_high_curve, '-', color='#EF9A9A', linewidth=2.2)
            ax.plot(xs_pts, y_high_pts, 's', color='#D32F2F', markersize=8,
                    label=f"{leg_fac} = {cl['hi']:.4g}{suf_u_leg}")

            ax.set_xticks([cx['lo'], cx['c'], cx['hi']])
            ax.set_xlabel(lbl_x)
            ax.set_ylabel(f"Prediccion {R['respuesta']}")
            ax.set_title(f"Interaccion {x_fac} * {leg_fac}")
            ax.legend(loc='best', framealpha=0.9)
            ax.grid(alpha=0.3)

        plt.tight_layout()
        fn = f"interacciones_ccd_{_safe_fname_ccd(R['respuesta'])}_{i // chunk_size + 1}.png"
        plt.savefig(fn, bbox_inches='tight', dpi=150)
        plt.close()
        files.append(fn)

    return files
