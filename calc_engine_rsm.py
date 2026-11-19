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
def ajustar_ccd(df_raw, factores, ycol, alpha_elim=0.05, eliminar=True):
    """Ajusta modelo cuadratico por OLS; si eliminar=True aplica backward elimination.
    Devuelve dict R compatible con la app (anova, R2, effects, eq, etc.)."""
    df, coding = codificar(df_raw, factores)

    terms_full = terminos_completos(factores)
    fml_full = _formula(ycol, factores, terms_full)
    m_full = smf.ols(fml_full, data=df).fit()

    if eliminar:
        terms, historial = _backward_elimination(df, ycol, factores, alpha_elim)
    else:
        terms, historial = list(terms_full), []

    fml = _formula(ycol, factores, terms)
    m = smf.ols(fml, data=df).fit()

    # ANOVA tipo II
    an = anova_lm(m, typ=3).reset_index().rename(
        columns={'index': 'Fuente', 'sum_sq': 'SS', 'df': 'GL', 'F': 'F', 'PR(>F)': 'p'})
    an = an[an['Fuente'] != 'Intercept'].copy()
    an['MS'] = an['SS'] / an['GL']
    an = an[['Fuente', 'SS', 'GL', 'MS', 'F', 'p']]
    an['Sig'] = ['OK' if (pd.notna(p) and p < 0.05) else ('-' if pd.isna(p) else 'NO') for p in an['p']]

    # Lack of Fit (solo si hay replicas en algun punto del diseno)
    grupos = df_raw.groupby(factores)[ycol]
    sizes = grupos.size()
    ss_pe = sum((n-1)*grupos.get_group(g).var(ddof=1)
                for g, n in sizes.items() if n >= 2 and not np.isnan(grupos.get_group(g).var(ddof=1)))
    df_pe = int((sizes[sizes >= 2] - 1).sum())
    lof = None
    sse = float((m.resid ** 2).sum())
    df_resid = int(m.df_resid)
    if df_pe >= 1 and df_resid > df_pe and sse > ss_pe:
        ss_lof = sse - ss_pe
        df_lof = df_resid - df_pe
        F_lof = (ss_lof/df_lof) / (ss_pe/df_pe)
        lof = {'SS_pe': ss_pe, 'df_pe': df_pe, 'SS_lof': ss_lof, 'df_lof': df_lof,
               'F': F_lof, 'p': stats.f.sf(F_lof, df_lof, df_pe)}

    # Efectos en unidades codificadas (para Pareto)
    effects = {}
    for t in terms:
        pn = _param(t, factores)
        effects[t] = {'coef': float(m.params.get(pn, np.nan)),
                      'p': float(m.pvalues.get(pn, np.nan)),
                      'efecto': float(2*m.params.get(pn, np.nan))}

    # Ecuaciones (codificada y NO codificada, estilo articulo)
    eq_coded = _eq_str(m, terms, factores, coding=None)
    eq_uncoded = _eq_str(m, terms, factores, coding=coding)

    R = dict(via=VIA_CCD, k=len(factores), factores=list(factores), respuesta=ycol,
             N=len(df), coding=coding, terms_full=terms_full, terms=terms,
             historial=historial, model=m, model_full=m_full,
             anova=an, R2=float(m.rsquared), R2_adj=float(m.rsquared_adj),
             S=float(np.sqrt(m.mse_resid)),   # <--- ESTA ES LA LÍNEA NUEVA
             lof=lof, effects=effects, y_bar=float(df[ycol].mean()),
             eq_coded=eq_coded, eq_uncoded=eq_uncoded,
             fitted=np.asarray(m.fittedvalues), resid=np.asarray(m.resid),
             tiene_bloques=False, tiene_centro=True)

    return R

# ------------------------------------------------------------------
# 6. ECUACIONES
# ------------------------------------------------------------------
def _eq_str(m, terms, factores, coding=None):
    """Ecuacion legible. Si coding=None, en unidades codificadas;
    si no, expandida a unidades reales (x_i = (X_i - c_i)/h_i) con numpy puro."""
    idx = {f: i for i, f in enumerate(factores)}
    n = len(factores)
    b = {}  # (i,) lineal, (i, i) cuadrado, (i, j) i<j cruzado; clave tupla ordenada
    b0 = float(m.params['Intercept'])
    for t in terms:
        pn = _param(t, factores)
        coef = float(m.params[pn])
        p = t.split('*')
        if len(p) == 1:
            b[(idx[p[0]],)] = b.get((idx[p[0]],), 0.0) + coef
        else:
            i, j = sorted((idx[p[0]], idx[p[1]]))
            b[(i, j)] = b.get((i, j), 0.0) + coef

    if coding is None:  # codificada
        s = f"Y = {b0:.4f} "
        for f, i in idx.items():
            if (i,) in b: s += f"{b[(i,)]:+.4f}*{f} "
            if (i, i) in b: s += f"{b[(i,i)]:+.4f}*{f}² "
        fs = list(factores)
        for x in range(n):
            for y in range(x + 1, n):
                if (x, y) in b:
                    s += f"{b[(x,y)]:+.4f}*{fs[x]}*{fs[y]} "
        return s.strip()

    # Expandida a unidades reales: x_i = (X_i - c_i)/h_i
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
    s = f"Y = {B0:.4f} "
    for f, i in idx.items():
        if abs(lin[i]) > 1e-12: s += f"{lin[i]:+.4f}*{f} "
        if abs(cuad[i]) > 1e-12: s += f"{cuad[i]:+.4f}*{f}² "
    fs = list(factores)
    for (i, j), v in sorted(cruz.items()):
        s += f"{v:+.4f}*{fs[i]}*{fs[j]} "
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
