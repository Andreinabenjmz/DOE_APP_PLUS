# calc_engine_plus.py
# Motor de cálculo y gráficos — sin dependencias de Streamlit ni Colab

import pandas as pd
import numpy as np
import os
import io
import itertools
import warnings
from datetime import date
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from scipy.interpolate import griddata

warnings.filterwarnings('default')

VIA_1, VIA_2, VIA_3 = 'via1', 'via2', 'via3'
VIA_INFO = {
    VIA_1: {'nombre': '2^k con replicas puras', 'emoji': 'A'},
    VIA_2: {'nombre': '2^k con bloques + centro', 'emoji': 'B'},
    VIA_3: {'nombre': '2^k no replicado (Lenth)', 'emoji': 'C'},
}


# ---------- Lenth ----------
def calcular_lenth(effects, df_resid=None):
    vals = np.array([e['efecto'] for e in effects.values()])
    abs_eff = np.abs(vals)
    m = len(abs_eff)

    if m == 0:
        return {
            'm': 0, 'PSE': np.nan, 'ME': np.nan, 'SME': np.nan, 'df_t': 0,
            'valido': False, 'mensaje': 'No se encontraron efectos para analizar.'
        }

    s0 = 1.5 * np.median(abs_eff)
    sub = abs_eff[abs_eff < 2.5 * s0]
    s1 = 1.5 * np.median(sub) if len(sub) > 0 else s0

    PSE = s1 if s1 > 0 else np.nan
    df_t = df_resid if (df_resid is not None and df_resid >= 1) else m

    if not np.isfinite(PSE) or PSE <= 0:
        return {
            'm': m,
            'PSE': np.nan,
            'ME': np.nan,
            'SME': np.nan,
            'df_t': m / 3,
            'valido': False,
            'mensaje': 'No se pudo estimar un PSE positivo.'
        }

    ME = stats.t.ppf(0.975, df_t) * PSE
    gamma = 0.5 * (1 + 0.95**(1.0 / m))
    SME = stats.t.ppf(gamma, df_t) * PSE

    return {
        'm': m,
        'PSE': PSE,
        'ME': ME,
        'SME': SME,
        'df_t': df_t,
        'valido': True,
        'mensaje': 'Cálculo exitoso.'
    }


# ---------- Preparacion ---------------------
def preparar_datos(raw, factores, respuestas):
    tmp = raw.copy()
    cg = '__G__'
    cy = '__Y__'
    while cg in tmp.columns:
        cg += '_'
    while cy in tmp.columns:
        cy += '_'

    id_vars = factores.copy()
    if 'Corrida_ID' in tmp.columns:
        id_vars.append('Corrida_ID')

    df = pd.melt(tmp, id_vars=id_vars, value_vars=respuestas,
                 var_name=cg, value_name=cy).rename(columns={cg: 'Grupo', cy: 'Y'})
    df['Curvatura'] = df[factores].apply(
        lambda r: 1 if all(v == 0 for v in r) else 0, axis=1)
    for f in factores:
        df[f] = pd.to_numeric(df[f], errors='raise')
    return df


def diagnosticar(raw, factores, respuestas):
    k = len(factores)
    n = len(raw)
    n_r = len(respuestas)
    if k < 2:
        return None, "Se requieren 2 o mas factores."
    for f in factores:
        vals = set(pd.to_numeric(raw[f], errors='coerce').dropna().unique().tolist())
        if not vals.issubset({-1, 0, 1}) or not {-1, 1}.issubset(vals):
            return None, f"Factor {f} mal codificado: {vals}."
    centros = int(raw[factores].eq(0).all(axis=1).sum())

    if centros > 0 and n_r >= 2:
        return VIA_2, f"{n} filas / {centros} centro(s) / {n_r} columnas."
    if centros == 0 and n_r >= 2:
        return VIA_1, f"{n} esquinas / {n_r} replicas."
    if centros == 0 and n_r == 1:
        return VIA_3, f"{n} esquinas / sin replicas."
    return None, f"No clasificable (n={n}, k={k}, centros={centros}, resp={n_r})."


# ---------- Calculos base ----------
def _calc_effects(df, factores, Nf_total=None):
    """Calcula efectos sobre el conjunto completo de esquinas (Nf_total = len(esq))."""
    k = len(factores)
    esq = df[df['Curvatura'] == 0].copy()
    if Nf_total is None or Nf_total <= 0:
        Nf_total = len(esq)
    # Nota: si alguien pasa Nf por bloque, garantizamos usar len(esq) cuando esq tiene todas las filas
    Nf_eff = len(esq) if len(esq) > 0 else Nf_total
    out = {}
    for mask in range(1, 2**k):
        inv = [factores[i] for i in range(k) if mask & (1 << i)]
        s = esq[inv].prod(axis=1)
        c = (esq['Y'] * s).sum()
        out['*'.join(inv)] = {
            'contraste': c,
            'SS': (c ** 2) / Nf_eff,
            'efecto': c / (Nf_eff / 2.0),
            'coef': c / Nf_eff,
            'inv': inv
        }
    return out


def _calc_main_effects(df, factores):
    esq = df[df['Curvatura'] == 0]
    return {
        f: {
            'low': esq[esq[f] == -1]['Y'].mean(),
            'high': esq[esq[f] == 1]['Y'].mean(),
            'eff': esq[esq[f] == 1]['Y'].mean() - esq[esq[f] == -1]['Y'].mean()
        }
        for f in factores
    }


def _calc_interactions(df, factores):
    esq = df[df['Curvatura'] == 0]
    k = len(factores)
    out = {}
    for i in range(k):
        for j in range(i + 1, k):
            f1, f2 = factores[i], factores[j]
            out[(f1, f2)] = {
                'mll': esq[(esq[f1] == -1) & (esq[f2] == -1)]['Y'].mean(),
                'mhl': esq[(esq[f1] == 1) & (esq[f2] == -1)]['Y'].mean(),
                'mlh': esq[(esq[f1] == -1) & (esq[f2] == 1)]['Y'].mean(),
                'mhh': esq[(esq[f1] == 1) & (esq[f2] == 1)]['Y'].mean()
            }
    return out


def _calcular_r2_pred(mfit, ss_total):
    """Calcula R-sq(pred) estilo Minitab mediante estadístico PRESS."""
    try:
        infl = mfit.get_influence()
        h = infl.hat_matrix_diag
        e = mfit.resid.values
        # Evitar división por cero en puntos con leverage == 1
        valid = (1.0 - h) > 1e-10
        if not np.all(valid):
            return np.nan
        press = np.sum((e / (1.0 - h)) ** 2)
        return 1.0 - (press / ss_total) if ss_total > 0 else np.nan
    except Exception:
        return np.nan


# ---------- Via 1 con soporte para terminos ----------
def calcular_via1_con_terminos(df, factores, terms=None):
    k = len(factores)
    n_r = df['Grupo'].nunique()
    N = len(df)
    Nf_total = len(df[df['Curvatura'] == 0])
    Nf = Nf_total // n_r if n_r > 0 else Nf_total
    G = df['Y'].sum()
    y = G / N
    SS_t = (df['Y'] ** 2).sum() - (G ** 2) / N

    ef_all = _calc_effects(df, factores, Nf_total)
    if terms is None:
        terms = [t for t in ef_all.keys() if len(t.split('*')) <= 2]
    ef = {t: ef_all[t].copy() for t in terms if t in ef_all}
    if len(ef) == 0:
        raise ValueError("Debes incluir al menos un termino en el modelo.")

    SS_m = sum(e['SS'] for e in ef.values())
    SS_r = SS_t - SS_m
    df_r = N - 1 - len(ef)
    if df_r < 1:
        raise ValueError(f"df_resid={df_r}<1. Demasiados terminos.")
    MS_r = SS_r / df_r

    for e in ef.values():
        e['MS'] = e['SS']
        e['F'] = e['MS'] / MS_r
        e['p'] = stats.f.sf(e['F'], 1, df_r)

    rows = [(n, e['SS'], 1, e['MS'], e['F'], e['p']) for n, e in ef.items()]
    rows.append(('Residual', SS_r, df_r, MS_r, np.nan, np.nan))
    anova = pd.DataFrame(rows, columns=['Fuente', 'SS', 'GL', 'MS', 'F', 'p'])
    anova['F_crit'] = [
        stats.f.ppf(0.95, gl, df_r) if pd.notna(gl) and gl > 0 else np.nan
        for gl in anova['GL']
    ]
    anova['Sig'] = [
        'OK' if (pd.notna(p) and p < 0.05) else ('-' if pd.isna(p) else 'NO')
        for p in anova['p']
    ]

    R2 = 1 - SS_r / SS_t
    R2a = 1 - (SS_r / df_r) / (SS_t / (N - 1))

    parts = [_term_to_formula(t) for t in terms]
    formula = 'Q("Y") ~ ' + ' + '.join(parts)
    try:
        mfit = smf.ols(formula, data=df).fit()
        R2p = _calcular_r2_pred(mfit, SS_t)
    except Exception:
        R2p = np.nan

    R = dict(
        via=VIA_1, k=k, factores=factores, N=N, Nf=Nf, Nf_total=Nf_total,
        n_replicas=n_r, n_bloques=0, tiene_bloques=False, tiene_centro=False,
        y_bar=y, SS_total=SS_t, SS_resid=SS_r, SS_model=SS_m,
        effects=ef, all_effects=ef_all, terms_included=list(terms),
        anova=anova, MS_resid=MS_r, df_resid=df_r,
        R2=R2, R2_adj=R2a, R2_pred=R2p,
        main_effects=_calc_main_effects(df, factores),
        interactions=_calc_interactions(df, factores),
        t_crit=stats.t.ppf(0.975, df_r)
    )
    R['lenth'] = calcular_lenth(ef, df_resid=df_r)
    return R


# ---------- Funciones de seleccion de modelo (estilo Minitab) ----------
def _term_to_formula(term):
    """Convierte 'A*B' a 'Q("A"):Q("B")' para statsmodels."""
    if '*' in term:
        return ':'.join(f'Q("{f}")' for f in term.split('*'))
    return f'Q("{term}")'


def _evaluar_interacciones_individualmente(df, factores, alpha=0.05):
    """
    Evalua cada interaccion doble individualmente contra el modelo base de solo principales
    usando OLS con Curvatura incluida.
    """
    n_b = df['Grupo'].nunique() if 'Grupo' in df.columns else 1
    nc = int((df['Curvatura'] == 1).sum())
    base_parts = [_term_to_formula(f) for f in factores]
    if n_b > 1:
        base_parts.append('Q("Grupo")')
    if nc > 0:
        base_parts.append('Q("Curvatura")')

    ints = [f"{factores[i]}*{factores[j]}"
            for i in range(len(factores))
            for j in range(i + 1, len(factores))]

    resultados = {}
    for term in ints:
        formula = 'Q("Y") ~ ' + ' + '.join(base_parts + [_term_to_formula(term)])
        try:
            m = smf.ols(formula, data=df).fit()
            aov = sm.stats.anova_lm(m, typ=3)
            key = _term_to_formula(term)
            ss_int = float(aov.loc[key, 'sum_sq'])
            df_int = float(aov.loc[key, 'df'])
            F_val = float(aov.loc[key, 'F'])
            p_val = float(aov.loc[key, 'PR(>F)'])
            resultados[term] = {'SS': ss_int, 'F': F_val, 'p': p_val, 'df_err': int(m.df_resid)}
        except Exception:
            resultados[term] = {'SS': 0.0, 'F': 0.0, 'p': 1.0, 'df_err': 0}
    return resultados


def seleccionar_modelo_minitab(df, factores, alpha=0.05):
    """
    Seleccion Stepwise Forward al estilo Minitab:
    Parte con efectos principales (+ Curvatura/Grupo) y agrega interacciones dobles con p < alpha.
    """
    n_b = df['Grupo'].nunique() if 'Grupo' in df.columns else 1
    nc = int((df['Curvatura'] == 1).sum())

    ints = [f"{factores[i]}*{factores[j]}"
            for i in range(len(factores))
            for j in range(i + 1, len(factores))]

    terms = list(factores)
    remaining = list(ints)

    while remaining:
        best_p = 1.0
        best_term = None

        extra_parts = []
        if n_b > 1:
            extra_parts.append('Q("Grupo")')
        if nc > 0:
            extra_parts.append('Q("Curvatura")')

        base_parts = [_term_to_formula(t) for t in terms] + extra_parts
        base_formula = 'Q("Y") ~ ' + ' + '.join(base_parts)

        for term in remaining:
            key = _term_to_formula(term)
            formula = base_formula + f' + {key}'
            try:
                m = smf.ols(formula, data=df).fit()
                if m.df_resid < 1:
                    continue
                aov = sm.stats.anova_lm(m, typ=3)
                if key in aov.index:
                    p_val = float(aov.loc[key, 'PR(>F)'])
                    if pd.notna(p_val) and p_val < best_p:
                        best_p = p_val
                        best_term = term
            except Exception:
                pass

        if best_p < alpha and best_term is not None:
            terms.append(best_term)
            remaining.remove(best_term)
        else:
            break

    return terms


def seleccionar_modelo_backward(df, factores, alpha=0.05, jerarquico=True):
    """
    Eliminacion hacia atras (Backward Elimination) estilo Minitab:
    Parte del modelo completo hasta orden 2 (+ Curvatura/Grupo) y elimina
    iterativamente el termino menos significativo si p > alpha.
    """
    n_b = df['Grupo'].nunique() if 'Grupo' in df.columns else 1
    nc = int((df['Curvatura'] == 1).sum())

    # Verificar que no haya colinealidad exacta por alias (ej. 2^(4-1) o 2^(5-2))
    # Filtramos interacciones cuyas columnas ya sean colineales con terminos previos
    esq = df[df['Curvatura'] == 0]
    candidatos = list(factores) + [
        f"{factores[i]}*{factores[j]}"
        for i in range(len(factores))
        for j in range(i + 1, len(factores))
    ]

    # Evitar columnas duplicadas (alias) si el diseno es fraccionado Res III/IV
    terms = []
    vistos = set()
    for t in candidatos:
        cols = t.split('*')
        patron = tuple(esq[cols].prod(axis=1).tolist())
        patron_neg = tuple((-esq[cols].prod(axis=1)).tolist())
        if patron not in vistos and patron_neg not in vistos:
            vistos.add(patron)
            terms.append(t)

    extra_parts = []
    if n_b > 1:
        extra_parts.append('Q("Grupo")')
    if nc > 0:
        extra_parts.append('Q("Curvatura")')

    while len(terms) > 1:
        parts = [_term_to_formula(t) for t in terms] + extra_parts
        formula = 'Q("Y") ~ ' + ' + '.join(parts)
        try:
            m = smf.ols(formula, data=df).fit()
            if m.df_resid < 1:
                # Si estamos saturados, quitar la ultima interaccion
                ints_act = [t for t in terms if '*' in t]
                if ints_act:
                    terms.remove(ints_act[-1])
                    continue
                break
            aov = sm.stats.anova_lm(m, typ=3)
        except Exception:
            break

        if jerarquico:
            elegibles = [
                t for t in terms
                if ('*' in t) or not any(t in x.split('*') for x in terms if '*' in x)
            ]
        else:
            elegibles = list(terms)

        worst_p = -1.0
        worst_term = None
        for t in elegibles:
            key = _term_to_formula(t)
            if key in aov.index:
                p_val = float(aov.loc[key, 'PR(>F)'])
                if pd.notna(p_val) and p_val > worst_p:
                    worst_p = p_val
                    worst_term = t

        if worst_term is not None and worst_p > alpha:
            terms.remove(worst_term)
        else:
            break

    return terms


def calcular_via2_con_terminos(df, factores, terms=None, alpha=0.05, modo='backward'):
    """
    Calcula ANOVA estilo Minitab (Adj SS Tipo III sobre matriz codificada)
    para disenos 2^k / 2^(k-p) con puntos centrales.

    terms:
        None  -> seleccion automatica segun `modo` ('backward' o 'forward')
        'all' -> incluye todos los terminos no colineales de orden <= 2
        lista -> terminos personalizados
    """
    k = len(factores)
    n_b = df['Grupo'].nunique() if 'Grupo' in df.columns else 1
    N = len(df)
    esq = df[df['Curvatura'] == 0]
    cen = df[df['Curvatura'] == 1]

    Nf_total = len(esq)
    Nf = Nf_total // n_b if n_b > 0 else Nf_total
    Nc = len(cen)

    G = df['Y'].sum()
    y = G / N
    SS_t = (df['Y'] ** 2).sum() - (G ** 2) / N

    bt = df.groupby('Grupo')['Y'].sum()
    sz = df.groupby('Grupo').size()
    SS_b = (bt ** 2 / sz).sum() - (G ** 2) / N
    df_b = n_b - 1

    ef_all = _calc_effects(df, factores, Nf_total)

    if terms is None:
        if modo == 'forward':
            terms = seleccionar_modelo_minitab(df, factores, alpha)
        else:
            terms = seleccionar_modelo_backward(df, factores, alpha, jerarquico=True)
    elif terms == 'all':
        # Evitar columnas aliadas exactas
        terms = []
        vistos = set()
        for t in [t for t in ef_all.keys() if len(t.split('*')) <= 2]:
            cols = t.split('*')
            patron = tuple(esq[cols].prod(axis=1).tolist())
            patron_neg = tuple((-esq[cols].prod(axis=1)).tolist())
            if patron not in vistos and patron_neg not in vistos:
                vistos.add(patron)
                terms.append(t)

    ef = {t: ef_all[t].copy() for t in terms if t in ef_all}
    yf = esq['Y'].mean()
    yc = cen['Y'].mean() if Nc > 0 else yf
    SS_c = (Nf_total * Nc / (Nf_total + Nc)) * ((yf - yc) ** 2) if (Nc > 0 and Nf_total > 0) else 0.0

    # Ajustar modelo OLS en unidades codificadas (igual que Minitab)
    parts = [_term_to_formula(t) for t in terms]
    if df_b > 0:
        parts = ['Q("Grupo")'] + parts
    if Nc > 0:
        parts.append('Q("Curvatura")')
    formula = 'Q("Y") ~ ' + ' + '.join(parts) if parts else 'Q("Y") ~ 1'

    mfit = smf.ols(formula, data=df).fit()
    aov_sm = sm.stats.anova_lm(mfit, typ=3)

    ss_r = float(mfit.ssr)
    df_r = int(mfit.df_resid)
    if df_r < 1:
        raise ValueError(f"df_resid={df_r}<1. Reduce los terminos del modelo.")
    MS_r = ss_r / df_r

    # Pure Error de puntos centrales (y replicas si existieran en esquinas)
    if Nc > 1:
        y_cen = cen['Y'].values
        ss_pe = float(((y_cen - y_cen.mean()) ** 2).sum())
        df_pe = Nc - 1
    else:
        ss_pe = 0.0
        df_pe = 0

    ss_lof = max(0.0, ss_r - ss_pe)
    df_lof = df_r - df_pe

    # Extraer Adj SS (Tipo III) por bloque, principales, interacciones 2-way y curvatura
    ss_b_adj = float(aov_sm.loc['Q("Grupo")', 'sum_sq']) if (df_b > 0 and 'Q("Grupo")' in aov_sm.index) else 0.0
    df_b_adj = int(aov_sm.loc['Q("Grupo")', 'df']) if (df_b > 0 and 'Q("Grupo")' in aov_sm.index) else 0

    lin_terms = [f for f in factores if f in terms and _term_to_formula(f) in aov_sm.index]
    int_terms = [t for t in terms if '*' in t and len(t.split('*')) == 2 and _term_to_formula(t) in aov_sm.index]

    ss_lin = sum(float(aov_sm.loc[_term_to_formula(f), 'sum_sq']) for f in lin_terms)
    df_lin = sum(int(aov_sm.loc[_term_to_formula(f), 'df']) for f in lin_terms)

    ss_2w = sum(float(aov_sm.loc[_term_to_formula(t), 'sum_sq']) for t in int_terms)
    df_2w = sum(int(aov_sm.loc[_term_to_formula(t), 'df']) for t in int_terms)

    ss_c_adj = float(aov_sm.loc['Q("Curvatura")', 'sum_sq']) if (Nc > 0 and 'Q("Curvatura")' in aov_sm.index) else SS_c
    df_c_adj = int(aov_sm.loc['Q("Curvatura")', 'df']) if (Nc > 0 and 'Q("Curvatura")' in aov_sm.index) else (1 if Nc > 0 else 0)

    rows = []

    # Model
    ss_model = SS_t - ss_r
    df_model = (N - 1) - df_r
    if df_model > 0:
        ms_model = ss_model / df_model
        F_model = ms_model / MS_r
        p_model = stats.f.sf(F_model, df_model, df_r)
        rows.append(('Model', ss_model, df_model, ms_model, F_model, p_model))

    # Bloques
    MS_b = F_b = p_b = np.nan
    if df_b_adj > 0:
        MS_b = ss_b_adj / df_b_adj
        F_b = MS_b / MS_r
        p_b = stats.f.sf(F_b, df_b_adj, df_r)
        rows.append(('Bloques', ss_b_adj, df_b_adj, MS_b, F_b, p_b))

    # Linear
    if df_lin > 0:
        ms_lin = ss_lin / df_lin
        F_lin = ms_lin / MS_r
        p_lin = stats.f.sf(F_lin, df_lin, df_r)
        rows.append(('Linear', ss_lin, df_lin, ms_lin, F_lin, p_lin))

    for f in lin_terms:
        key = _term_to_formula(f)
        ss = float(aov_sm.loc[key, 'sum_sq'])
        df_t = int(aov_sm.loc[key, 'df'])
        ms_t = ss / df_t
        F_t = ms_t / MS_r
        p_t = stats.f.sf(F_t, df_t, df_r)
        rows.append((f, ss, df_t, ms_t, F_t, p_t))

    # 2-Way Interactions
    if df_2w > 0:
        ms_2w = ss_2w / df_2w
        F_2w = ms_2w / MS_r
        p_2w = stats.f.sf(F_2w, df_2w, df_r)
        rows.append(('2-Way Interactions', ss_2w, df_2w, ms_2w, F_2w, p_2w))

    for t in int_terms:
        key = _term_to_formula(t)
        ss = float(aov_sm.loc[key, 'sum_sq'])
        df_t = int(aov_sm.loc[key, 'df'])
        ms_t = ss / df_t
        F_t = ms_t / MS_r
        p_t = stats.f.sf(F_t, df_t, df_r)
        rows.append((t, ss, df_t, ms_t, F_t, p_t))

    # Curvatura
    F_c = p_c = np.nan
    if df_c_adj > 0:
        ms_c = ss_c_adj / df_c_adj
        F_c = ms_c / MS_r
        p_c = stats.f.sf(F_c, df_c_adj, df_r)
        rows.append(('Curvatura', ss_c_adj, df_c_adj, ms_c, F_c, p_c))

    # Error / Lack-of-Fit / Pure Error
    rows.append(('Error', ss_r, df_r, MS_r, np.nan, np.nan))
    if df_lof > 0 and df_pe > 0:
        ms_lof = ss_lof / df_lof
        ms_pe = ss_pe / df_pe
        F_lof = ms_lof / ms_pe if ms_pe > 0 else np.nan
        p_lof = stats.f.sf(F_lof, df_lof, df_pe) if (pd.notna(F_lof) and ms_pe > 0) else np.nan
        rows.append(('  Lack-of-Fit', ss_lof, df_lof, ms_lof, F_lof, p_lof))
        rows.append(('  Pure Error', ss_pe, df_pe, ms_pe, np.nan, np.nan))
    elif df_pe > 0:
        rows.append(('  Pure Error', ss_pe, df_pe, ss_pe / df_pe, np.nan, np.nan))

    rows.append(('Total', SS_t, N - 1, np.nan, np.nan, np.nan))

    anova = pd.DataFrame(rows, columns=['Fuente', 'SS', 'GL', 'MS', 'F', 'p'])
    anova['F_crit'] = [
        stats.f.ppf(0.95, gl, df_r) if pd.notna(gl) and gl > 0 and fuente != 'Total' else np.nan
        for fuente, gl in zip(anova['Fuente'], anova['GL'])
    ]
    anova['Sig'] = [
        'OK' if (pd.notna(p) and p < 0.05) else ('-' if pd.isna(p) else 'NO')
        for p in anova['p']
    ]

    # Actualizar coeficientes y estadísticos de cada efecto con el ajuste OLS real
    for t in list(ef.keys()):
        key = _term_to_formula(t)
        if key in mfit.params.index:
            ef[t]['coef'] = float(mfit.params[key])
            ef[t]['efecto'] = 2.0 * float(mfit.params[key])
        for r in rows:
            if r[0] == t:
                ef[t]['SS'] = r[1]
                ef[t]['MS'] = r[3]
                ef[t]['F'] = r[4]
                ef[t]['p'] = r[5]
                break

    # Intercepto y bloques
    R_coef_bloques = {}
    intercepto_ols = float(mfit.params.get('Intercept', yf))
    if df_b > 0:
        medias_bloques = bt / sz
        R_intercepto_bloques = intercepto_ols
        for k_param, val_param in mfit.params.items():
            if k_param.startswith('Q("Grupo")'):
                nombre_b = k_param.split('[T.')[-1].rstrip(']')
                R_coef_bloques[nombre_b] = float(val_param)
    else:
        R_intercepto_bloques = intercepto_ols

    R2 = float(mfit.rsquared)
    R2a = float(mfit.rsquared_adj)
    R2p = _calcular_r2_pred(mfit, SS_t)

    R = dict(
        via=VIA_2, k=k, factores=factores, N=N, Nf=Nf, Nf_total=Nf_total, Nc=Nc,
        n_bloques=n_b, n_replicas=n_b, tiene_bloques=(df_b > 0), tiene_centro=(Nc > 0),
        y_bar=y, intercepto_ols=intercepto_ols,
        SS_total=SS_t, SS_blocks=ss_b_adj, SS_curv=ss_c_adj,
        intercepto_bloques=R_intercepto_bloques, coef_bloques=R_coef_bloques,
        SS_resid=ss_r, SS_model=ss_model,
        effects=ef, all_effects=ef_all,
        terms_included=list(terms), anova=anova,
        MS_resid=MS_r, MS_curv=ss_c_adj if Nc > 0 else np.nan,
        F_curv=F_c if Nc > 0 else np.nan, p_curv=p_c if Nc > 0 else np.nan,
        df_resid=df_r, yf_bar=yf, yc_bar=yc,
        R2=R2, R2_adj=R2a, R2_pred=R2p,
        main_effects=_calc_main_effects(df, factores),
        interactions=_calc_interactions(df, factores),
        t_crit=stats.t.ppf(0.975, df_r)
    )
    if df_b > 0:
        R['MS_blocks'] = MS_b
        R['F_blocks'] = F_b
        R['p_blocks'] = p_b
        R['df_blocks'] = df_b_adj
    R['lenth'] = calcular_lenth(ef, df_resid=df_r)
    return R


def calcular_via3(df, factores):
    k = len(factores)
    N = len(df)
    G = df['Y'].sum()
    y = G / N
    SS_t = (df['Y'] ** 2).sum() - (G ** 2) / N
    ef = _calc_effects(df, factores, Nf_total=N)
    L = calcular_lenth(ef, df_resid=0)
    if not L['valido']:
        raise ValueError(
            "No se puede determinar qué efectos están activos. "
            + L['mensaje']
        )

    for n_, e in ef.items():
        e['activo'] = abs(e['efecto']) > L['ME']
    act = [n_ for n_, e in ef.items() if e['activo']]
    ina = [n_ for n_, e in ef.items() if not e['activo']]
    SS_p = sum(ef[n_]['SS'] for n_ in ina)
    df_p = len(ina)
    MS_p = SS_p / df_p if df_p > 0 else np.nan
    for n_, e in ef.items():
        e['MS'] = e['SS']
        if e['activo'] and df_p >= 1 and MS_p > 0:
            e['F'] = e['MS'] / MS_p
            e['p'] = stats.f.sf(e['F'], 1, df_p)
        else:
            e['F'] = np.nan
            e['p'] = np.nan

    rows = [(n_, e['SS'], 1, e['MS'], e['F'], e['p']) for n_, e in ef.items() if e['activo']]
    if df_p > 0:
        rows.append(('Error (pooled)', SS_p, df_p, MS_p, np.nan, np.nan))
    anova = pd.DataFrame(rows, columns=['Fuente', 'SS', 'GL', 'MS', 'F', 'p'])
    if df_p > 0:
        anova['F_crit'] = [
            stats.f.ppf(0.95, gl, df_p) if pd.notna(gl) and gl > 0 else np.nan
            for gl in anova['GL']
        ]
    else:
        anova['F_crit'] = np.nan
    anova['Sig'] = [
        'OK' if (pd.notna(p) and p < 0.05) else ('EXP' if pd.isna(p) and n_ != 'Error (pooled)' else '-')
        for p, n_ in zip(anova['p'], anova['Fuente'])
    ]
    SS_m = sum(ef[n_]['SS'] for n_ in act)
    R2 = SS_m / SS_t if SS_t > 0 else 0
    R = dict(
        via=VIA_3, k=k, factores=factores, N=N, Nf=N, Nf_total=N,
        n_bloques=0, n_replicas=1, tiene_bloques=False, tiene_centro=False,
        y_bar=y, SS_total=SS_t, SS_resid=SS_p if df_p > 0 else np.nan, SS_model=SS_m,
        effects=ef, all_effects=ef, terms_included=act,
        anova=anova, activos=act, inactivos=ina,
        SS_pooled=SS_p, df_pooled=df_p, MS_pooled=MS_p,
        MS_resid=MS_p if df_p > 0 else 0.0, df_resid=df_p,
        R2=R2, R2_adj=None, R2_pred=None,
        main_effects=_calc_main_effects(df, factores),
        interactions=_calc_interactions(df, factores),
        t_crit=stats.t.ppf(0.975, max(1, L['m']))
    )
    R['lenth'] = L
    return R


# ---------- Graficos ----------
def _gen_pareto(R):
    terms = list(R['effects'].keys())

    tvals = []
    for n in R['effects']:
        F_val = R['effects'][n].get('F')
        if pd.notna(F_val) and F_val > 0:
            t = F_val ** 0.5
        else:
            t = abs(R['effects'][n]['efecto']) / max(R['lenth']['PSE'], 1e-9)
        if not np.isfinite(t):
            t = 50
        tvals.append(t)

    if R.get('tiene_centro') and pd.notna(R.get('F_curv')):
        terms.append('Curvatura')
        tvals.append(abs(R['F_curv'] ** 0.5) if np.isfinite(R['F_curv']) else 50)

    order = np.argsort(tvals)
    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(terms) + 2)))
    if R['via'] == VIA_3:
        umbral_visual = R['lenth']['ME'] / max(R['lenth']['PSE'], 1e-9)
    else:
        umbral_visual = R['t_crit']

    colors = ['#d9534f' if t >= umbral_visual else '#7fb3d5' for t in np.array(tvals)[order]]
    ax.barh([terms[i] for i in order], [tvals[i] for i in order], color=colors, edgecolor='black')

    if R['via'] == VIA_3:
        umb = R['lenth']['ME'] / max(R['lenth']['PSE'], 1e-9)
        lab = f'ME/PSE = {umb:.2f} (Lenth)'
    else:
        umb = R['t_crit']
        lab = f'|t| critico = {umb:.2f}'
    ax.axvline(umb, color='red', linestyle='--', label=lab)
    ylo, yhi = ax.get_ylim()
    ax.set_ylim(ylo, yhi + 0.8)
    ax.text(umb, yhi + 0.10, f'  {lab}  ', color='red', fontsize=10, fontweight='bold',
            ha='center', va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='red', alpha=0.95))
    ax.set_xlabel('|t| / |efecto| estandarizado')
    ax.set_title('Diagrama de Pareto' + (' (exploratorio)' if R['via'] == VIA_3 else ''))
    ax.legend(loc='lower right')
    plt.tight_layout()
    fn = 'pareto.png'
    plt.savefig(fn, bbox_inches='tight')
    plt.close()
    return fn


def _gen_seminormal(R):
    L = R['lenth']
    names = list(R['effects'].keys())
    ae = np.abs([R['effects'][n]['efecto'] for n in names])
    m = len(ae)
    o = np.argsort(ae)
    a = ae[o]
    ns = [names[i] for i in o]
    pp = (np.arange(1, m + 1) - 0.5) / m
    q = stats.norm.ppf((1 + pp) / 2)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.scatter(q, a, color='steelblue', s=90, edgecolor='black', zorder=3)
    for qi, yi, ni in zip(q, a, ns):
        ax.annotate(ni, (qi, yi), xytext=(5, 3), textcoords='offset points', fontsize=8)
    if L.get('valido') and pd.notna(L.get('ME')):
        nm = a < L['ME']
        sl, ic = np.polyfit(q[nm] if nm.sum() >= 2 else q, a[nm] if nm.sum() >= 2 else a, 1)
        lx = np.array([q.min() - 0.1, q.max() + 0.1])
        ax.plot(lx, sl * lx + ic, 'k--', label='Referencia')
        ax.axhline(L['ME'], color='red', linestyle='--', label=f'ME={L["ME"]:.3f}')
        ax.set_title(f'Semi-normal (Daniel) PSE={L["PSE"]:.3f}')
    else:
        ax.set_title('Semi-normal (Daniel)')
    ax.set_xlabel('Cuantiles half-normal')
    ax.set_ylabel('|Efecto|')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fn = 'seminormal.png'
    plt.savefig(fn, bbox_inches='tight')
    plt.close()
    return fn

def _gen_main_effects(R, spec=None, mapeos=None, unidades=None):
    nf = len(R['factores'])
    fig, axes = plt.subplots(1, nf, figsize=(4 * nf, 4), sharey=True)
    if nf == 1:
        axes = [axes]

    for i, (ax, f) in enumerate(zip(axes, R['factores'])):
        me = R['main_effects'][f]
        unidad = (unidades or {}).get(f, '').strip()
        etiqueta_f = f"{f} ({unidad})" if unidad else f

        if mapeos and f in mapeos and -1 in mapeos[f] and 1 in mapeos[f]:
            v_low = mapeos[f][-1]
            v_high = mapeos[f][1]
            ax.plot([v_low, v_high], [me['low'], me['high']], 'o-', markersize=10, color='#2196F3')
            ax.set_xticks([v_low, v_high])
            ax.set_xticklabels([f'{v_low}', f'{v_high}'])
            ax.set_xlabel(etiqueta_f)
        else:
            ax.plot([-1, 1], [me['low'], me['high']], 'o-', markersize=10, color='#2196F3')
            ax.set_xticks([-1, 1])
            ax.set_xticklabels(['-1', '+1'])
            ax.set_xlabel(etiqueta_f)

        if i == 0:
            ax.set_ylabel('Media de Y')
        ax.set_title(f'Efecto principal: {etiqueta_f}')
        ax.grid(alpha=0.3)

        if spec:
            ax.axhspan(spec['target'] - spec['tol'], spec['target'] + spec['tol'],
                       alpha=0.1, color='green')

    plt.tight_layout()
    fn = 'efectos_principales.png'
    plt.savefig(fn, bbox_inches='tight')
    plt.close()
    return fn

def _gen_interactions(R, custom_order=None, mapeos=None, unidades=None, eje_x_por_interaccion=None):
    sig = [n for n in R['effects'] if len(n.split('*')) == 2]

    if custom_order is not None and len(custom_order) > 0:
        sig = [n for n in custom_order if n in R['effects']]

    if not sig:
        return []
    files = []
    chunk_size = 2
    eje_x_map = eje_x_por_interaccion or {}

    for i in range(0, len(sig), chunk_size):
        chunk = sig[i:i+chunk_size]
        fig, axes = plt.subplots(1, len(chunk), figsize=(6*len(chunk), 4.5))
        if len(chunk) == 1:
            axes = [axes]

        for ax, name in zip(axes, chunk):
            parts = name.split('*')
            if len(parts) == 2:
                f1, f2 = parts
                it = R['interactions'][(f1, f2)]

                # Elegir que factor va en el eje X (por defecto f1)
                x_elegido = eje_x_map.get(name, f1)
                if x_elegido == f2:
                    x_fac, leg_fac = f2, f1
                    y_low = [it['mll'], it['mlh']]   # leg_fac = -1 mientras x_fac varia (-1 -> +1)
                    y_high = [it['mhl'], it['mhh']]  # leg_fac = +1 mientras x_fac varia (-1 -> +1)
                else:
                    x_fac, leg_fac = f1, f2
                    y_low = [it['mll'], it['mhl']]   # leg_fac = -1 mientras x_fac varia (-1 -> +1)
                    y_high = [it['mlh'], it['mhh']]  # leg_fac = +1 mientras x_fac varia (-1 -> +1)

                u_x = (unidades or {}).get(x_fac, '').strip()
                u_leg = (unidades or {}).get(leg_fac, '').strip()
                lbl_x = f"{x_fac} ({u_x})" if u_x else x_fac
                suf_u_leg = f" {u_leg}" if u_leg else ""

                # Usar valores reales si hay mapeos
                if mapeos and x_fac in mapeos and leg_fac in mapeos:
                    vx_low = mapeos[x_fac].get(-1, -1)
                    vx_high = mapeos[x_fac].get(1, 1)
                    vl_low = mapeos[leg_fac].get(-1, -1)
                    vl_high = mapeos[leg_fac].get(1, 1)

                    ax.plot([vx_low, vx_high], y_low,
                            'o-', color='#90CAF9', markersize=8, linewidth=2,
                            label=f'{leg_fac} = {vl_low}{suf_u_leg}')
                    ax.plot([vx_low, vx_high], y_high,
                            's-', color='#EF9A9A', markersize=8, linewidth=2,
                            label=f'{leg_fac} = {vl_high}{suf_u_leg}')
                    ax.set_xticks([vx_low, vx_high])
                    ax.set_xlabel(lbl_x)
                else:
                    ax.plot([-1, 1], y_low, 'o-', color='#90CAF9',
                            markersize=8, linewidth=2, label=f'{leg_fac} = -1')
                    ax.plot([-1, 1], y_high, 's-', color='#EF9A9A',
                            markersize=8, linewidth=2, label=f'{leg_fac} = +1')
                    ax.set_xticks([-1, 1])
                    ax.set_xticklabels([f'{x_fac} = -1', f'{x_fac} = +1'])
                    ax.set_xlabel(lbl_x)

                ax.set_ylabel('Media de Y')
                ax.legend(loc='best', framealpha=0.9)
                ax.grid(alpha=0.3)
                suf = ' (exploratoria)' if R['via'] == VIA_3 else ''
                ax.set_title(f'Interaccion {x_fac} * {leg_fac}{suf}')

        plt.tight_layout()
        fn_bloque = f'interacciones_grupo_{i//chunk_size + 1}.png'
        plt.savefig(fn_bloque, bbox_inches='tight', dpi=150)
        plt.close()
        files.append(fn_bloque)

    return files

def _gen_residuales(df, R):
    terms = R.get('terms_included', list(R['effects'].keys()))
    parts = [_term_to_formula(t) for t in terms]
    if R.get('tiene_bloques'):
        parts = ['Q("Grupo")'] + parts
    if R.get('tiene_centro'):
        parts.append('Q("Curvatura")')
    formula = 'Q("Y") ~ ' + ' + '.join(parts) if parts else 'Q("Y") ~ 1'

    mfit = smf.ols(formula, data=df).fit()
    df2 = df.copy()
    df2['fit'] = mfit.fittedvalues
    df2['res'] = mfit.resid
    df2 = df2.reset_index(drop=True)
    df2['orden'] = range(1, len(df2) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    stats.probplot(df2['res'], dist='norm', plot=axes[0])
    axes[0].set_title('(a) Normalidad de residuales')
    axes[1].scatter(df2['fit'], df2['res'], alpha=0.7, edgecolor='k')
    axes[1].axhline(0, color='red', linestyle='--')
    axes[1].grid(alpha=0.3)
    axes[1].set_title('(b) Residuales vs Ajustados')
    axes[2].scatter(df2['orden'], df2['res'], alpha=0.7, edgecolor='k')
    axes[2].axhline(0, color='red', linestyle='--')
    axes[2].grid(alpha=0.3)
    axes[2].set_title('(c) Residuales vs Orden')
    plt.tight_layout()
    fn = 'residuales.png'
    plt.savefig(fn, bbox_inches='tight')
    plt.close()
    return fn


def _gen_cubo(R):
    k = R['k']
    if k not in (2, 3):
        return None
    fn = 'cubo.png'
    if k == 3:
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection='3d')
        v = [(-1, -1, -1, '1'), (1, -1, -1, '2'), (-1, 1, -1, '3'), (1, 1, -1, '4'),
             (-1, -1, 1, '5'), (1, -1, 1, '6'), (-1, 1, 1, '7'), (1, 1, 1, '8')]
        e = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]
        for i, j in e:
            ax.plot([v[i][0], v[j][0]], [v[i][1], v[j][1]], [v[i][2], v[j][2]], 'b-', lw=1.5)
        for x, y, z, l in v:
            ax.scatter(x, y, z, color='red', s=100, edgecolor='black', zorder=5)
            ax.text(x * 1.15, y * 1.15, z * 1.15, l, fontsize=12, fontweight='bold')
        ax.scatter(0, 0, 0, color='green', s=200, marker='*', edgecolor='black', zorder=5)
        ax.set_xlabel(R['factores'][0])
        ax.set_ylabel(R['factores'][1])
        ax.set_zlabel(R['factores'][2])
        ax.set_xticks([-1, 0, 1])
        ax.set_yticks([-1, 0, 1])
        ax.set_zticks([-1, 0, 1])
        ax.set_title('Cubo 2^3 + centro')
    else:
        fig, ax = plt.subplots(figsize=(7, 7))
        v = [(-1, -1, '1'), (1, -1, '2'), (-1, 1, '3'), (1, 1, '4')]
        for a, b in [(0, 1), (0, 2), (1, 3), (2, 3)]:
            ax.plot([v[a][0], v[b][0]], [v[a][1], v[b][1]], 'b-', lw=1.5)
        for x, y, l in v:
            ax.scatter(x, y, color='red', s=120, edgecolor='black', zorder=5)
            ax.text(x * 1.12, y * 1.12, l, fontsize=13, fontweight='bold', ha='center')
        ax.scatter(0, 0, color='green', s=250, marker='*', edgecolor='black', zorder=5)
        ax.set_xlabel(R['factores'][0])
        ax.set_ylabel(R['factores'][1])
        ax.set_xticks([-1, 0, 1])
        ax.set_yticks([-1, 0, 1])
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.3, 1.3)
        ax.set_title('Diseno 2^2 + centro')
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(fn, dpi=150, bbox_inches='tight')
    plt.close()
    return fn


def _gen_curvatura(R):
    if not R['tiene_centro']:
        return None
    yf, yc = R['yf_bar'], R['yc_bar']
    d = yf - yc
    if abs(d) < 1e-9:
        return None
    fn = 'curvatura.png'
    x = np.linspace(-1, 1, 100)
    yc2 = d * x ** 2 + yc
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, np.full_like(x, yf), 'k--', lw=2, label=f'Lineal: {yf:.2f}')
    ax.plot(x, yc2, 'b-', lw=2, label='Centro-esquinas')
    ax.scatter([-1, 1], [yf, yf], color='red', s=120, edgecolor='black', zorder=5,
               label=f'Esquinas ({yf:.2f})')
    ax.scatter([0], [yc], color='green', s=120, edgecolor='black', zorder=5,
               label=f'Centro ({yc:.2f})')
    ax.annotate('', xy=(0, yf), xytext=(0, yc),
                arrowprops=dict(arrowstyle='<->', color='purple', lw=2))
    ax.text(0.06, (yf + yc) / 2, f'{d:+.2f}\n(curvatura)', color='purple', fontsize=11)
    ax.set_xlabel('Desalineacion')
    ax.set_ylabel('Y')
    ax.set_title('Curvatura')
    ax.set_xticks([-1, 0, 1])
    ax.set_xticklabels(['-1', '0', '+1'])
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(fn, dpi=150, bbox_inches='tight')
    plt.close()
    return fn


def _gen_spec(df, R, spec):
    cond = df.groupby(R['factores'])['Y'].mean().reset_index()
    lo = spec['target'] - spec['tol']
    hi = spec['target'] + spec['tol']

    fig, ax = plt.subplots(figsize=(14, 6))
    cols = ['green' if lo <= v <= hi else 'red' for v in cond['Y']]
    ax.bar(range(len(cond)), cond['Y'], color=cols, edgecolor='black')

    labels = []
    for _, row in cond.iterrows():
        partes = []
        for f in R['factores']:
            v = row[f]
            if v == -1:
                partes.append(f"{f}: -1")
            elif v == 1:
                partes.append(f"{f}: +1")
            else:
                partes.append(f"{f}: 0")
        labels.append("\n".join(partes))

    ax.axhline(spec['target'], color='green', linestyle='--', label=f"Objetivo {spec['target']}")
    ax.axhspan(lo, hi, alpha=0.15, color='green', label=f"Spec [{lo}, {hi}]")

    ax.set_xticks(range(len(cond)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Media de Y')
    ax.set_title('Medias por condicion vs Spec')
    ax.legend(loc='upper right')

    for i, v in enumerate(cond['Y']):
        ax.text(i, v + 0.3, f'{v:.2f}', ha='center', fontsize=8)

    plt.tight_layout()
    fn = 'especificacion.png'
    plt.savefig(fn, bbox_inches='tight')
    plt.close()
    return fn


# ---------- Ecuacion ----------
def _ecuacion_str(R, df=None):
    """Ecuación de regresión en unidades CODIFICADAS (exacta respecto al modelo OLS)."""
    if R['via'] == VIA_3:
        incluir = [n for n, e in R['effects'].items() if e.get('activo')]
    else:
        incluir = list(R.get('terms_included', R['effects'].keys()))

    terms = [f"{R['effects'][n]['coef']:+.4f}*{n}" for n in incluir if n in R['effects']]
    intercepto = R.get('intercepto_ols', R.get('yf_bar', R['y_bar']))

    if R['via'] == VIA_2:
        if R.get('tiene_bloques') and 'coef_bloques' in R:
            for b, coef_b in R['coef_bloques'].items():
                terms.append(f"{coef_b:+.4f}*{str(b).replace(' ', '_')}")
        if R.get('tiene_centro'):
            coef_curv = R['yc_bar'] - R['yf_bar']
            terms.append(f"{coef_curv:+.4f}*Ct_Pt")

    if not terms:
        return f"Y = {intercepto:.4f}"
    return f"Y = {intercepto:.4f} " + " ".join(terms)



def ecuacion_no_codificada(R, mapeos_inversos, unidades=None):
    """
    Convierte la ecuación en unidades codificadas a unidades reales (no codificadas),
    incluyendo las unidades de cada factor si fueron configuradas.
    """
    if not mapeos_inversos:
        return None

    factores = R['factores']
    m, r = {}, {}
    for f in factores:
        if f not in mapeos_inversos or -1 not in mapeos_inversos[f] or 1 not in mapeos_inversos[f]:
            return None
        v_low = float(mapeos_inversos[f][-1])
        v_high = float(mapeos_inversos[f][1])
        rango = (v_high - v_low) / 2.0
        if rango == 0:
            return None
        m[f] = (v_low + v_high) / 2.0
        r[f] = rango

    def _lbl(f):
        u = (unidades or {}).get(f, '').strip()
        return f"{f} ({u})" if u else f

    b0 = R.get('intercepto_ols', R.get('yf_bar', R['y_bar']))
    incluir = list(R.get('terms_included', R['effects'].keys()))

    u_int = b0
    u_lin = {f: 0.0 for f in factores}
    u_int2 = {}

    for t in incluir:
        if t not in R['effects']:
            continue
        coef = R['effects'][t]['coef']
        partes = t.split('*')
        if len(partes) == 1:
            f = partes[0]
            u_int -= coef * m[f] / r[f]
            u_lin[f] += coef / r[f]
        elif len(partes) == 2:
            f1, f2 = partes
            u_int += coef * (m[f1] * m[f2]) / (r[f1] * r[f2])
            u_lin[f1] -= coef * m[f2] / (r[f1] * r[f2])
            u_lin[f2] -= coef * m[f1] / (r[f1] * r[f2])
            u_int2[t] = coef / (r[f1] * r[f2])

    terminos = [f"{u_int:.4f}"]
    for f in factores:
        if abs(u_lin[f]) > 1e-12:
            terminos.append(f"{u_lin[f]:+.5g}*{_lbl(f)}")
    for t, c in u_int2.items():
        if abs(c) > 1e-12:
            f1, f2 = t.split('*')
            terminos.append(f"{c:+.5g}*{_lbl(f1)}*{_lbl(f2)}")
    if R.get('tiene_centro'):
        coef_curv = R['yc_bar'] - R['yf_bar']
        terminos.append(f"{coef_curv:+.4f}*Ct_Pt")

    return "Y = " + " ".join(terminos)


# ---------- Funciones de Reporte y Balance ----------
def generar_tabla_regresion(df, R):
    """Genera la tabla de coeficientes y el resumen de ajuste global."""
    terms = R.get('terms_included', list(R['effects'].keys()))
    parts = [_term_to_formula(t) for t in terms]
    if R.get('tiene_bloques'):
        parts = ['Q("Grupo")'] + parts
    if R.get('tiene_centro'):
        parts.append('Q("Curvatura")')

    formula = 'Q("Y") ~ ' + ' + '.join(parts) if parts else 'Q("Y") ~ 1'
    mfit = smf.ols(formula, data=df).fit()

    coef_df = pd.DataFrame({
        'Término': mfit.params.index,
        'Coef': mfit.params.values,
        'Error Est.': mfit.bse.values,
        't stat': mfit.tvalues.values,
        'Valor p': mfit.pvalues.values,
    })

    coef_df['Término'] = (
        coef_df['Término']
        .str.replace('Q("', '', regex=False)
        .str.replace('")', '', regex=False)
        .str.replace(':', ' * ', regex=False)
    )

    conf_int = mfit.conf_int()
    coef_df['Inferior 95%'] = conf_int[0].values
    coef_df['Superior 95%'] = conf_int[1].values
    coef_df['VIF'] = 1.0

    fit_df = pd.DataFrame({
        'Métrica': ['Multiple R', 'R Square', 'Adjusted R Square', 'Predicted R Square', 'Standard Error', 'Observations', 'AIC', 'BIC (SBC)'],
        'Valor': [
            np.sqrt(mfit.rsquared),
            mfit.rsquared,
            mfit.rsquared_adj,
            R.get('R2_pred', np.nan),
            np.sqrt(mfit.mse_resid),
            len(df),
            mfit.aic,
            mfit.bic
        ]
    })

    return coef_df, fit_df


def generar_texto_pareto(R):
    df_error = R['df_resid']
    t_crit = R['t_crit']
    total_terms = len(R['all_effects'])
    terms_usados = len(R['terms_included'])
    terminos_agrupados = total_terms - terms_usados
    df_total = R['N'] - 1
    df_puro = df_total - total_terms

    texto = (
        f"Para validar visualmente la significancia de los efectos, se utiliza el estadístico t de cada factor. "
        f"Este se compara contra un valor crítico de t derivado de los {df_error} grados de libertad del Error del ANOVA. "
        f"Con un nivel de significancia α = 0.05, el umbral es t ≈ {t_crit:.2f}."
    )
    if terminos_agrupados > 0:
        texto += (
            f"\n\nNota metodológica: El modelo inicial evaluó {total_terms} términos ({R['k']} factores principales y "
            f"{total_terms - R['k']} interacciones). Al excluir {terminos_agrupados} términos no significativos, el Residual final "
            f"cuenta con {df_error} grados de libertad ({df_puro} puros + {terminos_agrupados} agrupados)."
        )
    return texto


def generar_tabla_balance_neto(R, objetivo='max'):
    y_bar = R.get('intercepto_ols', R['y_bar'])
    sig_terms = [n for n, e in R['effects'].items() if (e.get('p', 1) < 0.05) or (R['via'] == VIA_3 and e.get('activo'))]

    rows = [{
        'Concepto (Factor / Interacción)': 'Base del modelo (Intercepto)',
        'Efecto (Puntos)': round(y_bar, 4),
        'Impacto porcentual (%)': 100.00,
        'Tipo de Efecto': 'Punto de partida'
    }]

    main_effects = []
    interactions = []

    for name in sig_terms:
        e = R['effects'][name]
        eff = e['efecto']
        pct = (eff / y_bar) * 100 if y_bar != 0 else 0.0

        if objetivo == 'max':
            tipo = 'Beneficio' if eff > 0 else 'Penalización'
        else:
            tipo = 'Beneficio (Reduce)' if eff < 0 else 'Penalización (Aumenta)'

        row_data = {
            'Concepto (Factor / Interacción)': name.replace('*', ' x '),
            'Efecto (Puntos)': round(eff, 4),
            'Impacto porcentual (%)': round(pct, 2),
            'Tipo de Efecto': tipo
        }
        if '*' not in name:
            main_effects.append(row_data)
        else:
            interactions.append(row_data)

    sum_main = sum(r['Efecto (Puntos)'] for r in main_effects)
    pct_main = sum(r['Impacto porcentual (%)'] for r in main_effects)
    sum_int = sum(r['Efecto (Puntos)'] for r in interactions)
    pct_int = sum(r['Impacto porcentual (%)'] for r in interactions)

    for r in main_effects:
        rows.append(r)
    if main_effects:
        rows.append({
            'Concepto (Factor / Interacción)': 'Suma de Efectos Principales',
            'Efecto (Puntos)': round(sum_main, 4),
            'Impacto porcentual (%)': round(pct_main, 2),
            'Tipo de Efecto': 'Totales'
        })

    for r in interactions:
        rows.append(r)
    if interactions:
        rows.append({
            'Concepto (Factor / Interacción)': 'Suma de Interacciones',
            'Efecto (Puntos)': round(sum_int, 4),
            'Impacto porcentual (%)': round(pct_int, 2),
            'Tipo de Efecto': 'Totales'
        })

    final_val = y_bar + sum_main + sum_int
    final_pct = 100.00 + pct_main + pct_int
    rows.append({
        'Concepto (Factor / Interacción)': 'Resultado Final Estimado',
        'Efecto (Puntos)': round(final_val, 4),
        'Impacto porcentual (%)': round(final_pct, 2),
        'Tipo de Efecto': 'Balance Neto'
    })

    return pd.DataFrame(rows)


def obtener_formulas_utilizadas(R):
    eq_str = _ecuacion_str(R) or "No disponible"
    terms = R.get('terms_included', list(R['effects'].keys()))
    parts = [_term_to_formula(t) for t in terms]
    if R.get('tiene_bloques'):
        parts = ['Q("Grupo")'] + parts
    if R.get('tiene_centro'):
        parts.append('Q("Curvatura")')
    formula = 'Q("Y") ~ ' + ' + '.join(parts) if parts else 'Q("Y") ~ 1'

    return {
        "Ecuación de Regresión Ajustada": eq_str,
        "Fórmula del Modelo (Statsmodels)": formula,
        "Error Estándar del Coeficiente (SE)": "SE = √(CM_residual / n)",
        "Estadístico t (Significancia)": "t = Coeficiente / SE",
        "Estadístico F (ANOVA)": "F = CM_efecto / CM_residual",
        "Estimación de Lenth (PSE)": "PSE = 1.5 * mediana(|efectos| < 2.5 * s0)",
        "Grados de Libertad (Lenth)": "df_t = m / 3"
    }


def predecir_y_combinacion(niveles, R):
    """Calcula Y predicho para niveles codificados (-1, 0, +1), incluyendo curvatura en el centro."""
    y_pred = R.get('intercepto_ols', R.get('yf_bar', R['y_bar']))
    terms = R.get('terms_included', list(R['effects'].keys()))

    for term in terms:
        if term not in R['effects']:
            continue
        coef = R['effects'][term]['coef']
        val = 1.0
        for f in term.split('*'):
            val *= niveles[f]
        y_pred += coef * val

    if R.get('tiene_centro') and all(abs(niveles[f]) < 1e-9 for f in R['factores']):
        y_pred += (R['yc_bar'] - R['yf_bar'])

    return y_pred


def generar_tabla_alias(df, factores):
    esq = df[df['Curvatura'] == 0].copy()
    patrones = {}
    for r in range(1, len(factores) + 1):
        for combo in itertools.combinations(factores, r):
            nombre = '*'.join(combo)
            patron = tuple(esq[list(combo)].prod(axis=1).tolist())
            patron_neg = tuple((-esq[list(combo)].prod(axis=1)).tolist())
            if patron in patrones:
                patrones[patron].append(nombre)
            elif patron_neg in patrones:
                patrones[patron_neg].append(f"-{nombre}")
            else:
                patrones[patron] = [nombre]

    filas = []
    for terminos in patrones.values():
        if len(terminos) > 1:
            principales = [t for t in terminos if '*' not in t]
            dobles = [t for t in terminos if len(t.split('*')) == 2]
            triples = [t for t in terminos if len(t.split('*')) == 3]
            mayores = [t for t in terminos if len(t.split('*')) > 3]
            filas.append({
                'Principales': ', '.join(principales) if principales else '—',
                'Dobles': ', '.join(dobles) if dobles else '—',
                'Triples': ', '.join(triples) if triples else '—',
                'Mayores': ', '.join(mayores) if mayores else '—'
            })
    return pd.DataFrame(filas)



def _gen_superficie_3d(R, resp_name='Respuesta (Y)', nombres_reales=None,
                       f1=None, f2=None, mapeos=None, unidades=None,
                       estilo='moderno', cmap=None):
    if R['via'] == VIA_3:
        return None, None, None, None

    factores = R['factores']
    if len(factores) < 2:
        return None, None, None, None

    f1_raw = f1 if (f1 in factores) else factores[0]
    f2_raw = f2 if (f2 in factores and f2 != f1_raw) else (factores[2] if len(factores) > 2 else factores[1])

    u1 = (unidades or {}).get(f1_raw, '').strip()
    u2 = (unidades or {}).get(f2_raw, '').strip()
    base1 = nombres_reales.get(f1_raw, f1_raw) if nombres_reales else f1_raw
    base2 = nombres_reales.get(f2_raw, f2_raw) if nombres_reales else f2_raw
    f1_label = f"{base1} ({u1})" if u1 else base1
    f2_label = f"{base2} ({u2})" if u2 else base2

    x = np.linspace(-1, 1, 50)
    y = np.linspace(-1, 1, 50)
    X, Y = np.meshgrid(x, y)

    Z = np.full_like(X, R.get('intercepto_ols', R['y_bar']), dtype=float)
    if f1_raw in R['effects']:
        Z += R['effects'][f1_raw]['coef'] * X
    if f2_raw in R['effects']:
        Z += R['effects'][f2_raw]['coef'] * Y

    for term_int in (f"{f1_raw}*{f2_raw}", f"{f2_raw}*{f1_raw}"):
        if term_int in R['effects']:
            Z += R['effects'][term_int]['coef'] * X * Y
            break

    # Convertir ejes a unidades reales si existen mapeos
    usar_real = (
        mapeos is not None
        and f1_raw in mapeos and -1 in mapeos[f1_raw] and 1 in mapeos[f1_raw]
        and f2_raw in mapeos and -1 in mapeos[f2_raw] and 1 in mapeos[f2_raw]
    )
    if usar_real:
        v1_lo, v1_hi = float(mapeos[f1_raw][-1]), float(mapeos[f1_raw][1])
        v2_lo, v2_hi = float(mapeos[f2_raw][-1]), float(mapeos[f2_raw][1])
        XX_plot = ((X + 1.0) / 2.0) * (v1_hi - v1_lo) + v1_lo
        YY_plot = ((Y + 1.0) / 2.0) * (v2_hi - v2_lo) + v2_lo
        suf_eje = ""
    else:
        XX_plot, YY_plot = X, Y
        suf_eje = " (coded)"

    idx_max = np.unravel_index(np.argmax(Z), Z.shape)
    x_opt = float(XX_plot[idx_max])
    y_opt = float(YY_plot[idx_max])
    z_opt = float(Z[idx_max])

    z_min = float(np.min(Z))
    z_max = float(np.max(Z))
    if z_max - z_min < 1e-9:
        z_min -= 0.1
        z_max += 0.1
    margen = (z_max - z_min) * 0.1
    limite_inf = z_min - margen
    limite_sup = z_max + margen

    y_span = float(np.max(YY_plot) - np.min(YY_plot)) or 2.0
    es_clasico = (estilo == 'clasico')
    cmap_usado = (cmap or 'terrain') if es_clasico else 'viridis'

    fig = plt.figure(figsize=(14, 6))

    # --- Panel 1: Superficie 3D ---
    ax1 = fig.add_subplot(121, projection='3d')
    if es_clasico:
        surf = ax1.plot_surface(XX_plot, YY_plot, Z, cmap=cmap_usado,
                                edgecolor='#333333', linewidth=0.25, alpha=0.92)
    else:
        surf = ax1.plot_surface(XX_plot, YY_plot, Z, cmap=cmap_usado,
                                edgecolor='none', alpha=0.90)

    ax1.scatter(x_opt, y_opt, z_opt, color='red', marker='*', s=280, zorder=10, label='Max. Predicho')
    ax1.text(x_opt, y_opt, z_opt + margen, f'{z_opt:.2f}', color='red', fontweight='bold', ha='center')
    ax1.legend()
    ax1.set_xlabel(f"{f1_label}{suf_eje}", labelpad=10)
    ax1.set_ylabel(f"{f2_label}{suf_eje}", labelpad=10)
    ax1.set_zlabel(resp_name, labelpad=15, rotation=90)
    ax1.set_title(f'Superficie 3D: {base1} vs {base2}')
    ax1.set_zlim(limite_inf, limite_sup)
    cbar1 = fig.colorbar(surf, ax=ax1, shrink=0.55, aspect=8, extend='both')
    cbar1.set_label(resp_name, rotation=270, labelpad=20)

    # --- Panel 2: Contorno 2D ---
    ax2 = fig.add_subplot(122)
    if es_clasico:
        cf = ax2.contourf(XX_plot, YY_plot, Z, levels=12, cmap=cmap_usado)
        line_lv = list(cf.levels)
        if len(line_lv) > 8:
            line_lv = line_lv[::2]
        lineas = ax2.contour(XX_plot, YY_plot, Z, levels=line_lv, colors='k', linewidths=0.7)
        ax2.clabel(lineas, inline=True, fontsize=8, fmt='%g')
        ax2.grid(color='white', alpha=0.3, linewidth=0.6)
    else:
        niveles_fondo = np.linspace(z_min, z_max, 50)
        cf = ax2.contourf(XX_plot, YY_plot, Z, levels=niveles_fondo, cmap='viridis', extend='both')
        niveles_lineas = np.linspace(z_min, z_max, 6)
        lineas = ax2.contour(XX_plot, YY_plot, Z, levels=niveles_lineas, colors='white', linewidths=1.2, alpha=1.0)
        import matplotlib.patheffects as path_effects
        txt = ax2.clabel(lineas, inline=True, inline_spacing=5, fontsize=9, fmt='%1.2f')
        for t in txt:
            t.set_fontweight('bold')
            t.set_color('white')
            t.set_path_effects([path_effects.withStroke(linewidth=2, foreground='black')])

    ax2.scatter(x_opt, y_opt, color='red', marker='*', s=200, zorder=10, label='Max. Predicho')
    ax2.text(x_opt, y_opt + 0.04 * y_span, f'{z_opt:.2f}', color='red', fontweight='bold', ha='center')
    ax2.legend(loc='upper right')
    ax2.set_xlabel(f"{f1_label}{suf_eje}")
    ax2.set_ylabel(f"{f2_label}{suf_eje}")
    ax2.set_title(f'Curvas de Nivel: {base1} vs {base2}')
    cbar2 = fig.colorbar(cf, ax=ax2, extend='both')
    cbar2.set_label(resp_name, rotation=270, labelpad=20)

    plt.tight_layout()
    fn = f'superficie_3d_{f1_raw}_{f2_raw}_{estilo}.png'
    plt.savefig(fn, bbox_inches='tight', dpi=150)
    plt.close()
    return fn, x_opt, y_opt, z_opt



