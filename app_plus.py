# app_plus.py
# Interfaz Streamlit - Flujo completo (Factorial 2^k + Rama CCD)
# 1. Cargar -> 2. Confirmar columnas -> 3A. Guardar niveles/unidades -> 3B. Matriz codificada -> 4. Centros y analisis -> 5. Resultados / CCD

import importlib
import os
import numpy as np
import pandas as pd
import streamlit as st

import calc_engine_plus as ce
import calc_engine_rsm as ce_rsm

importlib.reload(ce)
importlib.reload(ce_rsm)

st.set_page_config(
    page_title="DOE Tool - Analisis Factorial y RSM",
    layout="wide",
    initial_sidebar_state="expanded"
)


def _init_state():
    defaults = {
        'stage': 'upload',
        'raw': None,
        'df_work': None,
        'df_cod': None,
        'df_real': None,
        'factores': [],
        'respuestas': [],
        'respuesta': None,
        'corrida_col': None,
        'es_codificada': False,
        'niveles_guardados': {},
        'unidades_factores': {},
        'mapeos_niveles': {},
        'mapeos_inversos': {},
        'niveles_confirmados_paso3': False,
        'filas_centro_idx': [],
        'centros_confirmados': False,
        'analisis_tipo': None,
        'modelo_fraccionado': False,
        'resolucion': None,
        'R': None,
        'R_rsm': None,
        'df_ccd_raw': None,
        'factores_ccd': [],
        'respuesta_ccd': [],
        'R_ccd_full': None,
        'R_ccd': None,
        'opt_ccd': None,
        'nombre': '',
        'estudiante': '',
        'curso': '',
        'alpha_modelo': 0.05,
        'modo_seleccion': 'backward',
        'terms_manual': [],
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


_init_state()


def _to_numeric_clean(serie):
    if pd.api.types.is_numeric_dtype(serie):
        return pd.to_numeric(serie, errors='coerce')
    limpia = serie.astype(str).str.strip().str.replace(',', '.', regex=False)
    return pd.to_numeric(limpia, errors='coerce')


def _es_num_val(v):
    try:
        float(v)
        return not isinstance(v, bool)
    except (TypeError, ValueError):
        return False


def _fmt_nivel(v):
    if _es_num_val(v):
        return f"{float(v):g}"
    return str(v)


def _detectar_columnas(raw):
    factores, respuestas, descartar = [], [], []
    id_names = {
        'run', 'corrida', 'corrida_id', 'id', 'n_corrida', 'numero_corrida',
        'obs', 'observacion', 'index', 'unnamed: 0',
        'run order', 'std order', 'run type', 'order', 'tipo', 'type', 'tipopto',
        'block', 'bloque', 'order_run', 'orden', 'orden corrida', 'orden de corrida',
        'orden estandar', 'stdorder', 'runorder',
        'pt type', 'pt_type', 'tipo punto', 'tipo de punto',
        'replicate', 'replica', 'rep',
    }
    resp_names = {'y', 'response', 'respuesta', 'resp', 'rendimiento', 'yield', 'tamano', 'tamaño', 'potencial', 'zeta'}
    resp_hints = ('rep', 'replica', 'block', 'bloque', 'ensayo')

    for c in raw.columns:
        nombre_lower = str(c).strip().lower()
        vals = raw[c].dropna().unique().tolist()
        n_unicos = len(vals)

        if any(x in nombre_lower for x in ['corrida', 'run', 'orden', 'order']):
            if n_unicos == len(raw):
                descartar.append(c)
                continue

        if nombre_lower in id_names or n_unicos <= 1:
            descartar.append(c)
            continue

        es_resp_hint = (
            nombre_lower in resp_names
            or any(h in nombre_lower for h in resp_hints)
            or any(x in nombre_lower for x in ['resp', 'yield', 'rend'])
        )
        if es_resp_hint and pd.api.types.is_numeric_dtype(raw[c]):
            respuestas.append(c)
            continue

        if pd.api.types.is_numeric_dtype(raw[c]):
            if n_unicos in [2, 3]:
                factores.append(c)
                continue
            if len(raw) >= 11 and n_unicos in [4, 5]:
                s_num = np.sort(pd.to_numeric(raw[c], errors='coerce').dropna().unique())
                if len(s_num) >= 4:
                    mid = (s_num.min() + s_num.max()) / 2.0
                    simetrico = abs((s_num.mean()) - mid) < (s_num.max() - s_num.min()) * 0.25
                    tiene_centro = np.min(np.abs(s_num - mid)) < (s_num.max() - s_num.min()) * 0.1
                    if simetrico and tiene_centro:
                        factores.append(c)
                        continue
            respuestas.append(c)
            continue

        if n_unicos <= 5:
            factores.append(c)
            continue

        descartar.append(c)

    return factores, respuestas, descartar


def _es_codificada(raw, cols):
    """Detecta si las columnas de factores ya estan en escala codificada (-1, 0, +1 o CCD +/-alpha)."""
    if not cols:
        return False
    for c in cols:
        serie = _to_numeric_clean(raw[c])
        if serie.isna().any():
            return False
        vals = np.unique(np.round(serie.dropna().to_numpy(dtype=float), 4))
        if len(vals) < 2:
            return False
        if set(vals).issubset({-1.0, 0.0, 1.0}):
            continue
        es_simetrico = abs(vals.min() + vals.max()) < 1e-3
        tiene_base = {-1.0, 0.0, 1.0}.issubset(set(vals))
        en_rango_cod = (vals.min() >= -3.5) and (vals.max() <= 3.5)
        if not (es_simetrico and tiene_base and en_rango_cod):
            return False
    return True


def _detectar_centros(df_cod, cols):
    """Detecta filas donde TODOS los factores codificados estan cercanos a 0 (centros)."""
    df_num = df_cod[cols].apply(_to_numeric_clean)
    mascara = df_num.apply(lambda r: all(pd.notna(v) and abs(v) < 0.5 for v in r), axis=1)
    indices = df_cod[mascara].index.tolist()
    return len(indices), indices


def _detectar_replicas(df_cod, cols, resp_cols):
    """Cuenta replicas por filas duplicadas o por multiples columnas de respuesta/bloque."""
    if resp_cols is None:
        return 0
    if isinstance(resp_cols, (list, tuple)):
        valid_cols = [c for c in resp_cols if c in df_cod.columns]
    else:
        valid_cols = [resp_cols] if resp_cols in df_cod.columns else []

    if not valid_cols:
        return 0

    if len(valid_cols) >= 2:
        return len(valid_cols)

    df_esq = df_cod.copy()
    for c in cols:
        df_esq[c] = pd.to_numeric(df_esq[c], errors='coerce')
    if st.session_state.get('filas_centro_idx'):
        df_esq = df_esq.drop(st.session_state.filas_centro_idx, errors='ignore')
    else:
        _, idx_c = _detectar_centros(df_esq, cols)
        df_esq = df_esq.drop(idx_c, errors='ignore')
    if len(df_esq) == 0:
        return 0
    df_group = df_esq.groupby(cols)[valid_cols[0]].count()
    return int((df_group > 1).sum())


def _validar_analisis_vs_diseno(df_cod, factores, resp_cols, analisis):
    k = len(factores)
    if st.session_state.get('filas_centro_idx'):
        centros = len(st.session_state.filas_centro_idx)
    else:
        centros = _detectar_centros(df_cod, factores)[0]
    replicas = _detectar_replicas(df_cod, factores, resp_cols)
    msgs = []

    if analisis == 'factorial_via1':
        if centros > 0:
            msgs.append(f"Via 1 requiere SIN puntos centrales. Detectados: {centros} centros.")
        if replicas == 0:
            msgs.append("Via 1 requiere replicas puras. No se detectaron.")

    elif analisis == 'factorial_via2':
        if centros == 0:
            msgs.append("Via 2 requiere puntos centrales. No se detectaron.")

    elif analisis == 'factorial_via3':
        if centros > 0:
            msgs.append(f"Via 3 es para disenos NO replicados SIN centros. Detectados: {centros} centros.")
        if replicas > 0:
            msgs.append(f"Via 3 no usa replicas. Detectadas: {replicas} replicas.")

    elif analisis == 'ccd':
        if k < 2 or k > 5:
            msgs.append(f"CCD tipicamente usa 2-5 factores. Detectados: {k}.")

    return msgs


def _sugerir_analisis(df_cod, factores, resp_cols, n_respuestas=1):
    n = len(df_cod)
    if st.session_state.get('filas_centro_idx'):
        centros = len(st.session_state.filas_centro_idx)
    else:
        centros = _detectar_centros(df_cod, factores)[0]
    esquinas = n - centros
    replicas = _detectar_replicas(df_cod, factores, resp_cols)

    if hasattr(ce_rsm, 'es_diseno_ccd') and ce_rsm.es_diseno_ccd(df_cod, factores):
        return 'ccd', f"Detectado diseno CCD ({n} corridas con puntos axiales y {centros} centros). Sugerencia: Superficie de Respuesta (CCD)."
    if centros > 0:
        extra = f" × {n_respuestas} bloques/columnas" if n_respuestas >= 2 else ""
        return 'factorial_via2', f"Detectado: {esquinas} esquinas + {centros} centro(s){extra}. Sugerencia: Via 2."
    elif replicas > 0 or n_respuestas >= 2:
        return 'factorial_via1', f"Detectado: {esquinas} esquinas con replicas ({max(replicas, n_respuestas)}). Sugerencia: Via 1."
    else:
        return 'factorial_via3', f"Detectado: {esquinas} esquinas sin replicas ni centros. Sugerencia: Via 3."


def _codificar_factor(serie, v_bajo, v_alto):
    num = _to_numeric_clean(serie)
    v_medio = (v_bajo + v_alto) / 2.0
    rango = (v_alto - v_bajo) / 2.0
    if rango == 0:
        return num
    return (num - v_medio) / rango


def _limpiar_niveles_y_widgets(factores):
    """Restablece los niveles guardados y las claves de widgets a sus valores por defecto."""
    st.session_state.niveles_guardados = {}
    st.session_state.unidades_factores = {}
    st.session_state.mapeos_niveles = {}
    st.session_state.mapeos_inversos = {}
    st.session_state.niveles_confirmados_paso3 = False
    st.session_state.df_cod = None
    st.session_state.df_real = None
    st.session_state.df_work = None
    for f in factores:
        for pref in ('real_low_', 'real_high_', 'low_', 'high_', 'unit_', 'qual_low_', 'qual_high_'):
            st.session_state.pop(f"{pref}{f}", None)


def _info_box(titulo, texto, tipo='info'):
    emoji = {'info': '💡', 'warning': '⚠️', 'tip': '📌', 'success': '✅'}.get(tipo, '💡')
    color = '#2196F3' if tipo == 'info' else '#ff9800' if tipo == 'warning' else '#4caf50'
    st.markdown(f"""
    <div style="background-color:#f0f2f6;padding:12px 16px;border-radius:8px;margin:8px 0;border-left:4px solid {color};">
    <b>{emoji} {titulo}</b><br>{texto}
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.header("Configuracion")

    if st.button("Reiniciar todo", use_container_width=True):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

    if st.session_state.raw is not None:
        st.markdown("---")
        st.markdown("### Progreso")
        current = st.session_state.stage
        if current.startswith('ccd_'):
            stages = {
                'ccd_full': '1. CCD: Modelo completo',
                'ccd_refine': '2. CCD: Refinar modelo',
                'ccd_diag': '3. CCD: Contornos, 3D y Opt.'
            }
        else:
            stages = {
                'upload': '1. Cargar archivo',
                'preview': '2. Confirmar columnas',
                'encode': '3. Guardar niveles y matriz codificada',
                'analysis_select': '4. Puntos centrales y analisis',
                'results': '5. Resultados'
            }

        if current in stages:
            curr_idx = list(stages.keys()).index(current)
            for stage_key, stage_label in stages.items():
                idx = list(stages.keys()).index(stage_key)
                if stage_key == current:
                    st.markdown(f"<div style='background:#e3f2fd;padding:6px 10px;border-radius:6px;margin:2px 0;font-weight:bold;'>→ {stage_label}</div>", unsafe_allow_html=True)
                elif idx < curr_idx:
                    st.markdown(f"<div style='color:#4caf50;padding:2px 0;'>✅ {stage_label}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div style='color:#9e9e9e;padding:2px 0;'>⬜ {stage_label}</div>", unsafe_allow_html=True)
        st.markdown("---")

    st.markdown("### Generar datos")
    st.markdown("Necesitas crear un diseno experimental?")
    st.link_button("Ir a Bioprocess Tool", "https://bioprocesstool.com/", use_container_width=True)

    st.markdown("---")
    st.markdown("### Contexto")
    st.session_state.nombre = st.text_input("Proceso/Experimento", value=st.session_state.nombre, placeholder="Ej: Extraccion de proteina")
    st.session_state.estudiante = st.text_input("Estudiante", value=st.session_state.estudiante)
    st.session_state.curso = st.text_input("Curso/Asignatura", value=st.session_state.curso, placeholder="Ej: Diseno de Experimentos")


# ============================================================
# MAIN - FLUJO POR ETAPAS
# ============================================================

# ---------- ETAPA 1: CARGAR ARCHIVO ----------
if st.session_state.stage == 'upload':
    st.title("DOE Tool - Analisis Factorial y Superficie de Respuesta")

    st.markdown("""
    ## Bienvenido
    Esta herramienta te permite analizar disenos experimentales factoriales (2^k) y de superficie de respuesta (CCD).

    ### Flujo de trabajo:
    1. **Carga tu matriz** de diseno (Excel, CSV o tabla pegada)
    2. **Confirma las columnas** (factores y respuesta(s))
    3. **Guarda niveles reales y unidades** (soporta valores numericos, cualitativos como X/Y o continuar directo en -1, 0, +1)
    4. **Detecta puntos centrales / replicas y elige el analisis** (Via 1, 2, 3 o CCD)
    5. **Obtien resultados**: ANOVA, coeficientes, ecuacion, graficos, especificacion y superficies 3D
    """)

    uploaded = st.file_uploader("Sube tu archivo Excel o CSV", type=['xlsx', 'csv'])

    raw_nuevo = None
    msg_carga = ""

    if uploaded:
        try:
            if uploaded.name.endswith('.xlsx'):
                raw_nuevo = pd.read_excel(uploaded)
            else:
                raw_nuevo = pd.read_csv(uploaded)
            msg_carga = f"Archivo cargado: {raw_nuevo.shape[0]} filas × {raw_nuevo.shape[1]} columnas"
        except Exception as e:
            st.error(f"Error al cargar el archivo: {e}")

    with st.expander("📋 O pega tu tabla directamente (desde Excel o texto)", expanded=False):
        txt_pegado = st.text_area(
            "Pega aqui tus columnas y filas (1 sola fila de encabezados arriba):",
            height=180,
            placeholder="Diametro\tSuplemento\tTemperatura\tMicroburbujas\tRep1\tRep2\tRep3\n100\tX\t36.5\tSin\t76\t85\t85",
            key="txt_tabla_pegada"
        )
        if st.button("📥 Cargar tabla pegada", type='primary', key="btn_cargar_pegado"):
            if not txt_pegado.strip():
                st.warning("Pega primero la tabla en el recuadro.")
            else:
                try:
                    import io
                    muestra = txt_pegado.strip()
                    sep = '\t' if '\t' in muestra else (';' if ';' in muestra else (',' if ',' in muestra else r'\s+'))
                    raw_nuevo = pd.read_csv(io.StringIO(muestra), sep=sep)
                    msg_carga = f"Tabla pegada cargada: {raw_nuevo.shape[0]} filas × {raw_nuevo.shape[1]} columnas"
                except Exception as e:
                    st.error(f"No se pudo leer la tabla pegada: {e}")

    if raw_nuevo is not None:
        try:
            raw = raw_nuevo.copy()
            raw.columns = raw.columns.astype(str).str.strip()

            # Reset limpio del archivo y analisis previo (conservando niveles_guardados y unidades_factores)
            st.session_state.raw = raw
            st.session_state.df_cod = None
            st.session_state.df_real = None
            st.session_state.df_work = None
            st.session_state.niveles_confirmados_paso3 = False
            st.session_state.filas_centro_idx = []
            st.session_state.centros_confirmados = False
            st.session_state.factores = []
            st.session_state.respuestas = []
            st.session_state.respuesta = None
            st.session_state.corrida_col = None
            st.session_state.analisis_tipo = None
            st.session_state.modelo_fraccionado = False
            st.session_state.resolucion = None
            st.session_state.R = None
            st.session_state.R_rsm = None
            st.session_state.df_ccd_raw = None
            st.session_state.factores_ccd = []
            st.session_state.respuesta_ccd = []
            st.session_state.R_ccd_full = None
            st.session_state.R_ccd = None
            st.session_state.opt_ccd = None
            for wk in ('sel_factores', 'sel_respuestas', 'sel_analisis', 'confirm_centros',
                       'confirm_no_centros', 'modo_sel'):
                st.session_state.pop(wk, None)

            st.success(msg_carga)

            fact_cands, resp_cands, descartados = _detectar_columnas(raw)

            if len(descartados) > 0:
                st.info(f"Columnas descartadas (metadatos): {', '.join(descartados)}")

            if len(fact_cands) == 0:
                st.error("No se detectaron factores. Verifica tus columnas.")
            elif len(resp_cands) == 0:
                st.error("No se detecto columna de respuesta.")
            else:
                st.session_state.stage = 'preview'
                st.rerun()

        except Exception as e:
            st.error(f"Error al cargar: {e}")


# ---------- ETAPA 2: CONFIRMAR COLUMNAS ----------
elif st.session_state.stage == 'preview':
    st.title("2. Confirmar Columnas de la Matriz")

    raw = st.session_state.raw

    _info_box("Paso 2: Revision de columnas",
              "Confirma que el programa detecto correctamente los factores y la(s) respuesta(s). "
              "Si tus replicas o bloques estan en varias columnas (ej: Rep1, Rep2, Rep3 o Block 1..5), seleccionalas todas en Respuesta(s).",
              tipo='info')

    fact_cands, resp_cands, _ = _detectar_columnas(raw)

    corrida_opts = ['(Ninguna)'] + [c for c in raw.columns if any(x in str(c).lower() for x in ['corrida', 'run', 'orden', 'order', 'id'])]
    corrida_idx = corrida_opts.index(st.session_state.corrida_col) if st.session_state.corrida_col in corrida_opts else 0
    corrida_sel = st.selectbox("Columna de Corrida (opcional):", corrida_opts, index=corrida_idx)
    st.session_state.corrida_col = None if corrida_sel == '(Ninguna)' else corrida_sel

    default_factores = [f for f in st.session_state.factores if f in raw.columns] or fact_cands
    st.markdown("### Factores")
    factores = st.multiselect("Selecciona los factores:", raw.columns.tolist(), default=default_factores, key="sel_factores")

    opts_resp = [c for c in raw.columns if c not in factores and c != st.session_state.corrida_col]
    if not opts_resp:
        opts_resp = raw.columns.tolist()

    default_resps = [r for r in st.session_state.respuestas if r in opts_resp]
    if not default_resps:
        cands_validos = [r for r in resp_cands if r in opts_resp]
        default_resps = cands_validos if cands_validos else [opts_resp[0]]

    st.markdown("### Respuesta(s) / Columnas de replica o bloque")
    respuestas = st.multiselect(
        "Selecciona una o varias columnas de respuesta (o todas las columnas Rep / Block):",
        opts_resp,
        default=default_resps,
        key="sel_respuestas"
    )

    if len(factores) < 2:
        st.warning("Selecciona al menos 2 factores.")
    elif len(respuestas) == 0:
        st.warning("Selecciona al menos 1 variable de respuesta.")

    else:
        factores_cambiaron = (factores != st.session_state.factores)
        st.session_state.factores = factores
        st.session_state.respuestas = respuestas
        st.session_state.respuesta = respuestas[0]
        if factores_cambiaron:
            st.session_state.niveles_confirmados_paso3 = False

        es_cod = _es_codificada(raw, factores)
        st.session_state.es_codificada = es_cod

        cols_mostrar = []
        if st.session_state.corrida_col:
            cols_mostrar.append(st.session_state.corrida_col)
        cols_mostrar.extend(factores)
        cols_mostrar.extend(respuestas)
        st.dataframe(raw[cols_mostrar], use_container_width=True)

        st.markdown(f"""
        **Formato detectado:** {'Matriz codificada (-1, 0, +1)' if es_cod else 'Matriz NO codificada (valores reales / cualitativos)'}
        """)

        st.markdown("---")
        nav_c = st.columns([1, 3])
        with nav_c[0]:
            if st.button("← Regresar", use_container_width=True):
                st.session_state.stage = 'upload'
                st.rerun()
        with nav_c[1]:
            if st.button("Continuar a niveles y codificacion →", type='primary', use_container_width=True):
                st.session_state.stage = 'encode'
                st.rerun()


# ---------- ETAPA 3: NIVELES, UNIDADES Y MATRIZ CODIFICADA ----------
elif st.session_state.stage == 'encode':
    st.title("3. Niveles Reales, Unidades y Matriz Codificada")

    raw = st.session_state.raw
    factores = st.session_state.factores
    respuestas = st.session_state.get('respuestas') or (
        [st.session_state.respuesta] if st.session_state.get('respuesta') else []
    )
    es_cod = st.session_state.es_codificada

    if es_cod:
        _info_box(
            "Paso 3A: Matriz codificada detectada",
            "Tu matriz ya esta en escala codificada (-1, 0, +1). "
            "Si conoces los valores reales bajos y altos, configuralos abajo y pulsa **Guardar**. "
            "Si estas resolviendo un ejercicio sin niveles reales definidos, puedes continuar directo.",
            tipo='info'
        )

        if st.button(
            "⏩ Continuar directo sin definir niveles reales (usar solo -1, 0, +1)",
            use_container_width=True,
            key="btn_saltar_niveles"
        ):
            df_cod = raw.copy()
            for f in factores:
                df_cod[f] = pd.to_numeric(df_cod[f], errors='coerce')
            st.session_state.df_cod = df_cod.copy()
            st.session_state.df_real = df_cod.copy()
            st.session_state.df_work = df_cod.copy()
            st.session_state.mapeos_niveles = {}
            st.session_state.mapeos_inversos = {}
            st.session_state.unidades_factores = {}
            st.session_state.niveles_confirmados_paso3 = True
            st.session_state.centros_confirmados = False
            st.session_state.filas_centro_idx = []
            st.session_state.stage = 'analysis_select'
            st.rerun()

        st.markdown("---")
        head_c1, head_c2 = st.columns([3, 1])
        with head_c1:
            st.markdown("### Opcional: Configurar niveles reales y unidades")
        with head_c2:
            if st.button(
                "🗑️ Borrar valores (volver al default)",
                use_container_width=True,
                help="Restablece los niveles bajos, altos y unidades a sus valores por defecto"
            ):
                _limpiar_niveles_y_widgets(factores)
                st.session_state.niveles_confirmados_paso3 = False
                st.rerun()

        with st.form("form_guardar_niveles"):
            inputs_niveles = {}
            for f in factores:
                guardado = st.session_state.niveles_guardados.get(
                    f, {'low': -1.0, 'high': 1.0, 'unit': ''}
                )
                st.markdown(f"**Factor: {f}**")
                c1, c2, c3 = st.columns([2, 2, 1.5])
                v_bajo = c1.number_input(
                    f"Valor real para -1 (bajo) - {f}",
                    value=float(guardado.get('low', -1.0)),
                    format="%.4f",
                    key=f"real_low_{f}"
                )
                v_alto = c2.number_input(
                    f"Valor real para +1 (alto) - {f}",
                    value=float(guardado.get('high', 1.0)),
                    format="%.4f",
                    key=f"real_high_{f}"
                )
                unidad = c3.text_input(
                    f"Unidad - {f}",
                    value=str(guardado.get('unit', '')),
                    placeholder="ej: ppm, °C, min, %",
                    key=f"unit_{f}"
                )
                inputs_niveles[f] = (v_bajo, v_alto, unidad)

            guardar_niv = st.form_submit_button("💾 Guardar", type='primary', use_container_width=True)

        if guardar_niv:
            mapeos_niveles = {}
            mapeos_inversos = {}
            unidades_factores = {}
            df_cod = raw.copy()
            df_real = raw.copy()

            for f in factores:
                v_bajo, v_alto, unidad = inputs_niveles[f]
                v_medio = (v_bajo + v_alto) / 2.0
                rango = (v_alto - v_bajo) / 2.0
                st.session_state.niveles_guardados[f] = {
                    'low': float(v_bajo),
                    'high': float(v_alto),
                    'unit': unidad.strip()
                }
                mapeos_niveles[f] = {v_bajo: -1, v_medio: 0, v_alto: 1}
                mapeos_inversos[f] = {-1: v_bajo, 0: v_medio, 1: v_alto}
                unidades_factores[f] = unidad.strip()

                serie_cod = pd.to_numeric(raw[f], errors='coerce')
                df_cod[f] = serie_cod
                df_real[f] = serie_cod * rango + v_medio if rango != 0 else serie_cod

            st.session_state.mapeos_niveles = mapeos_niveles
            st.session_state.mapeos_inversos = mapeos_inversos
            st.session_state.unidades_factores = unidades_factores
            st.session_state.df_cod = df_cod.copy()
            st.session_state.df_real = df_real.copy()
            st.session_state.df_work = df_cod.copy()
            st.session_state.niveles_confirmados_paso3 = True
            st.rerun()

        if st.session_state.get('niveles_confirmados_paso3') and st.session_state.df_cod is not None:
            st.success("Niveles guardados correctamente.")
            df_cod = st.session_state.df_cod
            df_real = st.session_state.df_real

            st.markdown("---")
            st.markdown("### Paso 3B: Matriz Codificada (-1, 0, +1) para el Analisis")
            cols_ver = ([st.session_state.corrida_col] if st.session_state.get('corrida_col') else []) + factores + respuestas
            cols_ver = [c for c in cols_ver if c in df_cod.columns]
            st.dataframe(df_cod[cols_ver], use_container_width=True)

            st.markdown("---")
            nav_c = st.columns([1, 3])
            with nav_c[0]:
                if st.button("← Regresar a columnas", use_container_width=True):
                    st.session_state.stage = 'preview'
                    st.rerun()
            with nav_c[1]:
                if st.button("Continuar a puntos centrales y analisis →", type='primary', use_container_width=True):
                    st.session_state.df_work = df_cod.copy()
                    st.session_state.centros_confirmados = False
                    st.session_state.filas_centro_idx = []
                    st.session_state.stage = 'analysis_select'
                    st.rerun()
        else:
            st.markdown("---")
            if st.button("← Regresar a columnas", use_container_width=False):
                st.session_state.stage = 'preview'
                st.rerun()

    else:
        _info_box(
            "Matriz NO codificada detectada",
            "Tu archivo tiene valores reales. Indica/confirma los valores bajo (-1), alto (+1) y sus unidades para generar la matriz codificada (-1, 0, +1), "
            "que sera la matriz utilizada para buscar puntos centrales y ejecutar el analisis.",
            tipo='info'
        )

        mapeos_niveles = {}
        mapeos_inversos = {}
        unidades_factores = {}

        head_c1, head_c2 = st.columns([3, 1])
        with head_c1:
            st.markdown("### Configurar niveles reales y unidades")
        with head_c2:
            if st.button(
                "🗑️ Borrar valores (volver al default)",
                use_container_width=True,
                help="Restablece los niveles bajos, altos y unidades a sus valores por defecto"
            ):
                _limpiar_niveles_y_widgets(factores)
                st.rerun()

        for f in factores:
            vals = sorted(pd.to_numeric(raw[f], errors='coerce').dropna().unique().tolist())
            st.markdown(f"**Factor: {f}**")

            default_low = float(min(vals)) if len(vals) > 0 else 0.0
            default_high = float(max(vals)) if len(vals) > 0 else 100.0
            guardado = st.session_state.niveles_guardados.get(
                f, {'low': default_low, 'high': default_high, 'unit': ''}
            )
            c1, c2, c3 = st.columns([2, 2, 1.5])
            v_bajo = c1.number_input(
                f"Nivel BAJO (-1) para {f}",
                value=float(guardado.get('low', default_low)),
                format="%.4f",
                key=f"low_{f}"
            )
            v_alto = c2.number_input(
                f"Nivel ALTO (+1) para {f}",
                value=float(guardado.get('high', default_high)),
                format="%.4f",
                key=f"high_{f}"
            )
            unidad = c3.text_input(
                f"Unidad - {f}",
                value=str(guardado.get('unit', '')),
                placeholder="ej: ppm, °C, min, %",
                key=f"unit_{f}"
            )

            st.session_state.niveles_guardados[f] = {
                'low': float(v_bajo),
                'high': float(v_alto),
                'unit': unidad.strip()
            }
            v_medio = (v_bajo + v_alto) / 2.0
            suf_u = f" {unidad.strip()}" if unidad.strip() else ""
            st.caption(f"Centro (0) calculado: **{v_medio:.4f}{suf_u}**")

            mapeos_niveles[f] = {v_bajo: -1, v_medio: 0, v_alto: 1}
            mapeos_inversos[f] = {-1: v_bajo, 0: v_medio, 1: v_alto}
            unidades_factores[f] = unidad.strip()

        st.session_state.mapeos_niveles = mapeos_niveles
        st.session_state.mapeos_inversos = mapeos_inversos
        st.session_state.unidades_factores = unidades_factores

        df_cod = raw.copy()
        df_real = raw.copy()
        for f in factores:
            df_cod[f] = _codificar_factor(raw[f], mapeos_inversos[f][-1], mapeos_inversos[f][1]).round(6)
            df_real[f] = pd.to_numeric(raw[f], errors='coerce')

        st.markdown("---")
        st.markdown("### Matriz Codificada (-1, 0, +1) para el Analisis")
        cols_ver = ([st.session_state.corrida_col] if st.session_state.get('corrida_col') else []) + factores + respuestas
        cols_ver = [c for c in cols_ver if c in df_cod.columns]
        st.dataframe(df_cod[cols_ver], use_container_width=True)

        resumen_niveles = pd.DataFrame([
            {
                'Factor': f,
                'Nivel Bajo (-1)': mapeos_inversos[f][-1],
                'Centro (0)': mapeos_inversos[f][0],
                'Nivel Alto (+1)': mapeos_inversos[f][1],
                'Unidad': unidades_factores[f] if unidades_factores[f] else '—'
            }
            for f in factores
        ])
        with st.expander("📋 Ver resumen de niveles reales y unidades guardadas"):
            st.dataframe(resumen_niveles, use_container_width=True)

        st.markdown("---")
        nav_c = st.columns([1, 3])
        with nav_c[0]:
            if st.button("← Regresar a columnas", use_container_width=True):
                st.session_state.df_cod = df_cod.copy()
                st.session_state.df_real = df_real.copy()
                st.session_state.stage = 'preview'
                st.rerun()
        with nav_c[1]:
            if st.button("Continuar a puntos centrales y analisis →", type='primary', use_container_width=True):
                st.session_state.df_cod = df_cod.copy()
                st.session_state.df_real = df_real.copy()
                st.session_state.df_work = df_cod.copy()
                st.session_state.centros_confirmados = False
                st.session_state.filas_centro_idx = []
                st.session_state.stage = 'analysis_select'
                st.rerun()

# ---------- ETAPA 4: PUNTOS CENTRALES Y SELECCION DE ANALISIS ----------
elif st.session_state.stage == 'analysis_select':
    st.title("4. Puntos Centrales y Tipo de Analisis")

    df_cod = st.session_state.df_cod if st.session_state.df_cod is not None else st.session_state.df_work
    factores = st.session_state.factores
    respuestas = st.session_state.respuestas if st.session_state.respuestas else ([st.session_state.respuesta] if st.session_state.respuesta else [])
    respuesta = respuestas[0]
    k = len(factores)

    _info_box("Paso 4: Deteccion sobre matriz codificada",
              "Partiendo de tu matriz codificada guardada, el programa detecta los puntos centrales, las replicas y sugiere el mejor analisis.",
              tipo='info')

    # Busqueda de puntos centrales directamente sobre la matriz codificada
    n_centros_detect, idx_centros = _detectar_centros(df_cod, factores)
    if st.session_state.get('centros_confirmados'):
        idx_centros = st.session_state.filas_centro_idx
        n_centros_detect = len(idx_centros)
    elif n_centros_detect > 0:
        st.session_state.filas_centro_idx = idx_centros

    replicas = _detectar_replicas(df_cod, factores, respuestas)
    esq = len(df_cod) - n_centros_detect

    st.markdown("### ¿Que detecto el programa en la matriz codificada?")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Factores", k)
    with col2:
        st.metric("Esquinas", esq)
    with col3:
        st.metric("Puntos centrales", n_centros_detect, delta="(filas con todos ~ 0)")
    with col4:
        delta_rep = f"({len(respuestas)} columnas de replica)" if len(respuestas) >= 2 else "(filas identicas en factores)"
        st.metric("Replicas puras", replicas, delta=delta_rep)

    # Confirmacion de puntos centrales
    if n_centros_detect > 0:
        filas_str = ', '.join(str(i + 1) for i in idx_centros)
        st.info(f"Detecte **{n_centros_detect}** punto(s) central(es) en la(s) fila(s): **{filas_str}**")
        correcto = st.radio("¿Es correcto?", ["Si, es correcto", "No, quiero corregir"], horizontal=True, key="confirm_centros")
        if correcto == "Si, es correcto":
            st.session_state.filas_centro_idx = idx_centros
            st.session_state.centros_confirmados = True
        else:
            st.session_state.centros_confirmados = False
            st.markdown("Escribe los numeros de fila separados por coma (empezando desde 1):")
            filas_centro_input = st.text_input("Filas de centro", placeholder="ej: 3,6,11", key="filas_centro_manual")
            if filas_centro_input:
                try:
                    filas_idx = [int(x.strip()) - 1 for x in filas_centro_input.split(",")]
                    validas = [i for i in filas_idx if 0 <= i < len(df_cod)]
                    if len(validas) != len(filas_idx):
                        st.warning(f"Algunas filas estan fuera de rango (max: {len(df_cod)}). Se usaran solo las validas.")
                    if validas:
                        st.session_state.filas_centro_idx = validas
                        st.session_state.centros_confirmados = True
                        st.success(f"Marcadas como centros las filas: {', '.join(str(i + 1) for i in validas)}")
                except ValueError:
                    st.error("Formato invalido. Usa numeros separados por coma.")
    else:
        st.info("No detecte puntos centrales (filas donde todos los factores estan en 0).")
        correcto = st.radio("¿Tu matriz SI tiene puntos centrales?", ["No, no tiene", "Si, pero no los detecto"], horizontal=True, key="confirm_no_centros")
        if correcto == "Si, pero no los detecto":
            st.markdown("Escribe los numeros de fila separados por coma (empezando desde 1):")
            filas_centro_input = st.text_input("Filas de centro", placeholder="ej: 3,6,11", key="filas_centro_manual2")
            if filas_centro_input:
                try:
                    filas_idx = [int(x.strip()) - 1 for x in filas_centro_input.split(",")]
                    validas = [i for i in filas_idx if 0 <= i < len(df_cod)]
                    if validas:
                        st.session_state.filas_centro_idx = validas
                        st.session_state.centros_confirmados = True
                        st.success(f"Marcadas como centros las filas: {', '.join(str(i + 1) for i in validas)}")
                except ValueError:
                    st.error("Formato invalido. Usa numeros separados por coma.")
        else:
            st.session_state.centros_confirmados = True
            st.session_state.filas_centro_idx = []

    with st.expander("💡 ¿Que son los puntos centrales?"):
        st.markdown("""
        Los **puntos centrales** son corridas experimentales donde **todos los factores están en su valor intermedio** (nivel 0 en codificacion).

        **Ejemplo con 3 factores (A, B, C):**
        - Esquinas del diseno: A=-1, B=-1, C=-1  |  A=+1, B=-1, C=-1  |  etc.
        - **Punto central:** A=**0**, B=**0**, C=**0**

        **Para que sirven:**
        - Probar si existe **curvatura** en la superficie de respuesta
        - Estimar el **error puro** del experimento
        - Son la base del **Analisis Via 2**
        """)

    centros = len(st.session_state.get('filas_centro_idx', []))
    sug_analisis, sug_msg = _sugerir_analisis(df_cod, factores, respuestas, n_respuestas=len(respuestas))
    _info_box("Sugerencia del programa", sug_msg, tipo='tip')

    st.markdown("---")
    st.markdown("### Tipo de analisis")

    with st.expander("📖 Guia rapida: ¿Cual elegir?", expanded=True):
        st.markdown("""
        | Si tu diseno tiene... | Elige... |
        |---|---|
        | Solo esquinas (sin centros, sin replicas) | **Via 3** (Lenth) |
        | Esquinas + puntos centrales (filas con todos los factores = 0) | **Via 2** |
        | Esquinas + replicas completas del diseno | **Via 1** |
        | Esquinas + puntos axiales (valores como ±1.414) + centros | **CCD** |

        **Ejemplo de punto central:** Si tus factores son Temperatura (50, 125, 200) y pH (5, 7.5, 10),
        una fila con Temp=125 y pH=7.5 equivale en codificada a **Temp=0, pH=0** (punto central).
        """)

    opciones = {
        'factorial_via1': f'Via 1: 2^{k} con replicas puras',
        'factorial_via2': f'Via 2: 2^{k} con puntos centrales ({centros} detectados)',
        'factorial_via3': f'Via 3: 2^{k} no replicado (Lenth)',
        'ccd': 'Superficie de Respuesta (CCD)',
    }

    keys_op = list(opciones.keys())
    default_op = st.session_state.analisis_tipo if st.session_state.analisis_tipo in keys_op else sug_analisis
    idx_op = keys_op.index(default_op)
    analisis = st.radio("Selecciona tu analisis:", keys_op,
                        index=idx_op,
                        format_func=lambda x: opciones[x], key="sel_analisis")
    st.session_state.analisis_tipo = analisis

    explicaciones = {
        'factorial_via1': "**Via 1:** Analisis para disenos con **replicas completas**. Cada condicion experimental se repite varias veces.",
        'factorial_via2': f"**Via 2:** Analisis para disenos con **puntos centrales** ({centros} detectados en tu archivo). Permite probar curvatura.",
        'factorial_via3': "**Via 3:** Analisis para disenos **sin replicas ni centros**. Usa el metodo de Lenth para detectar efectos significativos.",
        'ccd': "**CCD:** Analisis de Superficie de Respuesta. Permite ajustar terminos cuadraticos (A²), superficies 3D, contornos individuales, contorno combinado y optimizacion multi-respuesta.",
    }
    st.markdown(explicaciones.get(analisis, ""))

    resp_ccd = list(respuestas)
    resp_fact_sel = list(respuestas)
    if analisis == 'ccd':
        raw_orig = st.session_state.raw
        cands_resp_ccd = [
            c for c in raw_orig.columns
            if c not in factores and c != st.session_state.get('corrida_col')
            and pd.api.types.is_numeric_dtype(raw_orig[c])
        ]
        for r_u in respuestas:
            if r_u not in cands_resp_ccd:
                cands_resp_ccd.insert(0, r_u)
        resp_ccd = st.multiselect(
            "Respuestas para CCD (puedes seleccionar 1 o varias para contorno combinado y optimizacion):",
            options=cands_resp_ccd,
            default=respuestas,
            key="sel_resp_ccd"
        )

    elif len(respuestas) > 1:
        es_bloque_o_rep = any(
            any(h in str(c).lower() for h in ('block', 'bloque', 'rep', 'replica', 'ensayo'))
            for c in respuestas
        )
        idx_multi_def = 1 if es_bloque_o_rep else 0
        modo_multi_fact = st.radio(
            f"Seleccionaste {len(respuestas)} columnas de respuesta ({', '.join(respuestas)}). ¿Como usarlas en Factorial?",
            ["Analizar una sola respuesta", "Unirlas como bloques / replicas de una misma respuesta"],
            index=idx_multi_def,
            horizontal=True,
            key="modo_multi_fact"
        )
        if modo_multi_fact == "Analizar una sola respuesta":
            resp_unica = st.selectbox("Elige la respuesta a analizar:", respuestas, key="resp_fact_unica")
            resp_fact_sel = [resp_unica]

    resp_para_validar = resp_fact_sel if analisis.startswith('factorial') else respuestas
    warnings = _validar_analisis_vs_diseno(df_cod, factores, resp_para_validar, analisis)
    if warnings:
        for w in warnings:
            _info_box("Advertencia", w, tipo='warning')
            if "centrales" in w.lower():
                st.markdown("""
                **Como solucionarlo:**
                1. Vuelve a tu Excel
                2. Añade 3-5 filas donde todos los factores tengan su **valor medio**
                3. Sube el archivo de nuevo
                4. Selecciona **Via 2**
                """)

    # DISEÑO FRACCIONADO - GUIA INTERACTIVA
    if analisis.startswith('factorial') and k >= 4:
        st.markdown("---")
        st.markdown("### 🔀 ¿Necesitas un diseño fraccionado?")

        with st.expander("📖 ¿Que es un diseño fraccionado y cuando usarlo?"):
            st.markdown(f"""
            Un **diseño fraccionado** te permite estudiar **{k} factores** con **menos experimentos** que un diseno completo.

            **Diseno completo:** 2^{k} = **{2**k} esquinas** (muchas corridas)
            **Diseno fraccionado:** 2^(k-p) = menos esquinas

            | Valor de p | Que significa | Esquinas | Resolucion | Cuando usarlo |
            |---|---|---|---|---|
            | **p=1** (media fraccion) | 2^({k}-1) = 2^{k-1} | **{2**(k-1)}** | {k} | Recomendado. Los principales no se confunden entre si. |
            | **p=2** (cuarto) | 2^({k}-2) = 2^{k-2} | **{2**(k-2)}** | {k-1} | Solo si necesitas MUY pocas corridas. |
            | **p=3** (octavo) | 2^({k}-3) = 2^{k-3} | **{2**(k-3)}** | {k-2} | No recomendado. Mucha confusion. |

            **Regla de oro:** Usa **p=1** (media fraccion) a menos que tengas una buena razon para usar otra.
            """)

        es_fracc = st.checkbox("Si, mi diseno es fraccionado", value=st.session_state.modelo_fraccionado)
        st.session_state.modelo_fraccionado = es_fracc

        if es_fracc:
            p = st.selectbox("Fraccion (p) - Recomendado: p=1:", [1, 2, 3], index=0,
                             help="p=1 = media fraccion (recomendado). p=2 = cuarto de fraccion.")
            resolucion = k - p + 1
            esquinas_frac = 2 ** (k - p)

            st.info(f"**Diseno 2^({k}-{p}) = 2^{k-p} = {esquinas_frac} esquinas | Resolucion: {resolucion}**")

            if resolucion >= 5:
                _info_box("Excelente", f"Resolucion {resolucion}: Los efectos principales y las interacciones de 2 vias NO estan confundidas entre si.", tipo='success')
            elif resolucion == 4:
                _info_box("Bueno", f"Resolucion {resolucion}: Los efectos principales estan libres de confusion, pero las interacciones de 2 vias pueden estar confundidas entre si.", tipo='tip')
            elif resolucion == 3:
                _info_box("Aceptable", f"Resolucion {resolucion}: Los efectos principales estan libres entre si, pero pueden estar confundidos con interacciones de 2 vias.", tipo='warning')
            else:
                _info_box("Precaucion", f"Resolucion {resolucion}: Los efectos principales estan confundidos entre si. No se recomienda.", tipo='warning')

            st.session_state.resolucion = resolucion

    st.markdown("---")
    nav_c = st.columns([1, 3])
    with nav_c[0]:
        if st.button("← Regresar a niveles y codificacion", use_container_width=True):
            st.session_state.stage = 'encode'
            st.rerun()
    with nav_c[1]:
        if st.button("Continuar al analisis →", type='primary', use_container_width=True):
            if warnings and any("requiere" in w.lower() for w in warnings):
                st.error("Corrige las advertencias antes de continuar, o elige un analisis diferente.")
            elif analisis == 'ccd':
                if not resp_ccd:
                    st.error("Selecciona al menos una respuesta para CCD.")
                else:
                    df_base_ccd = (st.session_state.df_real
                                   if st.session_state.df_real is not None
                                   else df_cod).copy()
                    for f in factores:
                        df_base_ccd[f] = _to_numeric_clean(df_base_ccd[f])
                    for r_c in resp_ccd:
                        df_base_ccd[r_c] = _to_numeric_clean(df_base_ccd[r_c])
                    st.session_state.df_ccd_raw = df_base_ccd
                    st.session_state.factores_ccd = list(factores)
                    st.session_state.respuesta_ccd = list(resp_ccd)
                    st.session_state.R_ccd_full = None
                    st.session_state.R_ccd = None
                    st.session_state.opt_ccd = None
                    st.session_state.stage = 'ccd_full'
                    st.rerun()
            else:
                st.session_state.respuestas_fact_activas = resp_fact_sel
                st.session_state.respuesta = resp_fact_sel[0]
                st.session_state.stage = 'results'
                st.rerun()


# ---------- ETAPA 5: RESULTADOS (SOLO FACTORIAL VIAS 1, 2 Y 3) ----------
elif st.session_state.stage == 'results':
    st.title("5. Resultados del Analisis")

    raw_cod = st.session_state.df_cod if st.session_state.df_cod is not None else st.session_state.df_work
    factores = st.session_state.factores
    resp_activas = st.session_state.get('respuestas_fact_activas') or st.session_state.respuestas or [st.session_state.respuesta]
    respuesta = resp_activas[0]
    analisis = st.session_state.analisis_tipo
    unidades = st.session_state.get('unidades_factores', {})

    if analisis == 'ccd':
        st.session_state.stage = 'ccd_full'
        st.rerun()

    centros_idx = set(st.session_state.get('filas_centro_idx', []))

    # Asegurar que Factorial trabaje SIEMPRE sobre la matriz codificada (-1, 0, +1)
    if len(resp_activas) > 1:
        df = ce.preparar_datos(raw_cod, factores, resp_activas)
        df['Y'] = _to_numeric_clean(df['Y'])
        if centros_idx:
            curv_orig = [1 if i in centros_idx else 0 for i in raw_cod.index]
            df['Curvatura'] = curv_orig * len(resp_activas)
        else:
            df['Curvatura'] = df[factores].apply(lambda r: 1 if all(abs(v) < 0.01 for v in r) else 0, axis=1)
        st.info(f"Analizando {len(resp_activas)} columnas como replicas/bloques: {', '.join(resp_activas)}")
    else:
        df = raw_cod.copy()
        df = df.rename(columns={respuesta: 'Y'})
        df['Grupo'] = 'G1'
        for f in factores:
            df[f] = _to_numeric_clean(df[f])
        df['Y'] = _to_numeric_clean(df['Y'])

        if centros_idx:
            df['Curvatura'] = [1 if i in centros_idx else 0 for i in df.index]
        else:
            df['Curvatura'] = df[factores].apply(lambda r: 1 if all(abs(v) < 0.01 for v in r) else 0, axis=1)

    st.markdown("---")
    st.markdown("### Configuracion del Modelo")

    modos_opts = ["Automatico (Backward)", "Automatico (Forward)", "Modelo completo (2 vias)", "Manual"]
    mapa_modo_idx = {'backward': 0, 'forward': 1, 'all': 2, 'manual': 3}
    idx_modo = mapa_modo_idx.get(st.session_state.modo_seleccion, 0)

    c1, c2 = st.columns(2)
    with c1:
        modo = st.radio("Seleccion de terminos:", modos_opts, index=idx_modo, key="modo_sel")
        if 'Backward' in modo:
            st.session_state.modo_seleccion = 'backward'
        elif 'Forward' in modo:
            st.session_state.modo_seleccion = 'forward'
        elif 'completo' in modo:
            st.session_state.modo_seleccion = 'all'
        else:
            st.session_state.modo_seleccion = 'manual'
    with c2:
        alpha = st.number_input("Nivel de significancia (alpha):", min_value=0.01, max_value=0.5,
                                value=float(st.session_state.alpha_modelo), step=0.01, format="%.2f", key="alpha_val")
        st.session_state.alpha_modelo = float(alpha)

    _info_box("Que es alpha?",
              f"alpha = {alpha} significa que aceptamos un {alpha*100:.0f}% de probabilidad de equivocarnos al declarar un efecto significativo.",
              tipo='tip')
    with st.expander("📖 ¿Cómo elige los términos el programa? (Backward, Forward y Jerarquía)"):
        st.markdown("""
        - **Backward (recomendado / estilo Minitab):** Empieza con todos los factores principales y sus interacciones dobles, y elimina uno por uno los menos significativos (`p > alpha`).
        - **Forward:** Empieza solo con los factores principales y agrega únicamente las interacciones con `p < alpha`.
        - **Principio de Jerarquía:** Si una interacción (ej. `A*C`) resulta significativa, el programa conserva automáticamente `A` y `C` en el modelo aunque alguno por separado tuviera `p > alpha`.
        """)

    R = None
    try:
        if analisis == 'factorial_via1':
            if st.session_state.modo_seleccion == 'manual':
                ef_all = ce._calc_effects(df, factores, len(df[df['Curvatura'] == 0]))
                all_terms = [t for t in ef_all.keys() if len(t.split('*')) <= 2]
                st.markdown("**Selecciona los terminos:**")
                cols_manual = st.columns(3)
                selected = []
                for i, t in enumerate(all_terms):
                    if cols_manual[i % 3].checkbox(t, value=True, key=f"term_{t}"):
                        selected.append(t)
                if len(selected) == 0:
                    st.warning("Selecciona al menos un termino.")
                    st.stop()
                R = ce.calcular_via1_con_terminos(df, factores, terms=selected)
            else:
                R = ce.calcular_via1_con_terminos(df, factores, terms=None)

        elif analisis == 'factorial_via2':
            if st.session_state.modo_seleccion == 'all':
                R = ce.calcular_via2_con_terminos(df, factores, terms='all', alpha=alpha)
            elif st.session_state.modo_seleccion in ('backward', 'forward'):
                R = ce.calcular_via2_con_terminos(df, factores, terms=None, alpha=alpha, modo=st.session_state.modo_seleccion)
            else:
                ef_all = ce._calc_effects(df, factores, len(df[df['Curvatura'] == 0]))
                all_terms = [t for t in ef_all.keys() if len(t.split('*')) <= 2]
                st.markdown("**Selecciona los terminos:**")
                cols_manual = st.columns(3)
                selected = []
                for i, t in enumerate(all_terms):
                    if cols_manual[i % 3].checkbox(t, value=True, key=f"term_{t}"):
                        selected.append(t)
                if len(selected) == 0:
                    st.warning("Selecciona al menos un termino.")
                    st.stop()
                R = ce.calcular_via2_con_terminos(df, factores, terms=selected, alpha=alpha)

        elif analisis == 'factorial_via3':
            R = ce.calcular_via3(df, factores)

        st.session_state.R = R

    except Exception as e:
        st.error(f"Error en el analisis: {e}")
        import traceback
        st.code(traceback.format_exc())
        st.stop()

    if R is not None:
        st.markdown("---")
        st.markdown("## Resumen del Modelo")

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("S (Error estandar)", f"{np.sqrt(R['MS_resid']):.7f}" if pd.notna(R.get('MS_resid')) else "-")
        col2.metric("R2", f"{R['R2']*100:.2f}%")
        col3.metric("R2 ajustado", f"{R['R2_adj']*100:.2f}%" if R.get('R2_adj') is not None else "-")
        col4.metric("R2 predicho", f"{R['R2_pred']*100:.2f}%" if pd.notna(R.get('R2_pred')) else "-")
        col5.metric("Terminos en modelo", len(R['terms_included']))

        _info_box("Que significa R2?",
                  f"R2 = {R['R2']*100:.2f}% indica que el modelo explica el {R['R2']*100:.2f}% de la variacion en la respuesta.",
                  tipo='tip')
        with st.expander("📖 ¿Cómo saber si el modelo es bueno? (S, R², R² ajustado y R² predicho)"):
            st.markdown("""
            - **S (Error estándar):** Desviación promedio entre los datos reales y lo que predice la ecuación (en las mismas unidades de `Y`). Mientras **más pequeño**, mejor.
            - **R²:** Porcentaje de la variación de `Y` explicada por el modelo. Siempre sube al agregar términos, incluso si no sirven.
            - **R² ajustado:** Corrige el `R²` penalizando términos innecesarios. Si agregas ruido, baja.
            - **R² predicho:** Mide qué tan bien predice el modelo **experimentos nuevos** (validación cruzada PRESS). Si `R²` y `R² predicho` están separados por más de **20%**, el modelo está sobreajustado.
            """)

        # ANOVA
        st.markdown("---")
        st.markdown("## Tabla ANOVA")
        _info_box("Analisis de Varianza",
                  "Descompone la variacion total en las fuentes que la explican. p < alpha = significativo.",
                  tipo='info')

        anova = R['anova'].copy()
        for col in ['SS', 'MS', 'F', 'F_crit']:
            if col in anova.columns:
                anova[col] = anova[col].apply(lambda x: f"{x:.5f}" if pd.notna(x) else "-")
        for col in ['p']:
            if col in anova.columns:
                anova[col] = anova[col].apply(lambda x: f"{x:.5f}" if pd.notna(x) else "-")

        st.dataframe(anova, use_container_width=True)

        with st.expander("📖 Guía para leer el ANOVA: Valor p, Curvatura y Lack of Fit (Falta de Ajuste)"):
            st.markdown("""
            - **Valor p (`< alpha`):** Si `p < 0.05` (o tu `alpha`), el factor o interacción tiene un efecto estadísticamente real sobre `Y`.
            - **Curvatura (`Ct Pt`):** Compara las esquinas frente a los puntos centrales. Si `p < 0.05`, la respuesta no es plana (existe curvatura) y conviene continuar con un diseño **CCD**.
            - **Lack of Fit (Falta de Ajuste) vs. Pure Error (Error Puro):**
              - **Pure Error:** Es la variación natural que ocurre al repetir exactamente la misma corrida en el laboratorio (por ejemplo, los puntos centrales).
              - **Lack of Fit:** Mide cuánto se desvía tu ecuación de los promedios reales.
              - ✅ **Si `p > 0.05` (No significativo):** ¡Excelente! Tu modelo ajusta bien los datos; los errores restantes son solo variabilidad normal del experimento.
              - ⚠️ **Si `p < 0.05` (Significativo):** El modelo no logra capturar toda la tendencia real (puede faltar una interacción o términos cuadráticos `A²`).
            """)

        # COEFICIENTES
        st.markdown("---")
        st.markdown("## Coeficientes del Modelo")
        try:
            coef_df, fit_df = ce.generar_tabla_regresion(df, R)
            st.dataframe(coef_df.round(5), use_container_width=True)
        except Exception as e:
            st.warning(f"No se pudieron generar coeficientes: {e}")

        # ECUACION
        st.markdown("---")
        st.markdown("## Ecuacion de Regresion")
        try:
            eq_cod = ce._ecuacion_str(R)
            st.markdown("**En unidades codificadas:**")
            st.code(eq_cod, language='text')

            eq_real = ce.ecuacion_no_codificada(R, st.session_state.mapeos_inversos, unidades=unidades)
            if eq_real:
                st.markdown("**En unidades reales (no codificadas, con tus unidades guardadas):**")
                st.code(eq_real, language='text')
            else:
                st.caption("Modo codificado (-1, 0, +1): no se definieron niveles reales numericos para traducir la ecuacion.")

            if hasattr(st, "copy_button"):
                st.copy_button("Copiar ecuacion", eq_real or eq_cod, key="copy_eq")
        except Exception as e:
            st.warning(f"No se pudo generar ecuacion: {e}")

        # RANKING DE EFECTOS Y CORRIDAS
        st.markdown("---")
        st.markdown("## Ranking de Efectos y Corridas")
        with st.expander("📊 Ver Ranking de Efectos (Balance Neto)", expanded=False):
            try:
                ranking = ce.generar_tabla_balance_neto(R)
                st.dataframe(ranking, use_container_width=True)
            except Exception as e:
                st.warning(f"No se pudo generar ranking: {e}")

        with st.expander("🏆 Ranking de Condiciones Observadas (Real vs. Ecuación)", expanded=False):
            st.markdown(
                "Compara las combinaciones experimentales que dieron mejores resultados reales frente a lo que predice tu ecuación. "
                "*(Si hay réplicas o centros repetidos, se muestra su promedio).*"
            )
            rk_c1, rk_c2 = st.columns([2, 1])
            rk_obj = rk_c1.radio(
                "Ordenar por:",
                ["Mayor respuesta (Maximizar Y)", "Menor respuesta (Minimizar Y)"],
                horizontal=True,
                key="rk_cond_obj"
            )
            rk_top = rk_c2.number_input("Mostrar top:", min_value=3, max_value=max(5, len(df)), value=5, step=1, key="rk_cond_top")

            try:
                cond_medias = df.groupby(factores, as_index=False).agg(
                    Y=('Y', 'mean'),
                    Replicas=('Y', 'count')
                )
                asc = rk_obj.startswith("Menor")
                cond_medias = cond_medias.sort_values('Y', ascending=asc).head(int(rk_top)).reset_index(drop=True)

                corr_col = st.session_state.get('corrida_col')
                ranking_data = []
                for idx, row in cond_medias.iterrows():
                    mask = pd.Series(True, index=raw_cod.index)
                    for f in factores:
                        mask &= np.isclose(_to_numeric_clean(raw_cod[f]), float(row[f]), atol=1e-4)
                    if corr_col and corr_col in raw_cod.columns:
                        corr_vals = [str(v) for v in raw_cod.loc[mask, corr_col].tolist()]
                    else:
                        corr_vals = [str(i + 1) for i in raw_cod.index[mask].tolist()]

                    fila = {
                        'Posición': idx + 1,
                        'Corrida(s)': ', '.join(corr_vals) if corr_vals else '-',
                        'Réplicas': int(row['Replicas'])
                    }

                    niveles_cod = {}
                    for f in factores:
                        val = float(row[f])
                        niveles_cod[f] = val
                        u_f = unidades.get(f, '').strip()
                        suf_u = f" {u_f}" if u_f else ""
                        m_inv = st.session_state.mapeos_inversos.get(f, {})

                        if np.isclose(val, -1.0, atol=1e-3):
                            fila[f] = f"{_fmt_nivel(m_inv[-1])}{suf_u} (-1)" if -1 in m_inv else "Bajo (-1)"
                        elif np.isclose(val, 1.0, atol=1e-3):
                            fila[f] = f"{_fmt_nivel(m_inv[1])}{suf_u} (+1)" if 1 in m_inv else "Alto (+1)"
                        elif abs(val) < 1e-3:
                            fila[f] = f"{_fmt_nivel(m_inv[0])}{suf_u} (0)" if 0 in m_inv else "Centro (0)"
                        else:
                            fila[f] = f"{val:+.3f} (cod)"

                    y_real = float(row['Y'])
                    y_pred_row = float(ce.predecir_y_combinacion(niveles_cod, R))
                    diff = y_real - y_pred_row
                    fila['Media Real (Y)'] = round(y_real, 4)
                    fila['Predicción (Ecuación)'] = round(y_pred_row, 4)
                    fila['Diferencia (Residuo)'] = round(diff, 4)
                    ranking_data.append(fila)

                st.dataframe(pd.DataFrame(ranking_data), use_container_width=True)
                st.caption("💡 **Diferencia (Residuo) pequeña** indica que la ecuación predice esa condición experimental con alta precisión.")
            except Exception as e:
                st.warning(f"No se pudo construir el ranking de condiciones: {e}")

        # GRAFICOS
        st.markdown("---")
        st.markdown("## Graficos Diagnostico")

        spec = None
        with st.expander("🎯 Evaluar cumplimiento de especificacion (Objetivo ± Tolerancia)", expanded=False):
            usar_spec = st.checkbox(
                "Activar especificacion (Target ± Tolerancia)",
                value=False,
                key="usar_spec_factorial"
            )
            if usar_spec:
                sc1, sc2 = st.columns(2)
                target_val = sc1.number_input(
                    "Objetivo (Target):",
                    value=float(round(df['Y'].mean(), 2)),
                    format="%.4g",
                    key="spec_target"
                )
                tol_val = sc2.number_input(
                    "Tolerancia (±):",
                    value=1.0,
                    min_value=0.0,
                    format="%.4g",
                    key="spec_tol"
                )
                spec = {'target': target_val, 'tol': tol_val}

        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "Pareto", "Semi-normal", "Efectos Principales", "Interacciones",
            "Superficie 3D y Contorno", "Residuales", "Especificacion"
        ])

        with tab1:
            try:
                fn = ce._gen_pareto(R)
                st.image(fn)
                _info_box("Diagrama de Pareto", "Efectos ordenados por magnitud. La linea vertical es el valor critico.", tipo='tip')
            except Exception as e:
                st.warning(f"No se pudo generar Pareto: {e}")

        with tab2:
            try:
                fn = ce._gen_seminormal(R)
                st.image(fn)
                _info_box("Grafico Semi-normal", "Efectos significativos se desvian de la linea recta.", tipo='tip')
            except Exception as e:
                st.warning(f"No se pudo generar semi-normal: {e}")

        with tab3:
            try:
                fn = ce._gen_main_effects(R, spec=spec, mapeos=st.session_state.mapeos_inversos, unidades=unidades)
                st.image(fn)
            except Exception as e:
                st.warning(f"No se pudieron generar efectos principales: {e}")

        with tab4:
            try:
                ints_modelo = [n for n in R['effects'] if len(n.split('*')) == 2]
                eje_x_sel = {}

                if ints_modelo:
                    st.markdown("**Elige que factor quieres mostrar en el eje X de cada grafica:**")
                    cols_ejex = st.columns(min(len(ints_modelo), 3))
                    for idx_int, inter in enumerate(ints_modelo):
                        fa, fb = inter.split('*')
                        eje_x_sel[inter] = cols_ejex[idx_int % len(cols_ejex)].radio(
                            f"Eje X para `{inter}`:",
                            options=[fa, fb],
                            index=0,
                            horizontal=True,
                            key=f"ejex_{inter}"
                        )

                fig_list = ce._gen_interactions(
                    R,
                    mapeos=st.session_state.mapeos_inversos,
                    unidades=unidades,
                    eje_x_por_interaccion=eje_x_sel
                )
                if not fig_list:
                    st.info("El modelo actual no tiene interacciones de 2 factores.")
                else:
                    for fn in fig_list:
                        st.image(fn)
                _info_box("Graficos de interaccion",
                          "Muestran como el efecto de un factor cambia segun el nivel del otro. "
                          "Lineas paralelas = sin interaccion. Lineas que se cruzan = interaccion significativa.",
                          tipo='tip')
            except Exception as e:
                st.warning(f"No se pudieron generar interacciones: {e}")

        with tab5:
            if R['via'] == ce.VIA_3:
                st.info("Los graficos de superficie 3D y contorno requieren Via 1 o Via 2.")
            else:
                try:
                    c3d_1, c3d_2 = st.columns(2)
                    f1_3d = c3d_1.selectbox("Factor en eje X:", factores, index=0, key="fact_3d_x")
                    f2_opts_3d = [f for f in factores if f != f1_3d]
                    f2_3d = c3d_2.selectbox("Factor en eje Y:", f2_opts_3d, index=min(len(f2_opts_3d) - 1, 1), key="fact_3d_y")

                    e3d_1, e3d_2 = st.columns([1.5, 1])
                    estilo_3d = e3d_1.radio(
                        "Estetica del grafico:",
                        ["Clasico (estilo articulo)", "Moderno (viridis)"],
                        horizontal=True,
                        key="fact_3d_estilo"
                    )
                    estilo_3d_key = 'clasico' if estilo_3d.startswith("Clasico") else 'moderno'
                    cmap_3d = e3d_2.selectbox(
                        "Colormap (modo clasico):",
                        ["terrain", "RdYlGn", "jet", "viridis"],
                        index=0,
                        disabled=(estilo_3d_key == 'moderno'),
                        key="fact_3d_cmap"
                    )

                    mapeos_3d = {}
                    for fk in (f1_3d, f2_3d):
                        mk = st.session_state.mapeos_inversos.get(fk, {})
                        if -1 in mk and 1 in mk and _es_num_val(mk[-1]) and _es_num_val(mk[1]):
                            mapeos_3d[fk] = mk

                    fn_3d, _, _, _ = ce._gen_superficie_3d(
                        R, resp_name=respuesta,
                        f1=f1_3d, f2=f2_3d,
                        mapeos=mapeos_3d,
                        unidades=unidades,
                        estilo=estilo_3d_key,
                        cmap=cmap_3d
                    )

                    if fn_3d:
                        st.image(fn_3d, use_container_width=True)

                    if R.get('tiene_centro') and pd.notna(R.get('p_curv')) and R['p_curv'] < 0.05:
                        _info_box(
                            "Nota sobre curvatura",
                            "La prueba de curvatura fue significativa (p < 0.05). El grafico anterior muestra el plano/silla del modelo factorial; "
                            "para modelar la curvatura cuadratica completa (A²) usa un diseno CCD.",
                            tipo='tip'
                        )
                except Exception as e:
                    st.warning(f"No se pudo generar la superficie 3D: {e}")

        with tab6:
            try:
                fn_res = ce._gen_residuales(df, R)
                st.image(fn_res, use_container_width=True)
                _info_box(
                    "¿Para qué sirven los gráficos de residuales?",
                    "Validan que tu modelo cumpla los supuestos del ANOVA: "
                    "<b>(a) Normalidad:</b> los puntos deben seguir la línea roja; "
                    "<b>(b) Varianza constante:</b> nube aleatoria sin forma de embudo ni curva; "
                    "<b>(c) Independencia (orden):</b> sin tendencias crecientes/decrecientes a lo largo de las corridas (confirma que la aleatorización funcionó).",
                    tipo='tip'
                )
            except Exception as e:
                st.warning(f"No se pudieron generar residuales: {e}")

        with tab7:
            if spec is None:
                st.info("Abre **'🎯 Evaluar cumplimiento de especificacion'** arriba de las pestañas y activa la casilla para generar el grafico (en Via 2 incluye tambien el punto central).")
            else:
                try:
                    fn_spec = ce._gen_spec(df, R, spec)
                    st.image(fn_spec, use_container_width=True)

                    lo = spec['target'] - spec['tol']
                    hi = spec['target'] + spec['tol']
                    cond_spec = df.groupby(R['factores'], sort=False)['Y'].mean().reset_index()
                    cond_spec['Cumple'] = cond_spec['Y'].apply(lambda v: 'Si' if lo <= v <= hi else 'No')
                    n_ok = int((cond_spec['Cumple'] == 'Si').sum())
                    n_tot = len(cond_spec)
                    _info_box(
                        "Cumplimiento de especificacion",
                        f"**{n_ok} de {n_tot}** condiciones cumplen con el rango **[{lo:.4g}, {hi:.4g}]** (barras verdes). "
                        "Las barras rojas quedan fuera de especificacion.",
                        tipo='success' if n_ok > 0 else 'warning'
                    )
                    with st.expander("📋 Ver tabla de medias por condicion vs especificacion"):
                        st.dataframe(cond_spec.round(4), use_container_width=True)
                except Exception as e:
                    st.warning(f"No se pudo generar el grafico de especificacion: {e}")

        # PREDICCION
        st.markdown("---")
        st.markdown("## Prediccion")
        _info_box("Prediccion del modelo", "Introduce los niveles de los factores para predecir la respuesta.", tipo='info')

        tiene_reales_pred = any(
            f in st.session_state.mapeos_inversos
            and -1 in st.session_state.mapeos_inversos[f]
            and _es_num_val(st.session_state.mapeos_inversos[f][-1])
            for f in factores
        )
        lbl_libre = "Ingresar valor real libre" if tiene_reales_pred else "Ingresar valor codificado (-1 a +1)"
        modo_pred = st.radio(
            "Como quieres ingresar los valores para predecir?",
            ["Seleccionar nivel (Bajo / Centro / Alto)", lbl_libre],
            horizontal=True,
            key="modo_pred_input"
        )

        pred_cols = st.columns(len(factores))
        niveles = {}
        for i, f in enumerate(factores):
            u = unidades.get(f, '').strip()
            etiqueta = f"{f} ({u})" if u else f
            suf_u = f" {u}" if u else ""
            mapeo = st.session_state.mapeos_inversos.get(f, {})
            tiene_m = (-1 in mapeo and 1 in mapeo)
            es_num = tiene_m and _es_num_val(mapeo[-1]) and _es_num_val(mapeo[1])

            if tiene_m and not es_num:
                opts = [(-1, f"{mapeo[-1]} (-1)")]
                if 0 in mapeo:
                    opts.append((0, f"{mapeo[0]} (0)"))
                opts.append((1, f"{mapeo[1]} (+1)"))
                sel = pred_cols[i].selectbox(etiqueta, opts, format_func=lambda x: x[1], key=f"pred_{f}")
                niveles[f] = sel[0]
            elif "Seleccionar nivel" in modo_pred:
                if es_num:
                    opts = [
                        (-1, f"Bajo ({mapeo[-1]}{suf_u})"),
                        (0, f"Centro ({mapeo.get(0, (float(mapeo[-1])+float(mapeo[1]))/2)}{suf_u})"),
                        (1, f"Alto ({mapeo[1]}{suf_u})")
                    ]
                else:
                    opts = [(-1, "Bajo (-1)"), (0, "Centro (0)"), (1, "Alto (+1)")]
                sel = pred_cols[i].selectbox(etiqueta, opts, format_func=lambda x: x[1], key=f"pred_{f}")
                niveles[f] = sel[0]
            else:
                if es_num:
                    v_lo, v_hi = float(mapeo[-1]), float(mapeo[1])
                    v_mid = float(mapeo.get(0, (v_lo + v_hi) / 2.0))
                    v_real_in = pred_cols[i].number_input(
                        etiqueta,
                        value=v_mid,
                        min_value=min(v_lo, v_hi),
                        max_value=max(v_lo, v_hi),
                        key=f"pred_real_{f}"
                    )
                    rango = (v_hi - v_lo) / 2.0
                    niveles[f] = (v_real_in - v_mid) / rango if rango != 0 else 0.0
                else:
                    niveles[f] = pred_cols[i].number_input(
                        f"{etiqueta} [codificado]",
                        value=0.0, min_value=-1.0, max_value=1.0, step=0.1,
                        key=f"pred_cod_{f}"
                    )

        try:
            y_pred = ce.predecir_y_combinacion(niveles, R)
            st.success(f"**Respuesta predicha: {y_pred:.4f}**")
        except Exception as e:
            st.warning(f"No se pudo predecir: {e}")

        # ALIAS (fraccionado)
        if st.session_state.modelo_fraccionado:
            st.markdown("---")
            st.markdown("## Estructura de Alias")
            try:
                alias_df = ce.generar_tabla_alias(df, factores)
                st.dataframe(alias_df, use_container_width=True)
            except Exception as e:
                st.warning(f"No se pudo generar alias: {e}")

    # PANEL DE NAVEGACION FINAL
    st.markdown("---")
    st.markdown("## Navegacion")

    _info_box("Que deseas hacer ahora?",
              "Puedes regresar a cualquier etapa anterior, cambiar la configuracion del modelo, "
              "o comenzar con un nuevo archivo.", tipo='info')

    nav_cols = st.columns(4)

    with nav_cols[0]:
        if st.button("⚙️ Cambiar modelo", use_container_width=True):
            st.session_state.R = None
            st.session_state.R_rsm = None
            st.session_state.stage = 'results'
            st.rerun()

    with nav_cols[1]:
        if st.button("📋 Cambiar analisis", use_container_width=True):
            st.session_state.R = None
            st.session_state.R_rsm = None
            st.session_state.stage = 'analysis_select'
            st.rerun()

    with nav_cols[2]:
        if st.button("🔢 Revisar niveles y matriz", use_container_width=True):
            st.session_state.R = None
            st.session_state.R_rsm = None
            st.session_state.stage = 'encode'
            st.rerun()

    with nav_cols[3]:
        if st.button("📁 Nuevo archivo", use_container_width=True):
            st.session_state.R = None
            st.session_state.R_rsm = None
            st.session_state.raw = None
            st.session_state.df_work = None
            st.session_state.df_cod = None
            st.session_state.df_real = None
            st.session_state.niveles_confirmados_paso3 = False
            st.session_state.stage = 'upload'
            st.rerun()


# ==================== RAMA CCD (LIMITADA EXCLUSIVAMENTE A VIA CCD) ====================
elif st.session_state.stage == 'ccd_full':
    st.title("Superficie de Respuesta (CCD) - Modelo completo")
    rawc = st.session_state.df_ccd_raw
    facc = st.session_state.factores_ccd
    respc = st.session_state.respuesta_ccd

    _info_box("Ruta CCD (Superficie de Respuesta)",
              f"Diseno CCD con **{len(rawc)} corridas** | Factores: **{', '.join(facc)}** | Respuesta(s): **{', '.join(respc)}**. "
              "Se ajustara un modelo cuadratico completo (lineales + cuadrados A² + interacciones 2 a 2).",
              tipo='info')

    st.dataframe(rawc[facc + respc], use_container_width=True)

    c1, c2 = st.columns([1, 3])
    with c1:
        if st.button("← Cambiar tipo de analisis", use_container_width=True):
            st.session_state.stage = 'analysis_select'
            st.rerun()
    with c2:
        if st.button("Ajustar modelos completos →", type='primary', use_container_width=True):
            try:
                Rs = {}
                for r in respc:
                    Rs[r] = ce_rsm.ajustar_ccd(rawc, facc, r, eliminar=False)
                st.session_state.R_ccd_full = Rs
                st.session_state.R_ccd = None
                st.session_state.stage = 'ccd_refine'
                st.rerun()
            except Exception as e:
                st.error(f"Error al ajustar CCD: {e}")


elif st.session_state.stage == 'ccd_refine':
    st.title("CCD - Refinar modelo (eliminacion hacia atras)")
    rawc = st.session_state.df_ccd_raw
    facc = st.session_state.factores_ccd
    respc = st.session_state.respuesta_ccd
    unidades_ccd = st.session_state.get('unidades_factores', {})

    _info_box("Eliminacion hacia atras (Backward)",
              "Puedes aplicar eliminacion automatica (respetando jerarquia: no quita un principal si participa en una interaccion activa) "
              "o trabajar con el modelo cuadratico completo.",
              tipo='tip')

    alfa = st.number_input("Alfa para eliminacion (p > alfa se elimina)", 0.01, 0.20, 0.05, 0.01, key="ccd_alfa")

    b1, b2 = st.columns(2)
    with b1:
        if st.button("Aplicar eliminacion hacia atras", type='primary', use_container_width=True):
            with st.spinner("Ajustando..."):
                Rs = {}
                for r in respc:
                    Rs[r] = ce_rsm.ajustar_ccd(rawc, facc, r, alpha_elim=alfa, eliminar=True)
                st.session_state.R_ccd = Rs
    with b2:
        if st.button("Usar modelo completo (sin eliminar)", use_container_width=True):
            if st.session_state.R_ccd_full:
                st.session_state.R_ccd = st.session_state.R_ccd_full
            else:
                Rs = {}
                for r in respc:
                    Rs[r] = ce_rsm.ajustar_ccd(rawc, facc, r, eliminar=False)
                st.session_state.R_ccd = Rs

    Rs = st.session_state.get('R_ccd') or st.session_state.get('R_ccd_full')
    if Rs:
        st.session_state.R_ccd = Rs
        for r, R in Rs.items():
            st.markdown("---")
            st.markdown(f"### Respuesta: {r}")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("S (Error estandar)", f"{np.sqrt(R['model'].mse_resid):.6f}")
            c2.metric("R2", f"{R['R2']*100:.2f}%")
            c3.metric("R2 ajustado", f"{R['R2_adj']*100:.2f}%")
            c4.metric("Terminos", len(R['terms']))

            st.dataframe(R['anova'].round(4), use_container_width=True)

            if R['lof']:
                ok_lof = R['lof']['p'] > 0.05
                st.write(
                    f"**Falta de ajuste:** F = {R['lof']['F']:.3f}, p = {R['lof']['p']:.4f} "
                    + ("-> no significativa (buena)" if ok_lof else "-> SIGNIFICATIVA (revisar modelo)")
                )
            else:
                st.caption("Sin replicas: no se puede calcular la falta de ajuste.")

            with st.expander("📖 ¿Qué significa la Falta de Ajuste (Lack of Fit) en CCD?"):
                st.markdown("""
                - Compara el error del modelo cuadrático frente al **error puro** de las réplicas (normalmente tus puntos centrales).
                - ✅ **`p > 0.05` (no significativa):** El modelo cuadrático describe bien la superficie real; las desviaciones son solo variabilidad natural del experimento.
                - ⚠️ **`p < 0.05` (significativa):** La superficie real tiene un comportamiento que el modelo reducido actual no alcanza a explicar del todo (prueba subir `alfa` o usar el modelo completo).
                """)

            st.markdown("**Ecuacion (unidades codificadas):**")
            st.code(R['eq_coded'], language=None)
            st.markdown("**Ecuacion (unidades reales):**")
            st.code(R['eq_uncoded'], language=None)
            if any(unidades_ccd.get(f) for f in facc):
                u_txt = " | ".join(f"{f}: {unidades_ccd[f]}" for f in facc if unidades_ccd.get(f))
                st.caption(f"Unidades guardadas -> {u_txt}")

        st.markdown("---")
        c1, c2 = st.columns(2)
        if c1.button("← Volver a modelo completo", use_container_width=True):
            st.session_state.stage = 'ccd_full'
            st.rerun()
        if c2.button("Ir a Diagnosticos, 3D, contornos y optimizacion →", type='primary', use_container_width=True):
            st.session_state.stage = 'ccd_diag'
            st.rerun()


elif st.session_state.stage == 'ccd_diag':
    st.title("CCD - Diagnosticos, Superficie 3D, contornos y optimizacion")
    Rs = st.session_state.get('R_ccd')
    if not Rs:
        st.error("Primero debes refinar el modelo (paso anterior).")
        st.stop()

    respc = list(Rs.keys())
    facc = Rs[respc[0]]['factores']
    unidades_ccd = st.session_state.get('unidades_factores', {})
    etiq_ccd = {f: f"{f} ({unidades_ccd[f]})" if unidades_ccd.get(f) else f for f in facc}

    for r, R in Rs.items():
        st.markdown(f"### Residuales - {r}")
        fn = ce_rsm.graf_residuales_ccd(R)
        st.image(fn, use_container_width=True)

    st.markdown("---")
    st.markdown("### Superficies 3D, Contornos individuales y combinado")
    st.caption("1) Define los ejes de cada grafico y los del combinado. "
               "2) Pulsa **Aplicar configuracion**. 3) Activa las casillas para mostrar los graficos.")

    with st.form("ccd_form_cfg"):
        st.markdown("**A. Ejes de las Superficies 3D y Contornos INDIVIDUALES** "
                    "(el/los factor(es) restante(s) queda(n) fijo(s)):")
        ejes_ind = {}
        for r in respc:
            cx, cy = st.columns(2)
            xi = cx.selectbox(f"Eje X - {r}", facc, index=0, key=f"xi_{r}")
            yi_opts = [f for f in facc if f != xi]
            yi = cy.selectbox(f"Eje Y - {r}", yi_opts, index=len(yi_opts) - 1, key=f"yi_{r}")
            ejes_ind[r] = (xi, yi)

        st.markdown("**B. Contorno COMBINADO** (elige los DOS factores de los ejes):")
        c1f, c2f = st.columns(2)
        f1 = c1f.selectbox("Eje X del combinado", facc, index=0, key="ccd_comb_x")
        f2_opts = [f for f in facc if f != f1]
        f2 = c2f.selectbox("Eje Y del combinado", f2_opts, index=len(f2_opts) - 1, key="ccd_comb_y")
        fijo_sel = [f for f in facc if f not in (f1, f2)]
        if len(fijo_sel) == 1:
            st.caption(f"Combinado: varian X = **{etiq_ccd[f1]}** e Y = **{etiq_ccd[f2]}**; queda fijo **{etiq_ccd[fijo_sel[0]]}**.")
        elif len(fijo_sel) > 1:
            st.caption(f"Combinado: varian X = **{etiq_ccd[f1]}** e Y = **{etiq_ccd[f2]}**; quedan fijos: {', '.join(etiq_ccd[x] for x in fijo_sel)}.")

        st.markdown("**C. Valores de los factores cuando estan FIJOS:**")
        fcols = st.columns(len(facc))
        fijos = {}
        for i, f in enumerate(facc):
            codf = Rs[respc[0]]['coding'][f]
            paso = max(codf['half'] / 10, 1e-6)
            with fcols[i]:
                fijos[f] = st.number_input(etiq_ccd[f], value=float(codf['c']),
                                           step=paso, format="%.6g", key=f"fijo_{f}")
        for f in facc:
            usado = any(f in t.split('*') for R in Rs.values() for t in R['terms'])
            if usado:
                st.info(f"'{etiq_ccd[f]}' esta en el modelo reducido: si algun grafico lo fija, ese valor SI afecta ese grafico.")
            else:
                st.caption(f"'{etiq_ccd[f]}' fue eliminado del modelo como no significativo: fijarlo no afecta los resultados.")

        estilo = st.radio("Estetica del grafico",
                          ["Clasico (estilo articulo)", "Moderno (viridis)"],
                          horizontal=True)
        cmaps, niveles_txt = {}, {}
        for r in respc:
            cmaps[r] = st.selectbox(f"Colormap para {r}",
                                    ["terrain", "RdYlGn", "jet", "viridis"],
                                    index=0, key=f"cmap_{r}")
            niveles_txt[r] = st.text_input(
                f"Niveles de contorno para {r} (opcional, separados por comas)",
                value="", key=f"niv_{r}",
                placeholder="ej: 0,25,50,75,100,125,150,175,200,240")
        st.form_submit_button("Aplicar configuracion", type='primary')

    estilo_key = 'clasico' if estilo.startswith("Clasico") else 'moderno'

    punto = {
        f1: float(st.session_state.get(f"pv_{f1}", Rs[respc[0]]['coding'][f1]['c'])),
        f2: float(st.session_state.get(f"pv_{f2}", Rs[respc[0]]['coding'][f2]['c']))
    }
    mostrar_estrella = bool(st.session_state.get('ccd_mostrar_punto', True))

    def _star_para(xi, yi):
        if not mostrar_estrella:
            return None
        return {xi: punto.get(xi, fijos.get(xi)), yi: punto.get(yi, fijos.get(yi))}

    chk_c1, chk_c2 = st.columns(2)
    mostrar_3d = chk_c1.checkbox("Mostrar superficies 3D individuales", value=True, key='ccd_mostrar_3d')
    mostrar_cont = chk_c2.checkbox("Mostrar contornos 2D individuales", value=True, key='ccd_mostrar_cont')

    if mostrar_3d or mostrar_cont:
        for r, R in Rs.items():
            xi, yi = ejes_ind[r]
            if xi == yi:
                st.error(f"Grafico de {r}: el eje X y el Y son el mismo factor ('{xi}'). "
                         "Elige dos factores distintos en la seccion A y pulsa 'Aplicar configuracion'.")
                continue

            txt = niveles_txt.get(r, '').strip()
            niv = None
            if txt:
                try:
                    niv = sorted(float(x) for x in txt.replace(';', ',').split(',') if x.strip())
                    if len(niv) < 2:
                        niv = None
                except ValueError:
                    niv = None
                    st.warning(f"Niveles de {r} no validos; se usaran niveles automaticos.")

            gcols = st.columns(2 if (mostrar_3d and mostrar_cont) else 1)
            col_idx = 0
            if mostrar_3d and hasattr(ce_rsm, 'graf_superficie_3d_ccd'):
                fn_3d_ccd = ce_rsm.graf_superficie_3d_ccd(
                    R, xi, yi, estilo=estilo_key, cmap=cmaps.get(r),
                    xlabel=etiq_ccd[xi], ylabel=etiq_ccd[yi], fijos=fijos
                )
                gcols[col_idx].image(fn_3d_ccd, use_container_width=True)
                col_idx += 1

            if mostrar_cont:
                fn_c = ce_rsm.graf_contorno_ccd(
                    R, xi, yi, estilo=estilo_key, cmap=cmaps.get(r),
                    xlabel=etiq_ccd[xi], ylabel=etiq_ccd[yi],
                    niveles=(niv if niv else None), punto=_star_para(xi, yi), fijos=fijos
                )
                gcols[col_idx].image(fn_c, use_container_width=True)

    ejes_comb_ok = (f1 != f2) and (f2 in [f for f in facc if f != f1])
    if not ejes_comb_ok:
        st.error(f"Los ejes del combinado (X = {f1}, Y = {f2}) no son validos. "
                 "Revisa la seccion B y pulsa 'Aplicar configuracion'.")

    # ---- Contorno combinado ----
    st.markdown("---")
    st.markdown("### Contorno combinado (region de interes)")
    st.caption("Define los rangos aceptables de cada respuesta y pulsa **Aplicar metas**.")
    rawc = st.session_state.df_ccd_raw
    with st.form("ccd_form_metas"):
        rangos = {}
        cc1, cc2 = st.columns(2)
        for i, r in enumerate(respc):
            rmin = float(rawc[r].min())
            rmax = float(rawc[r].max())
            paso = max((rmax - rmin) / 20, 1e-6)
            with (cc1 if i % 2 == 0 else cc2):
                lo = st.number_input(f"{r} - minimo aceptable", value=rmin,
                                     step=paso, format="%.6g", key=f"lo_{r}")
                hi = st.number_input(f"{r} - maximo aceptable", value=rmax,
                                     step=paso, format="%.6g", key=f"hi_{r}")
            rangos[r] = (lo, hi)
        st.form_submit_button("Aplicar metas", type='primary')

    mostrar_comb = st.checkbox("Mostrar contorno combinado", key='ccd_mostrar_comb')
    if mostrar_comb and ejes_comb_ok:
        fn_sup = ce_rsm.graf_superposicion_ccd(
            list(Rs.values()), rangos, f1, f2,
            xlabel=etiq_ccd[f1], ylabel=etiq_ccd[f2],
            punto=_star_para(f1, f2), fijos=fijos
        )
        st.image(fn_sup, use_container_width=True)

    # ---- Probar un punto ----
    st.markdown("---")
    st.markdown("### Probar un punto (opcional)")
    st.caption(f"Escribe los valores de **{etiq_ccd[f1]}** y **{etiq_ccd[f2]}** en unidades reales y pulsa **Aplicar punto**. "
               "La estrella se dibuja en los graficos de arriba y los factores fijos usan los valores de la seccion C.")
    with st.form("ccd_form_punto"):
        pcols = st.columns(2)
        for i, f in enumerate([f1, f2]):
            c_def = Rs[respc[0]]['coding'][f]['c']
            with pcols[i]:
                st.number_input(f"{etiq_ccd[f]} (unidades reales)", value=float(c_def),
                                step=0.05, format="%.6g", key=f"pv_{f}")
        st.form_submit_button("Aplicar punto")

    punto_vivo = {
        f1: float(st.session_state.get(f"pv_{f1}", Rs[respc[0]]['coding'][f1]['c'])),
        f2: float(st.session_state.get(f"pv_{f2}", Rs[respc[0]]['coding'][f2]['c']))
    }
    mcols = st.columns(len(respc))
    for i, (r, R) in enumerate(Rs.items()):
        pfull = {f: fijos.get(f, R['coding'][f]['c']) for f in R['factores']}
        pfull.update(punto_vivo)
        with mcols[i]:
            st.metric(f"Prediccion de {r} en el punto", f"{ce_rsm.predecir_ccd(R, pfull):.3f}")
    st.checkbox("Mostrar estrella en los graficos", value=True, key='ccd_mostrar_punto')

    # ---- Optimizacion ----
    st.markdown("---")
    st.markdown("### Optimizacion (deseabilidad)")
    st.caption("Configura los objetivos y pulsa **Aplicar objetivos**; luego usa **Optimizar ahora**.")
    with st.form("ccd_form_obj"):
        objetivos = {}
        for r in respc:
            rlo, rhi = rangos[r]
            paso = max((rhi - rlo) / 20, 1e-6)
            tipo = st.selectbox(f"Objetivo de {r}", ["max", "min", "target"],
                                format_func=lambda t: {"max": "Maximizar", "min": "Minimizar", "target": "Apuntar a un target"}[t],
                                key=f"obj_{r}")
            o = {'tipo': tipo, 'lo': rlo, 'hi': rhi}
            if tipo == 'target':
                o['target'] = st.number_input(f"Target de {r}",
                                              value=float((rlo + rhi) / 2),
                                              step=paso, format="%.6g", key=f"tgt_{r}")
            objetivos[r] = o
        st.form_submit_button("Aplicar objetivos", type='primary')

    if st.button("Optimizar ahora"):
        opt = ce_rsm.optimizar_ccd(list(Rs.values()), objetivos)
        st.session_state.opt_ccd = opt if opt is not None else 'SIN_SOLUCION'
    opt = st.session_state.get('opt_ccd')
    if opt == 'SIN_SOLUCION':
        st.warning("No existe ningun punto que cumpla todas las metas a la vez.")
    elif opt:
        punto_txt = " | ".join(
            f"{f} = {v:.4f}" + (f" {unidades_ccd[f]}" if unidades_ccd.get(f) else "")
            for f, v in opt['punto'].items()
        )
        pred_txt = " | ".join(f"{k} = {v:.4f}" for k, v in opt['predicciones'].items())
        st.success(f"**Mejor punto (deseabilidad = {opt['deseabilidad']:.3f}):** {punto_txt}  →  {pred_txt}")

    # ---- Navegacion final CCD ----
    st.markdown("---")
    nav_ccd = st.columns(4)
    with nav_ccd[0]:
        if st.button("← Volver a Refinar CCD", use_container_width=True):
            st.session_state.stage = 'ccd_refine'
            st.rerun()
    with nav_ccd[1]:
        if st.button("📋 Cambiar analisis", use_container_width=True):
            st.session_state.R_ccd = None
            st.session_state.R_ccd_full = None
            st.session_state.stage = 'analysis_select'
            st.rerun()
    with nav_ccd[2]:
        if st.button("🔢 Revisar niveles y matriz", use_container_width=True):
            st.session_state.R_ccd = None
            st.session_state.R_ccd_full = None
            st.session_state.stage = 'encode'
            st.rerun()
    with nav_ccd[3]:
        if st.button("📁 Nuevo archivo", use_container_width=True):
            st.session_state.R_ccd = None
            st.session_state.R_ccd_full = None
            st.session_state.raw = None
            st.session_state.df_work = None
            st.session_state.df_cod = None
            st.session_state.df_real = None
            st.session_state.niveles_confirmados_paso3 = False
            st.session_state.stage = 'upload'
            st.rerun()
