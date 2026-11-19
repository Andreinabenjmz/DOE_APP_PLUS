# app_plus.py
# Interfaz Streamlit - Flujo completo redisenado
# Cargar -> Confirmar columnas -> Niveles/Unidades + Codificar -> Puntos centrales y Analisis -> Resultados

import streamlit as st
import pandas as pd
import numpy as np
import os

import calc_engine_plus as ce
import calc_engine_rsm as ce_rsm

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
        'factores': [],
        'respuesta': None,
        'corrida_col': None,
        'es_codificada': False,
        'niveles_guardados': {},
        'unidades_factores': {},
        'mapeos_niveles': {},
        'mapeos_inversos': {},
        'filas_centro_idx': [],
        'centros_confirmados': False,
        'analisis_tipo': 'factorial_via2',
        'modelo_fraccionado': False,
        'resolucion': None,
        'R': None,
        'R_rsm': None,
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
    resp_names = {'y', 'response', 'respuesta', 'resp', 'rendimiento', 'yield'}

    for c in raw.columns:
        nombre_lower = str(c).strip().lower()
        nombre_norm = nombre_lower.replace('a', 'a').replace('e', 'e').replace('i', 'i').replace('o', 'o').replace('u', 'u').replace('n', 'n')
        vals = raw[c].dropna().unique().tolist()
        n_unicos = len(vals)

        if any(x in nombre_norm for x in ['corrida', 'run', 'orden', 'order']):
            if n_unicos == len(raw):
                descartar.append(c)
                continue

        if nombre_norm in id_names or n_unicos <= 1:
            descartar.append(c)
            continue

        if nombre_norm in resp_names or any(x in nombre_norm for x in ['resp', 'yield', 'rend']):
            if pd.api.types.is_numeric_dtype(raw[c]):
                respuestas.append(c)
                continue

        if pd.api.types.is_numeric_dtype(raw[c]):
            if n_unicos in [2, 3]:
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
    for c in cols:
        if not pd.api.types.is_numeric_dtype(raw[c]):
            return False
        vals = set(pd.to_numeric(raw[c], errors='coerce').dropna().unique())
        if not vals.issubset({-1, 0, 1}):
            return False
    return True


def _detectar_centros(df_cod, cols):
    """Detecta filas donde TODOS los factores codificados estan cercanos a 0 (centros).
    Devuelve (n_centros, lista_indices)."""
    df_num = df_cod[cols].apply(pd.to_numeric, errors='coerce')
    mascara = df_num.apply(lambda r: all(abs(v) < 0.5 for v in r), axis=1)
    indices = df_cod[mascara].index.tolist()
    return len(indices), indices


def _detectar_replicas(df_cod, cols, resp_col):
    """Cuenta replicas PURAS del diseno completo sobre la matriz codificada (excluye centros)."""
    if resp_col is None or resp_col not in df_cod.columns:
        return 0
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
    df_group = df_esq.groupby(cols)[resp_col].count()
    return int((df_group > 1).sum())


def _detectar_bloques(df_cod, cols):
    if 'Block' in df_cod.columns or 'Bloque' in df_cod.columns:
        return True
    k = len(cols)
    n = len(df_cod)
    if st.session_state.get('filas_centro_idx'):
        centros = len(st.session_state.filas_centro_idx)
    else:
        centros = _detectar_centros(df_cod, cols)[0]
    esquinas = n - centros
    potencia = 2 ** k
    return esquinas > potencia


def _validar_analisis_vs_diseno(df_cod, factores, respuesta, analisis):
    k = len(factores)
    if st.session_state.get('filas_centro_idx'):
        centros = len(st.session_state.filas_centro_idx)
    else:
        centros = _detectar_centros(df_cod, factores)[0]
    replicas = _detectar_replicas(df_cod, factores, respuesta)
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


def _sugerir_analisis(df_cod, factores, respuesta):
    n = len(df_cod)
    if st.session_state.get('filas_centro_idx'):
        centros = len(st.session_state.filas_centro_idx)
    else:
        centros = _detectar_centros(df_cod, factores)[0]
    esquinas = n - centros
    replicas = _detectar_replicas(df_cod, factores, respuesta)

    if centros > 0:
        return 'factorial_via2', f"Detectado: {esquinas} esquinas + {centros} centros. Sugerencia: Via 2."
    elif replicas > 0:
        return 'factorial_via1', f"Detectado: {esquinas} esquinas con {replicas} grupos de replicas. Sugerencia: Via 1."
    else:
        return 'factorial_via3', f"Detectado: {esquinas} esquinas sin replicas ni centros. Sugerencia: Via 3."


def _agregar_centros_cod(df_cod, factores, respuesta, n_centros=3):
    """Genera n puntos centrales (0 en escala codificada) y los añade a df_cod."""
    nuevas_filas = []
    for _ in range(n_centros):
        fila = {f: 0.0 for f in factores}
        fila[respuesta] = np.nan
        nuevas_filas.append(fila)
    return pd.concat([df_cod, pd.DataFrame(nuevas_filas)], ignore_index=True)


def _codificar_factor(serie, v_bajo, v_alto):
    v_medio = (v_bajo + v_alto) / 2.0
    rango = (v_alto - v_bajo) / 2.0
    if rango == 0:
        return pd.to_numeric(serie, errors='coerce')
    return (pd.to_numeric(serie, errors='coerce') - v_medio) / rango


def _limpiar_niveles_y_widgets(factores):
    """Restablece los niveles guardados y las claves de widgets a sus valores por defecto."""
    st.session_state.niveles_guardados = {}
    st.session_state.unidades_factores = {}
    for f in factores:
        for pref in ('real_low_', 'real_high_', 'low_', 'high_', 'unit_'):
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
        stages = {
            'upload': '1. Cargar archivo',
            'preview': '2. Confirmar columnas',
            'encode': '3. Niveles, unidades y codificacion',
            'analysis_select': '4. Puntos centrales y analisis',
            'results': '5. Resultados'
        }
        current = st.session_state.stage
        for stage_key, stage_label in stages.items():
            idx = list(stages.keys()).index(stage_key)
            curr_idx = list(stages.keys()).index(current)
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
    1. **Carga tu matriz** de diseno (Excel o CSV)
    2. **Confirma las columnas** (factores y respuesta)
    3. **Asigna niveles reales y unidades** (el programa genera la matriz codificada `-1, 0, +1`)
    4. **Detecta puntos centrales y elige el analisis** (Via 1, 2, 3 o CCD)
    5. **Obtien resultados**: ANOVA, coeficientes, ecuacion, graficos
    """)

    uploaded = st.file_uploader("Sube tu archivo Excel o CSV", type=['xlsx', 'csv'])

    if uploaded:
        try:
            if uploaded.name.endswith('.xlsx'):
                raw = pd.read_excel(uploaded)
            else:
                raw = pd.read_csv(uploaded)
            raw.columns = raw.columns.astype(str).str.strip()
            st.session_state.raw = raw
            st.session_state.df_cod = None
            st.session_state.df_work = None
            st.session_state.filas_centro_idx = []
            st.session_state.centros_confirmados = False
            st.session_state.factores = []
            st.session_state.respuesta = None
            # No borramos niveles_guardados para mantener los valores y unidades configurados

            st.success(f"Archivo cargado: {raw.shape[0]} filas × {raw.shape[1]} columnas")

            fact_cands, resp_cands, descartados = _detectar_columnas(raw)

            if len(descartados) > 0:
                st.info(f"Columnas descartadas (metadatos): {', '.join(descartados)}")

            if len(fact_cands) == 0:
                st.error("No se detectaron factores. Verifica que tus columnas numericas tengan 2 o 3 niveles.")
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
              "Confirma que el programa detecto correctamente los factores y la respuesta. "
              "En el siguiente paso configuraras los niveles reales y generaremos la matriz codificada para detectar los puntos centrales.",
              tipo='info')

    fact_cands, resp_cands, _ = _detectar_columnas(raw)

    corrida_opts = ['(Ninguna)'] + [c for c in raw.columns if any(x in str(c).lower() for x in ['corrida', 'run', 'orden', 'order', 'id'])]
    corrida_idx = corrida_opts.index(st.session_state.corrida_col) if st.session_state.corrida_col in corrida_opts else 0
    corrida_sel = st.selectbox("Columna de Corrida (opcional):", corrida_opts, index=corrida_idx)
    st.session_state.corrida_col = None if corrida_sel == '(Ninguna)' else corrida_sel

    default_factores = st.session_state.factores if st.session_state.factores else fact_cands
    st.markdown("### Factores")
    factores = st.multiselect("Selecciona los factores:", raw.columns.tolist(), default=default_factores, key="sel_factores")

    default_resp = st.session_state.respuesta if st.session_state.respuesta in raw.columns else (resp_cands[0] if resp_cands else raw.columns[0])
    st.markdown("### Respuesta")
    respuesta = st.selectbox("Selecciona la respuesta:", raw.columns.tolist(),
                              index=raw.columns.tolist().index(default_resp),
                              key="sel_respuesta")

    if len(factores) < 2:
        st.warning("Selecciona al menos 2 factores.")
    else:
        st.session_state.factores = factores
        st.session_state.respuesta = respuesta

        es_cod = _es_codificada(raw, factores)
        st.session_state.es_codificada = es_cod

        cols_mostrar = []
        if st.session_state.corrida_col:
            cols_mostrar.append(st.session_state.corrida_col)
        cols_mostrar.extend(factores)
        cols_mostrar.append(respuesta)
        st.dataframe(raw[cols_mostrar], use_container_width=True)

        st.markdown(f"""
        **Formato detectado:** {'Matriz codificada (-1, 0, +1)' if es_cod else 'Matriz NO codificada (valores reales)'}
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
    respuesta = st.session_state.respuesta
    es_cod = st.session_state.es_codificada

    if es_cod:
        _info_box("Matriz codificada detectada",
                  "Tu archivo ya viene en escala codificada (-1, 0, +1). "
                  "Asigna los valores reales bajos/altos y sus unidades para guardarlos (se usaran en la ecuacion real, graficos y prediccion). "
                  "El analisis trabajara exclusivamente con la matriz codificada.",
                  tipo='info')
    else:
        _info_box("Matriz NO codificada detectada",
                  "Tu archivo tiene valores reales. Indica/confirma los valores bajo (-1), alto (+1) y sus unidades para generar la matriz codificada (-1, 0, +1), "
                  "que sera la matriz utilizada para buscar puntos centrales y ejecutar el analisis.",
                  tipo='info')

    mapeos_niveles = {}
    mapeos_inversos = {}
    unidades_factores = {}

    head_c1, head_c2 = st.columns([3, 1])
    with head_c1:
        st.markdown("### Configurar niveles reales y unidades")
    with head_c2:
        if st.button("🗑️ Borrar valores (volver al default)", use_container_width=True, help="Restablece los niveles bajos, altos y unidades a sus valores por defecto"):
            _limpiar_niveles_y_widgets(factores)
            st.rerun()

    for f in factores:
        vals = sorted(pd.to_numeric(raw[f], errors='coerce').dropna().unique().tolist())
        st.markdown(f"**Factor: {f}**")

        if es_cod:
            guardado = st.session_state.niveles_guardados.get(f, {'low': 0.0, 'high': 100.0, 'unit': ''})
            c1, c2, c3 = st.columns([2, 2, 1.5])
            v_bajo = c1.number_input(
                f"Valor real para -1 (bajo) - {f}",
                value=float(guardado.get('low', 0.0)),
                format="%.4f",
                key=f"real_low_{f}"
            )
            v_alto = c2.number_input(
                f"Valor real para +1 (alto) - {f}",
                value=float(guardado.get('high', 100.0)),
                format="%.4f",
                key=f"real_high_{f}"
            )
            unidad = c3.text_input(
                f"Unidad - {f}",
                value=str(guardado.get('unit', '')),
                placeholder="ej: ppm, °C, min, %",
                key=f"unit_{f}"
            )
        else:
            default_low = float(min(vals)) if len(vals) > 0 else 0.0
            default_high = float(max(vals)) if len(vals) > 0 else 100.0
            guardado = st.session_state.niveles_guardados.get(f, {'low': default_low, 'high': default_high, 'unit': ''})
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

    # Generar matriz codificada (unica matriz con la que se trabaja)
    df_cod = raw.copy()
    if es_cod:
        for f in factores:
            df_cod[f] = pd.to_numeric(raw[f], errors='coerce')
    else:
        for f in factores:
            df_cod[f] = _codificar_factor(raw[f], mapeos_inversos[f][-1], mapeos_inversos[f][1]).round(6)

    st.markdown("---")
    st.markdown("### Matriz Codificada (-1, 0, +1) para el Analisis")
    st.dataframe(df_cod[factores + [respuesta]], use_container_width=True)

    # Tabla resumen de niveles guardados
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
            st.session_state.stage = 'preview'
            st.rerun()
    with nav_c[1]:
        if st.button("Continuar a puntos centrales y analisis →", type='primary', use_container_width=True):
            st.session_state.df_cod = df_cod.copy()
            st.session_state.df_work = df_cod.copy()
            # Al entrar desde aqui, detectar centros frescos sobre la matriz codificada
            st.session_state.centros_confirmados = False
            st.session_state.filas_centro_idx = []
            st.session_state.stage = 'analysis_select'
            st.rerun()


# ---------- ETAPA 4: PUNTOS CENTRALES Y SELECCION DE ANALISIS ----------
elif st.session_state.stage == 'analysis_select':
    st.title("4. Puntos Centrales y Tipo de Analisis")

    df_cod = st.session_state.df_cod if st.session_state.df_cod is not None else st.session_state.df_work
    factores = st.session_state.factores
    respuesta = st.session_state.respuesta
    k = len(factores)

    _info_box("Paso 4: Deteccion sobre matriz codificada",
              "Partiendo de tu matriz codificada (-1, 0, +1), el programa detecta los puntos centrales y sugiere el mejor analisis.",
              tipo='info')

    # Busqueda de puntos centrales directamente sobre la matriz codificada
    n_centros_detect, idx_centros = _detectar_centros(df_cod, factores)
    if st.session_state.get('centros_confirmados'):
        idx_centros = st.session_state.filas_centro_idx
        n_centros_detect = len(idx_centros)
    elif n_centros_detect > 0:
        st.session_state.filas_centro_idx = idx_centros

    replicas = _detectar_replicas(df_cod, factores, respuesta)
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
        st.metric("Replicas puras", replicas, delta="(filas identicas en factores)")

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

    # Cajita desplegable original: ¿Que son los puntos centrales?
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
    sug_analisis, sug_msg = _sugerir_analisis(df_cod, factores, respuesta)
    _info_box("Sugerencia del programa", sug_msg, tipo='tip')

    st.markdown("---")
    st.markdown("### Tipo de analisis")

    # Cajita desplegable original: Guia rapida
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

    warnings = _validar_analisis_vs_diseno(df_cod, factores, respuesta, analisis)
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

    explicaciones = {
        'factorial_via1': "**Via 1:** Analisis para disenos con **replicas completas**. Cada condicion experimental se repite varias veces.",
        'factorial_via2': f"**Via 2:** Analisis para disenos con **puntos centrales** ({centros} detectados en tu archivo). Permite probar curvatura.",
        'factorial_via3': "**Via 3:** Analisis para disenos **sin replicas ni centros**. Usa el metodo de Lenth para detectar efectos significativos.",
        'ccd': "**CCD:** Analisis de Superficie de Respuesta. Requiere puntos axiales (±α) ademas de esquinas y centros.",
    }
    st.markdown(explicaciones.get(analisis, ""))

    # DISEÑO FRACCIONADO - GUIA INTERACTIVA (cajita desplegable original)
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

            **Resolucion:** Indica que efectos pueden estar confundidos.
            - **Resolucion V** o mas: Efectos principales e interacciones de 2 vias son libres de confusion.
            - **Resolucion III**: Los principales pueden estar confundidos con interacciones de 2 vias.
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
            else:
                st.session_state.stage = 'results'
                st.rerun()


# ---------- ETAPA 5: RESULTADOS ----------
elif st.session_state.stage == 'results':
    st.title("5. Resultados del Analisis")

    raw = st.session_state.df_cod if st.session_state.df_cod is not None else st.session_state.df_work
    factores = st.session_state.factores
    respuesta = st.session_state.respuesta
    analisis = st.session_state.analisis_tipo
    unidades = st.session_state.get('unidades_factores', {})

    df = raw.copy()
    df = df.rename(columns={respuesta: 'Y'})
    df['Grupo'] = 'G1'
    for f in factores:
        df[f] = pd.to_numeric(df[f], errors='coerce')
    df['Y'] = pd.to_numeric(df['Y'], errors='coerce')

    centros_idx = set(st.session_state.get('filas_centro_idx', []))
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

        elif analisis == 'ccd':
            R_rsm = ce_rsm.ajustar_ccd(df, factores, 'Y')
            st.session_state.R_rsm = R_rsm
            R = None

        if analisis != 'ccd':
            st.session_state.R = R

    except Exception as e:
        st.error(f"Error en el analisis: {e}")
        import traceback
        st.code(traceback.format_exc())
        st.stop()

    # MOSTRAR RESULTADOS
    if analisis != 'ccd' and R is not None:
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

            if hasattr(st, "copy_button"):
                st.copy_button("Copiar ecuacion", eq_real or eq_cod, key="copy_eq")
        except Exception as e:
            st.warning(f"No se pudo generar ecuacion: {e}")

        # RANKING
        st.markdown("---")
        st.markdown("## Ranking de Efectos")
        try:
            ranking = ce.generar_tabla_balance_neto(R)
            st.dataframe(ranking, use_container_width=True)
        except Exception as e:
            st.warning(f"No se pudo generar ranking: {e}")

        # GRAFICOS
        st.markdown("---")
        st.markdown("## Graficos Diagnostico")

        tab1, tab2, tab3, tab4 = st.tabs(["Pareto", "Semi-normal", "Efectos Principales", "Interacciones"])

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
                fn = ce._gen_main_effects(R, spec=None, mapeos=st.session_state.mapeos_inversos, unidades=unidades)
                st.image(fn)
            except Exception as e:
                st.warning(f"No se pudieron generar efectos principales: {e}")

        with tab4:
            try:
                fig_list = ce._gen_interactions(R, mapeos=st.session_state.mapeos_inversos, unidades=unidades)
                for fn in fig_list:
                    st.image(fn)
                _info_box("Graficos de interaccion",
                          "Muestran como el efecto de un factor cambia segun el nivel del otro. "
                          "Lineas paralelas = sin interaccion. Lineas que se cruzan = interaccion significativa.",
                          tipo='tip')
            except Exception as e:
                st.warning(f"No se pudieron generar interacciones: {e}")

        # PREDICCION
        st.markdown("---")
        st.markdown("## Prediccion")
        _info_box("Prediccion del modelo", "Introduce los niveles de los factores para predecir la respuesta.", tipo='info')

        modo_pred = st.radio(
            "Como quieres ingresar los valores para predecir?",
            ["Seleccionar nivel (Bajo / Centro / Alto)", "Ingresar valor real libre"],
            horizontal=True,
            key="modo_pred_input"
        )

        pred_cols = st.columns(len(factores))
        niveles = {}
        for i, f in enumerate(factores):
            u = unidades.get(f, '').strip()
            etiqueta = f"{f} ({u})" if u else f
            suf_u = f" {u}" if u else ""
            if f in st.session_state.mapeos_inversos and -1 in st.session_state.mapeos_inversos[f]:
                mapeo = st.session_state.mapeos_inversos[f]
                if "Seleccionar nivel" in modo_pred:
                    opts = [
                        (-1, f"Bajo ({mapeo.get(-1, -1)}{suf_u})"),
                        (0, f"Centro ({mapeo.get(0, 0)}{suf_u})"),
                        (1, f"Alto ({mapeo.get(1, 1)}{suf_u})")
                    ]
                    sel = pred_cols[i].selectbox(etiqueta, opts, format_func=lambda x: x[1], key=f"pred_{f}")
                    niveles[f] = sel[0]
                else:
                    v_real_in = pred_cols[i].number_input(
                        etiqueta,
                        value=float(mapeo.get(0, 0.0)),
                        min_value=float(min(mapeo[-1], mapeo[1])),
                        max_value=float(max(mapeo[-1], mapeo[1])),
                        key=f"pred_real_{f}"
                    )
                    rango = (mapeo[1] - mapeo[-1]) / 2.0
                    niveles[f] = (v_real_in - mapeo[0]) / rango if rango != 0 else 0.0
            else:
                niveles[f] = pred_cols[i].number_input(etiqueta, value=0.0, key=f"pred_{f}")

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

    elif analisis == 'ccd':
        R_rsm = st.session_state.R_rsm
        st.markdown("## Resultados CCD")
        st.markdown("El analisis de Superficie de Respuesta (CCD) ajusta un modelo cuadratico completo.")
        if R_rsm:
            st.json(R_rsm)

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
            st.session_state.stage = 'upload'
            st.rerun()
