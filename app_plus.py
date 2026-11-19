# app_plus.py
# Interfaz Streamlit - Flujo completo redisenado
# Cargar -> Detectar -> Codificar/Descodificar -> Elegir analisis -> Resultados

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
        'factores': [],
        'respuesta': None,
        'corrida_col': None,
        'es_codificada': False,
        'mapeos_niveles': {},
        'mapeos_inversos': {},
        'analisis_tipo': None,
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
        'run','corrida','corrida_id','id','n_corrida','numero_corrida',
        'obs','observacion','index','unnamed: 0',
        'run order','std order','run type','order','tipo','type','tipopto',
        'block','bloque','order_run','orden','orden corrida','orden de corrida',
        'orden estandar','orden estandar','stdorder','runorder',
        'pt type','pt_type','tipo punto','tipo de punto',
        'replicate','replica','rep',
    }
    resp_names = {'y', 'response', 'respuesta', 'resp', 'rendimiento', 'yield'}

    for c in raw.columns:
        nombre_lower = str(c).strip().lower()
        nombre_norm = nombre_lower.replace('a','a').replace('e','e').replace('i','i').replace('o','o').replace('u','u').replace('n','n')
        vals = raw[c].dropna().unique().tolist()
        n_unicos = len(vals)

        if any(x in nombre_norm for x in ['corrida','run','orden','order']):
            if n_unicos == len(raw):
                descartar.append(c)
                continue

        if nombre_norm in id_names or n_unicos <= 1:
            descartar.append(c)
            continue

        if nombre_norm in resp_names or any(x in nombre_norm for x in ['resp','yield','rend']):
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


def _detectar_centros(raw, cols):
    """Detecta filas donde TODOS los factores estan cercanos a 0 (centros).
    Devuelve (n_centros, lista_indices)."""
    df_num = raw[cols].apply(pd.to_numeric, errors='coerce')
    mascara = df_num.apply(lambda r: all(abs(v) < 0.5 for v in r), axis=1)
    indices = raw[mascara].index.tolist()
    return len(indices), indices


def _detectar_replicas(raw, cols, resp_col):
    """Cuenta replicas PURAS del diseno completo (excluye puntos centrales)."""
    if resp_col is None or resp_col not in raw.columns:
        return 0
    df_esq = raw.copy()
    for c in cols:
        df_esq[c] = pd.to_numeric(df_esq[c], errors='coerce')
    # Excluir puntos centrales: usar filas confirmadas si existen, sino buscar ceros
    if 'filas_centro_idx' in st.session_state and st.session_state.filas_centro_idx:
        df_esq = df_esq.drop(st.session_state.filas_centro_idx, errors='ignore')
    else:
        df_esq = df_esq[~df_esq[cols].apply(lambda r: all(abs(v) < 0.5 for v in r), axis=1)]
    if len(df_esq) == 0:
        return 0
    df_group = df_esq.groupby(cols)[resp_col].count()
    return int((df_group > 1).sum())


def _detectar_bloques(raw, cols):
    if 'Block' in raw.columns or 'Bloque' in raw.columns:
        return True
    k = len(cols)
    n = len(raw)
    if 'filas_centro_idx' in st.session_state and st.session_state.filas_centro_idx:
        centros = len(st.session_state.filas_centro_idx)
    else:
        centros = _detectar_centros(raw, cols)[0]
    esquinas = n - centros
    potencia = 2 ** k
    return esquinas > potencia


def _validar_analisis_vs_diseno(raw, factores, respuesta, analisis):
    k = len(factores)
    n = len(raw)
    if 'filas_centro_idx' in st.session_state and st.session_state.filas_centro_idx:
        centros = len(st.session_state.filas_centro_idx)
    else:
        centros = _detectar_centros(raw, factores)[0]
    esquinas = n - centros
    replicas = _detectar_replicas(raw, factores, respuesta)
    tiene_bloques = _detectar_bloques(raw, factores)
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


def _sugerir_analisis(raw, factores, respuesta):
    n = len(raw)
    k = len(factores)
    # Usar filas confirmadas si existen
    if 'filas_centro_idx' in st.session_state and st.session_state.filas_centro_idx:
        centros = len(st.session_state.filas_centro_idx)
    else:
        centros = _detectar_centros(raw, factores)[0]
    esquinas = n - centros
    replicas = _detectar_replicas(raw, factores, respuesta)

    if centros > 0:
        return 'factorial_via2', f"Detectado: {esquinas} esquinas + {centros} centros. Sugerencia: Via 2."
    elif replicas > 0:
        return 'factorial_via1', f"Detectado: {esquinas} esquinas con {replicas} grupos de replicas. Sugerencia: Via 1."
    else:
        return 'factorial_via3', f"Detectado: {esquinas} esquinas sin replicas ni centros. Sugerencia: Via 3."


def _agregar_centros(raw, factores, respuesta, n_centros=3):
    """Genera n puntos centrales y los añade a la matriz."""
    df = raw.copy()
    # Calcular valor medio de cada factor
    centros_data = {}
    for f in factores:
        vals = pd.to_numeric(df[f], errors='coerce').dropna()
        # Excluir los 0 actuales (si los hay) para calcular el rango real
        vals_no_cero = vals[vals != 0]
        if len(vals_no_cero) > 0:
            v_min = vals_no_cero.min()
            v_max = vals_no_cero.max()
            v_medio = (v_min + v_max) / 2
        else:
            v_medio = vals.mean()
        centros_data[f] = v_medio

    # Crear filas de centros
    nuevas_filas = []
    for i in range(n_centros):
        fila = {f: centros_data[f] for f in factores}
        fila[respuesta] = np.nan  # Valor desconocido
        nuevas_filas.append(fila)

    df_centros = pd.DataFrame(nuevas_filas)
    df_completo = pd.concat([df, df_centros], ignore_index=True)
    return df_completo


def _codificar_factor(serie, v_bajo, v_alto):
    v_medio = (v_bajo + v_alto) / 2
    rango = (v_alto - v_bajo) / 2
    if rango == 0:
        return serie
    return (serie - v_medio) / rango


def _descodificar_factor(serie_cod, v_bajo, v_alto):
    v_medio = (v_bajo + v_alto) / 2
    rango = (v_alto - v_bajo) / 2
    return serie_cod * rango + v_medio


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
            'preview': '2. Revisar matriz',
            'encode': '3. Codificar / Descodificar',
            'analysis_select': '4. Elegir analisis',
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
    2. **El programa detecta** si esta codificada o no
    3. **Codifica o descodifica** segun necesites
    4. **Elige el tipo de analisis** (Via 1, 2, 3 o CCD)
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


# ---------- ETAPA 2: PREVIEW ----------
elif st.session_state.stage == 'preview':
    st.title("2. Revisar Matriz")

    raw = st.session_state.raw

    _info_box("Paso 2: Revision",
              "Confirma que el programa detecto correctamente los factores y la respuesta.",
              tipo='info')

    fact_cands, resp_cands, _ = _detectar_columnas(raw)

    corrida_opts = ['(Ninguna)'] + [c for c in raw.columns if any(x in str(c).lower() for x in ['corrida','run','orden','order','id'])]
    corrida_sel = st.selectbox("Columna de Corrida (opcional):", corrida_opts, index=0)
    st.session_state.corrida_col = None if corrida_sel == '(Ninguna)' else corrida_sel

    st.markdown("### Factores")
    factores = st.multiselect("Selecciona los factores:", raw.columns.tolist(), default=fact_cands, key="sel_factores")

    st.markdown("### Respuesta")
    respuesta = st.selectbox("Selecciona la respuesta:", raw.columns.tolist(),
                              index=raw.columns.tolist().index(resp_cands[0]) if resp_cands else 0,
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

        centros = _detectar_centros(raw, factores)[0]
        replicas = _detectar_replicas(raw, factores, respuesta)

        st.markdown("---")
        st.markdown("### ¿Que detecto el programa?")

        # Detectar centros automaticamente
        n_centros_detect, idx_centros = _detectar_centros(raw, factores)
        # Si el usuario ya confirmo filas, usar esas
        if 'filas_centro_idx' in st.session_state and st.session_state.filas_centro_idx:
            idx_centros = st.session_state.filas_centro_idx
            n_centros_detect = len(idx_centros)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Factores", len(factores))
        with col2:
            st.metric("Puntos centrales", n_centros_detect, delta="(filas con todos los factores ~ 0)")
        with col3:
            st.metric("Replicas puras", replicas, delta="(filas identicas en factores)")

        # Confirmacion de puntos centrales
        if n_centros_detect > 0:
            filas_str = ', '.join(str(i+1) for i in idx_centros)
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
                        validas = [i for i in filas_idx if 0 <= i < len(raw)]
                        if len(validas) != len(filas_idx):
                            st.warning(f"Algunas filas estan fuera de rango (max: {len(raw)}). Se usaran solo las validas.")
                        if validas:
                            st.session_state.filas_centro_idx = validas
                            st.session_state.centros_confirmados = True
                            st.success(f"Marcadas como centros las filas: {', '.join(str(i+1) for i in validas)}")
                    except ValueError:
                        st.error("Formato invalido. Usa numeros separados por coma.")
        else:
            st.info("No detecte puntos centrales (filas donde todos los factores estan cerca de 0).")
            correcto = st.radio("¿Tu matriz SI tiene puntos centrales?", ["No, no tiene", "Si, pero no los detecto"], horizontal=True, key="confirm_no_centros")
            if correcto == "Si, pero no los detecto":
                st.markdown("Escribe los numeros de fila separados por coma (empezando desde 1):")
                filas_centro_input = st.text_input("Filas de centro", placeholder="ej: 3,6,11", key="filas_centro_manual2")
                if filas_centro_input:
                    try:
                        filas_idx = [int(x.strip()) - 1 for x in filas_centro_input.split(",")]
                        validas = [i for i in filas_idx if 0 <= i < len(raw)]
                        if validas:
                            st.session_state.filas_centro_idx = validas
                            st.session_state.centros_confirmados = True
                            st.success(f"Marcadas como centros las filas: {', '.join(str(i+1) for i in validas)}")
                    except ValueError:
                        st.error("Formato invalido. Usa numeros separados por coma.")
            else:
                st.session_state.centros_confirmados = True
                st.session_state.filas_centro_idx = []

        # Explicacion de puntos centrales
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

        st.markdown(f"""
        **Codificacion detectada:** {'Codificada (-1, 0, +1)' if es_cod else 'NO codificada (valores reales)'}
        """)

        # Recalcular centros para la sugerencia
        centros_final = len(st.session_state.get('filas_centro_idx', []))
        sug_analisis, sug_msg = _sugerir_analisis(raw, factores, respuesta)
        _info_box("Sugerencia del programa", sug_msg, tipo='tip')

        # Boton para agregar centros automaticamente (si no hay y no confirmo)
        if centros_final == 0 and sug_analisis == 'factorial_via2' and not st.session_state.get('centros_confirmados'):
            st.markdown("---")
            _info_box("No tienes puntos centrales",
                      "El programa detecto que necesitas puntos centrales para el Analisis Via 2. "
                      "Puedes agregarlos automaticamente.",
                      tipo='warning')
            n_centros = st.number_input("Cuantos puntos centrales quieres agregar?", min_value=1, max_value=10, value=3, step=1)
            if st.button("➕ Agregar puntos centrales automaticamente", use_container_width=True):
                raw_nuevo = _agregar_centros(raw, factores, respuesta, n_centros=int(n_centros))
                st.session_state.raw = raw_nuevo
                st.success(f"Se agregaron {int(n_centros)} puntos centrales. La matriz ahora tiene {len(raw_nuevo)} filas.")
                st.rerun()

        # Botones de navegacion
        st.markdown("---")
        nav_c = st.columns([1, 3])
        with nav_c[0]:
            if st.button("← Regresar", use_container_width=True):
                st.session_state.stage = 'upload'
                st.rerun()
        with nav_c[1]:
            if st.button("Continuar a codificacion →", type='primary', use_container_width=True):
                st.session_state.stage = 'encode'
                st.rerun()


# ---------- ETAPA 3: CODIFICAR / DESCODIFICAR ----------
elif st.session_state.stage == 'encode':
    st.title("3. Codificar / Descodificar")

    raw = st.session_state.raw
    factores = st.session_state.factores
    respuesta = st.session_state.respuesta
    es_cod = st.session_state.es_codificada

    if es_cod:
        _info_box("Matriz codificada detectada",
                  "Tus factores estan en escala codificada (-1, 0, +1). Introduce los valores reales.",
                  tipo='info')
    else:
        _info_box("Matriz NO codificada detectada",
                  "Tus factores estan en valores reales. Introduce los niveles bajo y alto.",
                  tipo='info')

    mapeos_niveles = {}
    mapeos_inversos = {}

    st.markdown("### Configurar niveles")

    for f in factores:
        vals = sorted(raw[f].dropna().unique().tolist())
        st.markdown(f"**Factor: {f}**")

        if es_cod:
            c1, c2 = st.columns(2)
            # Leer valor previo de session_state si existe
            def_val_low = st.session_state.get(f"real_low_{f}", 0.0)
            def_val_high = st.session_state.get(f"real_high_{f}", 100.0)
            v_bajo = c1.number_input(f"Valor real para -1 (bajo) - {f}", value=def_val_low, key=f"real_low_{f}")
            v_alto = c2.number_input(f"Valor real para +1 (alto) - {f}", value=def_val_high, key=f"real_high_{f}")
            v_medio = (v_bajo + v_alto) / 2
            st.caption(f"Centro (0) calculado: **{v_medio:.4f}**")
            mapeos_niveles[f] = {v_bajo: -1, v_medio: 0, v_alto: 1}
            mapeos_inversos[f] = {-1: v_bajo, 0: v_medio, 1: v_alto}
        else:
            if len(vals) == 2:
                v_bajo = vals[0]
                v_alto = vals[1]
                v_medio = (v_bajo + v_alto) / 2
                st.info(f"Detectados 2 niveles: Bajo={v_bajo}, Alto={v_alto}. Centro: {v_medio:.4f}")
            elif len(vals) == 3:
                v_bajo, v_medio, v_alto = vals[0], vals[1], vals[2]
                st.info(f"Detectados 3 niveles: Bajo={v_bajo}, Centro={v_medio}, Alto={v_alto}")
            else:
                c1, c2 = st.columns(2)
                def_val_low = st.session_state.get(f"low_{f}", float(min(vals)))
                def_val_high = st.session_state.get(f"high_{f}", float(max(vals)))
                v_bajo = c1.number_input(f"Nivel BAJO para {f}", value=def_val_low, key=f"low_{f}")
                v_alto = c2.number_input(f"Nivel ALTO para {f}", value=def_val_high, key=f"high_{f}")
                v_medio = (v_bajo + v_alto) / 2
                st.caption(f"Centro calculado: **{v_medio:.4f}**")

            mapeos_niveles[f] = {v_bajo: -1, v_medio: 0, v_alto: 1}
            mapeos_inversos[f] = {-1: v_bajo, 0: v_medio, 1: v_alto}

    st.session_state.mapeos_niveles = mapeos_niveles
    st.session_state.mapeos_inversos = mapeos_inversos

    df_cod = raw.copy()
    df_real = raw.copy()

    if es_cod:
        for f in factores:
            df_real[f] = _descodificar_factor(raw[f], mapeos_inversos[f][-1], mapeos_inversos[f][1])
        df_cod = raw.copy()
    else:
        for f in factores:
            df_cod[f] = _codificar_factor(raw[f], mapeos_inversos[f][-1], mapeos_inversos[f][1])
        df_real = raw.copy()

    st.markdown("---")
    st.markdown("### Matrices disponibles")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Matriz Codificada (-1, 0, +1)**")
        st.dataframe(df_cod[factores + [respuesta]], use_container_width=True)
    with c2:
        st.markdown("**Matriz con Valores Reales**")
        st.dataframe(df_real[factores + [respuesta]], use_container_width=True)

    st.markdown("### ¿Con cual matriz quieres hacer el analisis?")
    eleccion = st.radio("Selecciona:", ["Codificada", "Valores Reales"], horizontal=True)

    st.markdown("---")
    nav_c = st.columns([1, 3])
    with nav_c[0]:
        if st.button("← Regresar a revision", use_container_width=True):
            st.session_state.stage = 'preview'
            st.rerun()
    with nav_c[1]:
        if st.button("Continuar a seleccion de analisis →", type='primary', use_container_width=True):
            if eleccion == "Codificada":
                st.session_state.df_work = df_cod.copy()
            else:
                st.session_state.df_work = df_real.copy()
            st.session_state.stage = 'analysis_select'
            st.rerun()


# ---------- ETAPA 4: SELECCION DE ANALISIS ----------
elif st.session_state.stage == 'analysis_select':
    st.title("4. Elegir Tipo de Analisis")

    raw = st.session_state.df_work
    factores = st.session_state.factores
    respuesta = st.session_state.respuesta
    k = len(factores)

    _info_box("Paso 4: Seleccion del analisis",
              "Elige el tipo de analisis que mejor se adapte a tu diseno.",
              tipo='info')

    centros = _detectar_centros(raw, factores)[0]
    replicas = _detectar_replicas(raw, factores, respuesta)
    esq = len(raw) - centros

    st.markdown(f"""
    **Caracteristicas de tu diseno:**
    - Factores: **{k}** ({', '.join(factores)})
    - Esquinas: **{esq}** | Centros: **{centros}** | Replicas: **{replicas}**
    - Total filas: **{len(raw)}**
    """)

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
        una fila con Temp=125 y pH=7.5 es un **punto central**.
        """)

    opciones = {
        'factorial_via1': f'Via 1: 2^{k} con replicas puras',
        'factorial_via2': f'Via 2: 2^{k} con puntos centrales ({centros} detectados)',
        'factorial_via3': f'Via 3: 2^{k} no replicado (Lenth)',
        'ccd': 'Superficie de Respuesta (CCD)',
    }

    analisis = st.radio("Selecciona tu analisis:", list(opciones.keys()),
                        format_func=lambda x: opciones[x], key="sel_analisis")
    st.session_state.analisis_tipo = analisis

    warnings = _validar_analisis_vs_diseno(raw, factores, respuesta, analisis)
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

            **Resolucion:** Indica que efectos pueden estar confundidos.
            - **Resolucion V** o mas: Efectos principales e interacciones de 2 vias son libres de confusion.
            - **Resolucion III**: Los principales pueden estar confundidos con interacciones de 2 vias.
            """)

        es_fracc = st.checkbox("Si, mi diseno es fraccionado", value=False)
        st.session_state.modelo_fraccionado = es_fracc

        if es_fracc:
            p = st.selectbox("Fraccion (p) - Recomendado: p=1:", [1, 2, 3], index=0,
                            help="p=1 = media fraccion (recomendado). p=2 = cuarto de fraccion.")
            resolucion = k - p + 1
            esquinas_frac = 2**(k-p)

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
        if st.button("← Regresar a matrices", use_container_width=True):
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

    raw = st.session_state.df_work
    factores = st.session_state.factores
    respuesta = st.session_state.respuesta
    analisis = st.session_state.analisis_tipo

    df = raw.copy()
    df = df.rename(columns={respuesta: 'Y'})
    df['Grupo'] = 'G1'
    df['Curvatura'] = df[factores].apply(lambda r: 1 if all(abs(v) < 0.01 for v in r) else 0, axis=1)
    for f in factores:
        df[f] = pd.to_numeric(df[f], errors='coerce')

    st.markdown("---")
    st.markdown("### Configuracion del Modelo")

    c1, c2 = st.columns(2)
    with c1:
        modo = st.radio("Seleccion de terminos:", ["Automatico (Backward)", "Manual"], index=0, key="modo_sel")
        st.session_state.modo_seleccion = 'backward' if 'Automatico' in modo else 'manual'
    with c2:
        alpha = st.number_input("Nivel de significancia (alpha):", min_value=0.01, max_value=0.5,
                               value=0.05, step=0.01, format="%.2f", key="alpha_val")
        st.session_state.alpha_modelo = alpha

    _info_box("Que es alpha?",
              f"alpha = {alpha} significa que aceptamos un {alpha*100:.0f}% de probabilidad de equivocarnos al declarar un efecto significativo.",
              tipo='tip')

    try:
        if analisis == 'factorial_via1':
            if st.session_state.modo_seleccion == 'backward':
                R = ce.calcular_via1_con_terminos(df, factores, terms=None)
            else:
                ef_all = ce._calc_effects(df, factores, len(df[df['Curvatura']==0]) // df['Grupo'].nunique())
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

        elif analisis == 'factorial_via2':
            if st.session_state.modo_seleccion == 'backward':
                R = ce.calcular_via2_con_terminos(df, factores, terms=None, alpha=alpha)
            else:
                ef_all = ce._calc_effects(df, factores, len(df[df['Curvatura']==0]) // df['Grupo'].nunique())
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

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("R2", f"{R['R2']:.4f}")
        col2.metric("R2 ajustado", f"{R['R2_adj']:.4f}")
        col3.metric("Error estandar", f"{np.sqrt(R['MS_resid']):.4f}")
        col4.metric("Terminos en modelo", len(R['terms_included']))

        _info_box("Que significa R2?",
                  f"R2 = {R['R2']:.4f} indica que el modelo explica el {R['R2']*100:.2f}% de la variacion en la respuesta.",
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
                anova[col] = anova[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "-")
        for col in ['p']:
            if col in anova.columns:
                anova[col] = anova[col].apply(lambda x: f"{x:.6f}" if pd.notna(x) else "-")

        st.dataframe(anova, use_container_width=True)

        # COEFICIENTES
        st.markdown("---")
        st.markdown("## Coeficientes del Modelo")
        try:
            coef_df, fit_df = ce.generar_tabla_regresion(df, R)
            st.dataframe(coef_df.round(4), use_container_width=True)
        except Exception as e:
            st.warning(f"No se pudieron generar coeficientes: {e}")

        # ECUACION
        st.markdown("---")
        st.markdown("## Ecuacion de Regresion")
        try:
            eq = ce._ecuacion_str(R)
            st.code(eq, language='text')
            st.copy_button("Copiar ecuacion", eq, key="copy_eq")
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
                fig = ce._gen_pareto(R)
                st.pyplot(fig)
                _info_box("Diagrama de Pareto", "Efectos ordenados por magnitud. La linea vertical es el valor critico.", tipo='tip')
            except Exception as e:
                st.warning(f"No se pudo generar Pareto: {e}")

        with tab2:
            try:
                fig = ce._gen_seminormal(R)
                st.pyplot(fig)
                _info_box("Grafico Semi-normal", "Efectos significativos se desvian de la linea recta.", tipo='tip')
            except Exception as e:
                st.warning(f"No se pudo generar semi-normal: {e}")

        with tab3:
            try:
                fig = ce._gen_main_effects(R, spec=None, mapeos=st.session_state.mapeos_inversos)
                st.pyplot(fig)
            except Exception as e:
                st.warning(f"No se pudieron generar efectos principales: {e}")

        with tab4:
            try:
                fig_list = ce._gen_interactions(R, mapeos=st.session_state.mapeos_inversos)
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

        pred_cols = st.columns(len(factores))
        niveles = {}
        for i, f in enumerate(factores):
            if f in st.session_state.mapeos_inversos and -1 in st.session_state.mapeos_inversos[f]:
                mapeo = st.session_state.mapeos_inversos[f]
                opts = [(-1, f"Bajo ({mapeo.get(-1, -1)})"), (0, f"Centro ({mapeo.get(0, 0)})"), (1, f"Alto ({mapeo.get(1, 1)})")]
                sel = pred_cols[i].selectbox(f, opts, format_func=lambda x: x[1], key=f"pred_{f}")
                niveles[f] = sel[0]
            else:
                niveles[f] = pred_cols[i].number_input(f, value=0.0, key=f"pred_{f}")

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
                alias_df = ce.generar_tabla_alias(factores, st.session_state.resolucion)
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
        if st.button("🔢 Revisar matrices", use_container_width=True):
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
            st.session_state.stage = 'upload'
            st.rerun()
