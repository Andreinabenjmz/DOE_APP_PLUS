# app_plus.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import io
import zipfile
from datetime import date
from docx import Document
import calc_engine_plus as ce
from docx.shared import Inches, Pt

def _insertar_tabla_word(doc, df_tabla):
    cols = list(df_tabla.columns)
    t = doc.add_table(rows=1, cols=len(cols)); t.style = 'Table Grid'
    hdr_cells = t.rows[0].cells
    for i, col_name in enumerate(cols):
        hdr_cells[i].text = str(col_name)
        for paragraph in hdr_cells[i].paragraphs:
            for run in paragraph.runs: run.font.bold = True; run.font.size = Pt(9)
    for _, row in df_tabla.iterrows():
        row_cells = t.add_row().cells
        for i, col_name in enumerate(cols):
            val = row[col_name]
            row_cells[i].text = "—" if pd.isna(val) else f"{val:.4f}" if isinstance(val, float) else str(val)
            for paragraph in row_cells[i].paragraphs:
                for run in paragraph.runs: run.font.size = Pt(9)
    doc.add_paragraph()

st.set_page_config(page_title="DOE 2^k - Modo Minitab", layout="wide")

for k, v in {'stage': 'upload', 'raw': None, 'df': None, 'via': None, 'factores': None, 'respuesta': None, 'spec': None, 'R_full': None, 'R_reduced': None, 'selected_terms': None, 'nombre': '', 'estudiante': '', 'curso': '', 'excel_bytes': None, 'word_bytes': None, 'figs_dict': None}.items():
    st.session_state.setdefault(k, v)

def _detectar_columnas(raw):
    factores, respuestas, descartar = [], [], []
    id_names = {'run','corrida','corrida_id','id','n_corrida','numero_corrida','obs','observacion','index','unnamed: 0'}
    resp_exact = {'y', 'response', 'respuesta', 'resp'}
    resp_hints = ['rep', 'replica', 'block', 'bloque']
    for c in raw.columns:
        nombre_lower = str(c).strip().lower()
        vals = raw[c].dropna().unique().tolist(); n_unicos = len(vals)
        if nombre_lower in id_names or n_unicos <= 1: descartar.append(c); continue
        if (nombre_lower in resp_exact or any(h in nombre_lower for h in resp_hints)) and pd.api.types.is_numeric_dtype(raw[c]): respuestas.append(c); continue
        if n_unicos == 2: factores.append(c); continue
        if n_unicos == 3:
            try:
                if set([float(v) for v in vals]) == {-1.0, 0.0, 1.0}: factores.append(c); continue
            except: pass
        if pd.api.types.is_numeric_dtype(raw[c]): respuestas.append(c); continue
        factores.append(c)
    return factores, respuestas, descartar

# --- SIDEBAR ---
with st.sidebar:
    st.header("Configuracion")
    if st.button("Reiniciar"):
        for k in list(st.session_state.keys()): del st.session_state[k]
        st.rerun()

    uploaded = st.file_uploader("Archivo Excel o CSV", type=['xlsx','csv'])
    if uploaded:
        try:
            raw = pd.read_excel(uploaded) if uploaded.name.endswith('.xlsx') else pd.read_csv(uploaded)
            raw.columns = raw.columns.astype(str).str.strip()
            st.session_state.raw = raw
            st.success(f"OK: {raw.shape[0]} filas x {raw.shape[1]} cols")
        except Exception as e: st.error(f"Error: {e}")

    if st.session_state.raw is not None:
        raw = st.session_state.raw
        st.markdown("### Columnas")
        fact_cands, resp_cands, _ = _detectar_columnas(raw)

        cols = st.multiselect("Factores", raw.columns.tolist(), default=fact_cands, key="mis_factores")
        resp = st.multiselect("Respuesta(s)", raw.columns.tolist(), default=resp_cands, key="mi_respuesta")

        mapping_nombres = {}
        if len(cols) > 0:
            with st.expander("✏️ Renombrar factores (opcional)"):
                for c in cols:
                    nuevo = st.text_input(f"Nombre para '{c}':", value=c, key=f"rename_{c}")
                    if nuevo.strip() != "" and nuevo.strip() != c: mapping_nombres[c] = nuevo.strip()
        
        mapeos_niveles = {}
        if len(cols) > 0:
            with st.expander("⚙️ Configurar Niveles (-1 / +1)", expanded=False):
                for c in cols:
                    col_real = mapping_nombres.get(c, c); vals = list(dict.fromkeys(raw[col_real].dropna().tolist()))
                    if set(vals).issubset({-1,0,1,-1.0,0.0,1.0}): continue
                    if len(vals) == 2:
                        v_bajo = st.selectbox(f"'{col_real}' - Nivel BAJO (-1):", vals, index=0, key=f"low_{c}")
                        v_alto = [v for v in vals if v != v_bajo][0]; st.caption(f"→ Nivel ALTO (+1): **{v_alto}**")
                        mapeos_niveles[col_real] = {v_bajo: -1, v_alto: 1}
                    elif len(vals) == 3:
                        v_bajo = st.selectbox(f"'{col_real}' - Nivel BAJO (-1):", vals, index=0, key=f"low_{c}")
                        v_alto = st.selectbox(f"'{col_real}' - Nivel ALTO (+1):", [v for v in vals if v != v_bajo], index=0, key=f"high_{c}")
                        v_mid = [v for v in vals if v != v_bajo and v != v_alto][0]; st.caption(f"→ Centro (0): **{v_mid}**")
                        mapeos_niveles[col_real] = {v_bajo: -1, v_mid: 0, v_alto: 1}
            st.markdown("---")

        via = st.radio("Via", options=[ce.VIA_1, ce.VIA_2, ce.VIA_3], format_func=lambda v: f"{ce.VIA_INFO[v]['nombre']}")
        objetivo = st.radio("Objetivo del análisis:", ["Maximizar la respuesta", "Minimizar la respuesta"], index=0)

        use_spec = st.checkbox("Especificacion (target +/- tol)")
        spec = None
        if use_spec:
            c1, c2 = st.columns(2); spec = {'target': c1.number_input("Target", value=24.0), 'tol': c2.number_input("Tolerancia", value=4.0)}

        st.markdown("### Contexto")
        nombre = st.text_input("Proceso", "Analisis 2^k")
        estudiante = st.text_input("Estudiante", "")
        curso = st.text_input("Curso", "Asignacion")

        st.markdown("### 🧪 Simulador de Datos")
        simular_datos = st.checkbox("Generar datos con ecuación matemática (añadir ruido)", key="sim_check")
        
        if simular_datos:
            if 'sim_b0' not in st.session_state:
                st.session_state.sim_b0, st.session_state.sim_bA, st.session_state.sim_bB, st.session_state.sim_bC, st.session_state.sim_bD, st.session_state.sim_bE, st.session_state.sim_bAA = 2.5, 0.6, 0.3, -0.5, 0.2, -0.4, -0.2

            if st.button("🎲 Autogenerar coeficientes plausibles"):
                st.session_state.sim_b0 = float(np.random.uniform(2.0, 3.5)); st.session_state.sim_bA = float(np.random.uniform(0.4, 0.8))
                st.session_state.sim_bB = float(np.random.uniform(0.2, 0.5)); st.session_state.sim_bC = float(-np.random.uniform(0.3, 0.7))
                st.session_state.sim_bD = float(np.random.uniform(0.1, 0.3)); st.session_state.sim_bE = float(-np.random.uniform(0.3, 0.6))
                st.session_state.sim_bAA = float(-np.random.uniform(0.1, 0.3)); st.rerun()

            c1, c2 = st.columns(2)
            b0 = c1.number_input("Intercepto (β0)", key="sim_b0", format="%.3f"); bA = c2.number_input("Efecto A (+)", key="sim_bA", format="%.3f")
            bB = c1.number_input("Efecto B (+)", key="sim_bB", format="%.3f"); bC = c2.number_input("Efecto C (-)", key="sim_bC", format="%.3f")
            bD = c1.number_input("Efecto D (+)", key="sim_bD", format="%.3f"); bE = c2.number_input("Efecto E (-)", key="sim_bE", format="%.3f")
            bAA = c1.number_input("Curvatura A² (-)", key="sim_bAA", format="%.3f")
            
            c3, c4 = st.columns(2)
            ruido = c3.number_input("Ruido (σ)", value=0.1)
            y_min = c4.number_input("Y mínimo", value=0.0) # <--- AQUÍ ESTÁ TU Y MÍNIMO
            y_max = st.number_input("Y máximo", value=5.0)

        # --- BOTÓN PROCESAR (En la barra lateral) ---
        if st.button("Procesar", type='primary'):
            try:
                if mapping_nombres:
                    raw = raw.rename(columns=mapping_nombres); cols = [mapping_nombres.get(c, c) for c in cols]
                rp = raw.copy(); rp.insert(0, 'Corrida_ID', range(1, len(rp) + 1))
                
                for f in cols:
                    if f in mapeos_niveles: rp[f] = rp[f].map(mapeos_niveles[f])
                    elif set(rp[f].dropna().unique()).issubset({-1,0,1,-1.0,0.0,1.0}): continue
                    else:
                        vals = list(dict.fromkeys(rp[f].dropna().tolist()))
                        if len(vals) == 2: lo, hi = min(vals), max(vals); rp[f] = rp[f].map({lo: -1, hi: 1})
                
                if simular_datos:
                    rp['Y_simulado'] = (b0 + bA * rp[cols[0]] + bB * rp[cols[1]] + bC * rp[cols[2]] + bD * rp[cols[3]] + bE * rp[cols[4]] + bAA * (rp[cols[0]]**2))
                    es_centro = rp[cols].eq(0).all(axis=1)
                    ruido_aleatorio = np.random.normal(0, ruido, len(rp))
                    rp.loc[~es_centro, 'Y_simulado'] += ruido_aleatorio[~es_centro]
                    rp['Y_simulado'] = np.clip(rp['Y_simulado'], y_min, y_max)
                    rp[resp[0]] = rp['Y_simulado']; rp = rp.drop(columns=['Y_simulado'])
                
                for c in resp: rp[c] = pd.to_numeric(rp[c], errors='raise')
                df = ce.preparar_datos(rp, cols, resp)

                st.session_state.df = df; st.session_state.via = via
                st.session_state.objetivo = 'max' if objetivo == "Maximizar la respuesta" else 'min'
                st.session_state.factores = cols; st.session_state.respuesta = resp; st.session_state.spec = spec
                st.session_state.nombre = nombre; st.session_state.estudiante = estudiante; st.session_state.curso = curso
                st.session_state.stage = 'design'
                st.session_state.R_full = None; st.session_state.R_reduced = None; st.session_state.selected_terms = None
                st.session_state.excel_bytes = None; st.session_state.word_bytes = None; st.session_state.figs_dict = None
                st.rerun()
            except Exception as e: st.error(f"Error: {e}")

# --- MAIN ---
if st.session_state.stage == 'upload':
    st.title("DOE 2^k - Modo Minitab")
    
    # PREVISUALIZACIÓN EN PANTALLA PRINCIPAL
    if st.session_state.raw is not None and 'simular_datos' in locals() and simular_datos:
        st.markdown("## 👀 Previsualización de Datos Simulados")
        st.markdown("Revisa la tabla generada. Si estás de acuerdo, presiona 'Procesar' en la barra lateral izquierda.")
        
        if len(cols) == 5 and len(resp) > 0:
            rp_temp = st.session_state.raw.copy()
            for f in cols:
                if f in mapeos_niveles: rp_temp[f] = rp_temp[f].map(mapeos_niveles[f])
                else:
                    vals = list(dict.fromkeys(rp_temp[f].dropna().tolist()))
                    if len(vals) == 2: lo, hi = min(vals), max(vals); rp_temp[f] = rp_temp[f].map({lo: -1, hi: 1})
            
            y_sim = (b0 + bA * rp_temp[cols[0]] + bB * rp_temp[cols[1]] + bC * rp_temp[cols[2]] + bD * rp_temp[cols[3]] + bE * rp_temp[cols[4]] + bAA * (rp_temp[cols[0]]**2))
            es_centro = rp_temp[cols].eq(0).all(axis=1)
            ruido_aleatorio = np.random.normal(0, ruido, len(rp_temp))
            y_sim.loc[~es_centro] += ruido_aleatorio[~es_centro]
            y_sim = np.clip(y_sim, y_min, y_max)
            
            df_preview = rp_temp[cols].copy()
            df_preview.insert(0, 'Corrida', range(1, len(df_preview) + 1))
            df_preview['Y_Simulada'] = y_sim.round(3)
            st.dataframe(df_preview, use_container_width=True)
        else:
            st.warning("Selecciona exactamente 5 factores y 1 respuesta para previsualizar.")
            
    elif st.session_state.raw is not None:
        st.markdown("## 📁 Datos Originales Cargados")
        st.dataframe(st.session_state.raw, use_container_width=True)
    else:
        st.markdown("### Como funciona\n\n1. Sube tu archivo Excel.\n2. Configura tus factores.\n3. Activa la simulación si lo deseas.\n4. Previsualiza los datos y presiona Procesar.")
        st.info("Empieza subiendo un archivo en la barra lateral")

elif st.session_state.stage == 'design':
    df = st.session_state.df; factores = st.session_state.factores; via = st.session_state.via
    st.title("Paso 1 - Diseno experimental")
    n_grupos = df['Grupo'].nunique(); n_total = len(df)
    n_esquinas = len(df[df['Curvatura']==0]) // n_grupos if n_grupos > 0 else len(df[df['Curvatura']==0])
    n_centros = len(df[df['Curvatura']==1]) // n_grupos if n_grupos > 0 else len(df[df['Curvatura']==1])
    if via == ce.VIA_1: expl = f"N = {n_total} = {n_esquinas} esquinas x {n_grupos} replicas puras"
    elif via == ce.VIA_2: expl = f"N = {n_total} = {n_esquinas + n_centros} puntos ({n_esquinas} esquinas + {n_centros} centros) x {n_grupos} bloque(s)"
    else: expl = f"N = {n_total} = {n_esquinas} esquinas (sin replicas)"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Factores", len(factores)); c2.metric("N total", n_total)
    c3.metric("Bloques" if via == ce.VIA_2 else "Replicas", n_grupos if via != ce.VIA_3 else 1); c4.metric("Via", via)
    st.info(f"**De donde viene N:** {expl}")
    
    cond = df[df['Curvatura']==0][factores].drop_duplicates().reset_index(drop=True)
    cond.insert(0, 'Condicion', range(1, len(cond)+1))
    st.markdown("### Combinaciones del diseno"); st.dataframe(cond, use_container_width=True)

    st.markdown("---"); st.markdown("### 🕵️ Estructura de Alias (Confundimiento)")
    st.dataframe(ce.generar_tabla_alias(df, factores), use_container_width=True)

    st.markdown("---")
    c1, c2 = st.columns(2)
    if c1.button("⬅️ Volver a Datos / Simulación", use_container_width=True):
        st.session_state.stage = 'upload'; st.session_state.df = None; st.session_state.R_full = None; st.rerun()
    if c2.button("Ir a Modelo completo ➡️", type='primary', use_container_width=True):
        st.session_state.stage = 'full_model'; st.rerun()

elif st.session_state.stage == 'full_model':
    st.title("Paso 2 - Modelo completo")
    df = st.session_state.df; factores = st.session_state.factores; via = st.session_state.via
    if st.session_state.R_full is None:
        with st.spinner("Calculando..."):
            try:
                if via == ce.VIA_1: R = ce.calcular_via1_con_terminos(df, factores, terms=None)
                elif via == ce.VIA_2: R = ce.calcular_via2_con_terminos(df, factores, terms=None)
                else: R = ce.calcular_via3(df, factores)
                st.session_state.R_full = R
            except Exception as e:
                st.error(f"ERROR EN EL CÁLCULO: {e}"); st.stop()
    R = st.session_state.R_full

    st.markdown("### 1. Tabla ANOVA"); st.dataframe(R['anova'].round(4), use_container_width=True)
    st.markdown("### 2. Coeficientes del Modelo")
    try:
        coef_df, fit_df = ce.generar_tabla_regresion(df, R)
        if via == ce.VIA_3: st.dataframe(coef_df[['Término', 'Coef']].round(4), use_container_width=True); st.info("Vía 3: modelo saturado, sin p-valores.")
        else: st.dataframe(coef_df.round(4), use_container_width=True)
    except Exception as e: st.warning(f"No se pudieron mostrar los coeficientes: {e}")
    st.write(f"**R2 = {R['R2']:.4f}**" + (f" | **R2 ajustado = {R['R2_adj']:.4f}**" if R.get('R2_adj') else ""))

    if via == ce.VIA_2:
        st.markdown("### 2. Diseño y Curvatura")
        c1, c2 = st.columns(2)
        fn_cub = ce._gen_cubo(R); fn_curv = ce._gen_curvatura(R)
        if fn_cub: c1.image(fn_cub, caption="Cubo 3D", use_container_width=True)
        if fn_curv: c2.image(fn_curv, caption="Curvatura", use_container_width=True)

    st.markdown("### 3. Diagrama de Pareto")
    if via == ce.VIA_3:
        if R.get('lenth', {}).get('valido'):
            c1, c2 = st.columns(2); c1.metric("PSE", f"{R['lenth']['PSE']:.4f}"); c2.metric("ME", f"{R['lenth']['ME']:.4f}")
            t_critico = R['lenth']['ME'] / R['lenth']['PSE']; st.info(f"Umbral |t| (Lenth): {t_critico:.2f}")
    else: st.info(f"Umbral |t| (Error Puro): {R['t_crit']:.2f}")
    st.image(ce._gen_pareto(R), use_container_width=True)

    st.markdown("### 4. Grafico semi-normal")
    st.image(ce._gen_seminormal(R), use_container_width=True)
    
    st.markdown("---")
    c1, c2 = st.columns(2)
    if c1.button("⬅️ Volver a Diseño"): st.session_state.stage = 'design'; st.rerun()
    if c2.button("Refinar modelo ➡️", type='primary'): st.session_state.stage = 'refine'; st.rerun()

elif st.session_state.stage == 'refine':
    st.title("Paso 3 - Refinar modelo")
    df = st.session_state.df; factores = st.session_state.factores; via = st.session_state.via
    R_full = st.session_state.R_full; all_terms = list(R_full['effects'].keys())

    if via == ce.VIA_3: st.markdown("**En esta Vía 3, el método de Lenth ya ha hecho el trabajo por ti.**")
    else: st.markdown("**Aqui construyes el modelo final.** Deja marcados los términos activos, desmarca el ruido y presiona 'Re-calcular'.")

    st.markdown("### Terminos disponibles")
    selected = []
    for term in all_terms:
        e = R_full['effects'][term]; c1, c2, c3, c4 = st.columns([3, 1, 1, 1]); display_term = term.replace('*', ' x ')
        if via == ce.VIA_3:
            activo = e.get('activo', False); c1.write(f"{'[ACTIVO]' if activo else '[RUIDO]'} {display_term}")
            if activo: selected.append(term)
        else:
            if c1.checkbox(display_term, value=True, key=f"chk_{term}"): selected.append(term)
        c2.write(f"Efecto: {e['efecto']:+.4f}"); p_val = e.get('p'); c3.write(f"p={p_val:.4f}" if pd.notna(p_val) else "p=Pooled" if via == ce.VIA_3 else "p=-")

    if len(selected) > 0:
        if via == ce.VIA_3:
            if st.button("Continuar al Diagnostico", type='primary'):
                st.session_state.R_reduced = R_full; st.session_state.selected_terms = selected; st.session_state.stage = 'diagnostics'; st.rerun()
        else:
            if st.button("Re-calcular con terminos seleccionados", type='primary'):
                with st.spinner("Ajustando modelo reducido..."):
                    try:
                        if via == ce.VIA_1: R_red = ce.calcular_via1_con_terminos(df, factores, terms=selected)
                        elif via == ce.VIA_2: R_red = ce.calcular_via2_con_terminos(df, factores, terms=selected)
                        st.session_state.R_reduced = R_red; st.session_state.selected_terms = selected; st.rerun()
                    except Exception as e: st.error(f"Error: {e}")

    if st.session_state.R_reduced is not None:
        R_red = st.session_state.R_reduced
        st.markdown("### 2. ANOVA del modelo reducido (Final)")
        st.dataframe(R_red['anova'].round(4), use_container_width=True)
        st.write(f"**R2 = {R_red['R2']:.4f}**" + (f" | **R2 ajustado = {R_red['R2_adj']:.4f}**" if R_red.get('R2_adj') else ""))
        with st.expander("🔬 Coeficientes e intervalos de confianza (Opcional)"):
            coef_df, _ = ce.generar_tabla_regresion(df, R_red)
            if R_red['via'] == ce.VIA_3: st.dataframe(coef_df[['Término', 'Coef']].round(4), use_container_width=True)
            else: st.dataframe(coef_df.round(4), use_container_width=True)

    c1, c2 = st.columns(2)
    if c1.button("Volver a Modelo completo"): st.session_state.stage = 'full_model'; st.rerun()
    if c2.button("Diagnostico", type='primary'): st.session_state.stage = 'diagnostics'; st.rerun()

elif st.session_state.stage == 'diagnostics':
    st.title("Paso 4 - Diagnostico")
    df = st.session_state.df; R = st.session_state.R_reduced if st.session_state.R_reduced else st.session_state.R_full

    st.markdown("### 1. Diagnostico de Residuales"); st.image(ce._gen_residuales(df, R), use_container_width=True)
    st.markdown("### 2. Efectos Principales"); st.image(ce._gen_main_effects(R, st.session_state.spec), use_container_width=True)
    
    st.markdown("#### Tabla Resumen de Efectos")
    filas_ef = [{'Factor': f, 'Media Nivel Bajo (-1)': round(R['main_effects'][f]['low'], 4), 'Media Nivel Alto (+1)': round(R['main_effects'][f]['high'], 4), 'Efecto (+1 vs -1)': round(R['main_effects'][f]['eff'], 4)} for f in st.session_state.factores]
    st.dataframe(pd.DataFrame(filas_ef), use_container_width=True)
    
    st.markdown("#### 🏆 Ranking de Condiciones Observadas (Top 5)")
    if 'Corrida_ID' in df.columns: cond_medias = df[df['Curvatura']==0].groupby(st.session_state.factores, sort=False).agg(Corrida=('Corrida_ID', 'first'), Y=('Y', 'mean')).reset_index()
    else: cond_medias = df[df['Curvatura']==0].groupby(st.session_state.factores, sort=False)['Y'].mean().reset_index(); cond_medias.insert(0, 'Corrida', range(1, len(cond_medias)+1))

    cond_medias['Y_Predicho'] = cond_medias.apply(lambda row: ce.predecir_y_combinacion(row.to_dict(), R), axis=1)
    cond_medias['Diferencia'] = cond_medias['Y'] - cond_medias['Y_Predicho']
    es_min = st.session_state.get('objetivo', 'max') == 'min'
    cond_medias = cond_medias.sort_values(by='Y', ascending=es_min).head(5).reset_index(drop=True)
    
    ranking_data = []
    for idx, row in cond_medias.iterrows():
        fila = {'Posición': idx + 1, 'Corrida': int(row['Corrida'])}
        for f in st.session_state.factores: fila[f] = "Nivel Bajo (-1)" if row[f] == -1 else "Nivel Alto (+1)" if row[f] == 1 else "Centro (0)"
        fila['Media Real (Y)'] = round(row['Y'], 4); fila['Predicción (Ecuación)'] = round(row['Y_Predicho'], 4); fila['Diferencia (Residuo)'] = round(row['Diferencia'], 4)
        ranking_data.append(fila)
    st.dataframe(pd.DataFrame(ranking_data), use_container_width=True)

    st.markdown("### 3. Interacciones")
    todas_interacciones = [n for n in R['effects'] if len(n.split('*'))==2]
    if todas_interacciones:
        orden_interacciones = st.multiselect("🛠️ Selecciona y ordena las interacciones a graficar:", options=todas_interacciones, default=todas_interacciones, key="selector_interacciones")
        st.session_state.orden_interacciones = orden_interacciones 
    fns_int = ce._gen_interactions(R, custom_order=st.session_state.get('orden_interacciones', []))
    if fns_int:
        for fn in fns_int:
            if os.path.exists(fn): st.image(fn, use_container_width=True)

    if st.session_state.spec:
        st.markdown("### 4. Cumplimiento de Especificacion"); st.image(ce._gen_spec(df, R, st.session_state.spec), use_container_width=True)
    
    c1, c2 = st.columns(2)
    if c1.button("Volver a Refinar"): st.session_state.stage = 'refine'; st.rerun()
    if c2.button("Avanzar al Reporte Final", type='primary'): st.session_state.stage = 'report'; st.rerun()

elif st.session_state.stage == 'report':
    st.title("Paso 5 - Reporte final")
    R = st.session_state.R_reduced if st.session_state.R_reduced else st.session_state.R_full
    df = st.session_state.df; nombre = st.session_state.nombre; slug = nombre.replace(' ', '_').replace('/', '-')

    st.markdown("### 1. Ecuacion de Regresion")
    if R['via'] == ce.VIA_2: st.write(f"**Media global** = {R['y_bar']:.4f}"); st.write(f"**Intercepto (Bloque 1)** = {R.get('intercepto_bloques', R['y_bar']):.4f}")
    else: st.write(f"**Beta 0** = {R['y_bar']:.4f}")

    eq = ce._ecuacion_str(R)
    if eq: st.code(eq, language=None)
    else: st.info("Ningun termino significativo. La ecuacion es solo la media (Beta 0).")

    if not R.get('tiene_bloques') and not R.get('tiene_centro'): st.dataframe(ce.generar_tabla_balance_neto(R, st.session_state.get('objetivo', 'max')), use_container_width=True)

    st.markdown("### 2. Generar Archivos")
    if st.button("Generar descargas (Excel, Word, PNGs)", type='primary'):
        with st.spinner("Generando reporte completo..."):
            tablas = {'ANOVA': R['anova']}
            Rf = st.session_state.R_full
            if Rf: tablas['Efectos_todos'] = pd.DataFrame([{'Termino': n, 'Efecto': e['efecto'], 'Coef': e['coef'], 'p': e.get('p', np.nan), 'En modelo final': 'Si' if n in (st.session_state.selected_terms or list(Rf['effects'].keys())) else 'No'} for n, e in Rf['effects'].items()])

            buf_xlsx = io.BytesIO()
            with pd.ExcelWriter(buf_xlsx) as w:
                for hoja, t in tablas.items(): t.to_excel(w, sheet_name=hoja[:31], index=False)

            figs = {'Pareto': ce._gen_pareto(R), 'SemiNormal': ce._gen_seminormal(R), 'Residuales': ce._gen_residuales(df, R), 'EfectosPrincipales': ce._gen_main_effects(R, st.session_state.spec)}
            orden_guardado = st.session_state.get('orden_interacciones', None)
            if not orden_guardado: orden_guardado = [n for n in R['effects'] if len(n.split('*'))==2]
            fns_int = ce._gen_interactions(R, custom_order=orden_guardado)
            if fns_int: figs['Interacciones'] = fns_int
            if st.session_state.via == ce.VIA_2:
                fn_cub = ce._gen_cubo(R); 
                if fn_cub: figs['Cubo'] = fn_cub
                fn_curv = ce._gen_curvatura(R); 
                if fn_curv: figs['Curvatura'] = fn_curv
            if st.session_state.spec: figs['Spec'] = ce._gen_spec(df, R, st.session_state.spec)

            doc = Document(); doc.add_heading(f"Informe de Análisis - {nombre}", 0)
            doc.add_paragraph(f"Estudiante: {st.session_state.estudiante}"); doc.add_paragraph(f"Curso: {st.session_state.curso}"); doc.add_paragraph(f"Fecha: {date.today().strftime('%Y-%m-%d')}")
            doc.add_heading("1. Diseño experimental", level=1); doc.add_paragraph(f"Factores: {', '.join(st.session_state.factores)}"); doc.add_paragraph(f"N = {len(df)}")
            doc.add_heading("2. Análisis y resultados", level=1)
            
            if not R.get('tiene_bloques') and not R.get('tiene_centro'): doc.add_heading("2.1. Balance Neto", level=2); _insertar_tabla_word(doc, ce.generar_tabla_balance_neto(R, st.session_state.get('objetivo', 'max'))); doc.add_heading("2.2. ANOVA", level=2)
            else: doc.add_heading("2.1. ANOVA", level=2)
            
            _insertar_tabla_word(doc, R['anova']); doc.add_paragraph(f"R2 = {R['R2']:.4f}" + (f" | R2 ajustado = {R['R2_adj']:.4f}" if R.get('R2_adj') else ""))
            
            doc.add_heading("2.2. Diagrama de Pareto", level=2)
            if 'Pareto' in figs and os.path.exists(figs['Pareto']): doc.add_picture(figs['Pareto'], width=Inches(6))
            
            doc.add_heading("2.3. Gráfico semi-normal", level=2)
            if 'SemiNormal' in figs and os.path.exists(figs['SemiNormal']): doc.add_picture(figs['SemiNormal'], width=Inches(6))
            
            doc.add_heading("2.4. Efectos principales", level=2)
            ef_princ_data = [{'Factor': f, 'Media Nivel Bajo (-1)': round(R['main_effects'][f]['low'], 4), 'Media Nivel Alto (+1)': round(R['main_effects'][f]['high'], 4), 'Efecto (+1 vs -1)': round(R['main_effects'][f]['eff'], 4)} for f in st.session_state.factores]
            _insertar_tabla_word(doc, pd.DataFrame(ef_princ_data))
            if 'EfectosPrincipales' in figs and os.path.exists(figs['EfectosPrincipales']): doc.add_picture(figs['EfectosPrincipales'], width=Inches(6))

            doc.add_heading("2.5. Condiciones Recomendadas (Ranking)", level=2)
            if 'Corrida_ID' in df.columns: cond_medias_word = df[df['Curvatura']==0].groupby(st.session_state.factores, sort=False).agg(Corrida=('Corrida_ID', 'first'), Y=('Y', 'mean')).reset_index()
            else: cond_medias_word = df[df['Curvatura']==0].groupby(st.session_state.factores, sort=False)['Y'].mean().reset_index(); cond_medias_word.insert(0, 'Corrida', range(1, len(cond_medias_word)+1))
            cond_medias_word['Y_Predicho'] = cond_medias_word.apply(lambda row: ce.predecir_y_combinacion(row.to_dict(), R), axis=1)
            cond_medias_word['Diferencia'] = cond_medias_word['Y'] - cond_medias_word['Y_Predicho']
            es_min = st.session_state.get('objetivo', 'max') == 'min'
            cond_medias_word = cond_medias_word.sort_values(by='Y', ascending=es_min).head(5).reset_index(drop=True)
            tabla_word = cond_medias_word.copy(); tabla_word.insert(0, 'Posición', range(1, len(tabla_word)+1))
            for f in st.session_state.factores: tabla_word[f] = tabla_word[f].apply(lambda x: "-1" if x==-1 else "+1" if x==1 else "0")
            tabla_word = tabla_word.rename(columns={'Y': 'Media Real', 'Y_Predicho': 'Predicción', 'Diferencia': 'Diferencia'})
            _insertar_tabla_word(doc, tabla_word)

            if 'Interacciones' in figs and figs['Interacciones']:
                doc.add_heading("2.6. Interacciones", level=2)
                for fn_int in figs['Interacciones']:
                    if os.path.exists(fn_int): doc.add_picture(fn_int, width=Inches(6))

            if st.session_state.via == ce.VIA_2:
                doc.add_heading("2.7. Curvatura y Diseño", level=2)
                if 'Cubo' in figs and os.path.exists(figs['Cubo']): doc.add_picture(figs['Cubo'], width=Inches(4.5))
                if 'Curvatura' in figs and os.path.exists(figs['Curvatura']): doc.add_picture(figs['Curvatura'], width=Inches(4.5))

            doc.add_heading("2.8. Diagnóstico de residuales", level=2)
            if 'Residuales' in figs and os.path.exists(figs['Residuales']): doc.add_picture(figs['Residuales'], width=Inches(6.5))
            
            doc.add_heading("2.9. Ecuación de regresión", level=2)
            if R['via'] == ce.VIA_2: doc.add_paragraph(f"Media global = {R['y_bar']:.4f}"); doc.add_paragraph(f"Intercepto base = {R.get('intercepto_bloques', R['y_bar']):.4f}")
            else: doc.add_paragraph(f"Beta 0 = {R['y_bar']:.4f}")
            if eq: doc.add_paragraph(eq, style='Intense Quote')

            doc.add_heading("3. Conclusiones", level=1); doc.add_paragraph("1. [Conclusión 1]"); doc.add_paragraph("2. [Conclusión 2]")
            doc.add_heading("4. Recomendaciones operativas", level=1); doc.add_paragraph("[Qué niveles fijar o evitar]")
            
            buf_word = io.BytesIO(); doc.save(buf_word)

            st.session_state.excel_bytes = buf_xlsx.getvalue(); st.session_state.word_bytes = buf_word.getvalue(); st.session_state.figs_dict = figs
        st.success("¡Reporte generado!")

    if st.session_state.excel_bytes:
        st.download_button("Descargar Tablas (Excel)", st.session_state.excel_bytes, file_name=f'{slug}_tablas.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        st.download_button("Descargar Plantilla de informe (Word)", st.session_state.word_bytes, file_name=f'{slug}_plantilla.docx', mime='application/vnd.openxmlformats-officedocument.wordprocessingml.document')
        buf_zip = io.BytesIO()
        with zipfile.ZipFile(buf_zip, 'w') as z:
            z.writestr(f'{slug}_tablas.xlsx', st.session_state.excel_bytes); z.writestr(f'{slug}_plantilla.docx', st.session_state.word_bytes)
            for name, fns in st.session_state.figs_dict.items():
                if isinstance(fns, list):
                    for fn in fns:
                        if fn and os.path.exists(fn): z.write(fn, arcname=os.path.basename(fn))
                elif fns and os.path.exists(fns): z.write(fns, arcname=os.path.basename(fns))
        st.download_button("Descargar Todo en ZIP", buf_zip.getvalue(), file_name=f'{slug}_analisis.zip', mime='application/zip')

    c1, c2 = st.columns(2)
    if c1.button("Volver a Diagnostico"): st.session_state.stage = 'diagnostics'; st.rerun()
    if c2.button("Empezar de nuevo"):
        for k in list(st.session_state.keys()): del st.session_state[k]
        st.rerun()