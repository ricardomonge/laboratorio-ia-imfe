import streamlit as st
import pandas as pd
from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import tempfile
import os
from datetime import datetime

# ==========================================
# 1. CONFIGURACIÓN Y SEGURIDAD
# ==========================================
st.set_page_config(page_title="IMFE - Laboratorio IA Colaborativa", layout="wide")

# Intentar obtener la API Key desde Streamlit Secrets
try:
    api_key = st.secrets["OPENAI_API_KEY"]
except KeyError:
    st.error("⚠️ No se encontró la 'OPENAI_API_KEY' en los Secrets de Streamlit.")
    st.stop()

client = OpenAI(api_key=api_key)

# Inicializar estados de sesión
if "messages" not in st.session_state:
    st.session_state.messages = []
if "log_data" not in st.session_state:
    st.session_state.log_data = []
if "configurado" not in st.session_state:
    st.session_state.configurado = False

# ==========================================
# 2. PANTALLA DE REGISTRO (DOCENTE)
# ==========================================
if not st.session_state.configurado:
    st.title("🚀 Configuración del Laboratorio Colaborativo")
    st.info("Esta sección debe completarse siguiendo las indicaciones del o la docente del curso antes de iniciar con los estudiantes.")
    
    with st.form("registro"):
        col1, col2 = st.columns(2)
        with col1:
            nrc = st.text_input("ID del Curso (Código/NRC)", placeholder="Ej: AES519/1375")
            grupo_id = st.text_input("ID del Grupo", placeholder="Ej: Grupo A")
        with col2:
            archivo_pdf = st.file_uploader("Subir apuntes del curso (PDF)", type="pdf")
            integrantes = st.text_area("Lista de Integrantes (uno por línea)")
        
        submit = st.form_submit_button("Inicializar Entorno de Aprendizaje")
        
        if submit:
            if nrc and grupo_id and archivo_pdf and integrantes:
                try:
                    with st.spinner("El alumno virtual está estudiando los apuntes del curso..."):
                        # Procesamiento de PDF con RAG
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                            tmp.write(archivo_pdf.getvalue())
                            tmp_path = tmp.name
                        
                        loader = PyPDFLoader(tmp_path)
                        docs = loader.load_and_split()
                        
                        # Usar la API Key de los secretos para las embeddings
                        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
                        st.session_state.vector_db = FAISS.from_documents(docs, embeddings)
                        
                        os.remove(tmp_path)
                        
                        # Guardar metadatos en session_state
                        st.session_state.nrc = nrc
                        st.session_state.grupo_id = grupo_id
                        st.session_state.integrantes = [i.strip() for i in integrantes.split("\n") if i.strip()]
                        st.session_state.configurado = True
                        st.rerun()
                except Exception as e:
                    st.error(f"Error al procesar el PDF: {e}")
            else:
                st.warning("Por favor, completa todos los campos y sube el PDF.")
    st.stop()

# ==========================================
# 3. INTERFAZ DE APRENDIZAJE (ESTUDIANTES)
# ==========================================
st.title(f"🎓 {st.session_state.grupo_id} | Curso: {st.session_state.nrc}")

# Sidebar con instrucciones y control
with st.sidebar:
    st.header("Instrucciones")
    st.write("1. Selecciona quién está hablando.")
    st.write("2. Explica el concepto al alumno virtual.")
    st.write("3. Si el alumno tiene dudas, utiliza los contenidos del curso para aclararlas.")
    
    st.divider()
    if st.button("🔴 Finalizar Sesión (Generar datos)"):
        st.session_state.finalizado = True

# Mostrar historial de mensajes
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Área de entrada de mensajes
col_u, col_t = st.columns([1, 3])
with col_u:
    autor_actual = st.selectbox("¿Quién escribe?", st.session_state.integrantes)

prompt = st.chat_input("Escribe tu explicación aquí...")

if prompt:
    # 1. Registro del mensaje del usuario
    display_text = f"**{autor_actual}:** {prompt}"
    st.session_state.messages.append({"role": "user", "content": display_text})
    with st.chat_message("user"):
        st.markdown(display_text)
    
    # 2. Búsqueda en el vector store (RAG)
    with st.spinner("El alumno está pensando..."):
        docs_rel = st.session_state.vector_db.similarity_search(prompt, k=3)
        contexto = "\n\n".join([d.page_content for d in docs_rel])
        
        # 3. Generación de respuesta con Prompt Ingenierizado para Educación
        sys_prompt = (
            "Eres un estudiante curioso, pero con dudas. Tu objetivo es aprender de los humanos. "
            "Usa el CONTEXTO DE LOS MATERIALES DEL CURSO  proporcionado para validar lo que dicen. "
            "Si lo que dicen es incompleto o incorrecto según el manual, expresa una duda socrática. "
            "No des la respuesta correcta directamente; haz que ellos piensen."
        )
        
        full_query = f"CONTEXTO DEL MATERIAL DEL CURSO:\n{contexto}\n\nEXPLICACIÓN DEL GRUPO:\n{prompt}"
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": full_query}
            ],
            temperature=0.7
        )
        
        ai_res = response.choices[0].message.content
        st.session_state.messages.append({"role": "assistant", "content": ai_res})
        with st.chat_message("assistant"):
            st.markdown(ai_res)
        
        # 4. Almacenamiento de datos 'Tidy' para R
        st.session_state.log_data.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "nrc": st.session_state.nrc,
            "grupo": st.session_state.grupo_id,
            "autor": autor_actual,
            "mensaje_estudiante": prompt,
            "respuesta_ia": ai_res,
            "longitud_respuesta": len(prompt)
        })

# ==========================================
# 4. EXPORTACIÓN PARA INVESTIGACIÓN (R)
# ==========================================
if st.session_state.get("finalizado"):
    st.divider()
    st.header("📊 Resultados del experimento")
    
    if st.session_state.log_data:
        df = pd.DataFrame(st.session_state.log_data)
        
        # SOLUCIÓN AL PROBLEMA DE TILDES: 
        # Usamos encoding='utf-8-sig' para que Excel y R reconozcan caracteres especiales
        csv_data = df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
        
        col_down1, col_down2 = st.columns(2)
        with col_down1:
            st.download_button(
                label="📥 Descargar dataset (.csv)",
                data=csv_data,
                file_name=f"data_{st.session_state.nrc}_{st.session_state.grupo_id}.csv",
                mime="text/csv"
            )
        
        st.dataframe(df)
    else:
        st.warning("No hay datos registrados en esta sesión.")