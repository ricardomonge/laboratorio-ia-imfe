import streamlit as st
import pandas as pd
from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import tempfile
import os
from datetime import datetime
from supabase import create_client, Client # <--- NUEVO

# ==========================================
# 1. CONFIGURACIÓN Y SEGURIDAD
# ==========================================
st.set_page_config(page_title="IMFE - Laboratorio IA Colaborativa", layout="wide")

# Conexión con Supabase
try:
    url: str = st.secrets["SUPABASE_URL"]
    key: str = st.secrets["SUPABASE_KEY"]
    supabase: Client = create_client(url, key)
except KeyError:
    st.error("⚠️ Faltan las credenciales de Supabase en Secrets.")
    st.stop()

# API OpenAI
try:
    api_key = st.secrets["OPENAI_API_KEY"]
except KeyError:
    st.error("⚠️ No se encontró la 'OPENAI_API_KEY' en Secrets.")
    st.stop()

client = OpenAI(api_key=api_key)

# Inicializar estados
if "messages" not in st.session_state:
    st.session_state.messages = []
if "log_data" not in st.session_state:
    st.session_state.log_data = []
if "configurado" not in st.session_state:
    st.session_state.configurado = False

# FUNCIÓN: Envía datos a Supabase
def guardar_en_supabase(registro):
    try:
        supabase.table("interacciones").insert(registro).execute()
    except Exception as e:
        st.error(f"Error al guardar en la nube (Supabase): {e}")

# ==========================================
# 2. PANTALLA DE REGISTRO
# ==========================================
if not st.session_state.configurado:
    st.title("🚀 Configuración del Laboratorio Colaborativo")
    with st.form("registro"):
        col1, col2 = st.columns(2)
        with col1:
            nrc = st.text_input("ID del Curso", placeholder="Ej: AES519")
            grupo_id = st.text_input("ID del Grupo", placeholder="Ej: Grupo A")
        with col2:
            archivo_pdf = st.file_uploader("Subir apuntes (PDF)", type="pdf")
            integrantes = st.text_area("Integrantes (uno por línea)")
        
        if st.form_submit_button("Inicializar"):
            if nrc and grupo_id and archivo_pdf and integrantes:
                try:
                    with st.spinner("Estudiando apuntes..."):
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                            tmp.write(archivo_pdf.getvalue())
                            tmp_path = tmp.name
                        loader = PyPDFLoader(tmp_path)
                        docs = loader.load_and_split()
                        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
                        st.session_state.vector_db = FAISS.from_documents(docs, embeddings)
                        os.remove(tmp_path)
                        
                        st.session_state.nrc = nrc
                        st.session_state.grupo_id = grupo_id
                        st.session_state.integrantes = [i.strip() for i in integrantes.split("\n") if i.strip()]
                        st.session_state.configurado = True
                        st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
    st.stop()

# ==========================================
# 3. INTERFAZ DE APRENDIZAJE
# ==========================================
st.title(f"🎓 {st.session_state.grupo_id} | Curso: {st.session_state.nrc}")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

col_u, col_t = st.columns([1, 3])
with col_u:
    autor_actual = st.selectbox("¿Quién escribe?", st.session_state.integrantes)

prompt = st.chat_input("Escribe tu explicación...")

# ... (mantenemos la carga de archivos y configuración de Supabase igual)

if prompt:
    # 1. Registro visual del mensaje del grupo
    display_text = f"**{autor_actual}:** {prompt}"
    st.session_state.messages.append({"role": "user", "content": display_text})
    with st.chat_message("user"):
        st.markdown(display_text)
    
    # 2. Búsqueda en los Materiales de la Unidad (RAG)
    with st.spinner("Tu alumno virtual está revisando los materiales de la unidad..."):
        # Buscamos en los documentos subidos
        docs_rel = st.session_state.vector_db.similarity_search(prompt, k=3)
        contexto = "\n\n".join([d.page_content for d in docs_rel])
        
        # ==========================================
        # SYSTEM PROMPT REFINADO: FOCO EN MATERIALES DE UNIDAD
        # ==========================================
        sys_prompt = (
            "Eres un estudiante novato que está cursando esta unidad de contenido. "
            "Tu única fuente de verdad son los MATERIALES DEL CURSO que el profesor te entregó. "
            "No eres un asistente; eres un compañero de clase al que el grupo debe enseñarle. "
            
            "\n\nTU COMPORTAMIENTO:"
            "\n1. Tono Curioso: Usa frases como 'No me queda claro...', 'En los apuntes entendí que...', '¿Podrían explicarme eso de otra forma?'."
            "\n2. Uso de Materiales: Tienes acceso a fragmentos de los materiales de la unidad (contexto). "
            "Si la explicación del grupo omite algo importante que aparece en los materiales, diles: 'Oigan, estaba leyendo los documentos de la unidad y mencionan algo sobre [Concepto], pero ustedes no lo han nombrado... ¿Cómo encaja eso aquí?'."
            "\n3. Resistencia Cognitiva: No aceptes una respuesta simple de 'Sí' o 'No'. Pide que te convenzan con ejemplos prácticos."
            "\n4. Meta: Solo si te explican el concepto de forma completa y coherente con los materiales, responde con entusiasmo: '¡Ahhh! Ahora sí entiendo la relación entre X e Y. ¡Gracias equipo!'"
        )
        
        # Estructura de la consulta enviada al modelo
        full_query = (
            f"FRAGMENTOS DE LOS MATERIALES DE LA UNIDAD:\n{contexto}\n\n"
            f"EXPLICACIÓN DE TUS COMPAÑEROS (Grupo {st.session_state.grupo_id}):\n{prompt}"
        )
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": full_query}
            ],
            temperature=0.8
        )
        
        ai_res = response.choices[0].message.content
        st.session_state.messages.append({"role": "assistant", "content": ai_res})
        with st.chat_message("assistant"):
            st.markdown(ai_res)
        
        # 3. Guardado en Supabase (Registro de investigación)
        registro = {
            "nrc": st.session_state.nrc,
            "grupo_id": st.session_state.grupo_id,
            "autor": autor_actual,
            "mensaje_estudiante": prompt,
            "respuesta_ia": ai_res,
            "longitud_respuesta": len(prompt)
        }
        guardar_en_supabase(registro)
        st.session_state.log_data.append(registro)

if st.sidebar.button("🔴 Finalizar y Descargar CSV"):
    df = pd.DataFrame(st.session_state.log_data)
    csv = df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
    st.sidebar.download_button("Descargar CSV", data=csv, file_name="data.csv", mime="text/csv")