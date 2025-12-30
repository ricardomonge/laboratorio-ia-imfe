# 🎓 IA Colaborativa - Laboratorio de Aprendizaje

Este proyecto es una herramienta de aprendizaje colaborativo basada en IA generativa, diseñada para fomentar el aprendizaje profundo mediante la técnica de *Learning by Teaching* (aprender enseñando). La aplicación utiliza un agente estudiante virtual (Teachable Agent) al que los grupos de estudiantes deben explicar conceptos específicos basándose en los materiales oficiales de la unidad.

## 🚀 Características principales

* **Agente estudiante finámico**: Una IA que asume el rol de un alumno novato, mostrando dudas socráticas y resistencia cognitiva para forzar explicaciones claras por parte de los estudiantes.
* **RAG (Retrieval-Augmented Generation)**: La IA utiliza materiales de la unidad cargados en PDF para validar la veracidad de lo que el grupo le explica en tiempo real.
* **Persistencia de Datos en Tiempo Real**: Conexión con **Supabase** (PostgreSQL) para registrar cada interacción de los grupos, lo que permite un análisis posterior de los datos (Tidy Data).
* **Despliegue Rápido**: Optimizado para funcionar en **Streamlit Cloud**.

## 🛠️ Infraestructura técnica para Learning Analytics

* **Lenguaje**: Python 3.x
* **Frontend**: [Streamlit](https://aprendizaje-colaborativo-imfe.streamlit.app/)
* **LLM**: OpenAI GPT-4o
* **Orquestación RAG**: LangChain & FAISS
* **Base de datos**: [Supabase](https://supabase.com/)
* **Análisis posterior sugerido**: R (Tidyverse, DBI, RPostgres)

## 📋 Configuración de variables de entorno

Para que la aplicación funcione correctamente en Streamlit Cloud, se deben configurar los siguientes **Secrets**:

```toml
OPENAI_API_KEY = "tu_clave_de_openai"
SUPABASE_URL = "https://tu_proyecto_id.supabase.co"
SUPABASE_KEY = "tu_clave_service_role"
```

## 📂 Estructura del repositorio

Para asegurar la replicabilidad del experimento, el repositorio se organiza de la siguiente manera:

* **`app_IMFE.py`**: El núcleo de la aplicación. Contiene la lógica del agente, la integración con LangChain para el RAG y las funciones de escritura en la base de datos de Supabase.
* **`requirements.txt`**: Archivo de dependencias.
* **`README.md`**: Este documento de documentación y guía metodológica.

## 🔬 Uso en investigación educativa

Este laboratorio ha sido diseñado con un enfoque de **Learning Analytics** y **Tidy Data**, optimizando la recolección de datos para su posterior análisis estadístico en **R** o **Python**.

### Análisis de interacciones
Al centralizar los datos en Supabase, el investigador puede analizar:
1.  **Densidad del discurso**: Relación entre la `longitud_respuesta` del estudiante humano y la complejidad de la respuesta de la IA.
2. **Análisis por grupo (NRC)**: Comparación de los niveles de participación y de la efectividad pedagógica entre distintas secciones de un mismo curso.
3.  **Patrones de enseñanza**: Identificación de los momentos exactos en los que el agente virtual muestra "insights" de aprendizaje basados en las explicaciones de los usuarios.

### Conexión con R

Se recomienda el uso de las librerías `DBI` y `RPostgres` para importar los datos directamente desde la nube:
```r
library(DBI)
library(RPostgres)

# Conexión directa a la infraestructura de investigación
con <- dbConnect(Postgres(), 
                 host = 'tu_host_pooler', 
                 port = 6543, 
                 user = 'postgres', 
                 password = 'tu_password')

dataset <- dbGetQuery(con, "SELECT * FROM interacciones")
