import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re
from nltk.stem import SnowballStemmer

# Configuración de la página
st.set_page_config(
    page_title="TF-IDF Español",
    page_icon="🔍",
    layout="centered"
)

# Tema minimalista con colores verdes
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0a2f1d 0%, #1a3d2e 50%, #2a4d3e 100%);
        color: #e0e0e0;
    }
    .main-title {
        font-size: 2.2rem;
        text-align: center;
        background: linear-gradient(45deg, #10b981, #34d399, #6ee7b7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        font-weight: 800;
    }
    .subtitle {
        text-align: center;
        color: #cbd5e1;
        margin-bottom: 2rem;
        font-size: 1rem;
    }
    .input-section {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .result-section {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .stButton button {
        background: linear-gradient(45deg, #10b981, #34d399);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.7rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        width: 100%;
        margin: 0.3rem 0;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(16, 185, 129, 0.4);
    }
    .suggestion-btn {
        background: rgba(255, 255, 255, 0.1) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
    }
    .suggestion-btn:hover {
        background: rgba(16, 185, 129, 0.2) !important;
        border: 1px solid #10b981 !important;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #10b981;
    }
    .similarity-high { color: #10b981; font-weight: 700; }
    .similarity-medium { color: #f59e0b; font-weight: 700; }
    .similarity-low { color: #ef4444; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# Título principal
st.markdown('<div class="main-title">🔍 Analizador TF-IDF</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Búsqueda semántica en documentos en español</div>', unsafe_allow_html=True)

# Nuevos documentos de ejemplo más interesantes
default_docs = """La inteligencia artificial está transformando la industria tecnológica.
El aprendizaje automático permite a las computadoras aprender sin programación explícita.
Las redes neuronales profundas son fundamentales para el reconocimiento de imágenes.
Los modelos de lenguaje como GPT-4 pueden generar texto similar al humano.
La ética en IA es crucial para el desarrollo responsable de la tecnología.
La ciencia de datos utiliza estadísticas y algoritmos para extraer insights.
El procesamiento de lenguaje natural ayuda a las máquinas a entender el texto."""

# Stemmer en español
stemmer = SnowballStemmer("spanish")

def tokenize_and_stem(text):
    text = text.lower()
    text = re.sub(r'[^a-záéíóúüñ\s]', ' ', text)
    tokens = [t for t in text.split() if len(t) > 1]
    stems = [stemmer.stem(t) for t in tokens]
    return stems

def analyze_documents(documents, question):
    """Función para analizar documentos y calcular similitudes"""
    if len(documents) < 1:
        return None
    
    # Crear vectorizador TF-IDF
    vectorizer = TfidfVectorizer(
        tokenizer=tokenize_and_stem,
        min_df=1
    )
    
    # Ajustar con documentos
    X = vectorizer.fit_transform(documents)
    
    # Calcular similitud con la pregunta
    question_vec = vectorizer.transform([question])
    similarities = cosine_similarity(question_vec, X).flatten()
    
    # Encontrar mejor respuesta
    best_idx = similarities.argmax()
    best_doc = documents[best_idx]
    best_score = similarities[best_idx]
    
    # Matriz TF-IDF
    df_tfidf = pd.DataFrame(
        X.toarray(),
        columns=vectorizer.get_feature_names_out(),
        index=[f"Doc {i+1}" for i in range(len(documents))]
    )
    
    # Stems coincidentes
    vocab = vectorizer.get_feature_names_out()
    q_stems = tokenize_and_stem(question)
    matched = [s for s in q_stems if s in vocab and df_tfidf.iloc[best_idx].get(s, 0) > 0]
    
    return {
        "best_idx": best_idx,
        "best_doc": best_doc,
        "best_score": best_score,
        "similarities": similarities,
        "documents": documents,
        "df_tfidf": df_tfidf,
        "matched_stems": matched
    }

# Inicializar session state
if 'question' not in st.session_state:
    st.session_state.question = "¿Qué es la inteligencia artificial?"
if 'analyze_triggered' not in st.session_state:
    st.session_state.analyze_triggered = False

# Sección de entrada
with st.container():
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    
    st.markdown("**📄 Documentos** (uno por línea)")
    text_input = st.text_area(
        "",
        default_docs,
        height=150,
        label_visibility="collapsed",
        key="documents"
    )
    
    st.markdown("**❓ Pregunta**")
    question = st.text_input(
        "", 
        st.session_state.question, 
        label_visibility="collapsed",
        key="question_input"
    )
    
    st.markdown('</div>', unsafe_allow_html=True)

# Preguntas sugeridas
st.markdown("**💡 Preguntas sugeridas:**")
col1, col2 = st.columns(2)

suggested_questions = [
    "¿Qué es el aprendizaje automático?",
    "¿Para qué sirven las redes neuronales?",
    "¿Qué pueden generar los modelos de lenguaje?",
    "¿Por qué es importante la ética en IA?",
    "¿Qué utiliza la ciencia de datos?",
    "¿Qué ayuda a entender el lenguaje natural?"
]

with col1:
    for i, suggested_q in enumerate(suggested_questions[:3]):
        if st.button(suggested_q, use_container_width=True, key=f"suggest_{i}"):
            st.session_state.question = suggested_q
            st.session_state.analyze_triggered = True
            st.rerun()

with col2:
    for i, suggested_q in enumerate(suggested_questions[3:]):
        if st.button(suggested_q, use_container_width=True, key=f"suggest_{i+3}"):
            st.session_state.question = suggested_q
            st.session_state.analyze_triggered = True
            st.rerun()

# Botón de análisis manual
analyze_clicked = st.button("Analizar Documentos", type="primary", use_container_width=True)

# Realizar análisis si se activó
if analyze_clicked or st.session_state.analyze_triggered:
    documents = [d.strip() for d in text_input.split("\n") if d.strip()]
    
    if len(documents) < 1:
        st.error("⚠️ Ingresa al menos un documento.")
    elif not st.session_state.question.strip():
        st.error("⚠️ Escribe una pregunta.")
    else:
        with st.spinner("🔍 Analizando documentos..."):
            results = analyze_documents(documents, st.session_state.question)
            
            if results:
                # RESULTADO PRINCIPAL
                st.markdown('<div class="result-section">', unsafe_allow_html=True)
                
                # Header del resultado
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown("Mejor Coincidencia")
                    st.markdown(f"**Documento {results['best_idx'] + 1}**")
                with col2:
                    if results['best_score'] > 0.3:
                        sim_class = "similarity-high"
                    elif results['best_score'] > 0.1:
                        sim_class = "similarity-medium"
                    else:
                        sim_class = "similarity-low"
                    st.markdown(f'<div class="{sim_class}" style="font-size: 1.5rem; text-align: center;">{results["best_score"]:.3f}</div>', unsafe_allow_html=True)
                
                # Pregunta y respuesta
                st.markdown("**Pregunta:**")
                st.info(f"\"{st.session_state.question}\"")
                
                st.markdown("**Respuesta encontrada:**")
                st.success(f"\"{results['best_doc']}\"")
                
                st.markdown('</div>', unsafe_allow_html=True)

                # MATRIZ TF-IDF
                with st.expander("Matriz TF-IDF", expanded=False):
                    st.dataframe(results['df_tfidf'].round(3), use_container_width=True)

                # TODAS LAS SIMILITUDES
                with st.expander("Todas las Similitudes", expanded=True):
                    sim_df = pd.DataFrame({
                        "Documento": [f"Doc {i+1}" for i in range(len(results['documents']))],
                        "Similitud": results['similarities'],
                        "Texto": results['documents']
                    })
                    
                    # Ordenar y mostrar
                    sim_df_sorted = sim_df.sort_values("Similitud", ascending=False)
                    
                    for _, row in sim_df_sorted.iterrows():
                        with st.container():
                            col1, col2, col3 = st.columns([1, 2, 1])
                            with col1:
                                st.markdown(f"**{row['Documento']}**")
                            with col2:
                                st.caption(row['Texto'][:70] + "..." if len(row['Texto']) > 70 else row['Texto'])
                            with col3:
                                score = row['Similitud']
                                if score > 0.3:
                                    sim_class = "similarity-high"
                                elif score > 0.1:
                                    sim_class = "similarity-medium"
                                else:
                                    sim_class = "similarity-low"
                                st.markdown(f'<div class="{sim_class}" style="text-align: right;">{score:.3f}</div>', unsafe_allow_html=True)
                            st.divider()

                # STEMS COINCIDENTES
                with st.expander("🔤 Términos Coincidentes", expanded=False):
                    if results['matched_stems']:
                        st.markdown("**Términos encontrados:**")
                        cols = st.columns(4)
                        for i, stem in enumerate(results['matched_stems']):
                            with cols[i % 4]:
                                st.markdown(f'<div style="background: rgba(16, 185, 129, 0.2); padding: 0.5rem; border-radius: 6px; text-align: center; margin: 0.2rem;">{stem}</div>', unsafe_allow_html=True)
                    else:
                        st.info("No se encontraron términos coincidentes significativos.")
    
    # Resetear el trigger después del análisis
    st.session_state.analyze_triggered = False

# Información en acordeón
with st.expander("ℹ️ Acerca del análisis", expanded=False):
    st.markdown("""
    **🔍 TF-IDF en Español**
    
    Esta herramienta analiza la similitud semántica entre preguntas y documentos usando:
    
    **📊 TF-IDF (Frecuencia de Término - Frecuencia Inversa de Documento)**
    - Mide la importancia de palabras en documentos
    - Considera frecuencia local y global
    
    **🎯 Similitud Coseno**
    - Compara vectores de características
    - Rango: 0 (sin relación) a 1 (muy similar)
    
    **Interpretación de resultados:**
    - 🟢 > 0.3: Alta similitud
    - 🟠 0.1 - 0.3: Similitud media  
    - 🔴 < 0.1: Baja similitud
    
    *Los documentos se procesan aplicando stemming para mejorar las coincidencias.*
    """)
