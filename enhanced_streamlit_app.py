"""
Enhanced Streamlit Application with Improved Automation

This enhanced version integrates the improved automation system that:
1. Parses LLM-generated workflows into executable steps
2. Dynamically executes GIS functions based on the workflow
3. Displays execution progress and results
4. Shows generated graphs and visualizations
"""

import os
import re
import streamlit as st
import time
from groq import Groq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import matplotlib.pyplot as plt
import traceback
from typing import Dict, List, Any

# Import the enhanced automation system
from improved_automation import (
    WorkflowExecutor, 
    create_enhanced_query_function,
    run_enhanced_automation,
    ENHANCED_WORKFLOW_PROMPT
)

# Import existing functions for fallback
from automation1 import (
    get_city_visualizations, get_and_save_lulc, fetch_and_plot_hydrology,
    get_city_bbox, download_worldpop_raster, clip_and_plot_population, 
    get_building_footprints, plot_buildings
)

# === CONFIGURATION ===
MODEL_NAME = "llama3-70b-8192"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOOLS = ["qgis", "gdal", "grass", "geopandas", "osmnx", "rasterio", "whitebox"]
EMBEDDING_BASE_PATH = "embeddings"
TOP_K = 2

# === Streamlit UI Setup ===
st.set_page_config(
    page_title="Enhanced Geo-LLM: Intelligent GIS Automation",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for better styling
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }
    
    .main-container {
        background: white;
        border-radius: 16px;
        padding: 2rem;
        margin: 1rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.12);
        backdrop-filter: blur(10px);
    }
    
    .step-container {
        background: #f8fafc;
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #3b82f6;
    }
    
    .step-success {
        border-left-color: #10b981;
        background: #f0fdf4;
    }
    
    .step-error {
        border-left-color: #ef4444;
        background: #fef2f2;
    }
    
    .step-running {
        border-left-color: #f59e0b;
        background: #fffbeb;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.8; }
    }
    
    .metric-card {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .success-rate {
        font-size: 2rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("# 🛰️ Enhanced Geo-LLM")
    st.markdown("### Configuration")
    
    GROQ_API_KEY = st.text_input(
        "🔑 GROQ API Key", 
        type="password", 
        value="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",
        help="Enter your GROQ API key for LLM access"
    )
    
    execution_mode = st.radio(
        "🚀 Execution Mode",
        ["Enhanced Automation", "Legacy Mode"],
        index=0,
        help="Choose between the new intelligent automation or legacy hardcoded execution"
    )
    
    show_debug = st.checkbox("🐛 Show Debug Info", False)
    auto_execute = st.checkbox("⚡ Auto-execute Workflow", True)
    
    st.markdown("### About Enhanced Mode")
    st.info("""
    **Enhanced Automation** features:
    - 🧠 Intelligent workflow parsing
    - 🔄 Dynamic function execution
    - 📊 Real-time progress tracking
    - 🎯 Context-aware reasoning
    - 📈 Execution analytics
    """)

# Main interface
st.markdown('<div class="main-container">', unsafe_allow_html=True)

st.title("🛰️ Enhanced Geo-LLM: Intelligent GIS Automation")
st.markdown("*Powered by advanced workflow parsing and dynamic execution*")

# === Context Retrieval Function ===
@st.cache_resource
def get_combined_context(user_query):
    """Retrieve context from GIS tool documentation"""
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    context_chunks = []
    
    for tool in TOOLS:
        try:
            vs_path = os.path.join(EMBEDDING_BASE_PATH, f"{tool}_faiss")
            if os.path.exists(vs_path):
                db = FAISS.load_local(vs_path, embedding_model, allow_dangerous_deserialization=True)
                docs = db.similarity_search(user_query, k=TOP_K)
                context_chunks.extend(docs)
        except Exception as e:
            if show_debug:
                st.warning(f"❌ Failed to load vector store for `{tool}`: {e}")
    
    combined_context = "\n\n".join(doc.page_content for doc in context_chunks)
    return combined_context, context_chunks

# === Input Section ===
st.markdown("## 💭 What geospatial challenge can I help you solve?")

# Example queries
example_queries = [
    "Generate a comprehensive flood risk analysis for Mumbai including DEM, land use, and population data",
    "Analyze urban heat island effect in Delhi using land cover and building density",
    "Create a multi-layer environmental assessment for Bangalore with terrain, hydrology, and LULC",
    "Perform watershed analysis for Chennai including flow accumulation and wetness index",
    "Generate a complete urban planning dataset for Pune with all available layers"
]

col1, col2 = st.columns([3, 1])

with col1:
    selected_example = st.selectbox(
        "🎯 Try an example query:",
        [""] + example_queries,
        index=0,
        help="Select an example or write your own query below"
    )

    query = st.text_area(
        "📝 Your Query",
        value=selected_example,
        height=120,
        placeholder="Describe your geospatial analysis task in detail...",
        help="The more specific you are, the better the workflow will be"
    )

with col2:
    st.markdown("### 🎛️ Quick Actions")
    run_button = st.button("🚀 Generate & Execute", type="primary", use_container_width=True)
    
    if st.button("📋 Generate Only", use_container_width=True):
        st.session_state.generate_only = True
    
    if st.button("🔄 Reset", use_container_width=True):
        st.session_state.clear()
        st.rerun()

# === Main Execution Logic ===
if run_button and query.strip():
    
    # Initialize session state
    if 'execution_results' not in st.session_state:
        st.session_state.execution_results = None
    
    # Create progress containers
    progress_container = st.container()
    results_container = st.container()
    
    with progress_container:
        st.markdown("## 🔄 Processing Your Request")
        
        # Step 1: Context Retrieval
        with st.spinner("🔍 Retrieving relevant GIS documentation..."):
            context, docs = get_combined_context(query)
        
        st.success("✅ Context retrieved successfully")
        
        # Step 2: Workflow Generation
        with st.spinner("🧠 Generating intelligent workflow..."):
            try:
                client = Groq(api_key=GROQ_API_KEY)
                enhanced_query_llama = create_enhanced_query_function(client, MODEL_NAME)
                workflow_text = enhanced_query_llama(query, context)
            except Exception as e:
                st.error(f"❌ Failed to generate workflow: {e}")
                st.stop()
        
        st.success("✅ Workflow generated successfully")
        
        # Display generated workflow
        with st.expander("📋 Generated Workflow", expanded=True):
            st.markdown("### 🧠 AI-Generated Workflow")
            st.write(workflow_text)
        
        # Step 3: Workflow Execution (if enabled)
        if execution_mode == "Enhanced Automation" and auto_execute:
            st.markdown("## ⚡ Executing Workflow")
            
            # Create execution progress
            execution_progress = st.progress(0)
            status_text = st.empty()
            
            try:
                # Run the enhanced automation
                with st.spinner("🚀 Executing workflow steps..."):
                    automation_results = run_enhanced_automation(query, workflow_text)
                
                # Store results in session state
                st.session_state.execution_results = automation_results
                
                execution_progress.progress(100)
                status_text.success("✅ Workflow execution completed!")
                
            except Exception as e:
                st.error(f"❌ Execution failed: {e}")
                if show_debug:
                    st.code(traceback.format_exc())
    
    # === Results Display ===
    if st.session_state.execution_results:
        results = st.session_state.execution_results
        
        with results_container:
            st.markdown("## 📊 Execution Results")
            
            # Summary metrics
            summary = results['summary']
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="success-rate">{summary['total_steps']}</div>
                    <div>Total Steps</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="success-rate">{summary['successful_steps']}</div>
                    <div>Successful</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                success_rate = f"{summary['success_rate']:.1%}"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="success-rate">{success_rate}</div>
                    <div>Success Rate</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                exec_time = f"{summary['total_execution_time']:.1f}s"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="success-rate">{exec_time}</div>
                    <div>Execution Time</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Detailed step results
            st.markdown("### 📋 Step-by-Step Results")
            
            for result in results['execution_results']:
                status_class = "step-success" if result.success else "step-error"
                status_icon = "✅" if result.success else "❌"
                
                st.markdown(f"""
                <div class="step-container {status_class}">
                    <h4>{status_icon} Step {result.step_number}</h4>
                    <p><strong>Status:</strong> {'Success' if result.success else 'Failed'}</p>
                    <p><strong>Execution Time:</strong> {result.execution_time:.2f}s</p>
                    {f'<p><strong>Error:</strong> {result.error_message}</p>' if result.error_message else ''}
                </div>
                """, unsafe_allow_html=True)
            
            # Display generated figures
            figures = results['generated_figures']
            if figures:
                st.markdown("### 🗺️ Generated Visualizations")
                
                cols = st.columns(2)
                for i, fig in enumerate(figures):
                    with cols[i % 2]:
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)  # Clean up memory
            
            # Debug information
            if show_debug:
                with st.expander("🐛 Debug Information"):
                    st.json({
                        'query': query,
                        'context_length': len(context),
                        'workflow_length': len(workflow_text),
                        'execution_summary': summary
                    })

# === Legacy Mode Fallback ===
elif execution_mode == "Legacy Mode" and run_button and query.strip():
    st.markdown("## 🔄 Running Legacy Mode")
    
    with st.spinner("🔍 Retrieving context and generating workflow..."):
        context, docs = get_combined_context(query)
        
        # Use basic LLM call
        client = Groq(api_key=GROQ_API_KEY)
        system_prompt = (
            "You are a GIS workflow planner. Given a user query and tool documentation, "
            "generate a step-by-step geospatial workflow including:\n"
            "- Data needed\n- Tools to use\n- Operations to perform\n"
            "Use libraries like Rasterio, GeoPandas, WhiteboxTools, OSMnx, or PyQGIS where needed."
        )
        full_prompt = f"Context from GIS docs:\n{context}\n\nUser Query:\n{query}"

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": full_prompt}
            ],
            temperature=0.3,
            max_tokens=2048
        )
        llama_response = response.choices[0].message.content

    st.markdown("### 🧠 Generated Workflow (Legacy)")
    st.success(llama_response)

    # Legacy hardcoded execution
    city_name = re.search(r"(?:in|for)\s+([A-Za-z\s]+)", query, re.IGNORECASE)
    if city_name:
        city_name = city_name.group(1).strip()
        st.markdown(f"### 🗺️ Visualizations for {city_name}")
        
        try:
            figs = get_city_visualizations(city_name)
            for fig in figs:
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
        except Exception as e:
            st.warning(f"Could not generate visualizations: {e}")

# === Footer ===
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    Enhanced Geo-LLM v2.0 | Powered by Advanced Workflow Automation
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

