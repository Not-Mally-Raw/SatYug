# 🛰️ Geo-LLM: Chain-of-Thought GIS Automation System

**ISRO Bhartiya Antariksh Hackathon 2025 - Problem Statement 4**

> *Designing a Chain-of-Thought-Based LLM System for Solving Complex Spatial Analysis Tasks Through Intelligent Geoprocessing Orchestration*

## 🎯 Project Overview

This project implements an advanced AI-powered geospatial analysis system that combines Large Language Models (LLMs) with comprehensive GIS workflows. The system can automatically generate and execute complex spatial analysis workflows from natural language queries using Chain-of-Thought reasoning.

## 🚀 Key Features

### 🧠 Enhanced Reasoning Engine
- **Structured Workflow Templates**: Pre-built patterns for common GIS tasks
- **Context-Aware Query Analysis**: Intelligent parameter extraction and workflow customization
- **Chain-of-Thought Implementation**: Step-by-step transparent reasoning process
- **Dynamic Code Generation**: Automatic Python code generation for GIS operations

### 🔄 Intelligent Automation System
- **Multi-Tool Integration**: GDAL, GeoPandas, GRASS GIS, OSMnx, QGIS, Rasterio, WhiteboxTools
- **Workflow Parsing & Execution**: Dynamic function mapping and execution
- **Real-time Progress Tracking**: Live execution monitoring with detailed analytics
- **Comprehensive Error Handling**: Robust error recovery and reporting

### 📚 Advanced RAG Pipeline
- **Multi-Source Knowledge Retrieval**: Semantic search across comprehensive GIS documentation
- **FAISS Vector Embeddings**: Pre-built indices for 7 major GIS tools
- **Quality Control Mechanisms**: Content validation and grounding verification
- **LangGraph Workflows**: Sophisticated reasoning chains with decision points

### 🎨 Professional User Interface
- **Streamlit Web Application**: Modern, responsive interface
- **Real-time Visualization**: Automatic chart and map generation
- **Execution Analytics**: Performance metrics and success tracking
- **Debug Capabilities**: Detailed logging and troubleshooting tools

## 🏗️ Architecture

```
├── automation1.py           # Core GIS automation functions
├── enhanced_reasoning.py    # Chain-of-Thought reasoning engine
├── enhanced_streamlit_app.py # Web application interface
├── fixed_automation1.py     # Stable GIS function implementations
├── improved_automation.py   # Enhanced workflow execution
├── integrated_system.py     # Complete end-to-end system
├── rag_pipeline.py          # RAG implementation with LangGraph
└── embeddings/              # FAISS vector stores for GIS documentation
    ├── gdal_faiss/
    ├── geopandas_faiss/
    ├── grass_faiss/
    ├── osmnx_faiss/
    ├── qgis_faiss/
    ├── rasterio_faiss/
    └── whitebox_faiss/
```

## 🛠️ Technical Stack

- **LLMs**: Groq (Llama3-70B), OpenAI (GPT-4)
- **GIS Libraries**: GDAL, GeoPandas, Rasterio, WhiteboxTools, OSMnx
- **AI/ML**: LangChain, LangGraph, FAISS, HuggingFace Embeddings
- **Frontend**: Streamlit with custom CSS styling
- **Data Processing**: NumPy, Pandas, Matplotlib
- **Geospatial Data**: SRTM, OpenStreetMap, WorldPop

### Requirements Coverage

| Requirement | Implementation | Status |
|-------------|---------------|--------|
| LLM Integration | Groq/OpenAI with multiple models | ✅ Complete |
| Chain-of-Thought Reasoning | Structured templates + CoT prompts | ✅ Complete |
| Natural Language Queries | Advanced query parsing + parameter extraction | ✅ Complete |
| GIS Tool Integration | 7+ tools with proper APIs | ✅ Complete |
| RAG Implementation | Multi-source documentation retrieval | ✅ Complete |
| Workflow Generation | JSON/structured + executable code | ✅ Complete |
| Error Handling | Comprehensive error management | ✅ Complete |
| User Interface | Professional Streamlit application | ✅ Complete |

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Not-Mally-Raw/SatYug.git
cd SatYug

# Install dependencies
pip install streamlit groq langchain faiss-cpu sentence-transformers
pip install gdal geopandas rasterio whitebox osmnx matplotlib

# Set up environment variables
export GROQ_API_KEY="your_groq_api_key"
export OPENAI_API_KEY="your_openai_api_key"  # Optional
```

### Running the Application

```bash
# Start the Streamlit interface
streamlit run enhanced_streamlit_app.py

# Or run the integrated system directly
python integrated_system.py
```

### Example Queries

- "Generate a comprehensive flood risk analysis for Mumbai including DEM, land use, and population data"
- "Analyze urban heat island effect in Delhi using land cover and building density"
- "Create a multi-layer environmental assessment for Bangalore with terrain, hydrology, and LULC"
- "Perform watershed analysis for Chennai including flow accumulation and wetness index"

## 📊 Demo Capabilities

### Supported Analysis Types
- **Flood Risk Assessment**: DEM analysis, flow accumulation, wetness index
- **Urban Planning**: Land use classification, building footprints, population analysis
- **Terrain Analysis**: Slope, aspect, curvature, topographic indices
- **Hydrological Modeling**: Watershed delineation, stream networks, drainage analysis
- **Environmental Assessment**: Multi-layer analysis with visualization

### Generated Outputs
- High-resolution maps and visualizations
- Quantitative analysis results
- GeoTIFF raster outputs
- Vector data in multiple formats
- Detailed execution reports

## 🏆 Competitive Advantages

1. **Advanced AI Integration**: Sophisticated LLM reasoning with structured workflows
2. **Comprehensive Tool Coverage**: 7+ major GIS tools in unified interface
3. **Professional Implementation**: Production-ready code with proper error handling
4. **Real-world Applications**: Practical solutions for actual geospatial challenges
5. **Scalable Architecture**: Modular design for easy extension and deployment

## 🎯 Future Enhancements

- Integration with Bhoonidhi and other Indian satellite data sources
- Advanced evaluation metrics and benchmarking
- Mobile-responsive interface for field work
- Collaborative features for team workflows
- Support for real-time satellite data processing
