"""
Streamlit UI for RAG System

Interactive web interface for document querying and system management.
"""

import os
import streamlit as st
import requests
import json
from datetime import datetime
from typing import Dict, Any, List
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
API_BASE_URL = f"http://{os.getenv('API_HOST', 'localhost')}:{os.getenv('API_PORT', 8000)}"
UI_TITLE = os.getenv("UI_TITLE", "RAG System - Document Q&A")

# Page configuration
st.set_page_config(
    page_title=UI_TITLE,
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    
    .citation-card {
        background: #f1f3f4;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 3px solid #34a853;
    }
    
    .error-card {
        background: #fef7f7;
        padding: 1rem;
        border-radius: 8px;
        border-left: 3px solid #dc3545;
        color: #721c24;
    }
    
    .success-card {
        background: #f0f9f4;
        padding: 1rem;
        border-radius: 8px;
        border-left: 3px solid #28a745;
        color: #155724;
    }
</style>
""", unsafe_allow_html=True)


def make_api_request(endpoint: str, method: str = "GET", data: Dict = None) -> Dict:
    """Make API request to the backend"""
    try:
        url = f"{API_BASE_URL}{endpoint}"
        
        if method == "GET":
            response = requests.get(url, timeout=30)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=60)
        else:
            return {"error": f"Unsupported HTTP method: {method}"}
        
        if response.status_code == 200:
            return response.json()
        else:
            return {
                "error": f"API request failed with status {response.status_code}",
                "details": response.text
            }
            
    except requests.exceptions.Timeout:
        return {"error": "Request timed out. The operation may take longer than expected."}
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to the API. Make sure the backend is running."}
    except Exception as e:
        return {"error": f"Request failed: {str(e)}"}


def display_health_status():
    """Display system health status in sidebar"""
    with st.sidebar:
        st.subheader("🏥 System Health")
        
        with st.spinner("Checking system health..."):
            health_data = make_api_request("/healthz")
        
        if "error" in health_data:
            st.error(f"Health check failed: {health_data['error']}")
            return
        
        # Overall status
        status = health_data.get("status", "unknown")
        status_colors = {
            "healthy": "🟢",
            "partial": "🟡", 
            "unhealthy": "🔴"
        }
        
        st.markdown(f"**Overall Status:** {status_colors.get(status, '⚪')} {status.title()}")
        
        # Component details
        components = health_data.get("components", {})
        
        with st.expander("Component Details", expanded=False):
            for component, details in components.items():
                if isinstance(details, dict):
                    comp_status = details.get("status", "unknown")
                    st.markdown(f"**{component.title()}:** {status_colors.get(comp_status, '⚪')} {comp_status}")
                    
                    # Show additional details for some components
                    if component == "elasticsearch" and "document_count" in details:
                        st.caption(f"Documents: {details['document_count']}")
                    elif component == "llm_client" and "model" in details.get("configuration", {}):
                        st.caption(f"Model: {details['configuration']['model']}")


def display_stats():
    """Display system statistics"""
    stats_data = make_api_request("/stats")
    
    if "error" not in stats_data:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="📄 Documents Indexed",
                value=stats_data.get("document_count", 0)
            )
        
        with col2:
            st.metric(
                label="🗄️ Index Name", 
                value=stats_data.get("index_name", "N/A")
            )
        
        with col3:
            current_time = datetime.now().strftime("%H:%M:%S")
            st.metric(
                label="🕐 Current Time",
                value=current_time
            )


def query_interface():
    """Main query interface"""
    st.markdown('<div class="main-header"><h1>📚 RAG System - Document Q&A</h1><p>Ask questions and get answers from your documents</p></div>', unsafe_allow_html=True)
    
    # Display stats
    display_stats()
    
    # Query form
    with st.form("query_form"):
        st.subheader("💬 Ask a Question")
        
        # Question input
        question = st.text_area(
            "Enter your question:",
            placeholder="e.g., What is machine learning? How does neural networks work?",
            height=100,
            help="Ask any question about the documents in your knowledge base."
        )
        
        # Query options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            retrieval_mode = st.selectbox(
                "Retrieval Mode:",
                options=["hybrid", "bm25", "dense", "elser"],
                index=0,
                help="Choose the search method"
            )
        
        with col2:
            top_k = st.slider(
                "Number of Documents:",
                min_value=1,
                max_value=20,
                value=5,
                help="How many relevant documents to consider"
            )
        
        with col3:
            include_citations = st.checkbox(
                "Include Citations",
                value=True,
                help="Show source documents with references"
            )
        
        # Advanced options
        with st.expander("🔧 Advanced Options", expanded=False):
            apply_guardrails = st.checkbox(
                "Apply Safety Guardrails",
                value=True,
                help="Apply safety checks and grounding verification"
            )
        
        # Submit button
        submit_button = st.form_submit_button("🚀 Ask Question", use_container_width=True)
    
    # Process query
    if submit_button and question.strip():
        process_query(question, retrieval_mode, top_k, include_citations, apply_guardrails)
    elif submit_button:
        st.warning("⚠️ Please enter a question before submitting.")


def process_query(question: str, retrieval_mode: str, top_k: int, include_citations: bool, apply_guardrails: bool):
    """Process user query and display results"""
    # Prepare request
    query_data = {
        "question": question,
        "retrieval_mode": retrieval_mode,
        "top_k": top_k,
        "include_citations": include_citations,
        "apply_guardrails": apply_guardrails
    }
    
    # Show query being processed
    st.subheader("🤔 Processing your question...")
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Make API request
    status_text.text("🔍 Searching for relevant documents...")
    progress_bar.progress(30)
    
    response = make_api_request("/query", method="POST", data=query_data)
    
    progress_bar.progress(70)
    status_text.text("🧠 Generating answer...")
    
    progress_bar.progress(100)
    status_text.text("✅ Complete!")
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    # Display results
    if "error" in response:
        st.markdown(f'<div class="error-card"><strong>❌ Error:</strong> {response["error"]}</div>', unsafe_allow_html=True)
        return
    
    # Answer section
    st.subheader("💡 Answer")
    answer = response.get("answer", "No answer generated")
    
    # Check if answer indicates lack of information
    if any(phrase in answer.lower() for phrase in ["don't have enough information", "cannot answer", "no relevant documents"]):
        st.info(answer)
    else:
        st.success(answer)
    
    # Metadata section
    metadata = response.get("metadata", {})
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🎯 Retrieval Mode", retrieval_mode.title())
    with col2:
        st.metric("📊 Results Found", metadata.get("retrieval_results", 0))
    with col3:
        confidence = metadata.get("confidence", 0)
        st.metric("🎪 Confidence", f"{confidence:.2f}" if confidence else "N/A")
    with col4:
        grounded = metadata.get("grounded", True)
        st.metric("🔒 Grounded", "✅ Yes" if grounded else "❌ No")
    
    # Citations section
    if include_citations and response.get("citations"):
        st.subheader("📚 Sources & Citations")
        
        citations = response["citations"]
        
        for i, citation in enumerate(citations, 1):
            with st.expander(f"📄 Source {i}: {citation.get('title', 'Unknown')}", expanded=i <= 2):
                
                # Citation details
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"**Snippet:** {citation.get('snippet', 'No snippet available')}")
                    
                    if citation.get("url") and citation["url"] != "#":
                        st.markdown(f"**Link:** [Open Document]({citation['url']})")
                
                with col2:
                    if citation.get("relevance_score"):
                        st.metric("Relevance", f"{citation['relevance_score']:.3f}")
                    
                    method = citation.get("retrieval_method", "unknown")
                    st.caption(f"Method: {method}")
        
        # Citation visualization
        if len(citations) > 1:
            st.subheader("📈 Citation Relevance Scores")
            
            # Prepare data for visualization
            titles = [f"Doc {i+1}" for i in range(len(citations))]
            scores = [c.get("relevance_score", 0) for c in citations]
            
            if any(score > 0 for score in scores):
                fig = px.bar(
                    x=titles,
                    y=scores,
                    title="Document Relevance Scores",
                    labels={"x": "Documents", "y": "Relevance Score"}
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)


def ingestion_interface():
    """Document ingestion interface"""
    st.subheader("📥 Document Ingestion")
    
    st.info("💡 **Tip:** Make sure your Google Drive credentials are configured and the folder ID is set in your environment variables.")
    
    with st.form("ingestion_form"):
        st.markdown("### Ingestion Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            folder_id = st.text_input(
                "Google Drive Folder ID (optional):",
                placeholder="Leave empty to use default from .env",
                help="The Google Drive folder ID containing PDFs"
            )
            
            max_files = st.number_input(
                "Maximum Files to Process:",
                min_value=1,
                max_value=100,
                value=10,
                help="Limit the number of files to process"
            )
        
        with col2:
            force_recreate = st.checkbox(
                "Force Recreate Index",
                value=False,
                help="Delete existing index and recreate (⚠️ This will remove all existing documents)"
            )
            
            if force_recreate:
                st.warning("⚠️ This will delete all existing documents!")
        
        # Submit button
        submit_ingestion = st.form_submit_button("📥 Start Ingestion", use_container_width=True)
    
    # Process ingestion
    if submit_ingestion:
        process_ingestion(folder_id, max_files, force_recreate)


def process_ingestion(folder_id: str, max_files: int, force_recreate: bool):
    """Process document ingestion"""
    # Prepare request
    ingestion_data = {
        "max_files": max_files,
        "force_recreate_index": force_recreate
    }
    
    if folder_id.strip():
        ingestion_data["folder_id"] = folder_id.strip()
    
    # Show ingestion progress
    st.subheader("📥 Ingesting Documents...")
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("🔍 Connecting to Google Drive...")
    progress_bar.progress(10)
    
    # Make API request (this may take a while)
    status_text.text("📄 Processing PDF documents...")
    progress_bar.progress(30)
    
    response = make_api_request("/ingest", method="POST", data=ingestion_data)
    
    progress_bar.progress(80)
    status_text.text("🧠 Generating embeddings and indexing...")
    
    progress_bar.progress(100)
    status_text.text("✅ Ingestion complete!")
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    # Display results
    if "error" in response:
        st.markdown(f'<div class="error-card"><strong>❌ Ingestion Failed:</strong> {response["error"]}</div>', unsafe_allow_html=True)
        
        # Show common troubleshooting tips
        with st.expander("🔧 Troubleshooting Tips", expanded=True):
            st.markdown("""
            **Common issues and solutions:**
            
            1. **Google Drive API not configured:**
               - Ensure `credentials.json` file exists
               - Check `GOOGLE_DRIVE_FOLDER_ID` in `.env` file
               - Verify service account has access to the folder
            
            2. **Elasticsearch connection issues:**
               - Check if Elasticsearch is running: `curl http://localhost:9200`
               - Restart Elasticsearch: `docker-compose restart elasticsearch`
            
            3. **Groq API issues:**
               - Verify `GROQ_API_KEY` in `.env` file
               - Check API quota and rate limits
            """)
    else:
        # Success message
        status = response.get("status", "unknown")
        message = response.get("message", "Ingestion completed")
        files_processed = response.get("files_processed", 0)
        chunks_created = response.get("chunks_created", 0)
        
        st.markdown(f'<div class="success-card"><strong>✅ {message}</strong></div>', unsafe_allow_html=True)
        
        # Show statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📁 Files Processed", files_processed)
        with col2:
            st.metric("📄 Chunks Created", chunks_created)
        with col3:
            avg_chunks = chunks_created / files_processed if files_processed > 0 else 0
            st.metric("📊 Avg Chunks/File", f"{avg_chunks:.1f}")
        
        # Show success message
        if files_processed > 0:
            st.balloons()
            st.success(f"🎉 Successfully processed {files_processed} files! You can now ask questions about these documents.")


def main():
    """Main application"""
    
    # Add navigation
    with st.sidebar:
        page = st.radio(
            "🧭 Navigation",
            ["💬 Query Documents", "📥 Ingest Documents"],
            index=0
        )
    
    # Main content based on selected page
    if page == "💬 Query Documents":
        query_interface()
    elif page == "📥 Ingest Documents":
        ingestion_interface()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666; font-size: 0.8em;'>"
        "RAG System built with Elasticsearch, Groq, and Streamlit | "
        f"<a href='{API_BASE_URL}/docs' target='_blank'>API Documentation</a>"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()