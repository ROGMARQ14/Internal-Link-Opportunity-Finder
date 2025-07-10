import streamlit as st
import pandas as pd
import numpy as np
import logging

from src.utils.csv_handler import CSVHandler
from src.core.data_processor import DataProcessor
from src.core.similarity_engine import SimilarityEngine
from src.analyzers.link_analyzer import LinkAnalyzer
from src.utils.export_utils import ExportManager
from config import (
    APP_NAME, APP_DESCRIPTION, DEFAULT_TOP_N, MIN_TOP_N, MAX_TOP_N,
    LOG_FORMAT, LOG_LEVEL
)

# Configure logging
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# Initialize helpers
csv_handler = CSVHandler()
data_processor = DataProcessor()
export_manager = ExportManager()

# Streamlit page config
st.set_page_config(
    page_title=APP_NAME,
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title(APP_NAME)
st.markdown(APP_DESCRIPTION)

# Sidebar - Settings
st.sidebar.header("Settings")

top_n = st.sidebar.slider(
    "Number of related URLs to find", 
    min_value=MIN_TOP_N, 
    max_value=MAX_TOP_N, 
    value=DEFAULT_TOP_N
)

cache_clear = st.sidebar.button("Clear Cache")

if cache_clear:
    SimilarityEngine(enable_cache=True).clear_cache()
    st.sidebar.success("Cache cleared!")

# Tabs for multi-step process
tab_upload, tab_process, tab_results = st.tabs(["Upload Data", "Process & Analyze", "Results"])

# Session state variables
if 'links_df' not in st.session_state:
    st.session_state.links_df = None
if 'embeddings_df' not in st.session_state:
    st.session_state.embeddings_df = None
if 'analysis_df' not in st.session_state:
    st.session_state.analysis_df = None
if 'link_stats' not in st.session_state:
    st.session_state.link_stats = None

# ------------------------- Tab 1: Upload Data -------------------------
with tab_upload:
    st.header("Step 1: Upload Your Data Files")
    st.subheader("Internal Links CSV")
    links_file = st.file_uploader("Upload internal links CSV", type=['csv'], key="links_csv")

    if links_file is not None:
        with st.spinner("Reading links CSV..."):
            links_df_raw = csv_handler.read_csv_with_auto_delimiter(links_file)
        if links_df_raw is not None:
            st.success(f"Loaded links CSV with {links_df_raw.shape[0]} rows")
            if st.button("Clean Links Data"):
                with st.spinner("Cleaning links data..."):
                    st.session_state.links_df = data_processor.clean_link_dataset(links_df_raw)
                st.success("Links data cleaned!")
                st.dataframe(st.session_state.links_df.head())

    st.subheader("Embeddings CSV")
    emb_file = st.file_uploader("Upload page embeddings CSV", type=['csv'], key="embeddings_csv")

    if emb_file is not None:
        with st.spinner("Reading embeddings CSV..."):
            embeddings_df_raw = csv_handler.read_csv_with_auto_delimiter(emb_file)
        if embeddings_df_raw is not None:
            st.success(f"Loaded embeddings CSV with {embeddings_df_raw.shape[0]} rows")
            if st.button("Clean Embeddings Data"):
                with st.spinner("Cleaning embeddings data..."):
                    cleaned_embeddings = data_processor.clean_embeddings_data(embeddings_df_raw)
                    if cleaned_embeddings is not None:
                        st.session_state.embeddings_df = data_processor.convert_embeddings_to_arrays(cleaned_embeddings)
                if st.session_state.embeddings_df is not None:
                    st.success("Embeddings data cleaned!")
                    st.dataframe(st.session_state.embeddings_df.head())

# ------------------------- Tab 2: Process & Analyze -------------------------
with tab_process:
    st.header("Step 2: Process & Analyze")

    if st.session_state.links_df is not None and st.session_state.embeddings_df is not None:
        st.success("Datasets ready for analysis")
        if st.button("Run Analysis"):
            with st.spinner("Calculating similarities and analyzing opportunities..."):
                # Initialize similarity engine
                sim_engine = SimilarityEngine()
                related_pages = sim_engine.find_related_pages(
                    st.session_state.embeddings_df, top_n=top_n
                )

                # Initialize link analyzer
                analyzer = LinkAnalyzer(st.session_state.links_df)
                analysis_df = analyzer.analyze_opportunities(related_pages, top_n=top_n)
                st.session_state.analysis_df = analysis_df
                st.session_state.link_stats = analyzer.get_link_statistics()
        if st.session_state.analysis_df is not None:
            st.success("Analysis completed!")
            st.dataframe(st.session_state.analysis_df.head())
    else:
        st.info("Please upload and clean both datasets in Step 1")

# ------------------------- Tab 3: Results -------------------------
with tab_results:
    st.header("Step 3: Results & Download")

    if st.session_state.analysis_df is not None:
        st.subheader("Filter Results")
        col1, col2 = st.columns(2)
        with col1:
            show_missing_only = st.checkbox("Show only rows with missing links", value=True)
        with col2:
            url_filter = st.text_input("Filter by URL contains")

        filtered_df = st.session_state.analysis_df.copy()
        if show_missing_only:
            mask = pd.Series([False] * len(filtered_df))
            for col in filtered_df.columns:
                if "links to A?" in col:
                    mask = mask | (filtered_df[col] == "Not Found")
            filtered_df = filtered_df[mask]
        if url_filter:
            mask = filtered_df['Target URL'].str.contains(url_filter, case=False, na=False)
            for col in filtered_df.columns:
                if "Related URL" in col:
                    mask = mask | filtered_df[col].str.contains(url_filter, case=False, na=False)
            filtered_df = filtered_df[mask]

        st.subheader("Analysis Table")
        st.dataframe(filtered_df, use_container_width=True)

        st.subheader("Download")
        col1, col2 = st.columns(2)
        with col1:
            excel_link = export_manager.get_excel_download_link(
                filtered_df,
                filename="internal_link_opportunities.xlsx",
                link_text="Download Excel",
                link_stats=st.session_state.link_stats
            )
            st.markdown(excel_link, unsafe_allow_html=True)
        with col2:
            csv_link = export_manager.get_csv_download_link(
                filtered_df, 
                filename="internal_link_opportunities.csv", 
                link_text="Download CSV"
            )
            st.markdown(csv_link, unsafe_allow_html=True)

        # Opportunity summary
        st.subheader("Summary")
        if st.session_state.link_stats:
            st.json(st.session_state.link_stats)
    else:
        st.info("Run the analysis in Step 2 to view results.")
