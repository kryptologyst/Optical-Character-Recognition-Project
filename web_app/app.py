"""
Streamlit web interface for OCR project.
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from PIL import Image
import io
import json
from typing import Dict, List

# Import our OCR modules
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from ocr_engine import OCREngineManager, OCREngine, OCRResult
from config import config_manager, setup_logging
from data_generator import SyntheticTextGenerator


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'ocr_manager' not in st.session_state:
        st.session_state.ocr_manager = OCREngineManager()
    
    if 'results_history' not in st.session_state:
        st.session_state.results_history = []


def display_image_with_boxes(image_path: str, bounding_boxes: List, text_results: List[str]) -> None:
    """
    Display image with bounding boxes and text annotations.
    
    Args:
        image_path: Path to the image
        bounding_boxes: List of bounding boxes
        text_results: List of extracted text for each box
    """
    img = Image.open(image_path)
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.imshow(img)
    ax.set_title("OCR Results with Bounding Boxes")
    ax.axis('off')
    
    # Draw bounding boxes
    for i, (bbox, text) in enumerate(zip(bounding_boxes, text_results)):
        if len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1
            
            # Draw rectangle
            rect = plt.Rectangle((x1, y1), width, height, 
                               fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)
            
            # Add text annotation
            ax.text(x1, y1-5, f"{i+1}: {text[:20]}...", 
                   fontsize=8, color='red', weight='bold')
    
    st.pyplot(fig)


def create_comparison_chart(results: Dict[OCREngine, OCRResult]) -> None:
    """
    Create comparison chart for multiple OCR engines.
    
    Args:
        results: Dictionary of OCR results
    """
    if len(results) < 2:
        return
    
    engines = [engine.value for engine in results.keys()]
    confidences = [result.confidence for result in results.values()]
    processing_times = [result.processing_time for result in results.values()]
    text_lengths = [len(result.text) for result in results.values()]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    # Confidence comparison
    ax1.bar(engines, confidences, color='skyblue')
    ax1.set_title('Confidence Scores')
    ax1.set_ylabel('Confidence')
    ax1.set_ylim(0, 1)
    
    # Processing time comparison
    ax2.bar(engines, processing_times, color='lightcoral')
    ax2.set_title('Processing Times')
    ax2.set_ylabel('Time (seconds)')
    
    # Text length comparison
    ax3.bar(engines, text_lengths, color='lightgreen')
    ax3.set_title('Text Length')
    ax3.set_ylabel('Characters')
    
    # Combined metrics
    ax4.bar(engines, confidences, alpha=0.7, label='Confidence', color='skyblue')
    ax4_twin = ax4.twinx()
    ax4_twin.bar(engines, processing_times, alpha=0.7, label='Time', color='lightcoral')
    ax4.set_title('Confidence vs Processing Time')
    ax4.set_ylabel('Confidence')
    ax4_twin.set_ylabel('Time (seconds)')
    
    plt.tight_layout()
    st.pyplot(fig)


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Modern OCR Engine",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Setup logging
    config = config_manager.get_config()
    setup_logging(config)
    
    # Title and description
    st.title("🔍 Modern Optical Character Recognition")
    st.markdown("""
    A comprehensive OCR solution supporting multiple state-of-the-art engines:
    - **Tesseract**: Traditional OCR with excellent accuracy
    - **EasyOCR**: Deep learning-based OCR with GPU support
    - **PaddleOCR**: High-performance OCR with angle detection
    """)
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Engine selection
    available_engines = st.session_state.ocr_manager.get_available_engines()
    engine_options = [engine.value for engine in available_engines]
    
    selected_engine = st.sidebar.selectbox(
        "Select OCR Engine",
        engine_options,
        index=0
    )
    
    # Image preprocessing options
    st.sidebar.header("Image Processing")
    enable_preprocessing = st.sidebar.checkbox("Enable Image Preprocessing", value=True)
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["📸 Upload Image", "📊 Compare Engines", "🎨 Generate Samples", "📈 Results History"])
    
    with tab1:
        st.header("Upload Image for OCR")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            help="Upload an image containing text to extract"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Convert to numpy array for processing
            img_array = np.array(image)
            
            # OCR processing
            if st.button("Extract Text", type="primary"):
                with st.spinner("Processing image..."):
                    try:
                        # Convert PIL image to OpenCV format
                        if len(img_array.shape) == 3:
                            img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                        else:
                            img_cv = img_array
                        
                        # Perform OCR
                        engine_enum = OCREngine(selected_engine)
                        result = st.session_state.ocr_manager.extract_text(img_cv, engine_enum)
                        
                        # Display results
                        st.success("Text extraction completed!")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("Extracted Text")
                            st.text_area("", result.text, height=200)
                            
                            st.subheader("Metadata")
                            st.write(f"**Engine:** {result.engine.value}")
                            st.write(f"**Confidence:** {result.confidence:.3f}")
                            st.write(f"**Processing Time:** {result.processing_time:.3f}s")
                            st.write(f"**Text Length:** {len(result.text)} characters")
                        
                        with col2:
                            if result.bounding_boxes:
                                st.subheader("Bounding Boxes")
                                for i, bbox in enumerate(result.bounding_boxes):
                                    st.write(f"Box {i+1}: {bbox}")
                            
                            # Download options
                            st.subheader("Download Results")
                            
                            # Text file
                            text_file = io.StringIO(result.text)
                            st.download_button(
                                label="Download as Text",
                                data=text_file.getvalue(),
                                file_name="extracted_text.txt",
                                mime="text/plain"
                            )
                            
                            # JSON file
                            result_dict = {
                                "text": result.text,
                                "confidence": result.confidence,
                                "processing_time": result.processing_time,
                                "engine": result.engine.value,
                                "bounding_boxes": result.bounding_boxes
                            }
                            json_str = json.dumps(result_dict, indent=2)
                            st.download_button(
                                label="Download as JSON",
                                data=json_str,
                                file_name="ocr_result.json",
                                mime="application/json"
                            )
                        
                        # Add to history
                        st.session_state.results_history.append({
                            "timestamp": pd.Timestamp.now(),
                            "engine": result.engine.value,
                            "text_length": len(result.text),
                            "confidence": result.confidence,
                            "processing_time": result.processing_time
                        })
                        
                    except Exception as e:
                        st.error(f"Error processing image: {str(e)}")
    
    with tab2:
        st.header("Compare Multiple OCR Engines")
        
        if len(available_engines) < 2:
            st.warning("At least 2 OCR engines are required for comparison.")
        else:
            uploaded_file_compare = st.file_uploader(
                "Choose an image for comparison",
                type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
                key="compare_upload"
            )
            
            if uploaded_file_compare is not None:
                image_compare = Image.open(uploaded_file_compare)
                st.image(image_compare, caption="Image for Comparison", use_column_width=True)
                
                if st.button("Compare All Engines", type="primary"):
                    with st.spinner("Comparing engines..."):
                        try:
                            img_array_compare = np.array(image_compare)
                            if len(img_array_compare.shape) == 3:
                                img_cv_compare = cv2.cvtColor(img_array_compare, cv2.COLOR_RGB2BGR)
                            else:
                                img_cv_compare = img_array_compare
                            
                            # Get results from all engines
                            comparison_results = st.session_state.ocr_manager.extract_text_multiple_engines(img_cv_compare)
                            
                            # Display comparison chart
                            create_comparison_chart(comparison_results)
                            
                            # Detailed results table
                            st.subheader("Detailed Comparison")
                            
                            comparison_data = []
                            for engine, result in comparison_results.items():
                                comparison_data.append({
                                    "Engine": engine.value,
                                    "Text": result.text[:100] + "..." if len(result.text) > 100 else result.text,
                                    "Confidence": f"{result.confidence:.3f}",
                                    "Processing Time": f"{result.processing_time:.3f}s",
                                    "Text Length": len(result.text)
                                })
                            
                            df = pd.DataFrame(comparison_data)
                            st.dataframe(df, use_container_width=True)
                            
                        except Exception as e:
                            st.error(f"Error comparing engines: {str(e)}")
    
    with tab3:
        st.header("Generate Sample Images")
        
        st.markdown("""
        Generate synthetic text images for testing OCR accuracy.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            num_images = st.slider("Number of images", 1, 20, 5)
            categories = st.multiselect(
                "Text categories",
                ["simple", "numbers", "mixed", "paragraphs"],
                default=["simple", "numbers", "mixed"]
            )
        
        with col2:
            variations = st.multiselect(
                "Image variations",
                ["clean", "noisy", "rotated", "blurred"],
                default=["clean", "noisy", "rotated"]
            )
        
        if st.button("Generate Sample Images", type="primary"):
            with st.spinner("Generating sample images..."):
                try:
                    generator = SyntheticTextGenerator()
                    dataset = generator.generate_dataset(
                        num_images=num_images,
                        categories=categories,
                        variations=variations
                    )
                    
                    st.success(f"Generated {len(dataset)} sample images!")
                    
                    # Display first few images
                    st.subheader("Sample Generated Images")
                    
                    for i, (image_path, text) in enumerate(dataset[:6]):  # Show first 6
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            img = Image.open(image_path)
                            st.image(img, caption=f"Sample {i+1}", width=200)
                        
                        with col2:
                            st.write(f"**Ground Truth:** {text}")
                            st.write(f"**File:** {Path(image_path).name}")
                
                except Exception as e:
                    st.error(f"Error generating samples: {str(e)}")
    
    with tab4:
        st.header("Results History")
        
        if st.session_state.results_history:
            # Convert to DataFrame
            df_history = pd.DataFrame(st.session_state.results_history)
            
            # Display summary statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Processings", len(df_history))
            
            with col2:
                avg_confidence = df_history['confidence'].mean()
                st.metric("Avg Confidence", f"{avg_confidence:.3f}")
            
            with col3:
                avg_time = df_history['processing_time'].mean()
                st.metric("Avg Processing Time", f"{avg_time:.3f}s")
            
            with col4:
                total_chars = df_history['text_length'].sum()
                st.metric("Total Characters", total_chars)
            
            # Engine usage chart
            st.subheader("Engine Usage")
            engine_counts = df_history['engine'].value_counts()
            st.bar_chart(engine_counts)
            
            # Detailed history table
            st.subheader("Detailed History")
            st.dataframe(df_history, use_container_width=True)
            
            # Clear history button
            if st.button("Clear History"):
                st.session_state.results_history = []
                st.rerun()
        
        else:
            st.info("No processing history available. Upload and process some images to see history here.")


if __name__ == "__main__":
    main()
