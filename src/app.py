import streamlit as st

import os

from constants import IMG_PATH
from utils import get_confidence_level
from models import BrandsClassifier, TfidfClassifier, EmbeddingClassifier, EnsembleModel, SentenceEmbeddingModel

# --- PAGE CONFIG ---
st.set_page_config(page_title="Prodify+ - Advanced Invoice Classifier", page_icon="🧾", layout="centered")

@st.cache_resource
def get_ensemble_model():
    tfidf_classifier =  TfidfClassifier()
    brands_classifier = BrandsClassifier()
    embedding_model = SentenceEmbeddingModel()
    embedding_classifier = EmbeddingClassifier(embedding_model)
    ensemble_model = EnsembleModel(brands_classifier, embedding_classifier, tfidf_classifier)
    return ensemble_model

# --- CLASSIFICATION LOGIC ---
def classify_product(ensemble_model, product):
    result = ensemble_model.run_ensemble(product)
    return result

# --- SIDEBAR ---
def render_sidebar() -> None:
    """Sidebar with logo, app name, and description"""
    with st.sidebar:
        logo_path = IMG_PATH / "td_new_trans.png"
        if os.path.exists(logo_path):
            st.image(logo_path)

        st.markdown(
            """
            <p style='color: grey; margin-bottom: 20px;'><b>Prodify+</b> is an AI-powered invoice product classification system.</p>
            <p style='color: grey; margin-bottom: 20px;'>Enter a product description and our ensemble model will predict it's corresponding three level GPC classification (Segment, Family, Class).</p>
            """,
            unsafe_allow_html=True
        )
        
        # Collapsible Ensemble Model Components section
        with st.expander("AI Models"):
            st.markdown(
                """
                <p style='color: grey; margin-bottom: 20px;'><b>1. Embedding Classifier:</b> E5-Large-Instruct for hierarchical semantic classification through cosine similarity.</p>
                <p style='color: grey; margin-bottom: 20px;'><b>2. Brand Classifier Model:</b> TF-IDF cosine similarity matching against a handpicked brand, product dataset.</p>
                <p style='color: grey; margin-bottom: 20px;'><b>3. TF-IDF + SVM Classifier:</b> Traditional ML pipeline for direct GPC prediction.</p>
                """,
                unsafe_allow_html=True
            )
        
        # Collapsible Datasets section
        with st.expander("Datasets"):
            st.markdown(
                """
                <p style='color: grey; margin-bottom: 20px;'><b>Training/Test Data:</b> 76k products merged from: <br> 1) <a href='https://www.kaggle.com/datasets/mohit2512/jio-mart-product-items/data' target='_blank' style='color: #2c7be5;'>Jio Mart</a> (category/sub-category mapped to GPC levels). <br> 2) <a href='https://github.com/ir-ischool-uos/mwpd' target='_blank' style='color: #2c7be5;'>MWPD</a> (pre-labeled GPC levels). <br> 3) <a href='https://fdc.nal.usda.gov/download-datasets/#bkmk-2' target='_blank' style='color: #2c7be5;'>USDA FoodData</a> (category mapped to four GPC levels). Split: 61k training, 15k test.</p>
                <p style='color: grey; margin-bottom: 20px;'><b>Brands Dataset:</b> 87 curated brands with 20 products each, manually mapped to first three GPC levels for similarity matching.</p>
                """,
                unsafe_allow_html=True
            )
        st.markdown("---")

        # Github repo link
        st.markdown(
            "<a href='https://github.com/OmarSamirz/Prodify-V2.0' target='_blank' style='color: #2c7be5; text-decoration: none;'>"
            "GitHub Repository</a>",
            unsafe_allow_html=True
        )

        # Copyright notice
        st.markdown(
            "<p style='color: grey;'>© 2025 Teradata.</p>",
            unsafe_allow_html=True
        )


# --- MAIN APP ---
def main():
    ensemble_model = get_ensemble_model()

    render_sidebar()

    st.markdown("# Prodify+")
    st.markdown(
        "<p style='margin-bottom:0px; font-size:20px;'><b>Enter invoice item:</b></p>",
        unsafe_allow_html=True
    )
    invoice_item = st.text_input("Invoice Item", key="invoice_item", label_visibility="collapsed")

    # --- CLASSIFICATION BUTTON ---
    if st.button("Classify"):
        if not invoice_item.strip():
            st.warning("Please enter an invoice item.")
        else:
            try:
                result = classify_product(ensemble_model, invoice_item)
                
                # Display main results
                st.success(f"**Segment:** {result['voted_segments'][0]}")
                st.success(f"**Family:** {result['voted_families'][0]}")
                st.success(f"**Class:** {result['voted_classes'][0]}")
                st.info(f"**Confidence:** {get_confidence_level(result['confidences'])[0]}")
                
                # Display individual model predictions in expandable section
                with st.expander("Individual Model Predictions"):
                    st.write("**Embedding Classifier:**")
                    st.write(f"- Segment: {result['embed_clf_preds'][0][0]}")
                    st.write(f"- Family: {result['embed_clf_preds'][1][0]}")
                    st.write(f"- Class: {result['embed_clf_preds'][2][0]}")
                    
                    st.write("**Brand Classifier Model:**")
                    st.write(f"- Segment: {result['brand_tfidf_sim_preds'][0][0]}")
                    st.write(f"- Family: {result['brand_tfidf_sim_preds'][1][0]}")
                    st.write(f"- Class: {result['brand_tfidf_sim_preds'][2][0]}")
                    
                    st.write("**TF-IDF + SVM Classifier:**")
                    st.write(f"- Segment: {result['tfidf_clf_preds'][0][0]}")
                    st.write(f"- Family: {result['tfidf_clf_preds'][1][0]}")
                    st.write(f"- Class: {result['tfidf_clf_preds'][2][0]}")
                    
            except Exception as e:
                st.error(f"Error: {e}")

if __name__ == "__main__":
    main()