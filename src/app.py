import torch
import streamlit as st
from teradataml import *

from modules.db import TeradataDatabase
from utils import load_tfidf_model, load_embedding_model
from constants import TFIDF_CLASSIFIER_CONFIG_PATH, E5_LARGE_INSTRUCT_CONFIG_PATH, IMG_PATH


# --- PAGE CONFIG ---
st.set_page_config(page_title="Prodify - Invoice Classifier", page_icon="🧾", layout="centered")


# --- CACHED LOADERS ---
@st.cache_resource
def get_db():
    db = TeradataDatabase()
    db.connect()
    return db


@st.cache_resource
def get_models():
    embed_model = load_embedding_model(E5_LARGE_INSTRUCT_CONFIG_PATH)
    tfidf_model = load_tfidf_model(TFIDF_CLASSIFIER_CONFIG_PATH)
    tfidf_model.load()
    return {"GenAI": embed_model, "ML Model": tfidf_model}


@st.cache_data
def get_classes():
    return DataFrame.from_table("classes").to_pandas().sort_values(by="id")["class_name"].tolist()


# --- CLASSIFICATION LOGIC ---
def classify_product(model, product, classes):
    if hasattr(model, "predict"):  # TF-IDF XGBoost
        prediction = model.predict([product])[0]
        return classes[prediction]

    elif hasattr(model, "get_scores"):  # Embedding model
        prediction = model.get_scores(product, classes)
        prediction = torch.argmax(prediction, dim=1)[0]
        return classes[prediction]
    else:
        return "Unknown model type"

# --- SIDEBAR ---
def render_sidebar() -> None:
    """Sidebar with logo, app name, and description"""
    with st.sidebar:
        logo_path = IMG_PATH / "td_new_trans.png"
        if os.path.exists(logo_path):
            st.image(logo_path)

        st.markdown(
            """
            <p style='color: grey; margin-bottom: 0px; font-size: 20px;'><b>App</b></p>
            <p style='color: grey; margin-bottom: 20px;'><b>Prodify</b> is an intelligent product classification system.</p>
            <p></p><p style='color: grey; margin-bottom: 20px;'>Enter a product description and the AI will predict the most relevant category.</p>
            <p></p><p style='color: grey; margin-bottom: 0px; font-size: 20px;'><b>Models</b></p>
            <p style='color: grey; margin-bottom: 20px;'><b>GenAI:</b> Uses E5-Large-Instruct an 0.6 B scale instruction-tuned text embedding model.</p>
            <p></p><p style='color: grey; margin-bottom: 20px;'><b>ML Model:</b> Uses a machine learing pipeline that utilizes TF-IDF and XGBoost.</p>
            <p></p><p style='color: grey; margin-bottom: 0px; font-size: 20px;'><b>Dataset</b></p>
            <p style='color: grey; margin-bottom: 20px;'><b>AMURD Dataset:</b> A human-annotated multilingual receipts dataset for key-information extraction and item classification across 44 product categories.</p>
            """,
            unsafe_allow_html=True
        )
        st.markdown("---")
        # Github repo link
        st.markdown(
            "<a href='https://github.com/OmarSamirz/AMuRD-Iteration-8' target='_blank' style='color: #2c7be5; text-decoration: none;'>"
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
    db = get_db()
    models = get_models()
    classes = get_classes()

    render_sidebar()

    st.markdown("# Prodify")
    st.markdown(
        "<p style='margin-bottom:0px; font-size:20px;'><b>Enter invoice item:</b></p>",
        unsafe_allow_html=True
    )
    invoice_item = st.text_input("", key="invoice_item", label_visibility="collapsed")

    st.markdown(
        "<p style='margin-bottom:0px; font-size:20px;'><b>Select model:</b></p>",
        unsafe_allow_html=True
    )
    model_choice = st.selectbox("", list(models.keys()), key="model_choice", label_visibility="collapsed")



    # --- CLASSIFICATION BUTTON ---
    if st.button("Classify"):
        if not invoice_item.strip():
            st.warning("Please enter an invoice item.")
        else:
            try:
                model = models[model_choice]
                if model_choice == "ML Model":
                    model.load()
                result = classify_product(model, invoice_item, classes)
                st.success(f"Item category is: **{result}**")
            except Exception as e:
                st.error(f"Error: {e}")


if __name__ == "__main__":
    main()
