import streamlit as st

from constants import OPUS_TRANSLATION_CONFIG_PATH
from gpc_agent import build_graph, translation_model
from utils import classify_product, load_translation_model

# --- Load the agent and DB/models ---
def load_agent():
    """Load model, translation, and build the agent."""
    with st.spinner("Loading model and preparing agent..."):
        agent = build_graph()  # expensive operation
    return agent

# --- Streamlit UI ---
def main_ui():
    st.set_page_config(page_title="Product Classifier", layout="centered")
    st.title("Product Classification Agent")

    # Load agent once
    translation_model = load_translation_model(OPUS_TRANSLATION_CONFIG_PATH)
    agent = load_agent()


    # Input and button
    product_name = st.text_input("Enter the product name:")

    if st.button("Classify"):
        if product_name.strip() == "":
            st.warning("Please enter a product name.")
        else:
            with st.spinner("Classifying product..."):
                result = classify_product(agent, translation_model, product_name)
            st.success("Classification Result")
            st.json(result)

# --- Run the app ---
if __name__ == "__main__":
    main_ui()
