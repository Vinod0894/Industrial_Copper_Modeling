"""
Industrial Copper Modeling — Streamlit app.

Reproduces the EXACT preprocessing pipeline used in the training notebook
(Industrial_Copper_leakage_fixed_V2.ipynb), using the artifacts saved by the
"save artifacts" cell added at the end of that notebook.

Regression features (order, from X.columns in the notebook):
    country, status, item type, application, width,
    material_ref_target_log, quantity_tons_log
  target: selling_price_log = log1p(selling_price)  ->  price = expm1(pred)

Classification features (order, from X1.columns in the notebook):
    country, item type, application, width,
    selling_price_log, material_ref_target_log, quantity_tons_log
  target: 1 = Won, 0 = Lost   (as encoded during training)

Expected files, all inside a "source" folder next to this script:
    source/artifacts.pkl
    source/best_regression_model.pkl
    source/best_classification_model.pkl
"""

import os
import pickle

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Industrial Copper Modeling", layout="wide")

SOURCE_DIR = "source"


# --------------------------------------------------------------------------
# Artifact loading
# --------------------------------------------------------------------------
@st.cache_resource
def load_pickle(filename):
    path = os.path.join(SOURCE_DIR, filename)
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_all_artifacts():
    try:
        artifacts = load_pickle("artifacts.pkl")
        reg_model = load_pickle("best_regression_model.pkl")
        clf_model = load_pickle("best_classification_model.pkl")
        return artifacts, reg_model, clf_model, None
    except FileNotFoundError as e:
        return None, None, None, str(e)


artifacts, reg_model, clf_model, load_error = load_all_artifacts()

st.markdown(
    """
    <div style='text-align:center'>
        <h1 style='color:#009999;'>Industrial Copper Modeling Application</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

if load_error:
    st.error(
        "Couldn't find the required pickle files.\n\n"
        f"Details: {load_error}\n\n"
        "Run the 'save artifacts' cell at the end of the notebook and make sure "
        "source/artifacts.pkl, source/best_regression_model.pkl and "
        "source/best_classification_model.pkl exist next to this script."
    )
    st.stop()

item_encoder = artifacts["item_encoder"]
status_encoder = artifacts["status_encoder"]
app_country_encoder = artifacts["app_country_encoder"]   # fit on ['application','country']
reg_scaler = artifacts["reg_scaler"]
clf_scaler = artifacts["clf_scaler"]
material_ref_target_mean = artifacts["material_ref_target_mean"]   # pandas Series
material_ref_global_mean = artifacts["material_ref_global_mean"]
reg_feature_order = artifacts["reg_feature_order"]
clf_feature_order = artifacts["clf_feature_order"]
item_type_rare = artifacts.get("item_type_rare", ["Others", "WI", "IPL", "SLAWR"])

# Dropdown option lists (raw / pre-grouping labels shown to the user)
status_options = list(status_encoder.categories_[0])
item_type_options_raw = ['W', 'WI', 'S', 'Others', 'PL', 'IPL', 'SLAWR']
# app_country_encoder was fit on columns ['application', 'country'] in that order
known_applications = set(app_country_encoder.categories_[0])
known_countries = set(app_country_encoder.categories_[1])
# Every material_ref seen at training time, for the dropdown (Streamlit selectboxes
# support type-ahead filtering, so this stays usable even with thousands of entries)
material_ref_options = ["(none / use average)"] + sorted(
    material_ref_target_mean.index.astype(str).tolist()
)


# --------------------------------------------------------------------------
# Shared preprocessing helpers (mirror the notebook exactly)
# --------------------------------------------------------------------------
def group_item_type(raw_item_type: str) -> str:
    """Rare item types were folded into 'Other' during training (cell 42)."""
    return "Other" if raw_item_type in item_type_rare else raw_item_type


def group_rare(value, known_values):
    """Application/country values with <500 train occurrences became 'Other'
    (cell 47). Anything the encoder never saw at fit time gets the same
    treatment at inference time."""
    value_str = str(value)
    return value_str if value_str in known_values else "Other"


def material_ref_to_target_log(material_ref: str) -> float:
    """Reproduces the train-only target-mean encoding + log1p from cells 15/32.
    Unseen or blank material_ref values fall back to the train-only global mean,
    exactly as at training time."""
    cleaned = str(material_ref).lstrip("0")
    mean_val = material_ref_target_mean.get(cleaned, material_ref_global_mean)
    return np.log1p(mean_val)


def encode_item_type(raw_item_type: str) -> float:
    grouped = group_item_type(raw_item_type)
    return item_encoder.transform([[grouped]])[0][0]


def encode_status(raw_status: str) -> float:
    return status_encoder.transform([[raw_status]])[0][0]


def encode_application_country(application, country):
    app_grouped = group_rare(application, known_applications)
    country_grouped = group_rare(country, known_countries)
    encoded = app_country_encoder.transform([[app_grouped, country_grouped]])[0]
    return encoded[0], encoded[1]  # application_encoded, country_encoded


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
tab1, tab2 = st.tabs(["PREDICT SELLING PRICE", "PREDICT STATUS"])

# ---------------------- TAB 1: Selling price regression -------------------
with tab1:
    with st.form("reg_form"):
        col1, col2 = st.columns(2)
        with col1:
            status = st.selectbox("Status", status_options)
            item_type = st.selectbox("Item Type", item_type_options_raw)
            country = st.number_input("Country code", value=28.0, step=1.0, format="%.0f")
            application = st.number_input("Application code", value=10.0, step=1.0, format="%.0f")
        with col2:
            width = st.number_input("Width", min_value=0.0, value=1000.0)
            quantity_tons = st.number_input("Quantity (tons)", min_value=0.0001, value=10.0)
            material_ref = st.text_input(
                "Material Reference",
                help="Used to look up the trained material-ref pricing signal. "
                     "Unrecognized references safely fall back to the average.",
            )

        reg_submit = st.form_submit_button("PREDICT SELLING PRICE")
        st.markdown(
            """
            <style>
            div.stButton > button:first-child {
                background-color: #009999;
                color: white;
                width: 100%;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

    if reg_submit:
        try:
            app_enc, country_enc = encode_application_country(application, country)
            status_enc = encode_status(status)
            item_enc = encode_item_type(item_type)
            material_ref_target_log = material_ref_to_target_log(material_ref)
            quantity_tons_log = np.log1p(quantity_tons)

            row = {
                "country": country_enc,
                "status": status_enc,
                "item type": item_enc,
                "application": app_enc,
                "width": width,
                "material_ref_target_log": material_ref_target_log,
                "quantity_tons_log": quantity_tons_log,
            }
            X_new = pd.DataFrame([row])[reg_feature_order]
            X_new_scaled = reg_scaler.transform(X_new)

            pred_log = reg_model.predict(X_new_scaled)[0]
            predicted_price = np.expm1(pred_log)

            st.write("## :green[Predicted selling price:] ", round(float(predicted_price), 2))
        except Exception as e:
            st.error(f"Couldn't compute a prediction: {e}")

# ---------------------- TAB 2: Won / Lost classification -------------------
with tab2:
    with st.form("clf_form"):
        col1, col2 = st.columns(2)
        with col1:
            citem_type = st.selectbox("Item Type", item_type_options_raw, key="c_item_type")
            ccountry = st.number_input("Country code", value=28.0, step=1.0, format="%.0f", key="c_country")
            capplication = st.number_input("Application code", value=10.0, step=1.0, format="%.0f", key="c_app")
        with col2:
            cwidth = st.number_input("Width", min_value=0.0, value=1000.0, key="c_width")
            cquantity_tons = st.number_input("Quantity (tons)", min_value=0.0001, value=10.0, key="c_qty")
            cselling_price = st.number_input("Quoted Selling Price", min_value=0.0001, value=500.0, key="c_price")
            cmaterial_ref = st.text_input("Material Reference", key="c_material_ref")

        clf_submit = st.form_submit_button("PREDICT STATUS")

    if clf_submit:
        try:
            app_enc, country_enc = encode_application_country(capplication, ccountry)
            item_enc = encode_item_type(citem_type)
            material_ref_target_log = material_ref_to_target_log(cmaterial_ref)
            quantity_tons_log = np.log1p(cquantity_tons)
            selling_price_log = np.log1p(cselling_price)

            row = {
                "country": country_enc,
                "item type": item_enc,
                "application": app_enc,
                "width": cwidth,
                "selling_price_log": selling_price_log,
                "material_ref_target_log": material_ref_target_log,
                "quantity_tons_log": quantity_tons_log,
            }
            X1_new = pd.DataFrame([row])[clf_feature_order]
            X1_new_scaled = clf_scaler.transform(X1_new)

            pred = clf_model.predict(X1_new_scaled)[0]

            if pred == 1:
                st.write("## :green[The Status is Won]")
            else:
                st.write("## :red[The status is Lost]")
        except Exception as e:
            st.error(f"Couldn't compute a prediction: {e}")

st.write(
    '<h6 style="color:rgb(0, 153, 153,0.35);">App Created by Vinod R</h6>',
    unsafe_allow_html=True,
)