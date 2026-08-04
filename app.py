"""
Industrial Copper Modeling — Streamlit app.

"""

import os
import pickle

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Industrial Copper Modeling", layout="wide")

SOURCE_DIR = "source"

# Shown at the top of the material_ref dropdown -- lets the user skip typing
# an exact reference and fall back to the average frequency instead.
NO_MATERIAL_REF_LABEL = "(none / use average)"


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
        "Run the 'Save Artifacts for Deployment' cell at the end of the "
        "notebook and make sure source/artifacts.pkl, "
        "source/best_regression_model.pkl and "
        "source/best_classification_model.pkl exist next to this script."
    )
    st.stop()

reg_feature_order = artifacts["regression_feature_columns"]
clf_feature_order = artifacts["classification_feature_columns"]
material_ref_freq_map = artifacts["material_ref_freq_map"]
material_ref_freq_default = artifacts["material_ref_freq_default"]
item_type_categories = artifacts["item_type_categories"]
status_categories = artifacts["status_categories"]
country_codes = artifacts["country_codes"]
application_codes = artifacts["application_codes"]
product_ref_codes = artifacts["product_ref_codes"]
reg_scaler = artifacts["regression_scaler"]
clf_scaler = artifacts["classification_scaler"]
defaults = artifacts.get("field_defaults", {})

material_ref_options = [NO_MATERIAL_REF_LABEL] + sorted(
    str(k) for k in material_ref_freq_map.keys()
)


# --------------------------------------------------------------------------
# Shared preprocessing helpers (mirror the notebook exactly)
# --------------------------------------------------------------------------
def material_ref_to_freq(material_ref: str) -> float:
    """Reproduces the frequency encoding from the notebook. Unseen, blank,
    or explicitly-skipped material_ref values fall back to the average
    frequency across all references seen at training time."""
    if material_ref == NO_MATERIAL_REF_LABEL or not material_ref:
        return material_ref_freq_default
    cleaned = str(material_ref).lstrip("0")
    return float(material_ref_freq_map.get(cleaned, material_ref_freq_default))


def build_feature_row(feature_order, base_values, item_type, status=None):
    """Builds a single-row DataFrame matching the training feature columns
    exactly, including one-hot columns for item type (and status, for the
    regression task only). Any dummy column not explicitly set stays 0 --
    which is also the correct encoding for whichever category drop_first
    dropped as the baseline during training."""
    row = {col: 0 for col in feature_order}
    row.update(base_values)

    item_col = f"item type_{item_type}"
    if item_col in row:
        row[item_col] = 1
    # if item_type is the dropped baseline category, leaving all
    # "item type_*" columns at 0 is exactly correct -- nothing more to do

    if status is not None:
        status_col = f"status_{status}"
        if status_col in row:
            row[status_col] = 1

    return pd.DataFrame([row])[feature_order]


def default_index(options, value):
    """Index of `value` in `options`, falling back to the closest match (or
    0) if the saved default isn't exactly in the list."""
    if value in options:
        return options.index(value)
    try:
        return min(range(len(options)), key=lambda i: abs(options[i] - value))
    except (TypeError, ValueError):
        return 0


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
tab1, tab2 = st.tabs(["PREDICT SELLING PRICE", "PREDICT STATUS"])

# ---------------------- TAB 1: Selling price regression -------------------
with tab1:
    with st.form("reg_form"):
        col1, col2 = st.columns(2)
        with col1:
            status = st.selectbox("Status", status_categories)
            item_type = st.selectbox("Item Type", item_type_categories)
            country = st.selectbox(
                "Country code", country_codes,
                index=default_index(country_codes, defaults.get("country", country_codes[0])),
            )
            application = st.selectbox(
                "Application code", application_codes,
                index=default_index(application_codes, defaults.get("application", application_codes[0])),
            )
            product_ref = st.selectbox(
                "Product reference code", product_ref_codes,
                index=default_index(product_ref_codes, defaults.get("product_ref", product_ref_codes[0])),
            )
        with col2:
            thickness = st.number_input(
                "Thickness", min_value=0.0, value=defaults.get("thickness", 1.0)
            )
            width = st.number_input(
                "Width", min_value=0.0, value=defaults.get("width", 1000.0)
            )
            quantity_tons = st.number_input(
                "Quantity (tons)", min_value=0.0001, value=defaults.get("quantity_tons", 10.0)
            )
            material_ref = st.selectbox(
                "Material Reference",
                material_ref_options,
                help="Pick the reference to look up how often it appeared in "
                     "training data. Choose the first option to use the "
                     "average frequency instead.",
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
            base_values = {
                "country": country,
                "application": application,
                "thickness": thickness,
                "width": width,
                "product_ref": product_ref,
                "quantity_tons_log": np.log1p(quantity_tons),
                "material_ref_freq": material_ref_to_freq(material_ref),
            }
            X_new = build_feature_row(reg_feature_order, base_values, item_type, status=status)
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
            citem_type = st.selectbox("Item Type", item_type_categories, key="c_item_type")
            ccountry = st.selectbox(
                "Country code", country_codes,
                index=default_index(country_codes, defaults.get("country", country_codes[0])),
                key="c_country",
            )
            capplication = st.selectbox(
                "Application code", application_codes,
                index=default_index(application_codes, defaults.get("application", application_codes[0])),
                key="c_app",
            )
            cproduct_ref = st.selectbox(
                "Product reference code", product_ref_codes,
                index=default_index(product_ref_codes, defaults.get("product_ref", product_ref_codes[0])),
                key="c_product_ref",
            )
        with col2:
            cthickness = st.number_input(
                "Thickness", min_value=0.0, value=defaults.get("thickness", 1.0), key="c_thickness"
            )
            cwidth = st.number_input(
                "Width", min_value=0.0, value=defaults.get("width", 1000.0), key="c_width"
            )
            cquantity_tons = st.number_input(
                "Quantity (tons)", min_value=0.0001, value=defaults.get("quantity_tons", 10.0), key="c_qty"
            )
            cmaterial_ref = st.selectbox(
                "Material Reference",
                material_ref_options,
                key="c_material_ref",
                help="Pick the reference to look up how often it appeared in "
                     "training data. Choose the first option to use the "
                     "average frequency instead.",
            )

        clf_submit = st.form_submit_button("PREDICT STATUS")

    if clf_submit:
        try:
            base_values = {
                "country": ccountry,
                "application": capplication,
                "thickness": cthickness,
                "width": cwidth,
                "product_ref": cproduct_ref,
                "quantity_tons_log": np.log1p(cquantity_tons),
                "material_ref_freq": material_ref_to_freq(cmaterial_ref),
            }
            X1_new = build_feature_row(clf_feature_order, base_values, citem_type)
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
