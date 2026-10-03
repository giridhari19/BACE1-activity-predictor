import streamlit as st
import joblib
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
st.set_page_config(
    page_title="BACE Activity Predictor",
    page_icon="🧬",
    layout="centered"
)

lr = joblib.load("lr_morgan.joblib")
rf = joblib.load("rf_morgan.joblib")

morgan_generator = rdFingerprintGenerator.GetMorganGenerator(
    radius=2,
    fpSize=2048
)

def predict_activity(smiles):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None, None, None, None, None, None

    fp = morgan_generator.GetFingerprintAsNumPy(mol).reshape(1, -1)

    lr_pred = lr.predict(fp)[0]
    lr_prob = lr.predict_proba(fp)[0, 1]
    lr_iprob = lr.predict_proba(fp)[0, 0]

    rf_pred = rf.predict(fp)[0]
    rf_prob = rf.predict_proba(fp)[0, 1]
    rf_iprob = rf.predict_proba(fp)[0, 0]

    return (
        lr_pred, lr_prob, lr_iprob,
        rf_pred, rf_prob, rf_iprob
    )


st.title("BACE Activity Predictor")

st.markdown(
    "Predict BACE activity from a molecular **SMILES** representation "
    "using two machine-learning models trained on Morgan fingerprints."
)
st.markdown(
    "Refer to the Github Page for detailed report on the project: [Github Repo](https://github.com/giridhari19/BACE1-activity-predictor)"
)
st.caption(
    "Predictions are model outputs and do not constitute experimental "
    "confirmation of biological activity."
)

st.subheader("Molecule")
smiles = st.text_area(
    "SMILES",
    placeholder="Enter a molecular SMILES string...",
    height=100,
    label_visibility="collapsed"
)

st.text(
    "Example: CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O"
)

predict_button = st.button(
    "Predict Activity",
    type="primary",
    use_container_width=True
)

if predict_button:

    if not smiles.strip():
        st.warning("Please enter a SMILES string.")
    else:
        results = predict_activity(smiles.strip())

        if results[0] is None:
            st.error("Invalid SMILES. Please check the molecular structure and try again.")
        else:
            (lr_pred, lr_prob, lr_iprob,rf_pred, rf_prob, rf_iprob) = results
            st.divider()
            st.subheader("Prediction Results")
            col1, col2 = st.columns(2)
            # Logistic Regression
            with col1:
                st.markdown("### Logistic Regression")
                if lr_pred == 1:
                    st.success("ACTIVE")
                else:
                    st.info("INACTIVE")
                st.metric(
                    "Active probability",
                    f"{lr_prob:.2%}"
                )
                st.metric(
                    "Inactive probability", 
                    f"{lr_iprob:.2%}"
                )

            # Random Forest
            with col2:
                st.markdown("### Random Forest")

                if rf_pred == 1:
                    st.success("ACTIVE")
                else:
                    st.info("INACTIVE")

                st.metric(
                    "Active probability",
                    f"{rf_prob:.2%}"
                )

                st.metric(
                    f"Inactive probability", 
                    f"{rf_iprob:.2%}"
                )

st.divider()

st.subheader("Methodology")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Dataset**")
    st.write("[BACE](https://huggingface.co/datasets/scikit-fingerprints/MoleculeNet_BACE)")

with col2:
    st.markdown("**Representation**")
    st.write("Morgan fingerprint")

with col3:
    st.markdown("**Models**")
    st.write("Logistic Regression\nRandom Forest")
