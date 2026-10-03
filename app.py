import streamlit as st
import joblib
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

lr = joblib.load("lr_morgan.joblib")
rf = joblib.load("rf_morgan.joblib")

morgan_generator = rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)

def predict_activity(smiles):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None, None

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
st.title('BACE Activity Predictor')
st.write('Predict BACE activity using Logistic Regression and Random Forest Classifier using Morgan Fingerprints.')
smiles=st.text_input('SMILES', placeholder="Enter a molecular SMILES string")
if st.button('Predict'):
  if not smiles:
    st.warning('Please enter a SMILES string')
  else:
    results=predict_activity(smiles)

    if results[0] is None:
      st.error("Invalid SMILES string. Please check it")
    else:
      (lr_pred, lr_prob, lr_iprob,
        rf_pred, rf_prob, rf_iprob
      )=results
        
