from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
model = joblib.load(APP_DIR / 'modele_decision_data_officer.joblib')

st.set_page_config(page_title='Assistant Data Officer', page_icon='📊', layout='wide')
st.title('Assistant Data Officer')
st.caption('Aide à la décision pour les dossiers de création de fonds d’investissement')

with st.sidebar:
    st.info('Démonstrateur académique. La validation finale appartient au Data Officer.')

c1, c2 = st.columns(2)
with c1:
    type_document = st.selectbox('Type de document', ['Prospectus','Règlement / By-Laws','By-Laws','Limited Partnership Deed','Limited Partnership Agreement','Legal document'])
    forme_juridique = st.selectbox('Forme juridique', ['FCP','FPCI','FPS','OPCVM-FCP','SLP','LP','SCSp-SICAV-RAIF','Scottish LP'])
    pays = st.selectbox('Pays de domiciliation', ['France','Luxembourg','Royaume-Uni','États-Unis'])
    devise = st.selectbox('Devise principale', ['EUR','USD','GBP'])
    langue = st.selectbox('Langue du document', ['Français','Anglais'])
with c2:
    document_present = st.radio('Document constitutif présent', ['Oui','Non'], horizontal=True)
    forme_juridique_presente = st.radio('Forme juridique présente', ['Oui','Non'], horizontal=True)
    pays_present = st.radio('Pays présent', ['Oui','Non'], horizontal=True)
    adresse_presente = st.radio('Adresse présente', ['Oui','Non'], horizontal=True)
    devise_presente = st.radio('Devise présente', ['Oui','Non'], horizontal=True)

c3, c4 = st.columns(2)
with c3:
    gp_requis = st.radio('General Partner requis', ['Non','Oui'], horizontal=True)
    gp_identifie = st.radio('General Partner identifié', ['Oui','Non'], horizontal=True) if gp_requis == 'Oui' else 'Non applicable'
    isin_statut = st.selectbox('Statut de l’ISIN', ['Présent','Provisoire','Ambigu'])
with c4:
    sfdr_statut = st.selectbox('Statut SFDR', ['Article 6','Article 8','Article 9','Non applicable','Absent alors qu’attendu'])
    conformite_adresse = st.selectbox('Conformité de l’adresse', ['Conforme','À vérifier'])
    incoherence_detectee = st.radio('Incohérence documentaire détectée', ['Non','Oui'], horizontal=True)
    version_document = st.selectbox('Version du document', ['Finale','Provisoire'])

def explain(d):
    missing, manual = [], []
    mapping = {
        'document_present': 'le document constitutif ou le prospectus à jour',
        'forme_juridique_presente': 'la forme juridique du fonds',
        'pays_present': 'le pays de domiciliation',
        'adresse_presente': 'l’adresse du fonds ou de l’entité',
        'devise_presente': 'la devise principale',
    }
    for key, text in mapping.items():
        if d[key] == 'Non':
            missing.append(text)
    if d['gp_requis'] == 'Oui' and d['gp_identifie'] == 'Non':
        missing.append('l’identification du General Partner')
    if d['sfdr_statut'] == 'Absent alors qu’attendu': manual.append('classification SFDR à vérifier')
    if d['isin_statut'] in ['Provisoire','Ambigu']: manual.append('statut de l’ISIN à contrôler')
    if d['incoherence_detectee'] == 'Oui': manual.append('incohérence documentaire détectée')
    if d['conformite_adresse'] == 'À vérifier': manual.append('conformité de l’adresse à vérifier')
    if d['version_document'] == 'Provisoire': manual.append('document provisoire')
    return missing, manual

if st.button('Analyser le dossier', type='primary', use_container_width=True):
    data = {
        'type_document': type_document, 'forme_juridique': forme_juridique,
        'pays_domiciliation': pays, 'devise_principale': devise,
        'langue_document': langue, 'gp_requis': gp_requis,
        'document_present': document_present,
        'forme_juridique_presente': forme_juridique_presente,
        'pays_present': pays_present, 'adresse_presente': adresse_presente,
        'devise_presente': devise_presente, 'gp_identifie': gp_identifie,
        'isin_statut': isin_statut, 'sfdr_statut': sfdr_statut,
        'conformite_adresse': conformite_adresse,
        'incoherence_detectee': incoherence_detectee,
        'version_document': version_document,
    }
    row = pd.DataFrame([data])
    pred = model.predict(row)[0]
    probs = model.predict_proba(row)[0]
    classes = model.named_steps['model'].classes_
    confidence = float(max(probs))
    missing, manual = explain(data)

    st.subheader('Résultat')
    if pred == 'Création possible': st.success('🟢 Création possible')
    elif pred == 'Analyse manuelle': st.warning('🟠 Analyse manuelle')
    else: st.error('🔴 Complément documentaire requis avant création')
    st.metric('Confiance du modèle', f'{confidence:.1%}')

    if missing:
        st.markdown('### Éléments à demander')
        for item in missing: st.write(f'• {item}')
        bullets = '\n'.join(f'- {item}' for item in missing)
        email = f"Bonjour,\n\nAprès analyse du dossier transmis, certaines informations indispensables à la création du fonds sont absentes.\n\nAfin de poursuivre le traitement, pourriez-vous nous transmettre les éléments suivants :\n\n{bullets}\n\nNous vous remercions par avance pour votre retour.\n\nBien cordialement,"
        st.markdown('### Brouillon d’email')
        st.text_area('Email généré', email, height=280)
    if manual:
        st.markdown('### Points à contrôler')
        for item in manual: st.write(f'• {item}')
    with st.expander('Détail des probabilités'):
        st.dataframe(pd.DataFrame({'Statut': classes, 'Probabilité': probs}).sort_values('Probabilité', ascending=False), hide_index=True, use_container_width=True)
