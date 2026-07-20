from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import streamlit as st

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "modele_decision_data_officer.joblib"

st.set_page_config(
    page_title="Assistant Data Officer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
:root {
    --ink: #172033;
    --muted: #667085;
    --panel: #ffffff;
    --line: #e4e7ec;
    --soft: #f5f7fa;
    --accent: #d83a56;
    --success: #16865c;
    --warning: #b87800;
    --danger: #c9364f;
}
.block-container {max-width: 1240px; padding-top: 1.2rem; padding-bottom: 3rem;}
[data-testid="stSidebar"] {background: linear-gradient(180deg, #f7f8fb 0%, #eef1f6 100%);}
.hero {padding: 1.7rem 1.9rem; border: 1px solid var(--line); border-radius: 22px; background: linear-gradient(135deg,#fff 0%,#f8f9fc 100%); box-shadow: 0 14px 35px rgba(23,32,51,.06); margin-bottom: 1.2rem;}
.hero h1 {margin:0; color:var(--ink); font-size:2.35rem;}
.hero p {margin:.55rem 0 0; color:var(--muted); font-size:1.02rem;}
.kpi {padding:1.1rem 1.2rem; border:1px solid var(--line); border-radius:18px; background:#fff; min-height:118px; box-shadow:0 7px 20px rgba(23,32,51,.04);}
.kpi-label {color:var(--muted); font-size:.9rem; margin-bottom:.4rem;}
.kpi-value {color:var(--ink); font-size:1.8rem; font-weight:750; line-height:1.15;}
.kpi-sub {color:var(--muted); font-size:.82rem; margin-top:.45rem;}
.section-card {padding:1.25rem 1.35rem; border:1px solid var(--line); border-radius:18px; background:#fff; margin-bottom:1rem;}
.result {padding:1.4rem 1.5rem; border-radius:20px; border:1px solid var(--line); margin-top:1rem;}
.result-success {background:#eefaf5; border-color:#b8e6d3;}
.result-warning {background:#fff8e8; border-color:#efd59a;}
.result-danger {background:#fff0f2; border-color:#efbec7;}
.result-title {font-size:1.45rem; font-weight:800; margin-bottom:.35rem;}
.muted {color:var(--muted);}
.badge {display:inline-block; padding:.3rem .65rem; border-radius:999px; background:#eef1f6; color:#344054; font-size:.82rem; font-weight:650;}
.step-row {display:flex; align-items:center; gap:.45rem; flex-wrap:wrap; margin:.8rem 0 1.1rem;}
.step {padding:.48rem .75rem; border-radius:999px; background:#eef1f6; color:#475467; font-size:.84rem;}
.step.done {background:#e9f8f1; color:#16744f;}
.step.current {background:#fff0f2; color:#b52842;}
.small {font-size:.86rem; color:var(--muted);}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def ensure_state() -> None:
    defaults = {
        "history": [],
        "document_present": True,
        "forme_juridique_presente": True,
        "pays_present": True,
        "adresse_presente": True,
        "devise_presente": True,
        "gp_requis": False,
        "gp_identifie": True,
        "incoherence_detectee": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_form() -> None:
    for key in [
        "document_present", "forme_juridique_presente", "pays_present",
        "adresse_presente", "devise_presente", "gp_identifie"
    ]:
        st.session_state[key] = True
    st.session_state["gp_requis"] = False
    st.session_state["incoherence_detectee"] = False


def extract_pdf_text(uploaded_file) -> str:
    if PdfReader is None:
        return ""
    try:
        reader = PdfReader(uploaded_file)
        parts = []
        for page in reader.pages[:8]:
            parts.append(page.extract_text() or "")
        return "\n".join(parts)
    except Exception:
        return ""


def suggest_from_text(text: str) -> dict[str, str]:
    low = text.lower()
    suggestions: dict[str, str] = {}
    if "luxembourg" in low:
        suggestions["pays"] = "Luxembourg"
    elif "france" in low or "français" in low:
        suggestions["pays"] = "France"
    elif "united kingdom" in low or "scotland" in low:
        suggestions["pays"] = "Royaume-Uni"
    if "eur" in low or "euro" in low:
        suggestions["devise"] = "EUR"
    elif "usd" in low or "dollar" in low:
        suggestions["devise"] = "USD"
    elif "gbp" in low or "sterling" in low:
        suggestions["devise"] = "GBP"
    for form in ["FPCI", "FCP", "FPS", "SLP", "LP", "SCSp-SICAV-RAIF", "Scottish LP"]:
        if form.lower() in low:
            suggestions["forme"] = form
            break
    if "prospectus" in low:
        suggestions["type_document"] = "Prospectus"
    elif "limited partnership agreement" in low:
        suggestions["type_document"] = "Limited Partnership Agreement"
    elif "by-laws" in low or "bylaws" in low:
        suggestions["type_document"] = "By-Laws"
    return suggestions


def build_email(missing_items: list[str]) -> str:
    bullets = "\n".join(f"- {item}" for item in missing_items)
    return f"""Bonjour,

Après analyse du dossier transmis, certaines informations indispensables au traitement sont absentes.

Afin de poursuivre la création du fonds, pourriez-vous nous transmettre les éléments suivants :

{bullets}

Nous vous remercions par avance pour votre retour.

Bien cordialement,"""


def explain_input(data: dict[str, Any]) -> tuple[list[str], list[str], list[str]]:
    blocking, manual, validated = [], [], []
    checks = [
        ("document_present", "le document constitutif ou le prospectus à jour", "Document constitutif présent"),
        ("forme_juridique_presente", "la forme juridique du fonds", "Forme juridique présente"),
        ("pays_present", "le pays de domiciliation", "Pays de domiciliation présent"),
        ("adresse_presente", "l’adresse du fonds ou de l’entité", "Adresse présente"),
        ("devise_presente", "la devise principale", "Devise principale présente"),
    ]
    for key, missing, ok in checks:
        if data[key] == "Non":
            blocking.append(missing)
        else:
            validated.append(ok)
    if data["gp_requis"] == "Oui":
        if data["gp_identifie"] == "Non":
            blocking.append("l’identification du General Partner")
        else:
            validated.append("General Partner identifié")
    if data["sfdr_statut"] == "Absent alors qu’attendu":
        manual.append("la classification SFDR doit être complétée ou vérifiée")
    if data["isin_statut"] in ["Provisoire", "Ambigu"]:
        manual.append("le statut de l’ISIN nécessite un contrôle")
    if data["incoherence_detectee"] == "Oui":
        manual.append("une incohérence documentaire a été détectée")
    if data["conformite_adresse"] == "À vérifier":
        manual.append("la conformité de l’adresse doit être vérifiée")
    if data["version_document"] == "Provisoire":
        manual.append("le document transmis est provisoire")
    return blocking, manual, validated


ensure_state()
model = load_model()

with st.sidebar:
    st.markdown("### 📊 Assistant métier")
    st.caption("Démonstrateur académique de Machine Learning supervisé")
    st.divider()
    st.markdown("**Objectif**")
    st.write("Classer un dossier en création possible, complément documentaire ou analyse manuelle.")
    st.info("La décision finale reste sous la responsabilité du Data Officer.")
    st.divider()
    st.button("Réinitialiser le formulaire", use_container_width=True, on_click=reset_form)

st.markdown(
    """<div class="hero"><h1>Assistant Data Officer</h1>
    <p>Aide à la décision pour le traitement des dossiers de création et de modification de fonds d’investissement.</p></div>""",
    unsafe_allow_html=True,
)

home_tab, analysis_tab, history_tab, method_tab = st.tabs([
    "🏠 Tableau de bord", "🔎 Analyse d’un dossier", "🕘 Historique", "ℹ️ Méthodologie"
])

with home_tab:
    history = st.session_state.history
    total = len(history)
    counts = pd.Series([h["Décision"] for h in history]).value_counts() if history else pd.Series(dtype=int)
    possible = int(counts.get("Création possible", 0))
    manual = int(counts.get("Analyse manuelle", 0))
    complement = int(counts.get("Complément documentaire requis avant création", 0))

    c1, c2, c3, c4 = st.columns(4)
    cards = [
        (c1, "Dossiers analysés", total, "Session actuelle"),
        (c2, "Créations possibles", possible, "Dossiers complets"),
        (c3, "Analyses manuelles", manual, "Contrôle humain requis"),
        (c4, "Compléments", complement, "Informations manquantes"),
    ]
    for col, label, value, sub in cards:
        with col:
            st.markdown(f'<div class="kpi"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div><div class="kpi-sub">{sub}</div></div>', unsafe_allow_html=True)

    st.markdown("### Vue d’ensemble")
    left, right = st.columns([1.3, 1])
    with left:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("#### Parcours du dossier")
        st.markdown('<div class="step-row"><span class="step done">1. Réception</span><span>→</span><span class="step done">2. Qualification</span><span>→</span><span class="step current">3. Analyse ML</span><span>→</span><span class="step">4. Validation humaine</span></div>', unsafe_allow_html=True)
        st.write("L’application structure les informations du dossier, prédit un statut et propose les actions à réaliser.")
        st.markdown('</div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("#### Indicateurs du démonstrateur")
        st.metric("Classes prédites", "3")
        st.metric("Scénarios métier", "300")
        st.metric("Validation finale", "Humaine")
        st.markdown('</div>', unsafe_allow_html=True)

    if history:
        chart = pd.DataFrame({"Statut": counts.index, "Nombre": counts.values}).set_index("Statut")
        st.bar_chart(chart)
    else:
        st.info("Aucune analyse réalisée dans cette session. Ouvrez l’onglet « Analyse d’un dossier » pour commencer.")

with analysis_tab:
    st.markdown("### Import optionnel d’un prospectus")
    uploaded_pdf = st.file_uploader("Déposer un document PDF", type=["pdf"], help="Le document sert uniquement à proposer un préremplissage. Les champs doivent être validés par l’utilisateur.")
    suggestions: dict[str, str] = {}
    if uploaded_pdf is not None:
        text = extract_pdf_text(uploaded_pdf)
        suggestions = suggest_from_text(text)
        st.success(f"Document chargé : {uploaded_pdf.name}")
        if suggestions:
            st.caption("Préremplissage indicatif détecté : " + ", ".join(f"{k} = {v}" for k, v in suggestions.items()))
        else:
            st.caption("Aucune information fiable n’a été extraite automatiquement. Complétez le formulaire manuellement.")

    form_left, form_right = st.columns(2)
    with form_left:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### 1. Informations générales")
        type_options = ["Prospectus", "Règlement / By-Laws", "By-Laws", "Limited Partnership Deed", "Limited Partnership Agreement", "Legal document"]
        type_default = type_options.index(suggestions.get("type_document", "Prospectus")) if suggestions.get("type_document") in type_options else 0
        type_document = st.selectbox("Type de document", type_options, index=type_default)
        form_options = ["FCP", "FPCI", "FPS", "OPCVM-FCP", "SLP", "LP", "SCSp-SICAV-RAIF", "Scottish LP"]
        form_default = form_options.index(suggestions.get("forme", "FCP")) if suggestions.get("forme") in form_options else 0
        forme_juridique = st.selectbox("Forme juridique", form_options, index=form_default)
        country_options = ["France", "Luxembourg", "Royaume-Uni", "États-Unis"]
        country_default = country_options.index(suggestions.get("pays", "France")) if suggestions.get("pays") in country_options else 0
        pays = st.selectbox("Pays de domiciliation", country_options, index=country_default)
        currency_options = ["EUR", "USD", "GBP"]
        currency_default = currency_options.index(suggestions.get("devise", "EUR")) if suggestions.get("devise") in currency_options else 0
        devise = st.selectbox("Devise principale", currency_options, index=currency_default)
        langue = st.selectbox("Langue du document", ["Français", "Anglais"])
        st.markdown("### 2. Structure")
        gp_requis_bool = st.toggle("General Partner requis", key="gp_requis")
        gp_identifie_bool = True
        if gp_requis_bool:
            gp_identifie_bool = st.toggle("General Partner identifié", key="gp_identifie")
        isin_statut = st.selectbox("Statut de l’ISIN", ["Présent", "Provisoire", "Ambigu"])
        st.markdown('</div>', unsafe_allow_html=True)

    with form_right:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### 3. Informations obligatoires")
        document_present_bool = st.toggle("Document constitutif présent", key="document_present")
        forme_presente_bool = st.toggle("Forme juridique présente", key="forme_juridique_presente")
        pays_present_bool = st.toggle("Pays présent", key="pays_present")
        adresse_presente_bool = st.toggle("Adresse présente", key="adresse_presente")
        devise_presente_bool = st.toggle("Devise présente", key="devise_presente")
        completeness = sum([document_present_bool, forme_presente_bool, pays_present_bool, adresse_presente_bool, devise_presente_bool]) / 5
        st.progress(completeness, text=f"Complétude des informations obligatoires : {completeness:.0%}")
        st.markdown("### 4. Contrôles complémentaires")
        sfdr_statut = st.selectbox("Statut SFDR", ["Article 6", "Article 8", "Article 9", "Non applicable", "Absent alors qu’attendu"])
        conformite_adresse = st.selectbox("Conformité de l’adresse", ["Conforme", "À vérifier"])
        incoherence_bool = st.toggle("Incohérence documentaire détectée", key="incoherence_detectee")
        version_document = st.selectbox("Version du document", ["Finale", "Provisoire"])
        st.markdown('</div>', unsafe_allow_html=True)

    analyze = st.button("Analyser le dossier", type="primary", use_container_width=True)

    if analyze:
        input_data = {
            "type_document": type_document,
            "forme_juridique": forme_juridique,
            "pays_domiciliation": pays,
            "devise_principale": devise,
            "langue_document": langue,
            "gp_requis": "Oui" if gp_requis_bool else "Non",
            "document_present": "Oui" if document_present_bool else "Non",
            "forme_juridique_presente": "Oui" if forme_presente_bool else "Non",
            "pays_present": "Oui" if pays_present_bool else "Non",
            "adresse_presente": "Oui" if adresse_presente_bool else "Non",
            "devise_presente": "Oui" if devise_presente_bool else "Non",
            "gp_identifie": "Oui" if (gp_requis_bool and gp_identifie_bool) else "Non applicable",
            "isin_statut": isin_statut,
            "sfdr_statut": sfdr_statut,
            "conformite_adresse": conformite_adresse,
            "incoherence_detectee": "Oui" if incoherence_bool else "Non",
            "version_document": version_document,
        }
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]
        classes = model.named_steps["model"].classes_
        confidence = float(max(probabilities))
        blocking, manual_points, validated = explain_input(input_data)

        if prediction == "Création possible":
            css_class, icon, risk, delay, action = "result-success", "🟢", "Faible", "≈ 3 minutes", "Poursuivre la création et le référencement"
            explanation = "Les informations obligatoires sont présentes et aucun point de contrôle majeur n’a été détecté."
        elif prediction == "Analyse manuelle":
            css_class, icon, risk, delay, action = "result-warning", "🟠", "Modéré", "≈ 10 à 15 minutes", "Faire valider les points de contrôle par un Data Officer"
            explanation = "Le dossier est suffisamment renseigné, mais un ou plusieurs contrôles nécessitent une validation humaine."
        else:
            css_class, icon, risk, delay, action = "result-danger", "🔴", "Élevé", "En attente de documentation", "Demander les informations manquantes avant création"
            explanation = "Le dossier ne peut pas être créé immédiatement car au moins une information obligatoire est absente."

        st.markdown(f'<div class="result {css_class}"><div class="result-title">{icon} {prediction.upper()}</div><div>{explanation}</div></div>', unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Confiance", f"{confidence:.1%}")
        m2.metric("Complétude", f"{completeness:.0%}")
        m3.metric("Niveau de risque", risk)
        m4.metric("Temps indicatif", delay)

        st.markdown("#### Action recommandée")
        st.info(action)

        detail1, detail2 = st.columns(2)
        with detail1:
            if validated:
                st.markdown("#### Éléments validés")
                for item in validated:
                    st.write(f"✅ {item}")
            if blocking:
                st.markdown("#### Informations à demander")
                for item in blocking:
                    st.write(f"❌ {item}")
        with detail2:
            if manual_points:
                st.markdown("#### Points à contrôler")
                for item in manual_points:
                    st.write(f"⚠️ {item}")
            probability_df = pd.DataFrame({"Statut": classes, "Probabilité": probabilities}).set_index("Statut")
            st.markdown("#### Probabilités du modèle")
            st.bar_chart(probability_df)

        if blocking:
            email = build_email(blocking)
            st.markdown("#### Brouillon d’email")
            st.text_area("Message généré", email, height=250)
            st.download_button("Télécharger le brouillon", email, file_name="demande_complement_documentaire.txt", mime="text/plain")

        record = {
            "Date": datetime.now().strftime("%d/%m/%Y %H:%M"),
            "Document": type_document,
            "Forme juridique": forme_juridique,
            "Pays": pays,
            "Décision": prediction,
            "Confiance": round(confidence, 4),
            "Complétude": round(completeness, 2),
        }
        if not st.session_state.history or st.session_state.history[-1] != record:
            st.session_state.history.append(record)

with history_tab:
    st.markdown("### Historique de la session")
    if st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df, use_container_width=True, hide_index=True)
        st.download_button("Exporter l’historique en CSV", history_df.to_csv(index=False).encode("utf-8-sig"), file_name="historique_analyses.csv", mime="text/csv")
    else:
        st.info("Aucune analyse n’a encore été réalisée pendant cette session.")

with method_tab:
    st.markdown("### Méthodologie du démonstrateur")
    st.markdown(
        """
1. **Formalisation des règles métier** à partir des informations nécessaires au traitement d’un dossier.
2. **Construction de 300 scénarios métier** dérivés de prototypes documentaires anonymisés.
3. **Apprentissage supervisé** avec comparaison de plusieurs algorithmes.
4. **Sélection et optimisation** d’un modèle Random Forest.
5. **Déploiement dans Streamlit** pour fournir une aide à la décision explicable.

#### Positionnement
Le modèle ne remplace pas le Data Officer. Il structure le contrôle, propose un statut et explicite les éléments manquants ou à vérifier.

#### Limites
Les 300 observations sont des scénarios métier construits, et non 300 dossiers historiques indépendants. Une utilisation opérationnelle nécessiterait une validation sur des données réelles nouvelles et un suivi continu des performances.

#### Lecture PDF
Le module PDF propose uniquement un préremplissage indicatif par recherche de mots-clés. Il ne constitue pas une extraction documentaire industrielle et chaque valeur doit être validée par l’utilisateur.
"""
    )
