from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import streamlit as st

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
}

.block-container {
    max-width: 1220px;
    padding-top: 1.4rem;
    padding-bottom: 3rem;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #f7f8fb 0%, #eef1f6 100%);
    border-right: 1px solid var(--line);
}

.hero {
    padding: 1.55rem 1.65rem;
    border: 1px solid var(--line);
    border-radius: 18px;
    background: linear-gradient(135deg, #ffffff 0%, #f7f8fb 100%);
    margin-bottom: 1.15rem;
    box-shadow: 0 8px 24px rgba(16, 24, 40, 0.06);
}

.hero h1 {
    color: var(--ink);
    margin: 0;
    font-size: 2.35rem;
    line-height: 1.08;
}

.hero p {
    color: var(--muted);
    margin: 0.55rem 0 0 0;
    font-size: 1rem;
}

.section-card {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 16px;
    padding: 1.05rem 1.1rem 0.25rem 1.1rem;
    margin-bottom: 0.85rem;
    box-shadow: 0 4px 14px rgba(16, 24, 40, 0.035);
}

.result-card {
    border-radius: 18px;
    padding: 1.35rem 1.45rem;
    border: 1px solid var(--line);
    box-shadow: 0 8px 24px rgba(16, 24, 40, 0.06);
    margin: 0.8rem 0 1rem 0;
}

.result-green { background: #ecfdf3; border-color: #abefc6; }
.result-orange { background: #fffaeb; border-color: #fedf89; }
.result-red { background: #fef3f2; border-color: #fecdca; }

.result-title {
    font-size: 1.45rem;
    font-weight: 800;
    color: var(--ink);
    margin-bottom: 0.35rem;
}

.result-text {
    color: #344054;
    margin: 0;
    line-height: 1.55;
}

.pill {
    display: inline-block;
    border-radius: 999px;
    padding: 0.3rem 0.65rem;
    background: #f2f4f7;
    color: #344054;
    font-size: 0.82rem;
    font-weight: 650;
    margin: 0.12rem 0.22rem 0.12rem 0;
}

.small-note {
    color: var(--muted);
    font-size: 0.9rem;
}

[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 0.85rem;
}

.stButton > button {
    border-radius: 12px;
    min-height: 3rem;
    font-weight: 750;
}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model() -> Any:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Le fichier modele_decision_data_officer.joblib est introuvable. "
            "Place-le dans le même dossier que l'application."
        )
    return joblib.load(MODEL_PATH)


def yes_no(value: bool) -> str:
    return "Oui" if value else "Non"


def build_email(missing_items: list[str]) -> str:
    bullets = "\n".join(f"- {item}" for item in missing_items)
    return f"""Objet : Éléments complémentaires requis pour le traitement du dossier

Bonjour,

Après analyse du dossier transmis, certaines informations indispensables à la création du fonds sont absentes.

Afin de poursuivre le traitement, pourriez-vous nous transmettre les éléments suivants :

{bullets}

Nous vous remercions par avance pour votre retour.

Bien cordialement,"""


def build_explanations(data: dict[str, str]) -> tuple[list[str], list[str], list[str]]:
    missing: list[str] = []
    manual: list[str] = []
    positives: list[str] = []

    mandatory_map = {
        "document_present": ("le prospectus ou document constitutif à jour", "Document présent"),
        "forme_juridique_presente": ("la forme juridique du fonds", "Forme juridique présente"),
        "pays_present": ("le pays de domiciliation", "Pays présent"),
        "adresse_presente": ("l’adresse du fonds ou de l’entité", "Adresse présente"),
        "devise_presente": ("la devise principale", "Devise présente"),
    }

    for key, (missing_label, positive_label) in mandatory_map.items():
        if data[key] == "Non":
            missing.append(missing_label)
        else:
            positives.append(positive_label)

    if data["gp_requis"] == "Oui":
        if data["gp_identifie"] == "Non":
            missing.append("l’identification du General Partner")
        else:
            positives.append("General Partner identifié")

    if data["sfdr_statut"] == "Absent alors qu’attendu":
        manual.append("La classification SFDR doit être complétée ou confirmée.")
    else:
        positives.append("Statut SFDR renseigné")

    if data["isin_statut"] in {"Provisoire", "Ambigu"}:
        manual.append("Le statut de l’ISIN nécessite une validation complémentaire.")
    else:
        positives.append("ISIN présent")

    if data["incoherence_detectee"] == "Oui":
        manual.append("Une incohérence documentaire a été signalée.")

    if data["conformite_adresse"] == "À vérifier":
        manual.append("La conformité de l’adresse doit être vérifiée.")

    if data["version_document"] == "Provisoire":
        manual.append("Le document transmis est une version provisoire.")
    else:
        positives.append("Version finale du document")

    return missing, manual, positives


def get_business_indicators(prediction: str, missing: list[str], manual: list[str]) -> tuple[str, str, str]:
    if prediction == "Création possible":
        return "Faible", "≈ 3 à 5 min", "Poursuivre la création et le référencement"
    if prediction == "Analyse manuelle":
        return "Modéré", "≈ 10 à 20 min", "Contrôler les points signalés avant validation"
    return "Élevé", "En attente du retour", f"Demander {len(missing)} élément(s) complémentaire(s)"


def append_history(record: dict[str, Any]) -> None:
    st.session_state.setdefault("history", [])
    st.session_state.history.insert(0, record)
    st.session_state.history = st.session_state.history[:20]


try:
    model = load_model()
except Exception as exc:
    st.error(f"Impossible de charger le modèle : {exc}")
    st.stop()

with st.sidebar:
    st.markdown("## 📊 Assistant métier")
    st.caption("Démonstrateur académique de Machine Learning supervisé")
    st.divider()
    st.markdown("**Objectif**")
    st.write(
        "Classer un dossier en création possible, complément documentaire ou analyse manuelle."
    )
    st.info("La décision finale reste sous la responsabilité du Data Officer.")
    st.divider()
    if st.button("Réinitialiser le formulaire", use_container_width=True):
        st.session_state.clear()
        st.rerun()

st.markdown(
    """
<div class="hero">
    <h1>Assistant Data Officer</h1>
    <p>Aide à la décision pour le traitement des dossiers de création et de modification de fonds d’investissement.</p>
</div>
""",
    unsafe_allow_html=True,
)

page_analyse, page_history, page_method = st.tabs(
    ["🔎 Analyse d’un dossier", "🕘 Historique", "ℹ️ Méthodologie"]
)

with page_analyse:
    with st.form("fund_form"):
        left, right = st.columns(2, gap="large")

        with left:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.subheader("1. Informations générales")
            type_document = st.selectbox(
                "Type de document",
                [
                    "Prospectus",
                    "Règlement / By-Laws",
                    "By-Laws",
                    "Limited Partnership Deed",
                    "Limited Partnership Agreement",
                    "Legal document",
                ],
            )
            forme_juridique = st.selectbox(
                "Forme juridique",
                ["FCP", "FPCI", "FPS", "OPCVM-FCP", "SLP", "LP", "SCSp-SICAV-RAIF", "Scottish LP"],
            )
            pays = st.selectbox(
                "Pays de domiciliation",
                ["France", "Luxembourg", "Royaume-Uni", "États-Unis"],
            )
            devise = st.selectbox("Devise principale", ["EUR", "USD", "GBP"])
            langue = st.selectbox("Langue du document", ["Français", "Anglais"])
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.subheader("2. Structure du fonds")
            gp_requis_bool = st.toggle("General Partner requis", value=False)
            gp_identifie_bool = st.toggle(
                "General Partner identifié",
                value=True,
                disabled=not gp_requis_bool,
            )
            isin_statut = st.selectbox("Statut de l’ISIN", ["Présent", "Provisoire", "Ambigu"])
            st.markdown("</div>", unsafe_allow_html=True)

        with right:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.subheader("3. Informations obligatoires")
            document_present_bool = st.toggle("Document constitutif présent", value=True)
            forme_presente_bool = st.toggle("Forme juridique présente", value=True)
            pays_present_bool = st.toggle("Pays présent", value=True)
            adresse_presente_bool = st.toggle("Adresse présente", value=True)
            devise_presente_bool = st.toggle("Devise présente", value=True)
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.subheader("4. Contrôles complémentaires")
            sfdr_statut = st.selectbox(
                "Statut SFDR",
                ["Article 6", "Article 8", "Article 9", "Non applicable", "Absent alors qu’attendu"],
            )
            conformite_adresse = st.selectbox("Conformité de l’adresse", ["Conforme", "À vérifier"])
            incoherence_bool = st.toggle("Incohérence documentaire détectée", value=False)
            version_document = st.selectbox("Version du document", ["Finale", "Provisoire"])
            st.markdown("</div>", unsafe_allow_html=True)

        submitted = st.form_submit_button(
            "Analyser le dossier",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        gp_requis = yes_no(gp_requis_bool)
        gp_identifie = yes_no(gp_identifie_bool) if gp_requis_bool else "Non applicable"

        input_data = {
            "type_document": type_document,
            "forme_juridique": forme_juridique,
            "pays_domiciliation": pays,
            "devise_principale": devise,
            "langue_document": langue,
            "gp_requis": gp_requis,
            "document_present": yes_no(document_present_bool),
            "forme_juridique_presente": yes_no(forme_presente_bool),
            "pays_present": yes_no(pays_present_bool),
            "adresse_presente": yes_no(adresse_presente_bool),
            "devise_presente": yes_no(devise_presente_bool),
            "gp_identifie": gp_identifie,
            "isin_statut": isin_statut,
            "sfdr_statut": sfdr_statut,
            "conformite_adresse": conformite_adresse,
            "incoherence_detectee": yes_no(incoherence_bool),
            "version_document": version_document,
        }

        input_df = pd.DataFrame([input_data])

        try:
            prediction = model.predict(input_df)[0]
            probabilities = model.predict_proba(input_df)[0]
            classes = model.named_steps["model"].classes_
        except Exception as exc:
            st.error(f"Erreur pendant la prédiction : {exc}")
            st.stop()

        confidence = float(max(probabilities))
        missing, manual, positives = build_explanations(input_data)
        risk, processing_time, recommendation = get_business_indicators(prediction, missing, manual)

        record = {
            "Date": datetime.now().strftime("%d/%m/%Y %H:%M"),
            "Document": type_document,
            "Forme juridique": forme_juridique,
            "Pays": pays,
            "Décision": prediction,
            "Confiance": confidence,
        }
        append_history(record)

        st.divider()
        st.markdown("## Résultat de l’analyse")

        if prediction == "Création possible":
            css_class = "result-green"
            icon = "🟢"
            explanation = (
                "Les informations obligatoires sont présentes et aucun contrôle complémentaire "
                "bloquant n’a été détecté."
            )
        elif prediction == "Analyse manuelle":
            css_class = "result-orange"
            icon = "🟠"
            explanation = (
                "Le dossier contient les informations minimales, mais certains contrôles nécessitent "
                "une validation humaine avant traitement."
            )
        else:
            css_class = "result-red"
            icon = "🔴"
            explanation = (
                "Le traitement ne peut pas être finalisé immédiatement car une ou plusieurs "
                "informations obligatoires sont absentes."
            )

        st.markdown(
            f"""
<div class="result-card {css_class}">
    <div class="result-title">{icon} {prediction}</div>
    <p class="result-text">{explanation}</p>
</div>
""",
            unsafe_allow_html=True,
        )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Confiance du modèle", f"{confidence:.1%}")
        m2.metric("Niveau de risque", risk)
        m3.metric("Temps indicatif", processing_time)
        m4.metric("Contrôles signalés", len(missing) + len(manual))

        detail_left, detail_right = st.columns([1.1, 0.9], gap="large")

        with detail_left:
            st.markdown("### Explication métier")
            if positives:
                st.markdown("**Éléments validés**")
                st.markdown(" ".join(f'<span class="pill">✓ {item}</span>' for item in positives), unsafe_allow_html=True)

            if missing:
                st.markdown("#### Éléments à demander")
                for item in missing:
                    st.write(f"- {item}")

            if manual:
                st.markdown("#### Points nécessitant une analyse manuelle")
                for item in manual:
                    st.write(f"- {item}")

            st.markdown("#### Action recommandée")
            st.info(recommendation)

        with detail_right:
            st.markdown("### Probabilités par statut")
            probability_df = pd.DataFrame(
                {"Statut": classes, "Probabilité": probabilities}
            ).sort_values("Probabilité", ascending=False)
            chart_df = probability_df.set_index("Statut")
            st.bar_chart(chart_df, y="Probabilité", horizontal=True)
            st.dataframe(
                probability_df.style.format({"Probabilité": "{:.1%}"}),
                use_container_width=True,
                hide_index=True,
            )

        if missing:
            st.markdown("### Brouillon d’email")
            email_text = build_email(missing)
            st.text_area("Email généré automatiquement", email_text, height=270)
            st.download_button(
                "Télécharger le brouillon (.txt)",
                data=email_text.encode("utf-8"),
                file_name="brouillon_demande_complementaire.txt",
                mime="text/plain",
                use_container_width=True,
            )

        result_export = pd.DataFrame(
            [{**record, **input_data, "Risque": risk, "Action recommandée": recommendation}]
        ).to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "Exporter le résultat de l’analyse (.csv)",
            data=result_export,
            file_name="resultat_analyse_data_officer.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.caption(
            "Démonstrateur académique : la prédiction est une aide à la décision et doit être confirmée par un Data Officer."
        )

with page_history:
    st.subheader("Historique de la session")
    history = st.session_state.get("history", [])
    if not history:
        st.info("Aucune analyse n’a encore été réalisée pendant cette session.")
    else:
        history_df = pd.DataFrame(history)
        st.dataframe(
            history_df.style.format({"Confiance": "{:.1%}"}),
            use_container_width=True,
            hide_index=True,
        )
        st.download_button(
            "Télécharger l’historique (.csv)",
            data=history_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="historique_analyses.csv",
            mime="text/csv",
        )

with page_method:
    st.subheader("Méthodologie de la preuve de concept")
    st.markdown(
        """
Cette application s’appuie sur un modèle de **Machine Learning supervisé** entraîné à partir de
**300 scénarios métier structurés**, dérivés de prototypes documentaires anonymisés.

Les trois classes prédites sont :

- **Création possible** ;
- **Complément documentaire requis avant création** ;
- **Analyse manuelle**.

Les informations comme la forme juridique, le pays, l’adresse et la devise sont considérées comme
bloquantes lorsqu’elles sont absentes. D’autres éléments, tels que le statut SFDR, un ISIN ambigu ou
une incohérence documentaire, conduisent plutôt à une analyse manuelle.

### Limites

- Les 300 lignes représentent des scénarios métier et non 300 prospectus historiques indépendants.
- Les performances du modèle doivent être confirmées sur des dossiers réels nouveaux.
- L’application ne remplace pas l’expertise et la validation du Data Officer.
"""
    )
