import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
from dateutil import parser
import json

with open('raw_data/journees.txt') as f:
    data = json.load(f)

# Concatene le chiffre de la journee avec "J-"
liste_journees = [f'J-{j}' for j in list(data.keys())]

# Récupère uniquement le chiffre de la journee selectionne par l'utilisateur
date = st.sidebar.selectbox('Sélectionnez la journée de votre choix', liste_journees)[2:]

liste_match = ([(match[0],f"{match[1]['home_club']} - {match[1]['away_club']}") for match in (data[date].items())])

# choix du match
match_id = st.sidebar.selectbox('Sélectionnez le match de votre choix', liste_match, format_func=lambda x: x[1])[0]

match = data[date][match_id]
date = parser.parse(match['date'])

CSS = """
h2,h3 {

    text-shadow: 1px 1px 2px;
}
h3 {
    text-align: center;
}

}
"""
st.write(f'<style>{CSS}</style>', unsafe_allow_html=True)
st.markdown(f"""## {f"{match['home_club']} - {match['away_club']}"} """)
st.markdown(f"### {date.strftime('%d %B %Y - %H:%M')}")

col_home, col_en_plus, col_away = st.columns(3)

col_home.markdown(f"**{match['home_club']}**")
col_en_plus.markdown(" ")
col_away.markdown(f"**{match['away_club']}**")

st.markdown(f"""## Prédiction : """)

# recuperation du logo home et away :
with open('raw_data/logo_equipes.txt') as f:
    data = json.load(f)

home_club=match['home_club']
away_club=match['away_club']
logo_home = data[home_club]["img120"]
logo_away = data[away_club]["img120"]

prediction='home' # resultat codé en dur

col_home.write(f"<img src=" + logo_home + ">", unsafe_allow_html=True)
if prediction == 'home' or prediction == 'draw' :
    # entourer le logo de la colonne de gauche
    pass


col_away.write(f"<img src=" + logo_away + ">", unsafe_allow_html=True)
if prediction == 'away' or prediction == 'draw' :
    # entourer le logo de la colonne de droite
    pass
