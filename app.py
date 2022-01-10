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
h2,h3,h5 {
    font-family: 'Roboto', sans-serif;
    font-weight: 700;
    text-shadow: 1px 1px 2px;
    text-align: center;
}
h3 {
    text-align: center;
}
h5 {
    font-weight: 300;
    font-size: 16px;
    font-family: 'Open Sans', sans-serif;
}
#win {
    border-radius: 50%;
    box-shadow: 0 1px 8px rgba(25, 233, 11, 0.93);
    border: rgba(25, 233, 11, 0.93) 8px  solid;
}
#loss {
    border-radius: 50%;
    box-shadow: 0 1px 4px rgba(233, 39, 11, 0.93);
    border: rgba(233, 39, 11, 0.93) 4px  solid;
}
#nul {
    text-align:center;
    font-weight: 600;
    font-family: 'Open Sans', sans-serif;
    font-size: 2.25rem;
    opacity: .8;
    text-shadow: 1px 1px 3px rgba(0,0,0,0.2);
    background-color: rgba(134, 134, 134, 0.13);
    border-radius: 85px 85px 85px 85px / 80px 80px 80px 80px;
    box-shadow: 0 0 8px rgba(0,0,0,0.2);
    
}
#nul_2 {
    text-align:center;
    font-weight: 800;
    font-family: 'Open Sans', sans-serif;
    font-size: 2.25rem;
    text-shadow: 1px 1px 3px rgba(0,0,0,0.2);
    background-color: rgba(134, 134, 134, 0.13);
    border-radius: 85px 85px 85px 85px / 80px 80px 80px 80px;
    box-shadow: 0 0 8px rgba(0,0,0,0.2);
    border: rgba(25, 233, 11, 0.93) 6px  solid;
}
}
"""
styles = """@import url('https://fonts.googleapis.com/css2?family=Open+Sans:wght@300;400;600;800&family=Roboto:wght@300;400;500;700&display=swap');"""
st.write(f'<style>{styles}</style>', unsafe_allow_html=True)
st.write(f'<style>{CSS}</style>', unsafe_allow_html=True)
st.markdown(f"""## {f"{match['home_club']} - {match['away_club']}"} """)
st.markdown(f"### {date.strftime('%d %B %Y - %H:%M')}")
st.markdown(f"##### {match['lieu']}")

col_home, col_en_plus, col_away = st.columns(3)

col_home.markdown(f"**{match['home_club']}**")
col_en_plus.markdown(" ")
col_away.markdown(f"**{match['away_club']}**")

prediction = st.selectbox('résultat', [3,1,0])
st.markdown(f"""## Prédiction : """)

# recuperation du logo home et away :
with open('raw_data/logo_equipes.txt') as f:
    logos = json.load(f)


logo_home = logos[match['home_club']]["img120"]
logo_away = logos[match['away_club']]["img120"]


col_home_res, col_en_plus_res, col_away_res = st.columns(3)

if prediction == 3 :
    # entourer le logo de la colonne de gauche
    col_home_res.write(f"<img src=\"{logo_home}\" id=\"win\">", unsafe_allow_html=True)
    col_en_plus_res.write(f"<p id=\"nul\">Match <br>Nul", unsafe_allow_html=True)
    col_away_res.write(f"<img src=\"{logo_away}\" id=\"loss\" >", unsafe_allow_html=True)
elif prediction == 0 :
    # entourer le logo de la colonne de droite
    col_home_res.write(f"<img src=\"{logo_home}\" id=\"loss\">", unsafe_allow_html=True)
    col_en_plus_res.write(f"<p id=\"nul\">Match <br>Nul", unsafe_allow_html=True)
    col_away_res.write(f"<img src=\"{logo_away}\" id=\"win\" >", unsafe_allow_html=True)
elif prediction == 1 :
    # entourer le logo de la colonne de droite
    col_home_res.write(f"<img src=\"{logo_home}\" id=\"loss\">", unsafe_allow_html=True)
    col_en_plus_res.write(f"<p id=\"nul_2\">Match <br>Nul", unsafe_allow_html=True)
    col_away_res.write(f"<img src=\"{logo_away}\" id=\"loss\" >", unsafe_allow_html=True)
