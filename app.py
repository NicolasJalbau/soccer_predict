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

col_home, col_away = st.columns(2)

col_home.markdown(f"**{match['home_club']}**")
col_away.markdown(f"**{match['away_club']}**")