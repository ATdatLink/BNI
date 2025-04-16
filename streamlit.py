import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import datetime
import re
from collections import Counter
from streamlit_option_menu import option_menu
import base64
from io import BytesIO
import spacy
import sys
import subprocess
from nltk.corpus import stopwords
import string
import re
import nltk

spacy.cli.download('fr_core_news_sm')
nltk.download('stopwords', quiet=True)

# Configuration de la page
st.set_page_config(
    page_title="Dashboard Narratifs Industriels | Nodelio",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Palette de couleurs Nodelio
NODELIO_COLORS = {
    "primary": "#3F5CC8",  # Bleu principal
    "secondary": "#D8DFF2",  # Bleu clair
    "accent": "#F2D8E1",  # Rose pâle
    "neutral": "#F5F7FA",  # Gris très clair
    "text": "#1A2342",  # Bleu foncé
    "positive": "#4AB58E",  # Vert
    "negative": "#E15554",  # Rouge
    "neutral_tone": "#F7CA57"  # Jaune
}

# CSS personnalisé
custom_css = f"""
<style>
    .main {{
        background-color: {NODELIO_COLORS["neutral"]};
        color: {NODELIO_COLORS["text"]};
    }}
    .stButton>button {{
        background-color: {NODELIO_COLORS["primary"]};
        color: white;
    }}
    .section-title {{
        color: {NODELIO_COLORS["primary"]};
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 20px;
    }}
    .card {{
        background-color: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
        transition: transform 0.3s ease;
    }}
    .card:hover {{
        transform: translateY(-5px);
    }}
    .card-metric {{
        font-size: 2.5rem;
        font-weight: bold;
        color: {NODELIO_COLORS["primary"]};
    }}
    .card-title {{
        font-size: 1rem;
        color: {NODELIO_COLORS["text"]};
        opacity: 0.8;
    }}
    .gradient-blue {{
        background: linear-gradient(120deg, {NODELIO_COLORS["primary"]}, {NODELIO_COLORS["secondary"]});
        padding: 2px;
        border-radius: 10px;
    }}
    .white-content {{
        background: white;
        border-radius: 8px;
        padding: 20px;
    }}
    .filter-container {{
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }}
    .nodelio-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 20px;
    }}
    .nodelio-logo {{
        background-color: white; 
        padding: 10px 20px; 
        border-radius: 5px; 
        display: inline-block;
    }}
    .footer {{
        text-align: center;
        color: {NODELIO_COLORS["text"]};
        font-size: 12px;
        margin-top: 30px;
        opacity: 0.8;
    }}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# Logo Nodelio
def add_logo():
    # Créer un logo factice avec du texte (à remplacer par une image réelle)
    st.markdown(
        f"""
        <div style="background-color: white; padding: 10px 20px; border-radius: 5px; display: inline-block;">
            <h2 style="color: {NODELIO_COLORS['primary']}; margin: 0;">NODELIO</h2>
            <p style="color: {NODELIO_COLORS['text']}; margin: 0; font-size: 12px;">SOCIAL INTELLIGENCE</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# Fonction pour créer des cartes métriques
def metric_card(title, value, delta=None, suffix="", prefix=""):
    card_html = f"""
    <div class="card">
        <div class="card-title">{title}</div>
        <div class="card-metric">{prefix}{value}{suffix}</div>
    """
    if delta is not None:
        delta_color = NODELIO_COLORS["positive"] if delta > 0 else NODELIO_COLORS["negative"]
        delta_sign = "+" if delta > 0 else ""
        card_html += f"""
        <div style="color: {delta_color}; font-size: 0.9rem;">
            {delta_sign}{delta}% par rapport à la période précédente
        </div>
        """
    card_html += "</div>"
    return card_html

# Fonction pour créer un en-tête de section avec gradient
def section_header(title, gradient_class="gradient-blue"):
    return f"""
    <div class="{gradient_class}">
        <div class="white-content">
            <h3 style="margin: 0; color: {NODELIO_COLORS['primary']};">{title}</h3>
        </div>
    </div>
    """

def preprocess_french_text(text_list):
    """
    Nettoie et prétraite une liste de textes français pour la génération de nuages de mots
    """
    import spacy
    import sys
    import subprocess
    from nltk.corpus import stopwords
    import string
    import re
    import nltk
    
    # Chargement du modèle français de spaCy avec installation automatique si nécessaire
    try:
        nlp = spacy.load('fr_core_news_sm')
    except OSError:
        # Installation automatique du modèle français de spaCy
        import streamlit as st
        with st.spinner("Installation du modèle français pour spaCy en cours... Cela peut prendre quelques instants."):
            # Utiliser spacy.cli.download au lieu de subprocess pour une meilleure compatibilité
            try:
                spacy.cli.download('fr_core_news_sm')
                st.success("Modèle français pour spaCy installé avec succès!")
                nlp = spacy.load('fr_core_news_sm')
            except Exception as e:
                st.error(f"Erreur lors de l'installation du modèle spaCy: {e}")
                # Fallback pour utiliser subprocess en cas d'échec avec spacy.cli
                try:
                    subprocess.check_call([sys.executable, "-m", "spacy", "download", "fr_core_news_sm"])
                    st.success("Modèle français pour spaCy installé avec succès!")
                    nlp = spacy.load('fr_core_news_sm')
                except:
                    st.error("L'installation automatique a échoué. Veuillez exécuter manuellement: python -m spacy download fr_core_news_sm")
                    return text_list
    
    # Récupération des stopwords français de NLTK avec installation automatique si nécessaire
    try:
        french_stopwords = set(stopwords.words('french'))
    except LookupError:
        # Télécharger les stopwords automatiquement
        with st.spinner("Téléchargement des stopwords français pour NLTK en cours..."):
            try:
                nltk.download('stopwords', quiet=True)
                st.success("Stopwords français pour NLTK installés avec succès!")
                french_stopwords = set(stopwords.words('french'))
            except Exception as e:
                st.error(f"Erreur lors du téléchargement des stopwords: {e}")
                # Fallback pour utiliser subprocess en cas d'échec avec nltk.download
                try:
                    subprocess.check_call([sys.executable, "-m", "nltk.downloader", "stopwords"])
                    french_stopwords = set(stopwords.words('french'))
                except:
                    st.error("L'installation automatique a échoué. Veuillez exécuter manuellement: python -m nltk.downloader stopwords")
                    french_stopwords = set()
    
    # Ajout de stopwords personnalisés spécifiques au domaine
    custom_stopwords = {'plus', 'très', 'après', 'avoir', 'être', 'faire', 'sans', 'avec', 'tout', 
                        'aussi', 'même', 'donc', 'alors', 'deux', 'trois', 'fois', 'comme', 'cette',
                        'ces', 'cet', 'celui', 'celle', 'ceux', 'celles', 'leur', 'leurs', 'dont',
                        'car', 'quand', 'comment', 'pourquoi', 'est', 'sont', 'sera', 'était'}
    french_stopwords.update(custom_stopwords)
    
    # Fonction de nettoyage du texte
    def clean_text(text):
        if not text or not isinstance(text, str):
            return ""
        
        # Conversion en minuscules
        text = text.lower()
        
        # Suppression des URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Suppression des adresses email
        text = re.sub(r'\S+@\S+', '', text)
        
        # Suppression de la ponctuation
        text = re.sub(f'[{string.punctuation}]', ' ', text)
        
        # Suppression des chiffres
        text = re.sub(r'\d+', '', text)
        
        # Suppression des espaces multiples
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    # Traitement de chaque texte dans la liste
    processed_texts = []
    
    for text in text_list:
        if not text:
            continue
            
        # Nettoyage basique
        clean = clean_text(text)
        
        # Traitement avec spaCy
        doc = nlp(clean)
        
        # Extraction des lemmes, filtrage des stopwords et mots courts
        lemmas = [token.lemma_ for token in doc 
                 if token.lemma_ not in french_stopwords 
                 and len(token.lemma_) > 2 
                 and not token.is_punct
                 and not token.is_space
                 and not token.is_digit]
        
        # Rejoindre les lemmes en une chaîne
        processed_text = ' '.join(lemmas)
        if processed_text:
            processed_texts.append(processed_text)
    
    return processed_texts


def generate_wordcloud(text_data, colormap='viridis', background_color='white', max_words=100):
    # Prétraitement des données textuelles
    processed_texts = preprocess_french_text(text_data)
    
    if not processed_texts:
        # Retourner un message si aucun texte ne reste après traitement
        return "<p style='text-align:center;'>Pas assez de données pour générer un nuage de mots</p>"
    
    # Joindre les textes traités
    text = ' '.join(processed_texts)
    
    # Créer le nuage de mots
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        max_words=max_words, 
        colormap=colormap, 
        background_color=background_color,
        collocations=False,
        min_font_size=10,
        max_font_size=None,
        relative_scaling=0.5
    ).generate(text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    
    # Convertir l'image en format base64 pour l'affichage
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    plt.close()
    
    return f'<img src="data:image/png;base64,{img_str}" style="width:50%; border-radius:5px;">'


# Fonction pour nettoyer les listes dans les colonnes
def clean_and_split(text):
    if pd.isna(text) or text == '':
        return []
    
    # Si c'est déjà une liste
    if isinstance(text, list):
        return text
    
    # tout lower
    text = text.lower()

    # Différents formats possibles
    # Format [item1, item2, item3]
    if text.startswith('[') and text.endswith(']'):
        items = re.findall(r'\'([^\']+)\'|\"([^\"]+)\"', text)
        if items:
            return [item[0] or item[1] for item in items]
    
    # Format "item1, item2, item3"
    return [item.strip() for item in text.split(',') if item.strip()]

# Fonction pour préparer les données
def prepare_data(df):
    # Convertir la colonne date au format datetime
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Filtrer les dates avant 01/04/2024
    df = df[df['date'] >= datetime.datetime(2024, 4, 1)]
    
    # Nettoyer les listes dans les colonnes
    list_columns = ['entreprises_mentionnees', 'personnalites_mentionnees', 
                    'organisations_institutions', 'enjeux_associes', 'topics', 'pays', "ville", 'region']
    
    for col in list_columns:
        if col in df.columns:
            df[col] = df[col].apply(clean_and_split)

    
    return df

# Fonction pour créer un graphique de tendance temporelle
def create_time_trend(df, freq='D', metric='count'):
    time_df = df.copy()
    
    # Grouper par date selon la fréquence choisie
    if metric == 'count':
        result = time_df.groupby(pd.Grouper(key='date', freq=freq)).size().reset_index(name='count')
        title = "Volume d'articles par jour"
        y_title = "Nombre d'articles"
    elif metric == 'sentiment':
        # Calculer la proportion de sentiment positif par jour
        grouped = time_df.groupby(pd.Grouper(key='date', freq=freq))
        result = pd.DataFrame({
            'date': grouped.groups.keys(),
            'positive': grouped.apply(lambda x: (x['sentiment'] == 'Positive').mean() if len(x) > 0 else 0),
            'negative': grouped.apply(lambda x: (x['sentiment'] == 'Négative').mean() if len(x) > 0 else 0),
            'neutral': grouped.apply(lambda x: (x['sentiment'] == 'Neutre').mean() if len(x) > 0 else 0),
        }).reset_index(drop=True)
        title = "Évolution du sentiment au fil du temps"
        y_title = "Proportion"
    
    # Créer le graphique
    if metric == 'count':
        fig = px.line(
            result, 
            x='date', 
            y='count',
            title=title,
            labels={'count': y_title, 'date': 'Date'},
            template='plotly_white'
        )
        # Personnaliser le graphique
        fig.update_traces(line_color=NODELIO_COLORS["primary"], line_width=3)
    else:
        fig = px.line(
            result,
            x='date',
            y=['positive', 'negative', 'neutral'],
            title=title,
            labels={'value': y_title, 'date': 'Date', 'variable': 'Sentiment'},
            template='plotly_white',
            color_discrete_map={
                'positive': NODELIO_COLORS["positive"],
                'negative': NODELIO_COLORS["negative"],
                'neutral': NODELIO_COLORS["neutral_tone"]
            }
        )
    
    fig.update_layout(
        font_family="Arial",
        title_font_size=20,
        title_font_color=NODELIO_COLORS["text"],
        legend_title_font_color=NODELIO_COLORS["text"],
        plot_bgcolor='white',
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial"
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(230, 230, 230, 0.8)'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(230, 230, 230, 0.8)'
        )
    )
    
    return fig

# Fonction pour créer un graphique de distribution
def create_distribution_chart(df, column, title, top_n=10, horizontal=True):
    # Si la colonne contient des listes, dépiler
    if df[column].apply(lambda x: isinstance(x, list)).any():
        # Crée une liste de tous les éléments dans toutes les listes
        all_items = [item for sublist in df[column].dropna() for item in sublist if item]
        # Compte les occurrences
        counter = Counter(all_items)
        # Converti en dataframe
        data = pd.DataFrame.from_dict(counter, orient='index').reset_index()
        data.columns = [column, 'count']
    else:
        # Compte directement pour les colonnes non-liste
        data = df[column].value_counts().reset_index()
        data.columns = [column, 'count']
    
    # Filtrer pour les N plus fréquents
    data = data.sort_values('count', ascending=False).head(top_n)
    
    # Créer le graphique
    if horizontal:
        fig = px.bar(
            data,
            y=column,
            x='count',
            title=title,
            orientation='h',
            template='plotly_white',
            color='count',
            color_continuous_scale=px.colors.sequential.Blues,
        )
    else:
        fig = px.bar(
            data,
            x=column,
            y='count',
            title=title,
            template='plotly_white',
            color='count',
            color_continuous_scale=px.colors.sequential.Blues,
        )
    
    fig.update_layout(
        font_family="Arial",
        title_font_size=18,
        title_font_color=NODELIO_COLORS["text"],
        showlegend=False,
        plot_bgcolor='white',
        coloraxis_showscale=False,
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial"
        ),
        margin=dict(l=10, r=10, t=50, b=10)
    )
    
    if horizontal:
        fig.update_yaxes(categoryorder='total ascending')
    else:
        fig.update_xaxes(categoryorder='total descending')
    
    return fig

# Fonction pour créer un graphique en camembert
def create_pie_chart(df, column, title):
    if df[column].apply(lambda x: isinstance(x, list)).any():
        all_items = [item for sublist in df[column].dropna() for item in sublist if item]
        data = pd.Series(all_items).value_counts().reset_index()
        data.columns = [column, 'count']
    else:
        data = df[column].value_counts().reset_index()
        data.columns = [column, 'count']
    
    # Limitez aux 6 premiers éléments et regroupez le reste
    if len(data) > 6:
        top_data = data.head(6)
        other_count = data['count'][6:].sum()
        other_data = pd.DataFrame({column: ['Autres'], 'count': [other_count]})
        data = pd.concat([top_data, other_data], ignore_index=True)
    
    # Créer le graphique
    fig = px.pie(
        data,
        values='count',
        names=column,
        title=title,
        color_discrete_sequence=px.colors.sequential.Blues_r,
        template='plotly_white',
        hole=0.4
    )
    
    fig.update_layout(
        font_family="Arial",
        title_font_size=18,
        title_font_color=NODELIO_COLORS["text"],
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=10, r=10, t=50, b=10)
    )
    
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hoverinfo='label+percent+value',
        marker=dict(line=dict(color='white', width=2))
    )
    
    return fig

# Fonction pour créer une carte de chaleur
def create_heatmap(df, col1, col2, title):
    # Créer une table de co-occurrence
    data = []
    
    # Calculer les co-occurrences
    for idx, row in df.iterrows():
        if isinstance(row[col1], list) and isinstance(row[col2], list):
            for item1 in row[col1]:
                for item2 in row[col2]:
                    data.append((item1, item2))
    
    # Compter les co-occurrences
    counter = Counter(data)
    
    # Convertir en DataFrame
    result = pd.DataFrame([(key[0], key[1], value) for key, value in counter.items()], 
                          columns=[col1, col2, 'count'])
    
    # Obtenir les items les plus fréquents pour chaque colonne
    top_col1 = pd.Series([x[0] for x in counter.keys()]).value_counts().head(10).index.tolist()
    top_col2 = pd.Series([x[1] for x in counter.keys()]).value_counts().head(10).index.tolist()
    
    # Filtrer les données
    result = result[result[col1].isin(top_col1) & result[col2].isin(top_col2)]
    
    # Pivoter pour créer une matrice
    matrix = result.pivot_table(values='count', index=col1, columns=col2, fill_value=0)
    
    # Créer la heatmap
    fig = px.imshow(
        matrix,
        labels=dict(x=col2, y=col1, color="Fréquence"),
        color_continuous_scale="Blues",
        title=title
    )
    
    fig.update_layout(
        font_family="Arial",
        title_font_size=18,
        title_font_color=NODELIO_COLORS["text"],
        xaxis=dict(tickangle=45),
        margin=dict(l=50, r=10, t=80, b=80)
    )
    
    return fig

# Fonction pour créer une treemap
def create_treemap(df, column, title):
    if df[column].apply(lambda x: isinstance(x, list)).any():
        # Dépiler les listes et compter
        all_items = [item for sublist in df[column].dropna() for item in sublist if item]
        counter = Counter(all_items)
        # Trier par fréquence et prendre les 15 premiers
        most_common = counter.most_common(15)
        data = pd.DataFrame(most_common, columns=[column, 'count'])
    else:
        data = df[column].value_counts().reset_index().head(15)
        data.columns = [column, 'count']
    
    # Créer la treemap
    fig = px.treemap(
        data,
        path=[column],
        values='count',
        title=title,
        color='count',
        color_continuous_scale='Blues',
        template='plotly_white'
    )
    
    fig.update_layout(
        font_family="Arial",
        title_font_size=18,
        title_font_color=NODELIO_COLORS["text"],
        margin=dict(l=10, r=10, t=50, b=10)
    )
    
    return fig

# Fonction pour créer une carte géographique
def create_geo_map(df, title):
    if 'region' in df.columns:
        # Géolocalisation simplifiée - utilise les régions
        geo_data = df['region'].value_counts().reset_index()
        geo_data.columns = ['region', 'count']
        
        # Créer une carte choroplèthe de la France
        fig = px.choropleth(
            geo_data,
            locations='region',  # colonne avec les identifiants des régions
            color='count',      # colonne avec les valeurs à représenter
            # Ici, vous auriez besoin d'un GeoJSON des régions françaises
            # scope="europe",
            title=title,
            color_continuous_scale="Blues",
        )
    else:
        # Fallback: créer un graphique à barres des pays/villes
        if 'pays' in df.columns:
            geo_col = 'pays'
        elif 'ville' in df.columns:
            geo_col = 'ville'
        else:
            geo_col = 'portee_geographique'
        
        geo_data = df[geo_col].value_counts().reset_index().head(10)
        geo_data.columns = [geo_col, 'count']
        
        fig = px.bar(
            geo_data,
            x=geo_col,
            y='count',
            title=title,
            template='plotly_white',
            color='count',
            color_continuous_scale=px.colors.sequential.Blues
        )
    
    fig.update_layout(
        font_family="Arial",
        title_font_size=18,
        title_font_color=NODELIO_COLORS["text"],
        margin=dict(l=10, r=10, t=50, b=10)
    )
    
    return fig

# Fonction principale pour l'application Streamlit
def main():
    add_logo()
    
    # Titre et description du dashboard
    st.title("Dashboard d'Analyse Narrative de l'Industrie")
    st.markdown("""
    Analysez les tendances narratives liées à l'industrie en France à travers des visualisations interactives.
    Importez votre fichier de données et explorez les insights clés.
    """)
    
    # Uploader le fichier
    uploaded_file = st.file_uploader("Importer un fichier CSV", type=["csv"])
    
    if uploaded_file is not None:
        # Charger et préparer les données
        try:
            df = pd.read_csv(uploaded_file, delimiter=',')
            df = prepare_data(df)
            
            # Navigation principale
            selected = option_menu(
                menu_title=None,
                options=["Vue d'ensemble", "Narratifs", "Acteurs", "Thématiques", "Géographie"],
                icons=["house", "chat-quote", "people", "tag", "geo"],
                menu_icon="cast",
                default_index=0,
                orientation="horizontal",
                styles={
                    "container": {"padding": "0!important", "background-color": "#fff", "border-radius": "10px", "margin-bottom": "20px"},
                    "icon": {"color": NODELIO_COLORS["primary"], "font-size": "16px"},
                    "nav-link": {"font-size": "16px", "text-align": "center", "margin":"0px", "--hover-color": NODELIO_COLORS["secondary"]},
                    "nav-link-selected": {"background-color": NODELIO_COLORS["primary"], "color": "white"},
                }
            )
            
            # Filtres communs à toutes les pages
            st.markdown("<div class='filter-container'>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            
            with col1:
                min_date = df['date'].min().date()
                max_date = df['date'].max().date()
                date_range = st.date_input(
                    "Période d'analyse",
                    [min_date, max_date],
                    min_value=min_date,
                    max_value=max_date
                )
            
            with col2:
                if 'tonalité' in df.columns or 'sentiment' in df.columns:
                    sentiment_col = 'sentiment' if 'sentiment' in df.columns else 'tonalité'
                    available_sentiments = df[sentiment_col].dropna().unique().tolist()
                    selected_sentiments = st.multiselect(
                        "Tonalité",
                        options=available_sentiments,
                        default=available_sentiments
                    )
            
            with col3:
                entity_columns = ['entreprises_mentionnees', 'personnalites_mentionnees', 
                                 'organisations_institutions']
                available_columns = [col for col in entity_columns if col in df.columns]
                
                if available_columns:
                    selected_entity_col = st.selectbox(
                        "Filtrer par entité",
                        options=['Aucun'] + available_columns,
                        format_func=lambda x: {
                            'Aucun': 'Aucun filtre', 
                            'entreprises_mentionnees': 'Entreprises',
                            'personnalites_mentionnees': 'Personnalités',
                            'organisations_institutions': 'Organisations'
                        }.get(x, x)
                    )
                    
                    if selected_entity_col != 'Aucun':
                        # Extraire toutes les entités uniques de la colonne sélectionnée
                        all_entities = []
                        for item_list in df[selected_entity_col].dropna():
                            if isinstance(item_list, list):
                                all_entities.extend(item_list)
                            else:
                                all_entities.append(item_list)
                        
                        unique_entities = sorted(list(set([e for e in all_entities if e])))
                        
                        selected_entity = st.selectbox(
                            f"Sélectionner une {selected_entity_col.replace('_mentionnees', '')}",
                            options=['Toutes'] + unique_entities
                        )
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Appliquer les filtres
            # Filtre de date
            if len(date_range) == 2:
                start_date, end_date = date_range
                filtered_df = df[(df['date'].dt.date >= start_date) & 
                                 (df['date'].dt.date <= end_date)]
            else:
                filtered_df = df.copy()
            
            # Filtre de sentiment
            if 'selected_sentiments' in locals() and len(selected_sentiments) > 0:
                sentiment_col = 'sentiment' if 'sentiment' in filtered_df.columns else 'tonalité'
                filtered_df = filtered_df[filtered_df[sentiment_col].isin(selected_sentiments)]
            
            # Filtre d'entité
            if 'selected_entity' in locals() and selected_entity != 'Toutes' and selected_entity_col != 'Aucun':
                if filtered_df[selected_entity_col].apply(lambda x: isinstance(x, list)).any():
                    # Pour les colonnes avec des listes
                    filtered_df = filtered_df[filtered_df[selected_entity_col].apply(
                        lambda x: selected_entity in x if isinstance(x, list) else False
                    )]
                else:
                    # Pour les colonnes avec des valeurs simples
                    filtered_df = filtered_df[filtered_df[selected_entity_col] == selected_entity]
            
            # Si aucune donnée après filtrage
            if len(filtered_df) == 0:
                st.warning("Aucune donnée ne correspond aux filtres sélectionnés. Veuillez ajuster vos critères.")
                return
            
            # Afficher le contenu en fonction de l'onglet sélectionné
            if selected == "Vue d'ensemble":
                st.markdown(section_header("Vue d'ensemble des tendances narratives"), unsafe_allow_html=True)
                
                # Métriques principales
                metrics_row1 = st.columns(4)
                
                with metrics_row1[0]:
                    total_articles = len(filtered_df)
                    # Calculer la variation par rapport à la période précédente
                    if len(date_range) == 2:
                        current_period = (date_range[1] - date_range[0]).days
                        previous_start = date_range[0] - datetime.timedelta(days=current_period)
                        previous_end = date_range[0] - datetime.timedelta(days=1)
                        previous_df = df[(df['date'].dt.date >= previous_start) & 
                                         (df['date'].dt.date <= previous_end)]
                        previous_count = len(previous_df)
                        
                        if previous_count > 0:
                            variation = ((total_articles - previous_count) / previous_count) * 100
                        else:
                            variation = None
                    else:
                        variation = None
                    
                    st.markdown(metric_card(
                        "Nombre total d'articles", 
                        total_articles, 
                        delta=round(variation) if variation is not None else None
                    ), unsafe_allow_html=True)
                
                with metrics_row1[1]:
                    # Sentiment moyen si disponible
                    if 'sentiment' in filtered_df.columns:
                        positive_ratio = (filtered_df['sentiment'] == 'Positive').mean() * 100
                        
                        # Calculer la variation
                        if len(date_range) == 2 and 'previous_df' in locals() and len(previous_df) > 0:
                            previous_positive = (previous_df['sentiment'] == 'Positive').mean() * 100
                            sentiment_variation = positive_ratio - previous_positive
                        else:
                            sentiment_variation = None
                            
                        st.markdown(metric_card(
                            "Tonalité positive", 
                            f"{round(positive_ratio, 1)}", 
                            delta=round(sentiment_variation, 1) if sentiment_variation is not None else None,
                            suffix="%"
                        ), unsafe_allow_html=True)
                    else:
                        st.markdown(metric_card("Tonalité", "N/A"), unsafe_allow_html=True)
                
                with metrics_row1[2]:
                    # Nombre d'entreprises uniques mentionnées
                    if 'entreprises_mentionnees' in filtered_df.columns:
                        all_companies = []
                        for companies in filtered_df['entreprises_mentionnees'].dropna():
                            if isinstance(companies, list):
                                all_companies.extend(companies)
                            else:
                                all_companies.append(companies)
                        
                        unique_companies = len(set([c for c in all_companies if c]))
                        st.markdown(metric_card("Entreprises mentionnées", unique_companies), unsafe_allow_html=True)
                    else:
                        st.markdown(metric_card("Entreprises mentionnées", "N/A"), unsafe_allow_html=True)
                
                with metrics_row1[3]:
                    # Nombre de sujets uniques
                    if 'topics' in filtered_df.columns:
                        all_topics = []
                        for topics in filtered_df['topics'].dropna():
                            if isinstance(topics, list):
                                all_topics.extend(topics)
                            else:
                                all_topics.append(topics)
                        
                        unique_topics = len(set([t for t in all_topics if t]))
                        st.markdown(metric_card("Sujets uniques", unique_topics), unsafe_allow_html=True)
                    else:
                        st.markdown(metric_card("Sujets uniques", "N/A"), unsafe_allow_html=True)
                
                # Tendances temporelles
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Tendance du volume d'articles
                st.plotly_chart(
                    create_time_trend(filtered_df, freq='D', metric='count'),
                    use_container_width=True
                )
                
                # Distribution de sentiment et types de narratifs
                row2 = st.columns(2)
                
                with row2[0]:
                    if 'sentiment' in filtered_df.columns:
                        st.plotly_chart(
                            create_pie_chart(filtered_df, 'sentiment', "Distribution des tonalités"),
                            use_container_width=True
                        )
                    elif 'tonalité' in filtered_df.columns:
                        st.plotly_chart(
                            create_pie_chart(filtered_df, 'tonalité', "Distribution des tonalités"),
                            use_container_width=True
                        )
                
                with row2[1]:
                    if 'cadrage' in filtered_df.columns:
                        st.plotly_chart(
                            create_pie_chart(filtered_df, 'cadrage', "Types de cadrage"),
                            use_container_width=True
                        )
                    elif 'narratif_paradigmatique' in filtered_df.columns:
                        st.plotly_chart(
                            create_pie_chart(filtered_df, 'narratif_paradigmatique', "Types de narratifs paradigmatiques"),
                            use_container_width=True
                        )
                
                # Treemap des enjeux associés
                if 'enjeux_associes' in filtered_df.columns:
                    st.plotly_chart(
                        create_treemap(filtered_df, 'enjeux_associes', "Enjeux associés à l'industrie"),
                        use_container_width=True
                    )
                
                # Nuage de mots des titres
                if 'title' in filtered_df.columns:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown(section_header("Analyse des titres d'articles", "gradient-rose"), unsafe_allow_html=True)
                    
                    titles = filtered_df['title'].dropna().tolist()
                    if titles:
                        wordcloud_html = generate_wordcloud(titles, colormap='Blues', background_color='white')
                        st.markdown(wordcloud_html, unsafe_allow_html=True)
            
            elif selected == "Narratifs":
                st.markdown(section_header("Analyse des narratifs"), unsafe_allow_html=True)
                
                # Analyse du top 20 des narratifs paradigmatiques sauf le narratif "Bruit"
                if 'nom_cluster_paradigmatique' in filtered_df.columns:
                    # Filtrer les narratifs pour exclure "Bruit"
                    filtered_df = filtered_df[filtered_df['nom_cluster_paradigmatique'] != 'Bruit']
                    narratif_counts = filtered_df['nom_cluster_paradigmatique'].value_counts().reset_index().head(20)
                    narratif_counts.columns = ['Narratif', 'Nombre']
                    
                    fig = px.bar(
                        narratif_counts,
                        y='Narratif',
                        x='Nombre',
                        title="Top 20 des narratifs paradigmatiques",
                        orientation='h',
                        template='plotly_white',
                        color='Nombre',
                        color_continuous_scale=px.colors.sequential.Blues,
                    )
                    
                    fig.update_layout(
                        font_family="Arial",
                        title_font_size=18,
                        title_font_color=NODELIO_COLORS["text"],
                        showlegend=False,
                        plot_bgcolor='white',
                        coloraxis_showscale=False,
                        hoverlabel=dict(
                            bgcolor="white",
                            font_size=12,
                            font_family="Arial"
                        ),
                        margin=dict(l=10, r=10, t=50, b=10),
                        height=600  # Hauteur ajustée pour les 20 narratifs
                    )
                    
                    fig.update_yaxes(categoryorder='total ascending')
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Évolution des principaux narratifs dans le temps avec sélection
                if 'nom_cluster_paradigmatique' in filtered_df.columns:
                    # Dépiler les narratifs et créer un DataFrame avec dates
                    narratifs_time = []
                    
                    for idx, row in filtered_df.iterrows():
                        if not pd.isna(row['nom_cluster_paradigmatique']) and not pd.isna(row['date']):
                            narratifs_time.append((row['date'], row['nom_cluster_paradigmatique']))
                    
                    if narratifs_time:
                        nt_df = pd.DataFrame(narratifs_time, columns=['date', 'narratif'])
                        
                        # Compter tous les narratifs et prendre les 15 plus fréquents pour le menu de sélection
                        all_narratifs = nt_df['narratif'].value_counts().head(15).index.tolist()
                        
                        # Créer un multiselect pour choisir les narratifs à afficher
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.markdown("### Sélectionnez les narratifs à afficher")
                        
                        # Par défaut, sélectionner les 5 narratifs les plus fréquents
                        default_narratifs = nt_df['narratif'].value_counts().head(5).index.tolist()
                        
                        selected_narratifs = st.multiselect(
                            "Choisir les narratifs à visualiser",
                            options=all_narratifs,
                            default=default_narratifs,
                            key="narratifs_timeline_selector"
                        )
                        
                        # Si aucun narratif n'est sélectionné, utiliser les 5 plus fréquents
                        if not selected_narratifs:
                            selected_narratifs = default_narratifs
                            st.info("Aucun narratif sélectionné. Affichage des 5 narratifs les plus fréquents par défaut.")
                        
                        # Filtrer pour les narratifs sélectionnés
                        nt_df_filtered = nt_df[nt_df['narratif'].isin(selected_narratifs)]
                        
                        # Grouper par jour et narratif
                        nt_grouped = nt_df_filtered.groupby([pd.Grouper(key='date', freq='W'), 'narratif']).size().reset_index(name='count')
                        
                        # Pivoter
                        nt_pivot = nt_grouped.pivot(index='date', columns='narratif', values='count').fillna(0)
                    
                        
                        # Créer une figure vide pour la version avec aires
                        fig2 = go.Figure()
                        
                        # Ajouter une trace pour chaque narratif
                        for narratif in selected_narratifs:
                            if narratif in nt_pivot.columns:
                                narratif_data = nt_pivot[narratif]
                                
                                fig2.add_trace(go.Scatter(
                                    x=nt_pivot.index,
                                    y=narratif_data,
                                    mode='lines',
                                    name=narratif,
                                    fill='tozeroy',  # Remplir l'aire sous la ligne
                                    line=dict(width=1),  # Ligne plus fine
                                    opacity=0.7  # Légère transparence
                                ))
                        
                        # Configuration du graphique
                        fig2.update_layout(
                            title="Évolution des narratifs sélectionnés au fil du temps",
                            title_font_size=18,
                            title_font_color=NODELIO_COLORS["text"],
                            legend_title="Narratif",
                            xaxis_title="Date",
                            yaxis_title="Nombre d'articles",
                            plot_bgcolor='white',
                            template='plotly_white',
                            font_family="Arial",
                            hovermode="x unified",
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=-0.4,
                                xanchor="center",
                                x=0.5
                            )
                        )
                        
                        st.plotly_chart(fig2, use_container_width=True)

                
                
                    # Analyse temporelle d'un narratif spécifique
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown(section_header("Analyse détaillée d'un narratif spécifique", "gradient-blue"), unsafe_allow_html=True)
                    
                    if 'nom_cluster_paradigmatique' in filtered_df.columns:
                        # Compter les occurrences de chaque narratif et les trier par fréquence
                        narratif_counts = filtered_df['nom_cluster_paradigmatique'].value_counts()
                        
                        # Créer une liste ordonnée des narratifs (du plus fréquent au moins fréquent)
                        all_narratifs = narratif_counts.index.tolist()
                        
                        # Ajouter des informations sur la fréquence dans les options du sélecteur
                        narratif_options = [f"{narratif} ({narratif_counts[narratif]} articles)" for narratif in all_narratifs]
                        
                        # Créer un dictionnaire de correspondance pour retrouver le narratif original
                        narratif_mapping = {option: narratif for option, narratif in zip(narratif_options, all_narratifs)}
                        
                        # Sélection par l'utilisateur avec les options enrichies
                        selected_narratif_option = st.selectbox(
                            "Sélectionner un narratif à analyser en détail",
                            options=narratif_options
                        )
                        
                        # Récupérer le narratif original à partir de l'option sélectionnée
                        selected_narratif = narratif_mapping[selected_narratif_option]
                        
                        # Filtrer les données pour ce narratif
                        narratif_df = filtered_df[filtered_df['nom_cluster_paradigmatique'] == selected_narratif]
                        
                        # Ajouter un filtre par entité
                        entity_columns = ['entreprises_mentionnees', 'personnalites_mentionnees', 'organisations_institutions']
                        available_entity_cols = [col for col in entity_columns if col in filtered_df.columns]
                        
                        if available_entity_cols:
                            # Créer un sélecteur pour le type d'entité
                            selected_entity_type = st.selectbox(
                                "Filtrer par type d'entité (optionnel)",
                                options=['Aucun filtre'] + available_entity_cols,
                                format_func=lambda x: {
                                    'Aucun filtre': 'Aucun filtre', 
                                    'entreprises_mentionnees': 'Entreprises',
                                    'personnalites_mentionnees': 'Personnalités',
                                    'organisations_institutions': 'Organisations'
                                }.get(x, x)
                            )
                            
                            # Si un type d'entité est sélectionné, proposer les entités spécifiques
                            if selected_entity_type != 'Aucun filtre':
                                # Extraire toutes les entités uniques de ce type dans le narratif sélectionné
                                all_entities = []
                                for entity_list in narratif_df[selected_entity_type].dropna():
                                    if isinstance(entity_list, list):
                                        all_entities.extend(entity_list)
                                    else:
                                        all_entities.append(entity_list)
                                
                                unique_entities = sorted(list(set([e for e in all_entities if e])))
                                
                                if unique_entities:
                                    selected_entity = st.selectbox(
                                        f"Sélectionner une {selected_entity_type.replace('_mentionnees', '').replace('_institutions', '')}",
                                        options=['Toutes'] + unique_entities
                                    )
                                    
                                    # Filtrer davantage si une entité spécifique est sélectionnée
                                    if selected_entity != 'Toutes':
                                        narratif_df = narratif_df[narratif_df[selected_entity_type].apply(
                                            lambda x: selected_entity in x if isinstance(x, list) else False
                                        )]
                                else:
                                    st.info(f"Aucune entité de type {selected_entity_type.replace('_mentionnees', '').replace('_institutions', '')} n'est associée à ce narratif.")
                        
                        # Vérifier si des données restent après filtrage
                        if len(narratif_df) == 0:
                            st.warning("Aucune donnée ne correspond aux filtres sélectionnés. Veuillez ajuster vos critères.")
                        else:
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                # Créer une série temporelle pour ce narratif
                                time_series = narratif_df.groupby(pd.Grouper(key='date', freq='D')).size().reset_index(name='count')
                                
                                # Créer le graphique
                                fig = px.line(
                                    time_series,
                                    x='date',
                                    y='count',
                                    title=f"Évolution temporelle",
                                    template='plotly_white'
                                )
                                
                                fig.update_traces(line_color=NODELIO_COLORS["primary"], line_width=3)
                                
                                fig.update_layout(
                                    font_family="Arial",
                                    title_font_size=16,
                                    title_font_color=NODELIO_COLORS["text"],
                                    xaxis_title="Date",
                                    yaxis_title="Nombre d'articles",
                                    plot_bgcolor='white',
                                    hovermode="x unified"
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                            
                            with col2:
                                # Analyse des enjeux associés à ce narratif spécifique
                                if 'enjeux_associes' in filtered_df.columns:
                                    # Extraire tous les enjeux associés à ce narratif
                                    all_enjeux = []
                                    for enjeux_list in narratif_df['enjeux_associes'].dropna():
                                        if isinstance(enjeux_list, list):
                                            all_enjeux.extend(enjeux_list)
                                    
                                    # Compter les occurrences
                                    if all_enjeux:
                                        enjeux_counter = Counter(all_enjeux)
                                        enjeux_df = pd.DataFrame.from_dict(enjeux_counter, orient='index').reset_index()
                                        enjeux_df.columns = ['Enjeu', 'Nombre']
                                        enjeux_df = enjeux_df.sort_values('Nombre', ascending=False).head(10)
                                        
                                        # Créer le graphique
                                        fig = px.pie(
                                            enjeux_df,
                                            values='Nombre',
                                            names='Enjeu',
                                            title="Enjeux associés",
                                            template='plotly_white',
                                            color_discrete_sequence=px.colors.sequential.Blues_r
                                        )

                                        # Mettre à jour les traces pour placer les étiquettes à l'extérieur avec des lignes de connexion
                                        fig.update_traces(
                                            textposition='outside',     # Place le texte à l'extérieur
                                            textinfo='label+percent',   # Affiche le nom de l'enjeu et le pourcentage
                                            pull=[0.01] * len(enjeux_df),  # Léger espacement pour améliorer la lisibilité
                                            marker=dict(
                                                line=dict(color='white', width=2)  # Bordure blanche entre les secteurs
                                            )
                                        )

                                        # Améliorer la mise en page pour accommoder les étiquettes externes
                                        fig.update_layout(
                                            font_family="Arial",
                                            title_font_size=16,
                                            title_font_color=NODELIO_COLORS["text"],
                                            margin=dict(l=30, r=30, t=50, b=30),  # Marges augmentées
                                            legend=dict(
                                                orientation="h",
                                                yanchor="bottom",
                                                y=-0.4,  # Position plus basse pour éviter le chevauchement
                                                xanchor="center",
                                                x=0.5
                                            ),
                                            showlegend=False  # Désactiver la légende puisque les labels sont déjà visibles
                                        )

                                        
                                        st.plotly_chart(fig, use_container_width=True)
                                    else:
                                        st.info("Aucun enjeu associé à ce narratif dans les données filtrées.")

                
                    # Remplacer le nuage de mots par un tableau paginé des articles filtrés
                    if 'nom_cluster_paradigmatique' in filtered_df.columns:
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.markdown(section_header("Articles correspondant aux filtres sélectionnés", "gradient-rose"), unsafe_allow_html=True)
                        
                        # Vérifier si nous avons déjà un narratif sélectionné et des données filtrées
                        if 'selected_narratif' in locals() and 'narratif_df' in locals():
                            # Créer un tableau des articles filtrés
                            if len(narratif_df) > 0:
                                # Sélectionner les colonnes pertinentes si elles existent
                                display_columns = []
                                if 'titre' in narratif_df.columns:
                                    display_columns.append('titre')
                                elif 'title' in narratif_df.columns:
                                    display_columns.append('title')
                                    
                                if 'url' in narratif_df.columns:
                                    display_columns.append('url')

                                if 'cadrage' in narratif_df.columns:
                                    display_columns.append('cadrage')     

                                if 'enjeux_associes' in narratif_df.columns:
                                    display_columns.append('enjeux_associes')
                                    
                                if 'nom_cluster_paradigmatique' in narratif_df.columns:
                                    display_columns.append('nom_cluster_paradigmatique')
                                    
                                # Vérifier qu'il y a des colonnes à afficher
                                if display_columns:
                                    # Créer une copie du dataframe avec seulement les colonnes sélectionnées
                                    display_df = narratif_df[display_columns].copy()
                                    
                                    # Formater les listes pour un meilleur affichage
                                    for col in display_df.columns:
                                        if display_df[col].apply(lambda x: isinstance(x, list)).any():
                                            display_df[col] = display_df[col].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
                                    
                                    # Rendre les URLs cliquables si présentes
                                    if 'url' in display_df.columns:
                                        # Fonction pour créer des liens cliquables
                                        def make_clickable(url):
                                            if pd.isna(url) or not isinstance(url, str):
                                                return ""
                                            # Extraire le nom de domaine pour un affichage plus propre
                                            try:
                                                from urllib.parse import urlparse
                                                domain = urlparse(url).netloc
                                                return f'<a href="{url}" target="_blank">{domain}</a>'
                                            except:
                                                return f'<a href="{url}" target="_blank">Lien</a>'
                                        
                                        # Appliquer le formatage aux URLs
                                        display_df['url'] = display_df['url'].apply(make_clickable)
                                    
                                    # Afficher le nombre total d'articles
                                    st.write(f"**{len(display_df)} articles trouvés**")
                                    
                                    
                                    # Pagination fixe à 10 articles par page
                                    rows_per_page = 10
                                    
                                    # Pagination
                                    total_pages = max(1, (len(display_df) + rows_per_page - 1) // rows_per_page)
                                    
                                    if total_pages > 1:
                                        col1, col2 = st.columns([1, 4])
                                        with col1:
                                            page_number = st.number_input("Page", min_value=1, max_value=total_pages, value=1)
                                        with col2:
                                            st.markdown(f"<div style='padding-top: 30px;'>sur {total_pages} pages</div>", unsafe_allow_html=True)
                                        
                                        start_idx = (page_number - 1) * rows_per_page
                                        end_idx = min(start_idx + rows_per_page, len(display_df))
                                        page_df = display_df.iloc[start_idx:end_idx]
                                        st.write(f"Affichage des articles {start_idx+1} à {end_idx} sur {len(display_df)}")
                                    else:
                                        page_df = display_df
                                    
                                    # Afficher le tableau avec les liens cliquables
                                    if 'url' in page_df.columns:
                                        st.write(page_df.to_html(escape=False), unsafe_allow_html=True)
                                    else:
                                        st.dataframe(page_df)
                                    
                                    # Bouton pour télécharger tous les résultats filtrés en CSV
                                    # Créer une version du dataframe sans les liens HTML pour le CSV
                                    csv_df = narratif_df[display_columns].copy()
                                    for col in csv_df.columns:
                                        if csv_df[col].apply(lambda x: isinstance(x, list)).any():
                                            csv_df[col] = csv_df[col].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
                                    
                                    csv = csv_df.to_csv(index=False)
                                    st.download_button(
                                        label="Télécharger tous les résultats en CSV",
                                        data=csv,
                                        file_name=f"articles_{selected_narratif.replace(' ', '_')}.csv",
                                        mime="text/csv"
                                    )
                                else:
                                    st.info("Aucune colonne pertinente trouvée dans les données.")
                            else:
                                st.info("Aucun article ne correspond aux filtres sélectionnés.")
                        else:
                            st.info("Veuillez sélectionner un narratif pour voir les articles correspondants.")



            
            elif selected == "Acteurs":
                st.markdown(section_header("Analyse des acteurs"), unsafe_allow_html=True)
                
                # Top entreprises et personnalités
                row1 = st.columns(2)
                
                with row1[0]:
                    if 'entreprises_mentionnees' in filtered_df.columns:
                        st.plotly_chart(
                            create_distribution_chart(filtered_df, 'entreprises_mentionnees', 
                                                     "Entreprises les plus mentionnées", top_n=10),
                            use_container_width=True
                        )
                
                with row1[1]:
                    if 'personnalites_mentionnees' in filtered_df.columns:
                        st.plotly_chart(
                            create_distribution_chart(filtered_df, 'personnalites_mentionnees', 
                                                     "Personnalités les plus mentionnées", top_n=10),
                            use_container_width=True
                        )
                
                # Organisations et institutions
                if 'organisations_institutions' in filtered_df.columns:
                    st.plotly_chart(
                        create_distribution_chart(filtered_df, 'organisations_institutions', 
                                                 "Organisations et institutions les plus mentionnées", top_n=10),
                        use_container_width=True
                    )
                
                # Relations entre acteurs
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(section_header("Relations entre acteurs", "gradient-rose"), unsafe_allow_html=True)
                
                actor_cols = [col for col in ['entreprises_mentionnees', 'personnalites_mentionnees', 
                                             'organisations_institutions'] if col in filtered_df.columns]
                
                if len(actor_cols) >= 2:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        actor_col1 = st.selectbox(
                            "Premier type d'acteur",
                            options=actor_cols,
                            index=0,
                            format_func=lambda x: {
                                'entreprises_mentionnees': 'Entreprises',
                                'personnalites_mentionnees': 'Personnalités',
                                'organisations_institutions': 'Organisations'
                            }.get(x, x)
                        )
                    
                    with col2:
                        remaining_cols = [col for col in actor_cols if col != actor_col1]
                        actor_col2 = st.selectbox(
                            "Second type d'acteur",
                            options=remaining_cols,
                            index=0,
                            format_func=lambda x: {
                                'entreprises_mentionnees': 'Entreprises',
                                'personnalites_mentionnees': 'Personnalités',
                                'organisations_institutions': 'Organisations'
                            }.get(x, x)
                        )
                    
                    # Créer la heatmap de co-occurrence
                    st.plotly_chart(
                        create_heatmap(
                            filtered_df, 
                            actor_col1, 
                            actor_col2, 
                            f"Co-occurrences entre {actor_col1.replace('_mentionnees', '')} et {actor_col2.replace('_mentionnees', '')}"
                        ),
                        use_container_width=True
                    )
                
                # Tonalité associée aux acteurs principaux
                if 'sentiment' in filtered_df.columns and 'entreprises_mentionnees' in filtered_df.columns:
                    # Créer une analyse de sentiment par entreprise
                    enterprise_sentiment = []
                    
                    for idx, row in filtered_df.iterrows():
                        if isinstance(row['entreprises_mentionnees'], list) and row['sentiment'] in ['Positive', 'Négative', 'Neutre']:
                            for company in row['entreprises_mentionnees']:
                                enterprise_sentiment.append((company, row['sentiment']))
                    
                    # Convertir en DataFrame
                    if enterprise_sentiment:
                        es_df = pd.DataFrame(enterprise_sentiment, columns=['entreprise', 'sentiment'])
                        
                        # Compter par entreprise et sentiment
                        es_pivot = pd.crosstab(es_df['entreprise'], es_df['sentiment'])
                        
                        # Calculer le total par entreprise
                        es_pivot['Total'] = es_pivot.sum(axis=1)
                        
                        # Trier et prendre les 10 premières entreprises
                        top_companies = es_pivot.sort_values('Total', ascending=False).head(10).index
                        es_pivot = es_pivot.loc[top_companies]
                        
                        # Calculer les pourcentages
                        for col in ['Positive', 'Négative', 'Neutre']:
                            if col in es_pivot.columns:
                                es_pivot[f'{col} %'] = es_pivot[col] / es_pivot['Total'] * 100
                        
                        # Créer le graphique
                        sentiment_cols = [col for col in ['Positive %', 'Neutre %', 'Négative %'] if col in es_pivot.columns]
                        
                        if sentiment_cols:
                            fig = go.Figure()
                            
                            # Ajouter une trace pour chaque sentiment
                            for col in sentiment_cols:
                                sentiment = col.replace(' %', '')
                                color = {
                                    'Positive': NODELIO_COLORS["positive"],
                                    'Négative': NODELIO_COLORS["negative"],
                                    'Neutre': NODELIO_COLORS["neutral_tone"]
                                }.get(sentiment, 'gray')
                                
                                fig.add_trace(go.Bar(
                                    y=es_pivot.index,
                                    x=es_pivot[col],
                                    name=sentiment,
                                    orientation='h',
                                    marker_color=color
                                ))
                            
                            # Mise en page
                            fig.update_layout(
                                title="Tonalité associée aux principales entreprises",
                                barmode='stack',
                                yaxis={'categoryorder': 'total ascending'},
                                xaxis_title="Pourcentage",
                                legend_title="Tonalité",
                                font_family="Arial",
                                title_font_size=18,
                                title_font_color=NODELIO_COLORS["text"],
                                template='plotly_white'
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
            
            elif selected == "Thématiques":
                st.markdown(section_header("Analyse thématique"), unsafe_allow_html=True)
                
                # Distribution des enjeux et topics
                row1 = st.columns(2)
                
                with row1[0]:
                    if 'enjeux_associes' in filtered_df.columns:
                        st.plotly_chart(
                            create_distribution_chart(filtered_df, 'enjeux_associes', 
                                                     "Principaux enjeux associés", top_n=10),
                            use_container_width=True
                        )
                
                with row1[1]:
                    if 'topics' in filtered_df.columns:
                        st.plotly_chart(
                            create_distribution_chart(filtered_df, 'topics', 
                                                     "Principaux sujets (topics)", top_n=10),
                            use_container_width=True
                        )
                
                # Treemap des enjeux
                if 'enjeux_associes' in filtered_df.columns:
                    st.plotly_chart(
                        create_treemap(filtered_df, 'enjeux_associes', "Hiérarchie des enjeux"),
                        use_container_width=True
                    )
                
                # Relations entre enjeux et acteurs
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(section_header("Relations entre thématiques et acteurs", "gradient-rose"), unsafe_allow_html=True)
                
                theme_cols = [col for col in ['enjeux_associes', 'topics'] if col in filtered_df.columns]
                actor_cols = [col for col in ['entreprises_mentionnees', 'personnalites_mentionnees', 
                                             'organisations_institutions'] if col in filtered_df.columns]
                
                if theme_cols and actor_cols:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        theme_col = st.selectbox(
                            "Type de thématique",
                            options=theme_cols,
                            index=0,
                            format_func=lambda x: {'enjeux_associes': 'Enjeux', 'topics': 'Topics'}.get(x, x)
                        )
                    
                    with col2:
                        actor_col = st.selectbox(
                            "Type d'acteur",
                            options=actor_cols,
                            index=0,
                            format_func=lambda x: {
                                'entreprises_mentionnees': 'Entreprises',
                                'personnalites_mentionnees': 'Personnalités',
                                'organisations_institutions': 'Organisations'
                            }.get(x, x)
                        )
                    
                    # Créer la heatmap de co-occurrence
                    st.plotly_chart(
                        create_heatmap(
                            filtered_df, 
                            theme_col, 
                            actor_col, 
                            f"Co-occurrences entre {theme_col.replace('_associes', '')} et {actor_col.replace('_mentionnees', '')}"
                        ),
                        use_container_width=True
                    )
                
                # Évolution des enjeux dans le temps avec sélection d'enjeux
                if 'enjeux_associes' in filtered_df.columns:
                    # Dépiler les enjeux et créer un DataFrame avec dates
                    enjeux_time = []
                    
                    for idx, row in filtered_df.iterrows():
                        if isinstance(row['enjeux_associes'], list) and not pd.isna(row['date']):
                            for enjeu in row['enjeux_associes']:
                                enjeux_time.append((row['date'], enjeu))
                    
                    if enjeux_time:
                        et_df = pd.DataFrame(enjeux_time, columns=['date', 'enjeu'])
                        
                        # Compter tous les enjeux et prendre les 15 plus fréquents pour le menu de sélection
                        all_enjeux = et_df['enjeu'].value_counts().head(30).index.tolist()
                        
                        # Créer un multiselect pour choisir les enjeux à afficher
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.markdown("### Sélectionnez les enjeux à afficher")
                        
                        # Par défaut, sélectionner les 5 enjeux les plus fréquents
                        default_enjeux = et_df['enjeu'].value_counts().head(5).index.tolist()
                        
                        selected_enjeux = st.multiselect(
                            "Choisir les enjeux à visualiser",
                            options=all_enjeux,
                            default=default_enjeux,
                            key="enjeux_timeline_selector"
                        )
                        
                        # Si aucun enjeu n'est sélectionné, utiliser les 5 plus fréquents
                        if not selected_enjeux:
                            selected_enjeux = default_enjeux
                            st.info("Aucun enjeu sélectionné. Affichage des 5 enjeux les plus fréquents par défaut.")
                        
                        # Filtrer pour les enjeux sélectionnés
                        et_df_filtered = et_df[et_df['enjeu'].isin(selected_enjeux)]
                        
                        # Grouper par jour et enjeu
                        et_grouped = et_df_filtered.groupby([pd.Grouper(key='date', freq='W'), 'enjeu']).size().reset_index(name='count')
                        
                        # Pivoter
                        et_pivot = et_grouped.pivot(index='date', columns='enjeu', values='count').fillna(0)
                        
                        # Créer une figure vide
                        fig = go.Figure()
                        
                        # Ajouter un histogramme pour chaque enjeu
                        for enjeu in selected_enjeux:
                            if enjeu in et_pivot.columns:
                                enjeu_data = et_pivot[enjeu]
                                
                                fig.add_trace(go.Scatter(
                                    x=et_pivot.index,
                                    y=enjeu_data,
                                    mode='lines',
                                    name=enjeu,
                                    fill='tozeroy',  # Remplir l'aire sous la ligne
                                    line=dict(width=1),  # Ligne plus fine
                                    opacity=1  # Légère transparence
                                ))
                        
                        # Configuration du graphique
                        fig.update_layout(
                            title="Évolution des enjeux sélectionnés au fil du temps",
                            title_font_size=18,
                            title_font_color=NODELIO_COLORS["text"],
                            legend_title="Enjeu",
                            xaxis_title="Date",
                            yaxis_title="Nombre de mentions",
                            plot_bgcolor='white',
                            template='plotly_white',
                            font_family="Arial",
                            hovermode="x unified",
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=-0.4,
                                xanchor="center",
                                x=0.5
                            )
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)


            
            elif selected == "Géographie":
                st.markdown(section_header("Analyse géographique"), unsafe_allow_html=True)
                
                # Distribution par portée géographique
                if 'portee_geographique' in filtered_df.columns:
                    st.plotly_chart(
                        create_pie_chart(filtered_df, 'portee_geographique', "Distribution par portée géographique"),
                        use_container_width=True
                    )
                
                # Distribution par pays et région
                row1 = st.columns(2)
                
                with row1[0]:
                    if 'pays' in filtered_df.columns:
                        st.plotly_chart(
                            create_distribution_chart(filtered_df, 'pays', 
                                                     "Distribution par pays", top_n=10, horizontal=False),
                            use_container_width=True
                        )
                
                with row1[1]:
                    if 'region' in filtered_df.columns:
                        st.plotly_chart(
                            create_distribution_chart(filtered_df, 'region', 
                                                     "Distribution par région", top_n=10, horizontal=False),
                            use_container_width=True
                        )
                
                # Carte des villes
                if 'ville' in filtered_df.columns:
                    # Nous aurions besoin des coordonnées pour une vraie carte
                    # À défaut, utiliser un graphique à barres
                    st.plotly_chart(
                        create_distribution_chart(filtered_df, 'ville', 
                                                 "Top villes mentionnées", top_n=15),
                        use_container_width=True
                    )
                
                # Relation entre géographie et enjeux
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(section_header("Enjeux par région", "gradient-rose"), unsafe_allow_html=True)
                
                geo_cols = [col for col in ['region', 'pays', 'ville', 'portee_geographique'] if col in filtered_df.columns]
                
                if geo_cols and 'enjeux_associes' in filtered_df.columns:
                    selected_geo = st.selectbox(
                        "Choisir une dimension géographique",
                        options=geo_cols,
                        format_func=lambda x: x.capitalize()
                    )
                    
                    # Créer la heatmap
                    st.plotly_chart(
                        create_heatmap(
                            filtered_df, 
                            selected_geo, 
                            'enjeux_associes', 
                            f"Enjeux principaux par {selected_geo}"
                        ),
                        use_container_width=True
                    )
            
            # Ajouter un pied de page
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.markdown(
                """
                <div class="footer">
                    <p>Dashboard réalisé par Nodelio Social Intelligence — Données actualisées au {}</p>
                </div>
                """.format(datetime.datetime.now().strftime('%d/%m/%Y')),
                unsafe_allow_html=True
            )
            
        except Exception as e:
            st.error(f"Une erreur s'est produite lors du chargement ou de l'analyse des données : {e}")
            st.error("Veuillez vérifier que le format du fichier CSV est correct et contient les colonnes attendues.")

if __name__ == "__main__":
    main()