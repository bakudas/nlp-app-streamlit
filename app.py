# Core Pkgs
import streamlit as st
st.set_page_config(page_title="NLP Web App", page_icon="🤯", layout="wide", initial_sidebar_state="collapsed")

# NLP Pkgs
from textblob import TextBlob
import neattext as nt
import nltk
nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('portuguese')
from nltk.probability import FreqDist
import spacy
from spacy import displacy
# from spacy.lang.pt.examples import sentences
from enelvo.normaliser import Normaliser
from deep_translator import GoogleTranslator

from collections import Counter
import re

# Viz Pkgs
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from wordcloud import WordCloud


################################
######## AUX FUNCTIONS #########
################################

def summarize_text(text, num_sentences=3):
    # Remove special characters and convert text to lower case
    clean_text = re.sub('[^a-zA-Z0-9]', ' ', text).lower()
    
    # Split the text into words
    words = clean_text.split()
    
    # Calculate the frequency of each word
    word_freq = Counter(words)
    
    # Sort the words based on frequency in descending order
    sorted_words = sorted(word_freq, key=word_freq.get, reverse=True)
    
    # Extract the top 'num_sentences' most frequent words
    top_words = sorted_words[:num_sentences]
    
    # Create a summary by joining the top words
    summary = ' '.join(top_words)
    
    return summary



@st.cache_data
# Lemma and Tokens Function
def text_analyzer(text):
    # import english library
    nlp = spacy.load("pt_core_news_sm")
    
    #create an nlp object
    doc = nlp(text)
    
    st.image(displacy.render(doc))
    
    #extract tokens and lemmas
    allData = [('"Tokens": {}, \n"Lemmas": {}'.format(token.text, token.lemma_)) for token in doc]
    
    return allData

# TEXT PRE PROCESS
# https://medium.com/turing-talks/uma-an%C3%A1lise-de-dom-casmurro-com-nltk-343d72dd47a7
def pre_process(texto):
    # seleciona apenas letras e coloca todas em minúsculo 
    letras_min =  re.findall(r'\b[A-zÀ-úü]+\b', texto.lower())

    # remove stopwords
    stop = set(stopwords)
    sem_stopwords = [w for w in letras_min if w not in stop]

    # juntando os tokens novamente em formato de texto
    texto_limpo = " ".join(sem_stopwords)

    return texto_limpo


################################
######### MAIN PROGRAM #########
################################

def main():        
    """NLP web app with Streamlit"""
    
    tittle_template = """
    <div style="background-color:blue; padding: 10px;">
    <h1 style="color:cyan;">NLP Web Application</h1>
    </div>
    """
    
    # PRINT TITLE
    st.markdown(tittle_template, unsafe_allow_html=True)
    
    subtittle_template = """
    <div style="background-color:cyan; padding: 10px;">
    <h3 style="color:blue;">by bakudas</h3>
    </div>
    """
    
    # PRINT SUBTITLE
    st.markdown(subtittle_template, unsafe_allow_html=True)

    # ADD IMAGE TO SIDEBAR
    st.sidebar.image("nlp.jpg", use_column_width=True)
    
    # ADD SELECTBOX TO SIDEBAR
    activity = ["Text Analysis", "Translation", "Sentiment Analysis", "About"]
    choice = st.sidebar.selectbox("Menu", activity)

    match choice:
        # TEXT ANALYSIS NLP
        case "Text Analysis":
            st.subheader("Text Analysis")
            st.write("")
            
            raw_text = st.text_area("Escreva aqui o texto para ser analisado", "Entre com um texto em inglês..", height=300)
            
            if st.button("Analisar!"):
                if len(raw_text) == 0:
                    st.warning("Entre com um texto para ser analisado..")
                else:
                    #blob = TextBlob(raw_text)
                    st.info("Funções Básicas")
                    
                    normalizador = Normaliser(tokenizer='readable', capitalize_inis=False, 
                          capitalize_pns=False, capitalize_acs=False, 
                          sanitize=True)
                    
                    text_normalized = normalizador.normalise(raw_text)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        with st.expander("Basic Information"):
                            st.write("Text Stats")
                            word_desc = nt.TextFrame(text_normalized).word_stats()
                            result_desc = {
                                "Length of text": word_desc['Length of Text'],
                                "Num of Vowels" : word_desc['Num of Vowels'],
                                "Num of Consonants" : word_desc['Num of Consonants'],
                                "Num of Stopwords" : word_desc['Num of Stopwords']
                                }
                            st.write(result_desc)
                            
                        with st.expander("Stopwords"):
                            st.success("Stopwords list:")
                            stop_w = nt.TextExtractor(text_normalized).extract_stopwords()
                            st.error(stop_w)
                            
                    with col2:
                        with st.expander("Processed Text"):
                            st.success("Stopwords Excluded Text")
                            processed_text = pre_process(text_normalized)
                            st.write(processed_text)
                            
                        with st.expander("Plot WordCloud"):
                            st.success("WordCloud")
                            wordcloud = WordCloud().generate(processed_text)
                            fig = plt.figure(1, figsize=(20, 10))
                            plt.imshow(wordcloud, interpolation='bilinear')
                            plt.axis('off')
                            st.pyplot(fig)
                            
                    st.write("")
                    st.write("")
                    st.info("Advanced Features")
                    
                    col3, col4 = st.columns(2)
                    
                    with col3:
                        with st.expander("Tokens&Lemmas"):
                            st.write("T&K")
                        
                            processed_text_ini = str(nt.TextFrame(text_normalized).remove_stopwords())
                            processed_text_mid = str(nt.TextFrame(processed_text_ini).remove_puncts())
                            #processed_text_fin = str(nt.TextFrame(processed_text_mid).remove_special_characters())
                        
                            tandl = text_analyzer(processed_text_mid)
                            
                            st.json(tandl)
                            
                    with col4:
                        with st.expander("Summarize"):
                            st.success("Summarization")
                            summary = summarize_text(text_normalized)
                            st.success(summary)
                    
                    
        # TRANSLATION           
        case "Translation":
            st.subheader("Translation")
            st.write("")
            st.write("")
            
            raw_text = st.text_area("Texto Original", "Escreva aqui um texto para ser traduzido..", height=200)
            if len(raw_text) < 3:
                st.warning("Por favor, escreva um texto maior que 3 letras...")
            else:
                target_lang = st.selectbox("Target language", ["Inglês", "Portuguese", "Espanhol", "Italiano"])
                
                match target_lang:
                    case "Inglês":
                        target_lang = 'en'
                    case "Portuguese":
                        target_lang = 'pt'
                    case "Espanhol":
                        target_lang = 'es'
                    case "Italiano":
                        target_lang = 'it'

                if st.button("Traduzir"):
                    translator = GoogleTranslator(source='auto', target=target_lang)
                    translated_text = translator.translate(raw_text)
                    st.write(translated_text)
                  
        # SENTIMENT ANALYSIS  
        case "Sentiment Analysis":
            st.subheader("Sentiment Analysis")
            st.write("")
            st.write("")
            
            raw_text = st.text_area("Texto para Analisar", "Escreva aqui um texto para ser analisado..", height=200)

            if st.button("Avaliar"):
                if len(raw_text) == 0:
                    st.warning("Escreva um texto...")
                else:
                    blob = TextBlob(raw_text)
                    st.info("Análise de Sentimento:")
                    st.write(blob.sentiment)
                    st.write("")
                    
        # QUICK ABOUT
        case "About":
            st.subheader("About")
            st.write("")
            
            st.markdown("""
            # NLP Web App made with Streamlit
            
            for more info drop me a line bakudas(at)vacaroxa.com
            """)
            
        # DEFAULT PASS
        case _:
            pass
            

if __name__ == "__main__":
    main()