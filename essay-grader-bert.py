import os
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import spacy
import language_tool_python
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity

# Environment Configuration
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Streamlit theme configuration (Dark theme with Sans Serif font)
st.set_page_config(page_title="Essay Grader", layout="wide")

# Constants and Model Names
BERT_MODEL_NAME = 'bert-base-uncased'
WRITING_STYLES = ['Formal', 'Technical', 'Creative', 'General', 'Academic']
STYLE_SENSITIVITY = {'Formal': 0.9, 'Technical': 1.0, 'Creative': 1.2, 'General': 1.0, 'Academic': 0.8}

# Initialize NLP Models and Tools
@st.cache_resource
def load_models():
    nlp = spacy.load('en_core_web_sm')
    language_tool = language_tool_python.LanguageTool('en-US')
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    sentiment_analyzer = SentimentIntensityAnalyzer()
    bert_tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    bert_model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL_NAME, num_labels=2)
    return nlp, language_tool, sentence_model, sentiment_analyzer, bert_tokenizer, bert_model

nlp, language_tool, sentence_model, sentiment_analyzer, bert_tokenizer, bert_model = load_models()

# Essay Scoring Functions
def get_grammar_score(text):
    matches = language_tool.check(text)
    grammar_score = max(0, 100 - 10 * len(matches))
    grammar_feedback = [f"- {match.message} (at: '{match.context}')" for match in matches[:5]]
    return grammar_score, " ".join(grammar_feedback) if grammar_feedback else "No significant grammar issues found."

def get_coherence_score(text):
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    embeddings = sentence_model.encode(sentences, convert_to_tensor=True)
    cosine_similarities = util.pytorch_cos_sim(embeddings, embeddings)
    mean_similarity = cosine_similarities.triu(diagonal=1).mean().item() * 100
    feedback = "Good logical flow between sentences." if mean_similarity > 70 else "Consider improving the logical flow and transitions between sentences."
    return mean_similarity, feedback

def get_semantic_score(text):
    inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    outputs = bert_model(**inputs)
    max_logits = outputs.logits.softmax(dim=-1).max().item() * 100
    feedback = "The essay demonstrates good depth and complexity." if max_logits > 50 else "The essay could benefit from more in-depth analysis and content richness."
    return max_logits, feedback

def get_sentiment_score(text, writing_style):
    sentiment = sentiment_analyzer.polarity_scores(text)
    base_score = sentiment['compound'] * 100
    adjusted_score = base_score * STYLE_SENSITIVITY[writing_style]
    feedback = "The tone is well-suited to the intended writing style." if adjusted_score > 50 else "Consider adjusting the tone to better suit the intended writing style."
    return adjusted_score, feedback

# Streamlit Interface
def main():
    st.title("Essay Grader")

    # User Inputs
    writing_style = st.selectbox("Select the writing style:", WRITING_STYLES)
    essay_text = st.text_area("Paste your essay here:", height=300)

    if st.button('Grade Essay'):
        with st.spinner('Analyzing...'):
            try:
                # Perform analysis
                grammar_score, grammar_feedback = get_grammar_score(essay_text)
                coherence_score, coherence_feedback = get_coherence_score(essay_text)
                semantic_score, semantic_feedback = get_semantic_score(essay_text)
                sentiment_score, sentiment_feedback = get_sentiment_score(essay_text, writing_style)

                # Compile scores
                scores = {
                    'Grammar': grammar_score,
                    'Coherence': coherence_score,
                    'Sentiment': sentiment_score,
                    'Semantic Content': semantic_score
                }

                # Ensure scores do not exceed 100%
                corrected_scores = {key: min(100, value) for key, value in scores.items()}

                # Display Score Summary
                st.write("### Score Summary")
                for aspect, score in corrected_scores.items():
                    st.metric(label=aspect, value=f"{score:.2f}%")

                # Visualization (Pie Chart)
                fig, ax = plt.subplots()
                ax.pie(corrected_scores.values(), labels=corrected_scores.keys(), autopct='%1.1f%%', startangle=90)
                ax.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.
                st.pyplot(fig)

                # Detailed Feedback
                st.write("### Detailed Feedback")
                st.write("**Grammar Feedback:**", grammar_feedback)
                st.write("**Coherence Feedback:**", coherence_feedback)
                st.write("**Semantic Content Feedback:**", semantic_feedback)
                st.write("**Sentiment Feedback:**", sentiment_feedback)

            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")


if __name__ == '__main__':
    main()