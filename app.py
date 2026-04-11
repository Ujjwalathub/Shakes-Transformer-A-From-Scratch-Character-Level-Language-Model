"""
Streamlit Frontend for Shakes-Transformer
Provides an interactive web interface for next-word prediction using the trained model
"""
import streamlit as st
import torch
from pathlib import Path

from scripts.transformer_model import TransformerModel
from scripts.data_pipeline import WordTokenizer
from scripts.inference import InferenceEngine

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
project_root = Path(__file__).parent

# ============================================================================
# Resource Caching - Load model and tokenizer once
# ============================================================================
@st.cache_resource
def load_resources():
    """
    Load tokenizer and model into memory.
    Resources are cached so they persist across interactions.
    """
    with st.spinner('🎭 Loading the Bard\'s wisdom...'):
        # Load Tokenizer
        tokenizer = WordTokenizer(vocab_size=4000)
        
        # Build vocabulary from Shakespeare text
        with open(project_root / 'data' / 'shakespeare.txt', 'r', encoding='utf-8') as f:
            text = f.read()
        tokenizer.build_vocabulary(text)
        
        # Initialize and Load Model
        vocab_size = len(tokenizer.word2idx)
        model = TransformerModel(
            vocab_size=vocab_size,
            d_model=128,
            num_heads=4,
            d_hidden=512,
            num_layers=4,
            seq_length=32,
            dropout=0.1
        ).to(device)
        
        # Load best checkpoint
        checkpoint_path = project_root / 'checkpoints' / 'best_model.pt'
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
        model.eval()
        
        return tokenizer, model


# ============================================================================
# Page Configuration
# ============================================================================
st.set_page_config(
    page_title="Shakes-Transformer",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# UI Design
# ============================================================================
st.title("🎭 Shakes-Transformer")
st.subheader("Next-Word Prediction in the Style of the Bard")
st.markdown("---")

# Load resources
tokenizer, model = load_resources()
inference_engine = InferenceEngine(model, tokenizer, device=device)

# ============================================================================
# Sidebar - Configuration
# ============================================================================
st.sidebar.header("⚙️ Configuration")

# Temperature slider for prediction creativity
temperature = st.sidebar.slider(
    "🌡️ Temperature (Creativity)",
    min_value=0.1,
    max_value=2.0,
    value=1.0,
    step=0.1,
    help="Higher temperature = more creative/random predictions. Lower = more deterministic."
)

# Number of predictions to show
num_predictions = st.sidebar.radio(
    "📊 Number of Predictions",
    options=[3, 5, 10],
    index=1,
    help="How many top predictions to display"
)

st.sidebar.markdown("---")
st.sidebar.info(
    "💡 **Model Info:**\n"
    f"- Vocab Size: {len(tokenizer.word2idx)}\n"
    f"- Max Sequence Length: {model.seq_length}\n"
    f"- Device: {device}\n"
    f"- Model: 4-layer Transformer"
)

# ============================================================================
# Main Interface
# ============================================================================
# Input section
col1, col2 = st.columns([4, 1])

with col1:
    user_prompt = st.text_input(
        "📝 Enter a Shakespearean phrase:",
        value="To be or not to",
        placeholder="Type your prompt and press Enter...",
        label_visibility="collapsed"
    )

with col2:
    predict_button = st.button("🔮 Predict", width='stretch')

# Input validation and warning
if user_prompt:
    num_tokens = len(tokenizer.encode(user_prompt))
    if num_tokens > model.seq_length:
        st.warning(
            f"⚠️ **Sequence Length Warning:** Your input has {num_tokens} tokens, "
            f"but the model was trained with max_seq_length={model.seq_length}. "
            f"Only the last {model.seq_length} tokens will be used."
        )

# ============================================================================
# Prediction Logic
# ============================================================================
if predict_button and user_prompt:
    with st.spinner('The Muse is thinking...'):
        try:
            # Get top-K predictions using InferenceEngine
            predictions = inference_engine.predict_top_k(user_prompt, k=num_predictions)
            
            # Display results in multiple formats
            st.success("✅ Predictions generated successfully!")
            st.markdown("---")
            
            # 1. Table display
            st.subheader(f"🎯 Top {num_predictions} Predicted Words")
            
            results_table = []
            for rank, (word, confidence) in enumerate(predictions, 1):
                results_table.append({
                    "Rank": rank,
                    "Word": word,
                    "Confidence": f"{confidence:.2%}"
                })
            
            st.dataframe(
                results_table,
                width='stretch',
                hide_index=True
            )
            
            # 2. Bar chart visualization
            st.subheader("📊 Probability Distribution")
            
            words = [p[0] for p in predictions]
            confidences = [p[1] * 100 for p in predictions]
            
            chart_data = {
                "Word": words,
                "Confidence (%)": confidences
            }
            
            st.bar_chart(
                data=chart_data,
                x="Word",
                y="Confidence (%)",
                width='stretch'
            )
            
            # 3. Top prediction highlight
            st.markdown("---")
            top_word, top_confidence = predictions[0]
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("🏆 Top Prediction", top_word, f"{top_confidence:.2%}")
            
            with col2:
                st.metric("📈 Entropy", f"{-sum(p * (p+1e-10) for _, p in predictions):.3f}")
            
            with col3:
                st.metric("🎲 Tokens", len(tokenizer.encode(user_prompt)))
            
        except Exception as e:
            st.error(f"❌ Error during prediction: {str(e)}")
            st.write("Please check your input and try again.")

else:
    if predict_button:
        st.warning("📝 Please enter a prompt to get predictions!")

# ============================================================================
# Footer / Instructions
# ============================================================================
st.markdown("---")
st.markdown(
    """
    ### 📖 Instructions
    1. **Enter a Prompt:** Type a phrase inspired by Shakespeare's works
    2. **Adjust Temperature:** Use the sidebar to control prediction creativity
    3. **Get Predictions:** Click the "🔮 Predict" button to see the top predicted next words
    4. **Interpret Results:** The confidence % shows how likely each word is to follow your prompt
    
    ### 🎭 Examples to Try
    - "To be or not to"
    - "All the world's a"
    - "Friends romans countrymen"
    - "What light through yonder"
    """
)

st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 12px; margin-top: 40px;'>
    Built with ❤️ using Streamlit | Transformer Model trained on Shakespeare's Complete Works
    </div>
    """,
    unsafe_allow_html=True
)
