import streamlit as st
from predict import predict_news

def main():
    # Set page configuration parameters
    st.set_page_config(
        page_title="Fake News Detector",
        page_icon="📰",
        layout="centered"
    )

    # Basic CSS Styling
    # Added CSS variables to format FAKE news warning as RED and REAL news as GREEN
    st.markdown("""
        <style>
        .main {
            background-color: #f5f7f9;
        }
        .title-text {
            color: #1E3A8A;
            text-align: center;
            font-family: 'Helvetica Neue', sans-serif;
            font-weight: bold;
        }
        .fake-box {
            background-color: #fee2e2;
            border-left: 5px solid #ef4444;
            padding: 20px;
            border-radius: 5px;
            margin-top: 20px;
        }
        .real-box {
            background-color: #dcfce3;
            border-left: 5px solid #22c55e;
            padding: 20px;
            border-radius: 5px;
            margin-top: 20px;
        }
        .result-text {
            font-size: 24px;
            font-weight: bold;
            margin: 0;
        }
        </style>
    """, unsafe_allow_html=True)

    # Application Title
    st.markdown("<h1 class='title-text'>📰 Fake News Detector</h1>", unsafe_allow_html=True)
    st.markdown("### Paste a news article below to check its authenticity.", unsafe_allow_html=True)

    # Text area for user input where article can be pasted
    news_input = st.text_area("News Article Content:", height=200, placeholder="Paste your news text here...")

    # Action button to trigger model classification
    if st.button("Check News", type="primary"):
        # Ensure the user has inputted text
        if not news_input.strip():
            st.warning("Please enter some text to check.")
        else:
            with st.spinner("Analyzing text..."):
                try:
                    # Get prediction using our predict.py script
                    label, confidence = predict_news(news_input)
                    
                    # Display the appropriate formatted box with Result and Confidence Score
                    if label == "FAKE":
                        st.markdown(f"""
                            <div class="fake-box">
                                <p class="result-text" style="color:#b91c1c;">🚨 FAKE NEWS DETECTED!</p>
                                <p style="color:#7f1d1d; margin-top:5px;">Confidence: {confidence * 100:.2f}%</p>
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                            <div class="real-box">
                                <p class="result-text" style="color:#15803d;">✅ REAL NEWS DETECTED!</p>
                                <p style="color:#166534; margin-top:5px;">Confidence: {confidence * 100:.2f}%</p>
                            </div>
                        """, unsafe_allow_html=True)
                        
                except FileNotFoundError:
                    # Display this error if train.py hasn't successfully generated the .pkl files
                    st.error("Model files not found! Please ensure you have run `python train.py` first.")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")

    # App footer
    st.markdown("---")
    st.caption("Developed with ❤️ using Python and Streamlit. Uses Multinomial Naive Bayes model with TF-IDF vectorization.")

if __name__ == "__main__":
    main()
