# import streamlit as st
# from transformers import pipeline
# from langchain_huggingface import HuggingFacePipeline
# from langchain.prompts import PromptTemplate
# from transformers.utils.logging import set_verbosity_error

# # Suppress verbose logging
# set_verbosity_error()

# def load_models():
#     # Display loading message
#     with st.spinner('Loading models... This may take a minute.'):
#         # Initialize the summarization models
#         summary_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")
#         summarizer = HuggingFacePipeline(pipeline=summary_pipeline)
        
#         refinement_pipeline = pipeline("summarization", model="facebook/bart-large")
#         refiner = HuggingFacePipeline(pipeline=refinement_pipeline)
        
#         # Initialize the Q&A model
#         qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")
        
#         # Create the summarization chain
#         summary_template = PromptTemplate.from_template("Summarize the following text in {length} way: \n\n{text}")
#         summarization_chain = summary_template | summarizer | refiner
        
#         return summarization_chain, qa_pipeline

# def main():
#     st.title("📝 Advanced Text Summarization")
#     st.write("This app summarizes text and allows you to ask questions about the summary.")
    
#     # Initialize or get the models
#     if 'models_loaded' not in st.session_state:
#         st.session_state.summarization_chain, st.session_state.qa_pipeline = load_models()
#         st.session_state.models_loaded = True
    
#     # Initialize session state variables for summary if not already there
#     if 'summary' not in st.session_state:
#         st.session_state.summary = None
    
#     # Text input area for the user
#     text_to_summarize = st.text_area("Enter the text to summarize:", height=200)
    
#     # Select the length of the summary
#     length_options = {"Short": "short", "Medium": "medium", "Long": "long"}
#     length = st.selectbox("Select summary length:", options=list(length_options.keys()))
    
#     # Button to generate summary
#     if st.button("Generate Summary"):
#         if text_to_summarize:
#             with st.spinner('Generating summary...'):
#                 # Map length to token count
#                 length_map = {"short": 50, "medium": 150, "long": 300}
#                 max_length = length_map.get(length_options[length].lower(), 150)
                
#                 # Generate the summary
#                 response = st.session_state.summarization_chain.invoke({
#                     "length": length_options[length], 
#                     "text": text_to_summarize
#                 })
                
#                 # Store the summary in session state
#                 st.session_state.summary = response
                
#                 # Display the summary
#                 st.subheader("Summary:")
#                 st.write(response)
#         else:
#             st.error("Please enter some text to summarize.")
    
#     # Display the summary if it exists
#     if st.session_state.summary:
#         st.subheader("Summary:")
#         st.write(st.session_state.summary)
        
#         # Question answering section
#         st.subheader("Ask a question about the summary:")
#         question = st.text_input("Your question:")
        
#         if st.button("Get Answer"):
#             if question:
#                 with st.spinner('Finding answer...'):
#                     qa_results = st.session_state.qa_pipeline(
#                         question=question, 
#                         context=st.session_state.summary
#                     )
                    
#                     # Display the answer
#                     st.subheader("Answer:")
#                     st.write(qa_results["answer"])
#             else:
#                 st.error("Please enter a question.")

# if __name__ == "__main__":
#     main()


import streamlit as st
from transformers import pipeline
from transformers.utils.logging import set_verbosity_error

# Suppress verbose logging
set_verbosity_error()

@st.cache_resource(show_spinner=False)
def load_models():
    """Load and cache Hugging Face models"""
    with st.spinner('Loading AI models... This may take 1-2 minutes.'):
        return {
            'summarizer': pipeline("summarization", model="facebook/bart-large-cnn"),
            'qa': pipeline("question-answering", model="deepset/roberta-base-squad2")
        }

def generate_summary(text, length):
    """Generate summary with length-appropriate parameters"""
    length_params = {
        'short': {'max_length': 100, 'min_length': 30},
        'medium': {'max_length': 250, 'min_length': 80},
        'long': {'max_length': 400, 'min_length': 120}
    }
    params = length_params[length.lower()]
    return models['summarizer'](
        text,
        max_length=params['max_length'],
        min_length=params['min_length'],
        do_sample=False,
        truncation=True
    )[0]['summary_text']

def main():
    st.title("📝 Advanced Text Summarization & Analysis")
    st.markdown("""
    **Key Features:**
    - AI-powered text summarization
    - Interactive Q&A about summaries
    - Adjustable summary length
    - Persistent session state
    """)
    
    # Initialize models and session state
    if 'models' not in st.session_state:
        st.session_state.models = load_models()
    
    if 'summary' not in st.session_state:
        st.session_state.summary = None

    # Main input section
    with st.form("input_form"):
        text = st.text_area("Paste your text here:", height=250,
                           placeholder="Enter at least 3-4 paragraphs for best results...")
        length = st.selectbox("Summary length:", 
                             ["Short", "Medium", "Long"], index=1)
        
        if st.form_submit_button("Generate Summary"):
            if len(text.split()) < 50:
                st.error("Please enter at least 50 words for meaningful summarization.")
            else:
                try:
                    with st.spinner('Analyzing text and generating summary...'):
                        st.session_state.summary = generate_summary(text, length)
                except Exception as e:
                    st.error(f"Error generating summary: {str(e)}")

    # Display summary
    if st.session_state.summary:
        st.subheader("Generated Summary")
        st.success(st.session_state.summary)
        
        # Q&A Section
        st.divider()
        st.subheader("Text Analysis Q&A")
        
        question = st.text_input("Ask about the content:", 
                                placeholder="What was the main conclusion?")
        if question and st.button("Get Answer"):
            try:
                result = st.session_state.models['qa'](
                    question=question,
                    context=st.session_state.summary
                )
                st.markdown(f"**Answer:** {result['answer']}")
                st.caption(f"Confidence score: {result['score']:.2f}")
            except ValueError:
                st.error("The generated summary is too short to answer this question.")
            except Exception as e:
                st.error(f"Error processing question: {str(e)}")

if __name__ == "__main__":
    main()
