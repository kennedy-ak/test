import streamlit as st
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers.utils.logging import set_verbosity_error

# Suppress verbose logging
set_verbosity_error()

def load_models():
    # Display loading message
    with st.spinner('Loading models... This may take a minute.'):
        # Initialize the summarization models
        summary_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")
        summarizer = HuggingFacePipeline(pipeline=summary_pipeline)
        
        refinement_pipeline = pipeline("summarization", model="facebook/bart-large")
        refiner = HuggingFacePipeline(pipeline=refinement_pipeline)
        
        # Initialize the Q&A model
        qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")
        
        # Create the summarization chain
        summary_template = PromptTemplate.from_template("Summarize the following text in {length} way: \n\n{text}")
        summarization_chain = summary_template | summarizer | refiner
        
        return summarization_chain, qa_pipeline

def main():
    st.title("📝 Advanced Text Summarization")
    st.write("This app summarizes text and allows you to ask questions about the summary.")
    
    # Initialize or get the models
    if 'models_loaded' not in st.session_state:
        st.session_state.summarization_chain, st.session_state.qa_pipeline = load_models()
        st.session_state.models_loaded = True
    
    # Initialize session state variables for summary if not already there
    if 'summary' not in st.session_state:
        st.session_state.summary = None
    
    # Text input area for the user
    text_to_summarize = st.text_area("Enter the text to summarize:", height=200)
    
    # Select the length of the summary
    length_options = {"Short": "short", "Medium": "medium", "Long": "long"}
    length = st.selectbox("Select summary length:", options=list(length_options.keys()))
    
    # Button to generate summary
    if st.button("Generate Summary"):
        if text_to_summarize:
            with st.spinner('Generating summary...'):
                # Map length to token count
                length_map = {"short": 50, "medium": 150, "long": 300}
                max_length = length_map.get(length_options[length].lower(), 150)
                
                # Generate the summary
                response = st.session_state.summarization_chain.invoke({
                    "length": length_options[length], 
                    "text": text_to_summarize
                })
                
                # Store the summary in session state
                st.session_state.summary = response
                
                # Display the summary
                st.subheader("Summary:")
                st.write(response)
        else:
            st.error("Please enter some text to summarize.")
    
    # Display the summary if it exists
    if st.session_state.summary:
        st.subheader("Summary:")
        st.write(st.session_state.summary)
        
        # Question answering section
        st.subheader("Ask a question about the summary:")
        question = st.text_input("Your question:")
        
        if st.button("Get Answer"):
            if question:
                with st.spinner('Finding answer...'):
                    qa_results = st.session_state.qa_pipeline(
                        question=question, 
                        context=st.session_state.summary
                    )
                    
                    # Display the answer
                    st.subheader("Answer:")
                    st.write(qa_results["answer"])
            else:
                st.error("Please enter a question.")

if __name__ == "__main__":
    main()