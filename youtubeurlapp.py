import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader

# Streamlit app
st.set_page_config(page_title="Langchain: Summarize Text from YT or Website", page_icon="smile")
st.title("Langchain: Summarize Text from YT or Website")
st.subheader("Summarize URL")
####
# Get the Groq API key and URL field
# with st.sidebar:
#     groq_api_key = st.text_input("GROQ API key", value="", type="password")

groq_api_key="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"

url = st.text_input("URL", label_visibility="collapsed")

from langchain_groq import ChatGroq
llm = ChatGroq(model="Gemma-7b-It", api_key=groq_api_key)

# Setting the prompt template
prompt_template = """
Provide the summary of the following content in 300 words:
Content: {text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=['text'])

# Creating the button
if st.button("Summarize the content from the YT or URL"):
    if not groq_api_key.strip() or not url.strip():
        st.error("Please enter the API key and URL")
    elif not validators.url(url):
        st.error("Please enter a valid URL")
    else:
        try:
            with st.spinner("Waiting...."):
                # Load the website or YT video data
                if "youtube.com" in url:
                    loader = YoutubeLoader.from_youtube_url(url,add_video_info=True,language="en-IN")
                else:
                    loader = UnstructuredURLLoader(
                        urls=[url],
                        ssl_verify=False,
                        headers={
                            "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"
                        }
                    )
                docs = loader.load()

                # Creating the chain for summarization
                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                output_summary = chain.run(docs)

                st.success(output_summary)
        except Exception as e:
            st.exception(f"Exception: {e}")
