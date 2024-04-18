# Description: Streamlit app to interact with the PDF using RAG with Text/Voice Inputs and get Voice Output

# Parameters for the LlamaCpp (LLM + Embeddings)
n_gpu_layers = 1  # Metal set to 1 is enough.
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.
use_mlock = True  # Force system to keep model in RAM.
model_path="Models/puma-3b.q4_1.gguf" # Make sure the model path is correct for your system!

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

from RAG.CustomEmbedding import SentenceTransformerEmbeddingFunction # type: ignore
embedding = SentenceTransformerEmbeddingFunction(model_name='sentence-transformers/msmarco-bert-base-dot-v5',device="mps")

def load_LLM(model_path):
    # Load the LLM
    from langchain.llms import LlamaCpp
    llm = LlamaCpp(
        model_path=model_path,
        n_gpu_layers=n_gpu_layers,
        n_batch=n_batch,
        n_ctx=4096,
        f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
        callback_manager=callback_manager,
        verbose=True,
        use_mlock = True
    ) # type: ignore
    return llm

# Load the vectorstore
from langchain.vectorstores import Chroma
from chromadb.config import Settings # type: ignore
vectorstore = Chroma(collection_name="edustore",embedding_function=embedding,
                     client_settings=Settings(chroma_db_impl="duckdb+parquet",
                                              persist_directory="RAG/Vectorstores/.chromaVS"))

#Make a prompt template for the adapting the QA Chain to the Situation
from langchain.prompts import PromptTemplate
prompt_template = """You are EVA - a Educational Chatbot. Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.
{context}
Question: {question}
Helpful Answer:"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

llm = load_LLM(model_path)

import time
import streamlit as st
from langchain.chains import RetrievalQA
qa_chain = RetrievalQA.from_chain_type(llm,
            retriever=vectorstore.as_retriever(),
            chain_type_kwargs={"prompt": PROMPT},
            verbose=True,)

# Function to perform voice search using the microphone
import speech_recognition as sr
def perform_voice_search():
    recognizer = sr.Recognizer()
    
    with sr.Microphone() as source:
        st.info("Listening... Speak now!")
        try:
            audio = recognizer.listen(source, timeout=5)
            st.info("Processing...")
            recognized_text = recognizer.recognize_google(audio)
            st.success("Recognition successful! You said: " + recognized_text)
            return recognized_text
        except sr.WaitTimeoutError:
            st.warning("Listening timed out. Try again.")
            return None
        except sr.UnknownValueError:
            st.error("Sorry, I could not understand the audio.")
            return None
        except sr.RequestError as e:
            st.error(f"Could not request results; {e}")
            return None
        
import os
import signal
user_question = None

def main():
    st.set_page_config(page_title="EVA: Edubot 💬", page_icon="chart_with_upwards_trend")
    st.header("EVA: Your very own Edubot 💬")
    st.subheader("Chat with EVA and get a personalized response!")

    video_process = None  # Initialize video_process variable
    prior_question = None  # Initialize prior_question variable

    # Define a column layout
    prompt_col1, prompt_col2 = st.columns([0.9, 0.1])
    # Text input box
    with prompt_col1:
        user_question = st.text_input(label="Ask your queries to EVA...",placeholder="Ask your queries to EVA...", label_visibility="collapsed")
    # Button
    with prompt_col2:
        voice_search_button = st.button("🎤")
    if voice_search_button:
        user_question = perform_voice_search()

    col_opt1, col_opt2, col_opt3 = st.columns([0.3, 0.3, 0.3])
    with col_opt1:
        # Checkbox to enable RAG Transcript (default: enabled)
        enable_transcript = st.checkbox("Enable EVA Transcript", value=True)
    with col_opt2:
        # Checkbox to enable Video response (default: enabled)
        enable_video_response = st.checkbox("Enable Video Response", value=True)
    with col_opt3:
        # Checkbox to enable Video enhancement (default: enabled)
        video_enhancement = st.checkbox("Enable Video Enhancement", value=True)

    if user_question and user_question != prior_question:
        # If there was a previous question, kill video_process and set prior_question
        if video_process is not None:
            os.killpg(os.getpgid(video_process.pid), signal.SIGTERM) #Doesnt Work still
        # Set last_user_question to the current user_question
        prior_question = user_question

        try:
            response = qa_chain({"query": user_question})
            if not enable_video_response:
                st.success("EVA: "+response["result"], icon="🤖")

            else:
                expander = st.expander("Transcript", expanded=False)
                expander.success("EVA: "+response["result"], icon="🤖")
                # Google TTS API
                from gtts import gTTS
                myobj = gTTS(text=response["result"], lang='en', slow=False)

                # Saving the converted audio in a mp3 file named with timestr
                timestr = time.strftime("%Y%m%d-%H%M%S")
                filepath = "Output/Audio/temp/QVS_Out_"+timestr+".wav"
                myobj.save(filepath)

                # Lip Syncing the audio with the video
                import subprocess

                # Define input paths and output path
                inputAudioPath = filepath
                inputVideoPath = 'Input_Vids/MO.mp4'
                lipSyncedOutputPath = 'Output/Video/result_' + timestr + '.mp4'

                # Construct the command as a list of arguments
                Wav2Lipcommand = [
                    "python",
                    "Wav2Lip/Wav2Lip-master/inference.py",
                    "--checkpoint_path",
                    "Wav2Lip/Wav2Lip-master/checkpoints/wav2lip.pth",
                    "--face", inputVideoPath,
                    "--audio", inputAudioPath,
                    "--outfile", lipSyncedOutputPath
                ]

                # Execute the command using subprocess.run()
                video_process = subprocess.run(Wav2Lipcommand, check=True)
                
                # Display Video and a message indicating where to check the saved video response
                st.video(lipSyncedOutputPath)
                st.write("You can check out the saved Video response at ", lipSyncedOutputPath)

        except Exception as e:
            #st.error(f"An error occurred: {e}")
            print("Video couldnt process due to multiple queries")


if __name__ == "__main__":
    main()


# Sample Questions
# what's the eligibility criteria for Gen C Pro role ?
# what are the technologies used for Gen C Pro role ?
# Where is ACM Winter School being conducted in 2023 ? 
