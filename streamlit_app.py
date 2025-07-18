import streamlit as st # web arayüzü oluşturmak için bir python kütüphanesi

from langchain_google_genai import ChatGoogleGenerativeAI #langchain google modulleri: chat llm 
from langchain_huggingface import HuggingFaceEmbeddings


from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

from dotenv import load_dotenv
import os 
import tempfile # geçici dosya işlemleri için

# .env dosyasından yükleme işlemlerini gereçekleştir ilk olarak load_dotenv
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY") 

# streamlit sayfa başlığını ve ikonunu ayarla 
st.set_page_config(page_title= "Müşteri Destek Botu", page_icon= "🤖")  # streamlit set page yani sayfa ayarı yapılan konfigürasyon dosyası
st.title("PDF Destek Botu (RAG + Memory)")
st.write("Bir pdf yükleyin, içeriğine dair sorular sorun. Türkçe desteklidir.")

# pdf yükleme bileşeni
uploaded_file = st.file_uploader("PDF dosyanızı yükleyin.", type= "pdf", key= "pdf_uploader")

# eğer kullanıcı yeni bir pdf yüklediyse ve daha önce yuklenen ile aynı değilse
if uploaded_file is not None:
    if "last_uploaded_name" not in st.session_state or uploaded_file.name != st.session_state.last_uploaded_name:
        #kullanıcya işleniyor bilgisi gönderelim
        with st.spinner("PDF işleniyor..."):
            #yüklenen pdf i gecici bir dosyaya yazdır
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name # geçici dosyanın yolu 
            # PyPDfLoader ile pdf içeriğini yükle
            loader = PyPDFLoader(tmp_path)
            documents = loader.load()

            # metinleri parçala yani chunklara böl
            splitter = RecursiveCharacterTextSplitter(chunk_size= 500, chunk_overlap= 50)
            docs = splitter.split_documents(documents)

            # HuggingfaceEmbedding ile vektörleştirme 
            embedding = HuggingFaceEmbeddings(model_name ="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

            #FAISS ile vektör veri tabanı
            vector_db  = FAISS.from_documents(docs, embedding)

            # memory ve gemini 1.5 ile llm oluşturma
            memory = ConversationBufferMemory(memory_key = "chat_history", return_messages=True)
            llm = ChatGoogleGenerativeAI(model = "gemini-1.5-flash", temperature = 0)

            # rag + memory zincirini oluşturma
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm = llm,
                retriever = vector_db.as_retriever(search_kwargs = {"k": 3}),
                memory = memory 
            )

            st.session_state.qa_chain = qa_chain
            st.session_state.chat_history = []
            st.session_state.last_uploaded_name = uploaded_file.name # aynı dosyanın yeniden işlenmesini engellemek için    

        st.success("Pdf başarıyla işlendi.")      

if "qa_chain" in st.session_state:  # eğer pdf işlendiyse 
    #kullanıcın sorusunu alır
    user_question = st.text_input ("💁🏻 Sorunuzu yazınız: ")

    if user_question:
        response = st.session_state.qa_chain.invoke(user_question) # langchain zincirine soruyu gönder
        st.session_state.chat_history.append(("💁🏻", user_question))  # kullanıcı mesajını geçmişe ekleme
        st.session_state.chat_history.append(("🤖", response["answer"])) # model yanıtını historye ekleme

    if st.session_state.chat_history:
        st.subheader("📋 Sohbet Geçmişi")
        for sender, msg in st.session_state.chat_history:
            st.markdown(f"**{sender}**: {msg}" )