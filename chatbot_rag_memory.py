"""
problem ismi: Akıllı Müşteri Destek Sistemi
            - müşteriler sık sık benzer soruları sorarlar:
                 - şifremi unuttum
                 - faturamı nereden alabilirim?
                 - iade süresi kaç gün
                 - yutdışına gönderim yapıyor musunuz?
            - çözüm:
                - .pdf dosyasını (yani sıkça sorulan sorular) vektör veritabanına dönüştür
                - kullanıcıdan gelen sorular veritabanında sorgulanır ve llm türkçe cevaplar üretir
                - 

Kullanılan teknolojiler:
    - Langchain : rag mimarisi kurmak için
    - faiss: embeddingleri saklamak için hızlı bir vektör veri tabanı
    - LLM (Gemini, HuggingFace, vb.): soru cevap için
    - streamlit: web arayüzü, son kullanıcı ile interaktif kullancıcı deneyimi 

    (python-dotenv: .env dosyasından api key çekmek için kullanılan kütpühane)
    (faiss: vektör database için kullanılan kütüphane)
    (langchain: rag mimarisi için kullanılan kütüphane(bunların hepsini bir zincir haline getirir) 
    (streamlit: web arayüzü için kullanılan kütüphane)


veri seti:
    - Soru: Yurtdışı satışlarınız bulunuyor mu?
    - Cevap: Hayır

    - Soru: Faturamı nereden alabilirim?
    - Cevap: Faturanız 3 iş günü içinde teslim edilecektir.

plan program:
    - Müşteri SSS bilgilerini içeren bir PDF dosyası oluştur
    - Kullanıcı bu dosyasyı arayüzden yükeleyecek
    - pdf metni parçaya ayrılacak ve embeddingler çıkartılacak
    - kullanıcı soru sorduğunda vektör db den benzer içerikler getirilir, LLM ile cevap oluşturulur
    - Konuşma geçmişi memort ile saklanır ve sonraki yanıtlara bağlam oluşturulur


install libraires: frezee
(requirements.txt dosyası ile kütüphaneleri dondurur(kütüphanelerin versiyonları ile birlikte))
- pip freeze >  requirement.txt

"""

from langchain_google_genai import ChatGoogleGenerativeAI #openai destekli llm modelleri için
from langchain.chains import ConversationalRetrievalChain #RAG + sohbet zinciri için
from langchain_community.vectorstores import FAISS #faiss vektör veritabanı
from langchain_huggingface import HuggingFaceEmbeddings # metni vektörleştrmek için embedding modeli
from langchain.memory import ConversationBufferMemory #sohbet geçmişini tutan hafıza yapısı

from dotenv import load_dotenv
import os

load_dotenv()  # ortam değişkenlerini .env dosyasından yükle
api_key = os.getenv("GEMINI_API_KEY")  # API anahtarını alır
if not api_key:
    raise ValueError("GEMİNİ_API_KEY is not found.")

os.environ["GOOGLE_API_KEY"] = api_key  # GOOGLE_API_KEY çevresel değişkenini ayarla

# embedding modelini başlat(text -> vektör dönüşümü)
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

#daha önce oluşturulmuş vektör veritabanını yükle
vectordb = FAISS.load_local(
       "sss_vectordb", #kaydedilmiş vektör veritabanı klasoru
       embedding,   #embedding modeli
       allow_dangerous_deserialization=True #pickle güvelik uyarisi bastırma
)

# konusma geçmişini tutmak için memory oluşturma
memory = ConversationBufferMemory(
    memory_key = "chat_history",  # konusma geçmişi bu anahtar ile saklanır
    return_messages= True # geçmiş mesajlar tam haliyle geri döner

)

# sıfır rastlantısallık ile çalışır, sabit cevaplar verir
llm= ChatGoogleGenerativeAI(
    model = "gemini-1.5-flash", #  kullanılan dil modeli
    temperature = 0  # determistinik (aynı girdiye aynı çıktıyı verir (kararlılık derecesi))
) 

# rag + memory zincir oluştur
# - llm
# - faiss retriever: en benzer 3 bege getirilsin (k=3)
# - memory

qa_chain = ConversationalRetrievalChain.from_llm(
    llm = llm,
    retriever = vectordb.as_retriever(search_kwargs ={"k": 3}),     
    memory = memory,
    verbose = True
)

print("Destek Botuna Hoş Geldiniz !")
while True:
    #
    user_input = input("Siz: ")
    if user_input.lower == "çık":
        break
    
    # kullanıcı sorusu + llm + rag + memory zincirine verilir
    response = qa_chain.run(user_input)
    print("Destek Botu: ", response)