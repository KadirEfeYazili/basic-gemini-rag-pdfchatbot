"""
embeddingleri oluşturma ve bunları vektör veritabanına kaydetme işlemlerini gerçekleştir
"""
from langchain_community.embeddings import HuggingFaceEmbeddings #langchain huggingface tabanlı vektör temsil modeli
from langchain_community.vectorstores import FAISS #faiss kütüphanesini kullanarak vektörleri depolayacağız
from langchain.document_loaders import PyPDFLoader #pdf dosyasından metin çıkarmak için
from langchain.text_splitter import RecursiveCharacterTextSplitter #metni daha küçük parçalara ayırmak için (chunk)

from dotenv import load_dotenv
import os

load_dotenv()  # .env dosyasından API anahtarlarını yükler

api_key = os.getenv("GEMINI_API_KEY")  # API anahtarını alır

if not api_key:
    raise ValueError("GEMİNİ_API_KEY environment variable is not set.")

# Google API anahtarını ayarlıyoruz
os.environ["GOOGLE_API_KEY"] = api_key #GOOGLE_API_KEY çevresel değişkenini api_key ile ayarlıyoruz

# sık sorulan sorular dosyasını yükle
loader = PyPDFLoader("musteri_destek_SSS.pdf")

# langchain documents obejesi oluştur
documents = loader.load()

# metni parçalamak için
# splitter metni anlamlı parçalara ayırırken cümle veya paragraf bütünlüğünü korumaya çalılıyor
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500, # Her parça max 500 karakter içerecek demek
    chunk_overlap = 50 # Her parçanın bir öncekinden 50 karakter alabilir
)

# chunk'ları oluştur
docs = text_splitter.split_documents(documents)


# HuggingFace modelini kullanarak embedding oluştur (metinleri gömmek için)
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# faiis vektör veritabanı, paraçlara ayrılmış metni embedding ile vektör haline getirir ve index oluşturur
vectordb = FAISS.from_documents(docs, embedding)

# oluşturualan vektör veritabanını local diske kaydet
vectordb.save_local("sss_vectordb")  # Faiss vektör veritabanını yerel diske kaydediyoruz

print("Embedding ve vektör veritabanı başarıyla oluşturuldu ve kaydedildi.")