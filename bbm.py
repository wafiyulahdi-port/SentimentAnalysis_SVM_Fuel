import streamlit as st
import pandas as pd
import numpy as np
import re
from tqdm import tqdm
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sn
import io
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, f1_score
import os
from PIL import Image

# Set up the stemming factory
factory = StemmerFactory()
stemmer = factory.create_stemmer()

st.markdown(
    """
    <style>
        [data-testid=stSidebar] {
            background-color: #e4f6ff;
        }
        [data-testid=stSidebarUserContent] {
            padding-top: 3.5rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Fungsi case folding


def casefolding(text):
    text = re.sub(r'\n', ' ', text)  # Menghilangkan Enter
    text = re.sub(r'\bhttps?\S*\b', ' ', text)  # Menghilangkan URL
    # Mengganti spasi ganda menjadi spasi tunggal
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()  # Mengubah huruf menjadi huruf kecil
    text = text.replace(":", " ")  # Hapus :
    # Menghapus kata yang diawali dengan @
    text = re.sub(r'@\w+\s*', ' ', text)
    text = re.sub(r'[-+]?[0-9]+', ' ', text)  # Menghapus angka
    text = re.sub(r'[^\w\s]', ' ', text)  # Menghapus karakter tanda baca
    text = text.strip()  # Menghapus whitespace di awal dan di akhir
    return text

# Fungsi normalisasi teks


def text_normalize(text, key_norm):
    text = ' '.join([key_norm[key_norm['singkat'] == word]['hasil'].values[0] if
                     (key_norm['singkat'] == word).any() else word for word in text.split()])
    text = str.lower(text)
    return text

# Fungsi untuk memuat stopwords


def load_stopwords():
    stopwords_df = pd.read_csv('stopwords_indonesian.csv')
    return set(stopwords_df['stopwords'].tolist())


stopwords_ind = load_stopwords()

# Fungsi untuk menghapus stopwords


def remove_stop_words(text):
    clean_words = [word for word in text.split() if word not in stopwords_ind]
    return " ".join(clean_words)


# Fungsi stemming
def stemming(text):
    return stemmer.stem(text)

# Fungsi preprocessing terpisah


def apply_casefolding(df):
    df["casefolding"] = df["full_text"].apply(casefolding)
    return df


def apply_text_normalization(df, key_norm):
    df['normalisasi_teks'] = df['casefolding'].apply(
        lambda x: text_normalize(x, key_norm))
    return df


def apply_stopword_removal(df):
    df['remove_stopwords'] = df['normalisasi_teks'].apply(remove_stop_words)
    return df


def apply_stemming(df):
    df['stemming'] = ""
    progress_bar = st.progress(0)
    total = len(df)
    for i, text in enumerate(df['remove_stopwords']):
        df.at[i, 'stemming'] = stemming(text)
        progress = (i + 1) / total
        progress_bar.progress(progress)
    df.to_excel('clean_text.xlsx', index=False)
    df = df[df["stemming"] != ""]
    return df[['full_text', 'casefolding', 'normalisasi_teks', 'remove_stopwords', 'stemming']]


# Baca file positif.csv dan negatif.csv
positif_df = pd.read_csv('positif.csv', sep=';', encoding='utf-8')
negatif_df = pd.read_csv('negatif.csv', sep=';', encoding='utf-8')

# Konversi ke kamus dengan kata sebagai kunci dan bobot sebagai nilai
kamus_positif = dict(zip(positif_df['word'], positif_df['weight']))
kamus_negatif = dict(zip(negatif_df['word'], negatif_df['weight']))

# Fungsi untuk menghitung sentimen berdasarkan bobot


def hitung_sentimen(teks):
    kata_kata = teks.lower().split()
    total_bobot = 0

    for kata in kata_kata:
        if kata in kamus_positif:
            total_bobot += kamus_positif[kata]
        elif kata in kamus_negatif:
            total_bobot += kamus_negatif[kata]

    if total_bobot > 0:
        return 'Positif'
    elif total_bobot < 0:
        return 'Negatif'
    else:
        return 'Netral'

# Fungsi untuk memprediksi sentimen


def predict_sentiment(text, vectorizer, model, key_norm, stopwords_ind):
    text = casefolding(text)
    text = text_normalize(text, key_norm)
    text = remove_stop_words(text)
    text = stemming(text)

    text_tfidf = vectorizer.transform([text]).toarray()
    prediction = model.predict(text_tfidf)

    if prediction == 'Positif':
        return 'Positif'
    elif prediction == 'Negatif':
        return 'Negatif'
    else:
        return 'Netral'


# Load logo
logo = Image.open("logo.png")
st.sidebar.image(logo, use_container_width=True)

# Menu di sidebar
menu = st.sidebar.selectbox("Pilih Menu", [
    "Dashboard",
    "Preprocessing",
    "EDA",
    "TF-IDF dan Feature Selection",
    "Model SVM",
    "Prediksi Sentimen Model"
])

# Load file key_norm.csv dan stopwords_indonesian.csv
key_norm = pd.read_csv('key_norm.csv')

# Membaca file scrapped_data.csv
file_path = os.path.join(os.getcwd(), 'bbm.csv')

if os.path.exists(file_path):
    df = pd.read_csv(file_path)
    df = df[['full_text']]
else:
    st.error(f"File {file_path} tidak ditemukan.")

if menu == "Dashboard":
    st.title("ANALISIS SENTIMEN DENGAN ALGORITMA SUPPORT VECTOR MACHINE PADA TWEETS KUALITAS BBM DI PLATFORM X")

    image = Image.open("oilgas.png")
    st.image(image, caption="Lorem Ipsum", use_container_width=True)

    st.header("Lorem Ipsum")
    st.markdown("""
    <div style='text-align: justify'>
    Lorem ipsum dolor sit amet, consectetur adipiscing elit. Integer pretium orci elit, 
    quis pretium enim lacinia lacinia. Vestibulum tristique lacus et libero interdum bibendum. 
    Nulla molestie quam eu nisl bibendum consequat. Curabitur consequat est ligula, 
    vel interdum dui rhoncus eu. Aliquam erat volutpat. Nam accumsan nibh volutpat, mattis mauris non, 
    facilisis elit. Vestibulum mollis, diam ultricies ultrices facilisis, quam lorem facilisis mauris, 
    eu euismod tortor ipsum sed nulla. Aliquam a neque et ipsum venenatis fermentum id vitae lacus. 
    Quisque cursus nec dui et vulputate.
    </div>
    """, unsafe_allow_html=True)


elif menu == "Preprocessing":
    st.title("Sentiment Analysis Preprocessing")
    st.write("Data sebelum preprocessing:")
    st.write(df.head())
    # Cek apakah file hasil preprocessing sudah ada
    if os.path.exists('clean_text.xlsx'):
        st.info("File hasil preprocessing ditemukan. Data akan langsung dimuat.")
        df_clean = pd.read_excel('clean_text.xlsx')
        st.session_state.df = df_clean
        st.success("Data hasil preprocessing berhasil dimuat!")
        st.write(df_clean.head())
    else:
        if 'df' not in st.session_state:
            st.session_state.df = df
        with st.spinner('Memulai Preprocessing...'):
            st.session_state.df = apply_casefolding(st.session_state.df)
            st.write("Data setelah casefolding:")
            st.write(st.session_state.df.head())
            st.session_state.df = apply_text_normalization(
                st.session_state.df, key_norm)
            st.write("Data setelah normalisasi teks:")
            st.write(st.session_state.df.head())
            st.session_state.df = apply_stopword_removal(st.session_state.df)
            st.write("Data setelah penghapusan stopwords:")
            st.write(st.session_state.df.head())
            df_clean = apply_stemming(st.session_state.df)
            st.success("Preprocessing selesai!")
            st.write("Data setelah stemming:")
            st.write(df_clean.head())
        if 'df' not in st.session_state:
            st.session_state.df = df

# Menu baru: Prediksi Sentimen Berdasarkan Model
elif menu == "Prediksi Sentimen Model":
    st.title("Prediksi Sentimen Berdasarkan Model")
    model_choice = st.radio("Pilih Model", ["SVM 500 Fitur", "SVM Best Fitur"])
    input_text = st.text_area("Masukkan kalimat untuk diprediksi sentimennya:")
    if st.button("Prediksi"):
        if input_text.strip() != "":
            # Pilih model dan selector sesuai pilihan
            if model_choice == "SVM 500 Fitur":
                vectorizer = st.session_state.get('vectorizer', None)
                model = st.session_state.get('svm_model', None)
                selector = st.session_state.get('selector', None)
            else:
                vectorizer = st.session_state.get('vectorizer', None)
                model = st.session_state.get('svm_model', None)
                selector = st.session_state.get('selector', None)
            if vectorizer is not None and model is not None and selector is not None:
                # Preprocessing input
                text = casefolding(input_text)
                text = text_normalize(text, key_norm)
                text = remove_stop_words(text)
                text = stemming(text)
                text_tfidf = vectorizer.transform([text])
                text_selected = selector.transform(text_tfidf)
                pred = model.predict(text_selected)[0]
                st.success(f"Hasil prediksi sentimen: **{pred}**")
            else:
                st.warning("Model belum tersedia. Silakan lakukan pelatihan model terlebih dahulu di menu 'Model SVM'.")
        else:
            st.warning("Teks input tidak boleh kosong.")

elif menu == "EDA":
    st.title("Exploratory Data Analysis (EDA)")

    if 'df' in st.session_state:
        # Ganti 'clean_text' dengan 'stemming' sebagai hasil akhir preprocessing
        st.session_state.df = st.session_state.df[st.session_state.df["stemming"] != ""]
        st.session_state.df['sentimen'] = st.session_state.df['stemming'].apply(
            hitung_sentimen)

        data = st.session_state.df

        st.write("Data setelah analisis sentimen:")
        st.write(data)

        st.write("Jumlah data per sentimen:")
        st.write(data['sentimen'].value_counts())

        data_positif = data[data['sentimen'] == 'Positif']
        data_negatif = data[data['sentimen'] == 'Negatif']

        text_positif = ' '.join(data_positif["stemming"].values.tolist())
        text_negatif = ' '.join(data_negatif["stemming"].values.tolist())

        wordcloud_positif = WordCloud(width=800, height=800, background_color='white',
                                      stopwords=None, min_font_size=10).generate(text_positif)
        wordcloud_negatif = WordCloud(width=800, height=800, background_color='white',
                                      stopwords=None, min_font_size=10).generate(text_negatif)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Wordcloud Positif")
            st.image(wordcloud_positif.to_array())

        with col2:
            st.subheader("Wordcloud Negatif")
            st.image(wordcloud_negatif.to_array())

elif menu == "TF-IDF dan Feature Selection":
    st.title("TF-IDF dan Feature Selection")

    st.session_state.df = st.session_state.df[st.session_state.df['sentimen'] != 'Netral']

    if 'df' in st.session_state:
        st.write("Data untuk TF-IDF dan Feature Selection:")
        X_raw = st.session_state.df["stemming"]
        y_raw = st.session_state.df["sentimen"]
        X_train, X_test, y_train, y_test = train_test_split(
            X_raw.values, y_raw.values, test_size=0.2, random_state=42)

        vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        vectorizer.fit(X_train)

        kolom = vectorizer.get_feature_names_out()
        X_train_TFIDF = vectorizer.transform(X_train).toarray()
        X_test_TFIDF = vectorizer.transform(X_test).toarray()
        X = vectorizer.transform(st.session_state.df["stemming"]).toarray()
        train_tf_idf = pd.DataFrame(X_train_TFIDF, columns=kolom)
        test_tf_idf = pd.DataFrame(X_test_TFIDF, columns=kolom)

        # --- Transformasi TF, IDF, TF-IDF ke format long table ---
        X_train_tfidf = vectorizer.transform(X_train)
        X_train_tf = vectorizer.transform(X_train).toarray() / vectorizer.idf_  # TF ≈ TF-IDF / IDF
        fitur = vectorizer.get_feature_names_out()
        idf_values = vectorizer.idf_
        idf_dict = dict(zip(fitur, idf_values))
        tf_df = pd.DataFrame(X_train_tf, columns=fitur)
        tfidf_df = pd.DataFrame(X_train_tfidf.toarray(), columns=fitur)
        tabel_fitur_aktif = []
        for idx in range(tf_df.shape[0]):
            tf_row = tf_df.iloc[idx]
            tfidf_row = tfidf_df.iloc[idx]
            for fitur_aktif in tfidf_row[tfidf_row != 0].index:
                tabel_fitur_aktif.append({
                    'Dokumen': idx,
                    'Fitur': fitur_aktif,
                    'TF': tf_row[fitur_aktif],
                    'IDF': idf_dict[fitur_aktif],
                    'TF-IDF': tfidf_row[fitur_aktif]
                })
        feature_df = pd.DataFrame(tabel_fitur_aktif)
        st.write("Tabel TF, IDF, dan TF-IDF (long format):")
        st.write(feature_df.head(50))

        st.write("Fitur TF-IDF dari data latih:")
        st.write(train_tf_idf.head())

        chi2_features = SelectKBest(chi2, k=500)
        X_kbest_features = chi2_features.fit_transform(train_tf_idf, y_train)

        st.write('Banyaknya fitur awal:', train_tf_idf.shape[1])
        st.write('Banyaknya fitur setelah di seleksi:',
                 X_kbest_features.shape[1])

        features = chi2_features.get_support(indices=True)
        feature_names = np.array(kolom)[features]
        scores = chi2_features.scores_[features]

        sorted_indices = np.argsort(scores)[::-1]

        top_features = sorted_indices[:10]
        top_feature_names = feature_names[top_features]
        top_scores = scores[top_features]

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(range(len(top_scores)), top_scores, align='center')
        ax.set_yticks(range(len(top_feature_names)));
        ax.set_yticklabels(top_feature_names)
        ax.set_xlabel('Chi-Squared Score')
        ax.set_title('Top 10 Features by Chi-Squared Score')

        st.pyplot(fig)
    else:
        st.error("Silakan upload file dan lakukan preprocessing terlebih dahulu.")

elif menu == "Model SVM":
    st.title("Model SVM")

    if 'df' in st.session_state:
        st.write("Pilih skenario evaluasi model:")
        skenario = st.radio(
            "Skenario", ["Skenario 1: 500 Fitur", "Skenario 2: Best Performance"])

        # Ambil data
        X_raw = st.session_state.df["stemming"]
        y_raw = st.session_state.df["sentimen"]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_raw.values, y_raw.values, test_size=0.2, random_state=42)

        # TF-IDF
        vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        vectorizer.fit(X_train)

        train_tf_idf = vectorizer.transform(X_train)
        test_tf_idf = vectorizer.transform(X_test)

        if skenario == "Skenario 1: 500 Fitur":
            st.subheader("Skenario 1: 500 Fitur (Chi-Square)")

            selector = SelectKBest(chi2, k=500)
            X_train_selected = selector.fit_transform(train_tf_idf, y_train)
            X_test_selected = selector.transform(test_tf_idf)

            model = SVC()
            model.fit(X_train_selected, y_train)
            y_pred = model.predict(X_test_selected)

            # Simpan model dan vectorizer ke file pkl
            import pickle
            with open('svm_500.pkl', 'wb') as f:
                pickle.dump(model, f)
            with open('vectorizer_500.pkl', 'wb') as f:
                pickle.dump(vectorizer, f)
            st.success("Model dan vectorizer SVM 500 fitur berhasil disimpan!")
        else:
            st.subheader("Skenario 2: Best Performance (Chi-Square)")

            max_f1 = 0
            k_feature_f1 = 0

            st.subheader("Mencari jumlah fitur terbaik...")
            progress_bar = st.progress(0)
            status_text = st.empty()

            total_steps = min(1000, train_tf_idf.shape[1]) - 10
            for i, k in enumerate(range(10, min(1000, train_tf_idf.shape[1]), 1)):
                selector = SelectKBest(chi2, k=k)
                X_train_k = selector.fit_transform(train_tf_idf, y_train)
                X_test_k = selector.transform(test_tf_idf)

                model_k = SVC()
                model_k.fit(X_train_k, y_train)
                y_pred_k = model_k.predict(X_test_k)

                f1 = f1_score(y_test, y_pred_k, average="weighted")

                if f1 > max_f1:
                    max_f1 = f1
                    k_feature_f1 = k
                    best_model = model_k
                    best_selector = selector
                    best_pred = y_pred_k

                progress_bar.progress((i + 1) / total_steps)
                status_text.text(f"Evaluasi {k} fitur...")

            st.success(
                f"F1-score terbaik: {max_f1:.4f} dengan {k_feature_f1} fitur")

            model = best_model
            selector = best_selector
            y_pred = best_pred

            # Simpan model dan vectorizer ke file pkl
            import pickle
            with open('svm_best.pkl', 'wb') as f:
                pickle.dump(model, f)
            with open('vectorizer_best.pkl', 'wb') as f:
                pickle.dump(vectorizer, f)
            st.success("Model dan vectorizer SVM best fitur berhasil disimpan.")

        # Evaluasi Model
        unique_labels = sorted(list(set(y_test)))  # Label dinamis

        st.write("Classification Report:")
        report = classification_report(
            y_test, y_pred, target_names=unique_labels)
        st.markdown(f"```\n{report}\n```")

        # Confusion Matrix
        confm = confusion_matrix(y_test, y_pred, labels=unique_labels)
        df_cm = pd.DataFrame(confm, index=unique_labels, columns=unique_labels)

        st.subheader('Confusion Matrix')
        fig, ax = plt.subplots()
        ax = sn.heatmap(df_cm, cmap='Greens', annot=True, fmt=".0f")
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Label Prediksi')
        ax.set_ylabel('Label Sebenarnya')
        ax.invert_xaxis()
        ax.invert_yaxis()
        ax.xaxis.set_ticks_position('top')
        ax.xaxis.set_label_position('top')
        st.pyplot(fig)

        # Simpan ke session state jika diperlukan di menu lain
        st.session_state.vectorizer = vectorizer
        st.session_state.svm_model = model
        st.session_state.selector = selector

    else:
        st.error("Silakan lakukan preprocessing terlebih dahulu.")

