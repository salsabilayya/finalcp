import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Load data
def load_data(filename):
    data = pd.read_csv(filename)
    return data

# Normalisasi data
def normalize_data(data):
    scaler = MinMaxScaler()
    return scaler.fit_transform(data)

# Buat prediksi dari model
def make_predictions(models, x_test_norm):
    predictions = {}
    for name, model in models.items():
        predictions[name] = model.predict(x_test_norm)
    return predictions

# Confusion Matrix
def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap='inferno')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(np.unique(y_true)))
    plt.xticks(tick_marks, np.unique(y_true))
    plt.yticks(tick_marks, np.unique(y_true))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.grid(False)
    for i in range(len(np.unique(y_true))):
        for j in range(len(np.unique(y_true))):
            plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > cm.max() / 2. else "black")
    plt.tight_layout()
    return plt

def main():
    # Sidebar
    st.sidebar.title("Best Ketchup in Supermarket")
    dropdown_options = ["Business Understanding & EDA", "Evaluation"]
    selected_dropdown = st.sidebar.selectbox("Ketchup Data Modelling", dropdown_options)

    # Main content
    st.write(""" 
            # Best Ketchup in Supermarket
             """)
    st.caption('''Selamat datang di dashboard "Best Ketchup in Supermarket"!
               Dashboard ini didedikasikan untuk memberikan wawasan tentang penjualan dan popularitas berbagai merk saus tomat di supermarket. Dengan berbagai merk dan variasi yang tersedia, saus tomat menjadi salah satu produk penting di rak-rak supermarket. Dashboard ini menampilkan data penjualan, analisis statistik, serta visualisasi yang menarik tentang preferensi konsumen terhadap saus tomat.
               ''')

    # Load data
    Ketchup = load_data('Ketchup.csv')
    DataCleaned = load_data('Data Cleaned.csv')

    # Tampilkan grafik sesuai dropdown
    if selected_dropdown == "Business Understanding & EDA":
        # Visualisasi harga ketchup
        fig = px.bar(Ketchup, x='Ketchup.choice', y='price.heinz', title='Ketchup Prices')
        fig.update_xaxes(title='Brand')
        fig.update_yaxes(title='Price')
        st.plotly_chart(fig)
        st.caption('Ini adalah tampilan harga Ketchup dalam diagram berbentuk bar. Diagram ini menunjukkan bahwa merk saus tomat yang memiliki harga paling tinggi adalah Heinz, disusul dengan STB, Hunts, dan yang terakhir adalah Delmonte. Jarak antara perbedaan harga merk Heinz dengan Delmonte sangat jauh, sedangkan jarak perbedaan antara merk Hunts dengan STB tidak begitu jauh.')

        # Menghitung jumlah preferensi dan persentasenya
        Ketchup['preference_count'] = Ketchup.groupby('Ketchup.choice')['Ketchup.choice'].transform('count')
        Ketchup['preference_percent'] = (Ketchup['preference_count'] / Ketchup['preference_count'].sum()) * 100

        # GRAFIK PIE PREFERENCE
        fig = px.pie(Ketchup, values='preference_percent', names='Ketchup.choice', title='Ketchup Preference')
        st.plotly_chart(fig)
        st.caption('''Preferensi pelanggan terhadap saus tomat lebih tinggi di merk Heinz dengan persentase 72.4%. Merk STB memiliki persentase 15.1% sebagai pilihan pelanggan, Hunts dengan 11.8%, dan Delmonte dengan persentase terendah yaitu 0,743%. Ini menunjukkan bahwa merk Heinz lebih banyak digemari oleh pelanggan daripada merk Delmonte.
                   Interpretasi: meskipun merk Heinz memiliki harga yang paling tinggi  dari merk lainnya, namun tidak menghalanginya untuk tetap menjadi pilihan favorit bagi banyak pelanggan.
                   Insight: Adanya loyalitas pelanggan yang kuat terhadap merk yang dapat dipicu oleh kualitas yang konsisten, rasa yang disukai, ataupun pengalaman positif sebelumnya.
                   Actionable Insight: Mempertahankan standar kualitas yang tinggi dan terus mengembangkan produk baru untuk tetap relevan dengan tren pasar dan memenuhi kebutuhan konsumen.
                   ''')

        # GRAFIK BOX KETCHUP PRICES
        fig = px.box(Ketchup, x='Ketchup.choice', y='price.heinz', title='Ketchup Prices', color='Ketchup.choice')
        fig.update_xaxes(title='Brand')
        fig.update_yaxes(title='Price')
        st.plotly_chart(fig)
        st.caption('''Boxplot diatas ini merepresentasikan nilai mean, median, min, max, q1, dan q3. Berikut adalah rinciannya.
                   [Heinz: max = 1.47, min = 0.79, q1 = 0.99, q3 = 1.39.]
                   [Hunts: max = 1.47, min = 0.79, q1 = 1.19, q3 = 1.46.]
                   [STB: max = 1.47, min = 0.79, q1 = 1.19, q3 = 1.46.]
                   [Delmonte: max = 1.47, min = 0.99, q1 = 1.19, q3 = 1.46.]
                   ''')

    elif selected_dropdown == "Evaluation":
        # Hapus kolom kategorikal
        numeric_columns = Ketchup.select_dtypes(include=[np.number]).columns
        Ketchup_corr = Ketchup[numeric_columns].corr()

        # Tampilkan heatmap dengan layout yang diperbarui
        fig = px.imshow(Ketchup_corr)
        fig.update_layout(title='Correlation Heatmap', width=800, height=600)
        st.plotly_chart(fig)
        st.caption('Heatmap adalah visualisasi yang menampakkan korelasi antara variabel dalam bentuk matriks. Korelasi ini ditampilkan dalam bentuk warna, di mana warna yang lebih gelap menunjukkan korelasi yang lebih kuat, sedangkan warna yang lebih terang menunjukkan korelasi yang lemah. Pada heatmap di atas, dapat dilihat bahwa korelasi antara percentage count dan percentage preferences terhadap merk Heinz memiliki warna yang gelap, yang berarti korelasinya kuat.')

        # Implementasi prediksi model
        st.write(""" 
                #### Confusion Matrix
                 """)

        # Load model yang telah dilatih
        gnb = GaussianNB()  # Isi dengan model Gaussian Naive Bayes yang telah dilatih
        knn = KNeighborsClassifier()  # Isi dengan model KNN yang telah dilatih
        dtc = DecisionTreeClassifier()  # Isi dengan model Decision Tree yang telah dilatih
        models = {"GNB": gnb, "KNN": knn, "DTC": dtc}

        # Normalisasi data Ketchup
        Ketchup_norm = normalize_data(Ketchup.drop(columns=['Ketchup.choice']))

        # Bagi data menjadi fitur (X) dan target (y)
        X = Ketchup_norm
        y = Ketchup['Ketchup.choice']

        # Latih model
        for name, model in models.items():
            model.fit(X, y)

        # Membuat prediksi
        predictions = make_predictions(models, Ketchup_norm)

        # Menyusun hasil prediksi ke dalam DataFrame
        result_df = pd.DataFrame(predictions)

        # Menampilkan confusion matrix untuk setiap model
        for name, model in models.items():
            y_test = Ketchup['Ketchup.choice']  # Sesuaikan dengan kolom yang berisi label aktual
            y_pred = result_df[f"{name}"]
            cm_plot = plot_confusion_matrix(y_test, y_pred, title=f"{name} Confusion Matrix")
            st.pyplot(cm_plot)
            st.caption('Confusion Matrix merupakan tabel yang digunakan untuk mengevaluasi kinerja suatu model klasifikasi pada set data uji di mana nilai sebenarnya sudah diketahui. Confusion matrix juga menampilkan jumlah prediksi yang benar dan salah yang dibuat oleh model dalam setiap kelas target. Matriks ini membantu dalam mengevaluasi seberapa baik model dapat membedakan kelas positif dan negatif.')
            

        # Plot ROC curves
        st.write(""" 
                #### ROC Curves
                 """)
        fig_roc, ax_roc = plt.subplots(1, 3, figsize=(18, 6))

        for model, name, ax in zip(models.values(), models.keys(), ax_roc):
            y_pred_proba = model.predict_proba(X)

            fpr, tpr, _ = roc_curve(y == np.unique(y)[1], y_pred_proba[:, 1])
            roc_auc = roc_auc_score(y == np.unique(y)[1], y_pred_proba[:, 1])

            ax.plot(fpr, tpr, label=f'{name} (ROC-AUC = {roc_auc:.2f})')
            ax.plot([0, 1], [0, 1], color='navy', linestyle='--')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'ROC Curve - {name}')
            ax.legend(loc='lower right')

        plt.tight_layout()
        st.pyplot(fig_roc)
        st.caption('Skor ROC AUC dapat berkisar dari0 hingga 1, yang di mana skor 1 menunjukkan kinerja sempurna. Pada diagram kurva di atas dapat dilihat bahwa kurva dengan algoritma decision tree classifier memiliki skor ROC AUC 1.00. Artinya diagram kurva algoritma ini kinerjanya sempurna.')

        # Cross-validation
        st.write(""" 
                #### Cross-Validation
                 """)
        # Inisialisasi model
        models = [gnb, knn, dtc]
        model_names = ['Gaussian Naive Bayes', 'K-Nearest Neighbors', 'Decision Tree Classifier']

        # Load data cleaned
        X_cleaned = DataCleaned.drop(columns=['Ketchup.id'])  # Ganti 'target' dengan nama kolom target yang sesuai
        y_cleaned = DataCleaned['Ketchup.id']  # Ganti 'target' dengan nama kolom target yang sesuai

        # Normalisasi data cleaned
        X_cleaned_norm = normalize_data(X_cleaned)

        # Lakukan validasi silang untuk setiap model
        cv_scores = []
        for model in models:
            scores = cross_val_score(model, X_cleaned_norm, y_cleaned, cv=5)
            cv_scores.append(scores)

        # Buat dataframe dari hasil validasi silang
        df_cv_scores = pd.DataFrame(cv_scores, index=model_names).T

        # Tampilkan visualisasi hasil validasi silang dengan lineplot
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df_cv_scores, markers=True)
        plt.title('Cross-Validation Scores for Different Models')
        plt.xlabel('Fold')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.legend(title='Model', loc='lower right')
        plt.xticks(ticks=range(5), labels=['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5'])
        st.pyplot(plt)
        st.caption('Pada gambar diagram diatas, dapat disimpulkan bahwa garis decision tree classifier memiliki nilai cross validation yang stabil dan model bekerja dengan baik pada dataset.')

main()
