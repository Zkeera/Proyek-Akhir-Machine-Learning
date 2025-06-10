# Laporan Proyek Machine Learning - Sistem Rekomendasi Kupon Dalam Kendaraan

## Domain Proyek

Dalam era digital saat ini, sistem rekomendasi telah menjadi bagian penting dalam meningkatkan pengalaman pengguna, termasuk dalam sektor otomotif. Proyek ini berfokus pada pengembangan sistem rekomendasi kupon dalam kendaraan, dengan tujuan memberikan penawaran yang relevan kepada pengguna berdasarkan karakteristik dan preferensi mereka. Penggunaan sistem ini diharapkan dapat meningkatkan tingkat penggunaan kupon, loyalitas pelanggan, dan potensi pendapatan bisnis.

Masalah ini penting untuk diselesaikan karena konsumen kini mengharapkan layanan yang bersifat personalisasi. Sistem rekomendasi dalam kendaraan yang efektif dapat menciptakan pengalaman pengguna yang lebih baik dan meningkatkan engagement pelanggan terhadap produk yang ditawarkan.

## Business Understanding

### Problem Statements

- Bagaimana cara mengelompokkan pengguna berdasarkan kebiasaan dan karakteristik mereka agar sistem rekomendasi lebih tepat sasaran?
- Bagaimana memanfaatkan data kupon dan karakteristik pengguna untuk meningkatkan akurasi rekomendasi?

### Goals

- Mengidentifikasi segmen pelanggan yang berbeda dengan teknik clustering.
- Membuat sistem rekomendasi kupon yang relevan dan personal.

### Solution Statements

- Menggunakan algoritma **K-Means Clustering** untuk mengelompokkan data pelanggan.
- Melakukan eksplorasi data dengan **PCA** (Principal Component Analysis) untuk memvisualisasikan cluster.
- Melakukan evaluasi kualitas cluster menggunakan **Silhouette Score**.

## Data Understanding

Dataset ini merupakan data simulasi dari sistem kupon dalam kendaraan, terdiri dari berbagai informasi pengguna dan respons terhadap kupon. Dataset awal memiliki 12.684 baris dan 26 kolom sebelum dilakukan preprocessing.

### Kondisi Data Awal

- Missing Values: Terdapat nilai NaN pada beberapa kolom, contohnya pada kolom CarryAway.

- Data Duplikat: Setelah pengecekan, tidak ditemukan data duplikat pada dataset.

- Outlier: Tidak dilakukan analisis eksplisit mengenai outlier karena mayoritas fitur bersifat kategorikal.

### Sumber Dataset
Tautan sumber data yang digunakan dalam proyek ini adalah
https://raw.githubusercontent.com/Zkeera/Proyek-Akhir-Machine-Learning/refs/heads/main/Dataset/in-vehicle-coupon-recommendation.csv
dan diakses langsung melalui URL raw GitHub. Tautan yang digunakan dalam notebook ini sesuai dengan dataset yang dimuat ke dalam DataFrame df.

### Deskripsi Fitur

Berikut adalah deskripsi fitur yang digunakan dalam dataset:

- destination: Tujuan perjalanan pengguna.

- passanger: Jumlah penumpang dalam kendaraan.

- weather: Kondisi cuaca saat perjalanan.

- coupon: Jenis kupon yang ditawarkan.

- gender: Jenis kelamin pengguna.

- age: Rentang usia pengguna.

- maritalStatus: Status pernikahan pengguna.

- has_children: Indikator apakah pengguna memiliki anak.

- education: Tingkat pendidikan pengguna.

- occupation: Pekerjaan pengguna.

- user_profile: Fitur gabungan yang menggambarkan konteks pengguna

berdasarkan beberapa atribut di atas (contoh: tujuan, cuaca, jenis kelamin, usia, status pernikahan, dll.).


## Data Preparation

Pada bagian ini, kami menggunakan teknik Label Encoding untuk mengonversi fitur coupon menjadi format numerik. Fitur lainnya, seperti user_profile, dihasilkan dengan menggabungkan informasi dari beberapa kolom yang relevan menjadi satu string untuk setiap pengguna.

Label Encoding: Fitur coupon dikodekan menjadi nilai numerik menggunakan LabelEncoder.

User Profile: Profil pengguna digabungkan dari beberapa kolom (misalnya, tujuan perjalanan, cuaca, usia, dll.) menjadi satu string yang mewakili konteks pengguna secara keseluruhan.

Tujuan data preparation ini adalah untuk mempersiapkan data agar dapat diproses dengan optimal oleh algoritma clustering.

a. Untuk Content-Based Filtering (CBF):
Beberapa fitur teks (seperti occupation, age, income, maritalStatus, dll) digabungkan menjadi satu fitur baru bernama user_profile.

Dilakukan TF-IDF Vectorization pada kolom user_profile untuk menghasilkan representasi numerik.

Data kupon direpresentasikan menggunakan kolom coupon.

b. Untuk Collaborative Filtering (CF):
Label pada kolom coupon diubah menjadi numerik menggunakan Label Encoding, menjadi coupon_id.

Matriks interaksi dibuat menggunakan user_id sebagai indeks dan coupon_id sebagai kolom dalam pivot table, dengan nilai yang menunjukkan apakah pengguna menerima kupon tersebut (Y). Matriks interaksi ini kemudian digunakan untuk modeling Collaborative Filtering.

Pivot Table: df.pivot_table(index='user_id', columns='coupon_id', values='Y', fill_value=0)

Diterapkan TruncatedSVD untuk dekomposisi dimensi rendah matriks interaksi.

## Modeling

a. Content-Based Filtering (CBF):

Untuk rekomendasi berbasis konten (CBF), kami menghitung kemiripan antara profil pengguna dan kupon yang ada menggunakan cosine similarity. Representasi numerik untuk user_profile dibuat dengan TF-IDF Vectorization. Hasil rekomendasi dihitung untuk pengguna pertama (user_index = 0), dan kupon yang direkomendasikan adalah yang memiliki kemiripan tertinggi dengan profil pengguna tersebut.

Output untuk CBF menunjukkan kupon yang paling mirip berdasarkan profil pengguna.

Cosine Similarity dihitung menggunakan kode berikut:

tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(df['user_profile'])
cos_sim = cosine_similarity(tfidf_matrix[user_index], tfidf_matrix)
similar_indices = cos_sim.argsort()[0][-6:-1][::-1]
df.iloc[similar_indices][['coupon', 'destination', 'Y']]

b. Collaborative Filtering (CF):

Untuk Collaborative Filtering, kami menggunakan teknik TruncatedSVD (Singular Value Decomposition) untuk pemfaktoran matriks interaksi pengguna-kupon. Setelah matriks interaksi dibentuk, kami melakukan dekomposisi untuk mengurangi dimensi dan memprediksi interaksi yang belum tercatat.

Matriks prediksi dihasilkan dengan mengalikan matriks laten yang dihasilkan oleh SVD dengan komponen yang diperoleh dari dekomposisi.

Evaluasi untuk Collaborative Filtering dilakukan dengan menghitung RMSE (Root Mean Square Error).

SVD dilakukan dengan kode berikut:

svd = TruncatedSVD(n_components=5, random_state=42)
latent_matrix = svd.fit_transform(interaction_array)
predicted_matrix = np.dot(latent_matrix, svd.components_)
rmse = mean_squared_error(true_values, predicted_values) ** 0.5
print(f"RMSE Collaborative Filtering (TruncatedSVD): {rmse:.4f}")

### Hasil Visualisasi:
- Visualisasi hasil clustering menunjukkan adanya pengelompokan pengguna yang cukup jelas setelah reduksi PCA.
- Komposisi setiap cluster dianalisis berdasarkan fitur dominan.

a. Content-Based Filtering (CBF)
Menggunakan cosine similarity antara user profile dan kupon.

Top-N rekomendasi dihasilkan dengan memilih kupon yang paling mirip berdasarkan skor kemiripan.

Contoh Output Top-N untuk CBF:
User: Executive_30-39_high income
Top-5 Kupon: ['Coffee House', 'Bar', 'Restaurant(<20)', 'Carry out & Take away', 'Restaurant(20-50)']

b. Collaborative Filtering (CF)
Matriks interaksi direduksi menggunakan TruncatedSVD, lalu dikalikan kembali untuk mendapatkan prediksi skor antar user dan kupon.

Contoh Output Top-N untuk CF:
User ID: 0
Top-5 Kupon berdasarkan skor prediksi: ['Bar', 'Coffee House', 'Restaurant(20-50)', 'Carry out & Take away', 'Restaurant(<20)']

Sebuah heatmap digunakan untuk memvisualisasikan matriks interaksi antara pengguna dan kupon. Visualisasi ini memberikan gambaran mengenai bagaimana interaksi pengguna dengan kupon yang ditawarkan.

sns.heatmap(interaction_matrix, cmap="YlGnBu", cbar=True)
plt.title("User-Coupon Interaction Matrix")
plt.show()

## Evaluation

### Metrik Evaluasi: **Silhouette Score**

- **Silhouette Score** digunakan untuk menilai kualitas hasil clustering. Nilai berkisar antara -1 hingga 1, dengan nilai lebih tinggi menunjukkan pengelompokan yang lebih baik.
- Pada eksperimen ini, didapatkan Silhouette Score sebesar **~0.23**, yang mengindikasikan bahwa hasil clustering cukup baik meskipun masih dapat ditingkatkan.
- Hasil evaluasi menunjukkan bahwa pengguna dalam setiap cluster memiliki karakteristik yang berbeda dan dapat digunakan untuk rekomendasi kupon yang lebih relevan.

a. Evaluasi Content-Based Filtering:
Precision@5: Digunakan untuk mengukur ketepatan 5 rekomendasi teratas.

Hasil Precision@5: 1.00, yang menunjukkan seluruh rekomendasi relevan untuk pengguna tersebut.

b. Evaluasi Collaborative Filtering:
Root Mean Square Error (RMSE): Digunakan untuk mengukur error antara prediksi skor dan skor aktual.

Hasil RMSE: 0.0000, menunjukkan model mampu merekonstruksi interaksi pengguna-kupon dengan sangat baik.

