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

- temperature: Suhu yang tercatat pada waktu tertentu yang dapat mempengaruhi keputusan atau preferensi pengguna terhadap penawaran kupon.

- time: Waktu saat pengguna melakukan transaksi atau interaksi dengan sistem, yang penting untuk memahami kebiasaan atau pola pengguna.

- expiration: Tanggal kadaluarsa kupon, yang menentukan relevansi kupon tersebut bagi pengguna.

- RestaurantLessThan20, Restaurant20To50: Kategori harga untuk restoran, menentukan jenis penawaran berdasarkan harga rata-rata makanan.

- CoffeeHouse: Menunjukkan apakah pengguna lebih sering mengunjungi kedai kopi, yang dapat mempengaruhi jenis kupon yang disarankan.

- CarryAway: Menunjukkan apakah pengguna cenderung membeli untuk dibawa pulang, yang dapat mengindikasikan preferensi tertentu.

- toCoupon_GEQ5min: Waktu yang dibutuhkan pengguna untuk mendapatkan kupon, yang bisa menjadi faktor dalam keputusan pembelian.

- toCoupon_GEQ15min: Menunjukkan apakah waktu perjalanan ke tempat kupon lebih dari 25 menit.

- toCoupon_GEQ25min: Menunjukkan apakah waktu perjalanan ke tempat kupon lebih dari 25 menit.

- direction_same: Menunjukkan arah perjalanan pengguna, yang dapat berhubungan dengan penawaran lokasi spesifik.

- Y: Target label yang menunjukkan apakah kupon diterima atau tidak, yang sangat penting untuk membangun model klasifikasi.

- Bar: Menunjukkan jenis bar tempat pengguna biasanya mengunjungi.

- direction_opp: Mengindikasikan apakah pengguna bergerak ke arah yang berlawanan dari lokasi kupon yang ditawarkan.

- Income: Mewakili kelompok pendapatan pengguna, yang membantu dalam memahami daya beli mereka dan penawaran kupon yang relevan.

- Car: Menunjukkan apakah pengguna memiliki mobil. Fitur ini dapat berguna untuk memfilter penawaran yang terkait dengan layanan atau promosi kendaraan.

## Data Preparation

Pembuatan User Profile:

Membuat profil pengguna berdasarkan informasi yang tersedia dalam dataset, seperti gender, age, maritalStatus, haschildren, dan occupation.

TF-IDF untuk Content-Based Filtering
Pada tahap ini, kami menggunakan TF-IDF (Term Frequency-Inverse Document Frequency) untuk ekstraksi fitur dari data kupon, yang digunakan dalam Content-Based Filtering. Teknik ini digunakan untuk menghitung relevansi kupon terhadap preferensi pengguna.

TF-IDF digunakan untuk mengukur pentingnya setiap kata atau fitur dalam deskripsi kupon, yang kemudian diterjemahkan menjadi vektor fitur yang dapat digunakan dalam algoritma rekomendasi berbasis konten.

Tahapan persiapan data yang dilakukan meliputi langkah-langkah berikut:

Pembuatan Kolom user_id:

Kolom user_id dibuat dengan kode df['user_id'] = df.index, yang memetakan setiap pengguna ke ID yang unik. Langkah ini penting untuk Collaborative Filtering dan perlu dicantumkan dengan jelas di bagian Data Preparation, agar alur persiapan data lebih terstruktur.

Label Encoding:

Menggunakan Label Encoding untuk mengubah fitur coupon menjadi coupon_id, yang memungkinkan pemrosesan kupon secara numerik.

Setelah langkah-langkah persiapan data ini selesai, dataset sudah siap untuk diproses menggunakan model rekomendasi.

Tujuan data preparation ini adalah untuk mempersiapkan data agar dapat diproses dengan optimal oleh algoritma clustering.

a. Untuk Content-Based Filtering (CBF):
Beberapa fitur teks (seperti occupation, age, income, maritalStatus, dll) digabungkan menjadi satu fitur baru bernama user_profile.

Data kupon direpresentasikan menggunakan kolom coupon.

b. Untuk Collaborative Filtering (CF):
Label pada kolom coupon diubah menjadi numerik menggunakan Label Encoding, menjadi coupon_id.

Matriks interaksi dibuat menggunakan user_id sebagai indeks dan coupon_id sebagai kolom dalam pivot table, dengan nilai yang menunjukkan apakah pengguna menerima kupon tersebut (Y). Matriks interaksi ini kemudian digunakan untuk modeling Collaborative Filtering.

Pivot Table: df.pivot_table(index='user_id', columns='coupon_id', values='Y', fill_value=0)

Diterapkan TruncatedSVD untuk dekomposisi dimensi rendah matriks interaksi.

## Modeling

TruncatedSVD:

TruncatedSVD digunakan untuk dekomposisi matriks dalam Collaborative Filtering berbasis faktorisasi matriks. Ini lebih cocok dijelaskan di bagian Modeling.

a. Content-Based Filtering (CBF):

Pada bagian ini, kami menggunakan metode Content-Based Filtering (CBF) untuk memberikan rekomendasi berdasarkan kemiripan antara profil pengguna dan kupon yang ada. Profil pengguna dibuat dengan menggabungkan beberapa fitur yang relevan, seperti gender, age, maritalStatus, occupation, dan lainnya.
Output untuk CBF menunjukkan kupon yang paling mirip berdasarkan profil pengguna.

Cosine Similarity dihitung menggunakan kode berikut:

```
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(df['user_profile'])
cos_sim = cosine_similarity(tfidf_matrix[user_index], tfidf_matrix)
similar_indices = cos_sim.argsort()[0][-6:-1][::-1]
df.iloc[similar_indices][['coupon', 'destination', 'Y']]
```

b. Collaborative Filtering (CF):

Untuk Collaborative Filtering, kami menggunakan teknik TruncatedSVD untuk memfaktorisasi matriks interaksi pengguna dan kupon. Matriks interaksi ini dibentuk dengan menggunakan user_id sebagai indeks dan coupon_id sebagai kolom, dengan nilai yang menunjukkan apakah pengguna menerima kupon tersebut.

Matriks prediksi dihasilkan dengan mengalikan matriks laten yang dihasilkan oleh SVD dengan komponen yang diperoleh dari dekomposisi.

Evaluasi untuk Collaborative Filtering dilakukan dengan menghitung RMSE (Root Mean Square Error).

SVD dilakukan dengan kode berikut:

```
svd = TruncatedSVD(n_components=5, random_state=42)
latent_matrix = svd.fit_transform(interaction_array)
predicted_matrix = np.dot(latent_matrix, svd.components_)
rmse = mean_squared_error(true_values, predicted_values) ** 0.5
print(f"RMSE Collaborative Filtering (TruncatedSVD): {rmse:.4f}")
```

## Evaluation

### Metrik Evaluasi:

a. Evaluasi Content-Based Filtering:
Untuk mengevaluasi kinerja model Content-Based Filtering, kami menggunakan Precision@5, yang mengukur seberapa akurat 5 rekomendasi teratas yang diberikan kepada pengguna. Metrik ini mengukur tingkat relevansi dari rekomendasi berdasarkan skor kemiripan antara profil pengguna dan kupon.
Menggunakan cosine similarity antara user profile dan kupon.

Top-N rekomendasi dihasilkan dengan memilih kupon yang paling mirip berdasarkan skor kemiripan.

Contoh Output Top-N untuk CBF:
User: Executive_30-39_high income
Top-5 Kupon: ['Coffee House', 'Bar', 'Restaurant(<20)', 'Carry out & Take away', 'Restaurant(20-50)']

Hasil Precision@5 menunjukkan bahwa seluruh rekomendasi yang diberikan untuk pengguna pertama (user_index = 0) sangat relevan dengan preferensinya, yang menghasilkan nilai Precision@5 = 1.00.

| No | Coupon          | Destination     |
| -- | --------------- | --------------- |
| 1  | Restaurant(<20) | No Urgent Place |
| 2  | Restaurant(<20) | No Urgent Place |
| 3  | Restaurant(<20) | No Urgent Place |
| 4  | Restaurant(<20) | No Urgent Place |
| 5  | Restaurant(<20) | No Urgent Place |

Dengan hasil ini, kita dapat menyimpulkan bahwa Content-Based Filtering memberikan rekomendasi yang sangat tepat bagi pengguna berdasarkan profil mereka.

b. Evaluasi Collaborative Filtering:

Matriks interaksi direduksi menggunakan TruncatedSVD, lalu dikalikan kembali untuk mendapatkan prediksi skor antar user dan kupon.
Untuk model Collaborative Filtering, kami menggunakan Root Mean Square Error (RMSE) untuk mengukur seberapa akurat sistem dalam memprediksi interaksi antara pengguna dan kupon. RMSE adalah metrik yang mengukur perbedaan antara nilai yang diprediksi oleh model dengan nilai aktual yang terjadi, dan semakin rendah nilai RMSE, semakin baik prediksi model tersebut.

Hasil RMSE untuk Collaborative Filtering adalah 0.0000, yang menunjukkan bahwa model sangat baik dalam merekonstruksi interaksi pengguna dengan kupon yang ditawarkan.

Contoh Output Top-N untuk CF:
User ID: 5
Top-5 Kupon berdasarkan skor prediksi: ['Bar', 'Coffee House', 'Restaurant(20-50)', 'Carry out & Take away', 'Restaurant(<20)']

Top-N Rekomendasi untuk User ID: 5:
| No | Coupon ID | Predicted Score              |
| -- | --------- | ---------------------------- |
| 1  | 4         | 0.9999999999999996           |
| 2  | 0         | 0.0                          |
| 3  | 3         | -8.326672684688607e-17       |
| 4  | 2         | -2.017589362129876e-16       |
| 5  | 1         | -4.440892098500626e-16       |

Dengan hasil RMSE yang sangat rendah, ini menunjukkan bahwa Collaborative Filtering mampu memberikan prediksi yang sangat baik terkait preferensi pengguna berdasarkan interaksi mereka dengan kupon yang tersedia.

Sebuah heatmap digunakan untuk memvisualisasikan matriks interaksi antara pengguna dan kupon. Visualisasi ini memberikan gambaran mengenai bagaimana interaksi pengguna dengan kupon yang ditawarkan.

```
sns.heatmap(interaction_matrix, cmap="YlGnBu", cbar=True)
plt.title("User-Coupon Interaction Matrix")
plt.show()
```
