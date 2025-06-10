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
Dataset diambil dari GitHub dan dapat diakses melalui tautan berikut:  
https://github.com/taqi1502/Proyek-Akhir-Machine-Learning/blob/main/dataset.csv

### Deskripsi Fitur

Beberapa fitur penting dalam dataset antara lain:
- `Bar`, `CoffeeHouse`, `CarryAway`, `RestaurantLessThan20`, `Restaurant20To50`: Frekuensi kunjungan ke tempat tersebut dalam sebulan terakhir.
- `toCoupon_GEQ5min`: Estimasi waktu menuju lokasi penggunaan kupon.
- `direction_same`: Apakah arah tujuan pengguna sama dengan arah ke lokasi kupon.
- `Y`: Label target, menunjukkan apakah pengguna akan menggunakan kupon (yes/no).

### Variabel-variabel dalam dataset:
- `gender`: Jenis kelamin pengguna
- `age`: Umur pengguna
- `car_owner`: Kepemilikan kendaraan
- `income`: Pendapatan pengguna
- `marital_status`: Status pernikahan
- `children`: Jumlah anak
- `education`: Tingkat pendidikan
- `occupation`: Jenis pekerjaan
- `coupon`: Jenis kupon yang diberikan
- `expiration`: Durasi masa berlaku kupon
- `coffeehouse`, `bar`, `carryaway`, dst: Frekuensi kunjungan ke tempat-tempat tersebut
- `destination`: Tujuan perjalanan
- `passanger`: Penumpang yang ikut dalam perjalanan
- `weather`, `temperature`, `time`, `weekday`: Kondisi saat perjalanan

## Data Preparation

- **Encoding variabel kategorik** menggunakan One-Hot Encoding karena sebagian besar fitur bersifat kategorikal.
- **Normalisasi fitur numerik** menggunakan StandardScaler untuk memastikan bahwa setiap fitur memiliki skala yang sebanding sebelum dilakukan clustering.
- **Menggabungkan** fitur yang telah diencoding dan dinormalisasi menjadi satu matriks fitur.

Tujuan data preparation ini adalah untuk mempersiapkan data agar dapat diproses dengan optimal oleh algoritma clustering.

a. Untuk Content-Based Filtering (CBF):
Beberapa fitur teks (seperti occupation, age, income, maritalStatus, dll) digabungkan menjadi satu fitur baru bernama user_profile.

Dilakukan TF-IDF Vectorization pada kolom user_profile untuk menghasilkan representasi numerik.

Data kupon direpresentasikan menggunakan kolom coupon.

b. Untuk Collaborative Filtering (CF):
Label pada kolom coupon diubah menjadi numerik menggunakan Label Encoding, menjadi coupon_id.

Dibuat pivot table untuk menghasilkan matrix interaksi user-item, dengan baris sebagai user_profile dan kolom coupon_id.

Diterapkan TruncatedSVD untuk dekomposisi dimensi rendah matriks interaksi.

## Modeling

- Algoritma utama yang digunakan adalah **K-Means Clustering**.
- Dilakukan pencarian jumlah cluster optimal menggunakan **Metode Elbow** dan evaluasi dengan **Silhouette Score**.
- Setelah ditentukan jumlah cluster yang optimal (misalnya k=4), dilakukan pemodelan ulang dan analisis komposisi tiap cluster.
- Untuk visualisasi, dilakukan reduksi dimensi menggunakan **PCA** ke dalam 2 dimensi.

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

