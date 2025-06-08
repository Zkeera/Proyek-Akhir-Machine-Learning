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

Dataset ini merupakan data simulasi dari sistem kupon dalam kendaraan, terdiri dari berbagai informasi pengguna dan respons terhadap kupon.

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

## Modeling

- Algoritma utama yang digunakan adalah **K-Means Clustering**.
- Dilakukan pencarian jumlah cluster optimal menggunakan **Metode Elbow** dan evaluasi dengan **Silhouette Score**.
- Setelah ditentukan jumlah cluster yang optimal (misalnya k=4), dilakukan pemodelan ulang dan analisis komposisi tiap cluster.
- Untuk visualisasi, dilakukan reduksi dimensi menggunakan **PCA** ke dalam 2 dimensi.

### Hasil Visualisasi:
- Visualisasi hasil clustering menunjukkan adanya pengelompokan pengguna yang cukup jelas setelah reduksi PCA.
- Komposisi setiap cluster dianalisis berdasarkan fitur dominan.

## Evaluation

### Metrik Evaluasi: **Silhouette Score**

- **Silhouette Score** digunakan untuk menilai kualitas hasil clustering. Nilai berkisar antara -1 hingga 1, dengan nilai lebih tinggi menunjukkan pengelompokan yang lebih baik.
- Pada eksperimen ini, didapatkan Silhouette Score sebesar **~0.23**, yang mengindikasikan bahwa hasil clustering cukup baik meskipun masih dapat ditingkatkan.
- Hasil evaluasi menunjukkan bahwa pengguna dalam setiap cluster memiliki karakteristik yang berbeda dan dapat digunakan untuk rekomendasi kupon yang lebih relevan.

---

