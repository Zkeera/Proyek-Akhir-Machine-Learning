# Laporan Proyek Machine Learning - Sistem Rekomendasi Kupon Dalam Kendaraan

## Domain Proyek

Sistem rekomendasi kupon dalam kendaraan (*In-Vehicle Coupon Recommendation*) merupakan bagian penting dalam industri transportasi modern dan periklanan berbasis konteks. Pengemudi atau penumpang sering kali dihadapkan pada situasi di mana penawaran kupon diskon untuk restoran, kedai kopi, atau minimarket dapat menjadi sangat relevan berdasarkan tujuan perjalanan, cuaca, waktu, maupun profil demografis mereka. Proyek ini bertujuan untuk membangun sistem rekomendasi yang mampu mencocokkan kupon yang tepat kepada pengguna yang berada di dalam kendaraan berdasarkan konteks situasi dan profil mereka.

Dataset yang digunakan berasal dari UCI Machine Learning Repository yang berisi kombinasi data demografis, perilaku, dan konteks pengguna saat ditawari kupon di dalam kendaraan.

## Business Understanding

### Problem Statements

* Bagaimana cara memberikan rekomendasi kupon yang relevan kepada pengguna berdasarkan profil demografis dan konteks perjalanan mereka di dalam kendaraan?
* Pendekatan sistem rekomendasi apa yang paling efektif untuk memprediksi penerimaan kupon oleh pengguna (*Content-Based Filtering* atau *Collaborative Filtering*)?

### Goals

* Membangun model *Content-Based Filtering* untuk merekomendasikan kupon berdasarkan kemiripan profil pengguna dan konteks.
* Membangun model *Collaborative Filtering* menggunakan teknik *matrix factorization* (*Truncated SVD*) untuk memprediksi preferensi atau interaksi pengguna terhadap kupon.


* Mengevaluasi performa model menggunakan metrik yang sesuai seperti Akurasi/Precision dan *Root Mean Squared Error* (RMSE).



### Solution Statements

* Mengembangkan sistem berbasis **Content-Based Filtering** menggunakan *TF-IDF Vectorizer* dan *Cosine Similarity* pada profil gabungan pengguna (*user profile*).


* Mengembangkan sistem berbasis **Collaborative Filtering** menggunakan *Truncated SVD* pada matriks interaksi pengguna-kupon.


* Melakukan evaluasi kinerja model menggunakan metrik *Precision@5* untuk *Content-Based Filtering* dan *RMSE* untuk *Collaborative Filtering*.



## Data Understanding

Dataset memuat kombinasi data demografis, perilaku, dan konteks pengguna dalam kendaraan saat ditawari kupon. Dataset ini mencakup berbagai variabel seperti tujuan perjalanan (*destination*), penumpang (*passanger*), cuaca (*weather*), suhu, waktu, jenis kupon, masa berlaku (*expiration*), hingga informasi demografi seperti usia, jenis kelamin, status pernikahan, dan pekerjaan.

Kolom target utama adalah kolom `Y` yang menunjukkan apakah kupon tersebut diterima atau digunakan oleh pengguna (1 = Ya, 0 = Tidak).

## Data Preparation

Berikut adalah langkah-langkah *data preparation* yang dilakukan:

1. **Penggabungan Fitur Profil Pengguna**: Menggabungkan beberapa kolom demografis dan kontekstual (seperti *destination, passanger, weather, coupon, gender, age, maritalStatus, has_children, education, occupation*) menjadi satu string teks tunggal bernama `user_profile`.


2. **Transformasi Teks dengan TF-IDF**: Mengubah fitur teks `user_profile` menjadi representasi matriks numerik menggunakan *TF-IDF Vectorizer* untuk kebutuhan *Content-Based Filtering*.


3. **Penyusunan Matriks Interaksi**: Membuat matriks interaksi antara pengguna (*user_id*) dan item (*coupon_id*) berdasarkan nilai penerimaan kupon (`Y`) untuk kebutuhan *Collaborative Filtering*.



## Modeling

### Algoritma yang digunakan:

* **Content-Based Filtering**
* **Cara Kerja**: Membandingkan profil pengguna menggunakan *cosine similarity* berdasarkan representasi vektor TF-IDF dari fitur-fitur kontekstual dan demografis. Model ini merekomendasikan kupon yang memiliki karakteristik paling mirip dengan profil atau preferensi kupon yang pernah direspons positif oleh pengguna.




* **Collaborative Filtering (Truncated SVD)**
* **Cara Kerja**: Menggunakan teknik *matrix factorization* dengan *Truncated SVD* dari pustaka `scikit-learn` untuk mengurai matriks interaksi user-item menjadi dimensi laten yang lebih kecil, lalu merekonstruksinya guna memprediksi skor interaksi kupon yang belum dievaluasi oleh pengguna.





## Evaluation

### Model dievaluasi menggunakan metrik yang relevan untuk masing-masing pendekatan:

* **Content-Based Filtering**: Dievaluasi menggunakan metrik akurasi rekomendasi (*Precision@5*), di mana model diuji untuk melihat seberapa relevan top-5 kupon yang direkomendasikan. Hasil evaluasi menunjukkan nilai akurasi top-5 mencapai `1.00`, yang berarti item yang direkomendasikan selaras dengan konteks data aktual.


* **Collaborative Filtering**: Dievaluasi menggunakan *Root Mean Squared Error* (RMSE) pada matriks interaksi. Nilai RMSE yang diperoleh adalah `0.0000`, menunjukkan tingkat kecocokan rekonstrusi matriks yang optimal pada data latih interaksi non-nol.



**Insight:**

* *Content-Based Filtering* sangat handal dalam menangani informasi kontekstual langsung seperti cuaca, tujuan, dan profil demografi pengguna.


* *Collaborative Filtering (SVD)* memberikan fondasi yang kuat dalam memprediksi kecenderungan pola penerimaan kupon berbasis histori matriks interaksi.



## Conclusion

Sistem rekomendasi kupon dalam kendaraan berhasil dibangun dengan menerapkan dua pendekatan utama, yaitu *Content-Based Filtering* dan *Collaborative Filtering*.

**Kesimpulan Utama:**

* Penggabungan fitur kontekstual dan profil ke dalam representasi teks terbukti efektif diterapkan pada *Content-Based Filtering*.


* Pendekatan *Collaborative Filtering* dengan *Truncated SVD* mampu memetakan interaksi pengguna terhadap variasi kupon secara sistematis.


* Dalam implementasi sistem nyata di dalam kendaraan (*in-vehicle system*), pendekatan *hybrid* yang menggabungkan kedua metode ini akan jauh lebih optimal guna mengatasi tantangan seperti *cold-start problem* sekaligus meningkatkan personalisasi kupon bagi pengemudi.
