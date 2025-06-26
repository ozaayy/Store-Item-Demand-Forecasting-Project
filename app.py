import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import joblib
import io
import os

st.set_page_config(layout="wide")

st.title("Mağaza Ürün Talep Tahmin Uygulaması")
st.write("Bu uygulama, LSTM modeli kullanarak mağaza ürün satışlarını tahmin eder. Lütfen gerekli dosyaları yükleyin ve tahmin işlemini başlatın.")

# Fonksiyon tanımlamaları
@st.cache_data
def prepare_sales_data(train_data_path, test_data_path):
    df_train = pd.read_csv(train_data_path)
    df_test = pd.read_csv(test_data_path)

    if 'sales' in df_test.columns:
        df_test = df_test.drop(columns=['sales'])
    if 'id' not in df_train.columns:
         df_train['id'] = np.nan

    df_combined = pd.concat([df_train, df_test], ignore_index=True, sort=False)
    df_combined['date'] = pd.to_datetime(df_combined['date'], format='mixed', dayfirst=False, errors='raise')
    df_combined.sort_values(by=['date', 'store', 'item'], inplace=True)
    df_combined.set_index('date', inplace=True)
    df_combined.index = pd.DatetimeIndex(df_combined.index)

    # --- Orijinal Mağaza/Ürün ID'lerini Koru ---
    df_combined['store_id'] = df_combined['store']
    df_combined['item_id'] = df_combined['item']

    # --- Temel Zaman Tabanlı Özellikler ---
    df_combined['year'] = df_combined.index.year
    df_combined['month'] = df_combined.index.month
    df_combined['day'] = df_combined.index.day
    df_combined['dayofweek'] = df_combined.index.dayofweek
    df_combined['dayofyear'] = df_combined.index.dayofyear
    df_combined['weekofyear'] = df_combined.index.isocalendar().week.astype(int)
    df_combined['quarter'] = df_combined.index.quarter
    df_combined['is_weekend'] = (df_combined.index.dayofweek >= 5).astype(int)
    df_combined['is_new_year_day'] = ((df_combined.index.month == 1) & (df_combined.index.day == 1)).astype(int)

    windows = [7, 30, 90]
    for window in windows:
        df_combined[f'sales_rolling_mean_{window}d'] = df_combined.groupby(['store', 'item'])['sales'].transform(lambda x: x.shift(1).rolling(window=window).mean())
        df_combined[f'sales_rolling_median_{window}d'] = df_combined.groupby(['store', 'item'])['sales'].transform(lambda x: x.shift(1).rolling(window=window).median())
        df_combined[f'sales_rolling_std_{window}d'] = df_combined.groupby(['store', 'item'])['sales'].transform(lambda x: x.shift(1).rolling(window=window).std())
        df_combined[f'sales_rolling_min_{window}d'] = df_combined.groupby(['store', 'item'])['sales'].transform(lambda x: x.shift(1).rolling(window=window).min())
        df_combined[f'sales_rolling_max_{window}d'] = df_combined.groupby(['store', 'item'])['sales'].transform(lambda x: x.shift(1).rolling(window=window).max())

    df_combined['sales_lag_365d'] = df_combined.groupby(['store', 'item'])['sales'].transform(lambda x: x.shift(365))
    df_combined['sales_lag_30d'] = df_combined.groupby(['store', 'item'])['sales'].transform(lambda x: x.shift(30))
    df_combined['sales_lag_7d'] = df_combined.groupby(['store', 'item'])['sales'].transform(lambda x: x.shift(7))

    df_combined = pd.get_dummies(df_combined, columns=['store', 'item'], drop_first=True)
    df_combined['month_dayofweek_interaction'] = df_combined['month'] * (df_combined['dayofweek'] + 1)

    feature_columns = [col for col in df_combined.columns if col not in ['sales', 'id', 'store_id', 'item_id']]
    
    # Lag ve rolling özelliklerini belirle
    lag_and_roll_cols = [col for col in feature_columns if 'lag' in col or 'rolling' in col]

    # Her mağaza-ürün grubu içinde bu özelliklerdeki NaN'leri ileriye doğru doldur (ffill)
    # Bu, test setindeki boşlukları train setinin son verileriyle doldurur.
    df_combined[lag_and_roll_cols] = df_combined.groupby(['store_id', 'item_id'])[lag_and_roll_cols].ffill()

    # ffill sonrası hala kalabilecek NaN'leri SADECE özellik sütunlarında 0 ile doldur.
    # Bu, test setindeki 'sales' sütununun NaN kalmasını sağlar, böylece test seti doğru ayırt edilebilir.
    df_combined[feature_columns] = df_combined[feature_columns].fillna(0)

    target_column = 'sales'

    print(f"Toplam özellik sayısı (model girdisi): {len(feature_columns)}")

    return df_combined, feature_columns, target_column

def create_timewindow(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X[i:(i + time_steps)]
        Xs.append(v)
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

# Streamlit Arayüzü
with st.sidebar:
    st.header("Dosyaları Yükleyin")
    uploaded_train_csv = st.file_uploader("Eğitim Veri Setini Yükleyin (train.csv)", type="csv")
    uploaded_test_csv = st.file_uploader("Test Veri Setini Yükleyin (test.csv)", type="csv")
    uploaded_model = st.file_uploader("Model Dosyasını Yükleyin (.keras)", type=["keras", "h5"])

    st.info("Not: Ölçekleyici dosyaları (`scalerX.save`, `scalerY.save`) proje klasöründen otomatik olarak yüklenecektir.")

    predict_button = st.button("Tahminleri Oluştur", type="primary")

if predict_button:
    # Ölçekleyici dosyalarının yollarını koda sabit olarak ekle
    scaler_x_path = 'scalerX.save'
    scaler_y_path = 'scalerY.save'

    if all([uploaded_train_csv, uploaded_test_csv, uploaded_model]):
        with st.spinner('Lütfen bekleyin... Dosyalar işleniyor ve tahminler yapılıyor.'):
            try:
                # 1. Modeli ve ölçekleyicileri yükle
                st.subheader("Adım 1: Model ve Ölçekleyiciler Yükleniyor")

                with open("temp_model.keras", "wb") as f:
                    f.write(uploaded_model.getbuffer())
                loaded_model = tf.keras.models.load_model("temp_model.keras")

                # Ölçekleyici dosyalarının varlığını kontrol et
                if not os.path.exists(scaler_x_path) or not os.path.exists(scaler_y_path):
                    st.error(f"Ölçekleyici dosyaları proje klasöründe bulunamadı. Lütfen '{scaler_x_path}' ve '{scaler_y_path}' dosyalarının `app.py` ile aynı dizinde olduğundan emin olun.")
                    st.stop()
                
                # Joblib dosyalarını sabit yoldan yükle
                loaded_scaler_x = joblib.load(scaler_x_path)
                loaded_scaler_y = joblib.load(scaler_y_path)
                st.success("Model ve ölçekleyiciler başarıyla yüklendi.")

                # 2. Veriyi hazırla ve doğru şekilde böl
                st.subheader("Adım 2: Veri Hazırlanıyor ve Bölünüyor")
                df_processed, feature_cols, target_col = prepare_sales_data(uploaded_train_csv, uploaded_test_csv)
                
                # Veriyi train ve test olarak doğru şekilde ayır.
                # Test verisi, 'sales' sütununun başta NaN olduğu kısımdır.
                train_df = df_processed[df_processed[target_col].notna()].copy()
                test_df = df_processed[df_processed[target_col].isna()].copy()
                st.success(f"Test verisi doğru şekilde ayrıldı. Tahmin edilecek {test_df.shape[0]} satır bulundu.")
                
                # Analiz sayfasında kullanmak için geçmiş veriyi de kaydet
                st.session_state['historical_df'] = train_df
                
                test_x = loaded_scaler_x.transform(test_df[feature_cols])

                time_steps = 30
                # X_test'i oluştururken artık y_test'e ihtiyacımız yok, create_timewindow fonksiyonunu buna göre düzenleyelim
                X_test, _ = create_timewindow(test_x, np.zeros(len(test_x)), time_steps)
                st.success(f"Test verisi (X_test) tahmin için hazırlandı. Boyut: {X_test.shape}")

                # 3. Tahmin yap
                st.subheader("Adım 3: Tahminler Yapılıyor")
                test_predictions_scaled = loaded_model.predict(X_test, verbose=0)
                st.success(f"Tahminler tamamlandı. Boyut: {test_predictions_scaled.shape}")

                # 4. Tahminleri orijinal ölçeğe dönüştür
                st.subheader("Adım 4: Tahminler Geri Dönüştürülüyor")
                test_predictions_original_scale = loaded_scaler_y.inverse_transform(test_predictions_scaled).flatten()
                st.success(f"Orijinal ölçekteki tahminlerin boyutu: {test_predictions_original_scale.shape}")

                # 5. Sonuçları birleştir ve göster
                st.subheader("Adım 5: Sonuçlar Hazırlanıyor")
                test_df_for_predictions = test_df.iloc[time_steps:].copy()
                test_df_for_predictions['sales_predicted'] = test_predictions_original_scale

                final_prediction_df = test_df_for_predictions.copy()

                if 'id' in final_prediction_df.columns:
                    # --- Session State'e kaydetme ---
                    # Analiz sayfasında kullanmak için sonuçları session state'e kaydedin.
                    st.session_state['final_prediction_df'] = final_prediction_df.copy()

                    submission_df = final_prediction_df[final_prediction_df['id'].notna()].copy()
                    submission_df = submission_df[['id', 'sales_predicted']].copy()
                    submission_df['id'] = submission_df['id'].astype(int)
                    submission_df['sales_predicted'] = submission_df['sales_predicted'].apply(lambda x: max(0.0, x)).round().astype(int)
                    submission_df_sorted = submission_df.sort_values(by='id').copy()

                    st.success("Nihai tahmin DataFrame'i başarıyla oluşturuldu.")
                    st.dataframe(submission_df_sorted.head())

                    # Download button
                    csv = submission_df_sorted.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Tahminleri İndir (submission.csv)",
                        data=csv,
                        file_name='submission.csv',
                        mime='text/csv',
                    )

                    st.success("Tahmin sonuçlarının detaylı analizini ve görselleştirmelerini görmek için yandaki menüden 'Analiz' sayfasına geçebilirsiniz.")

                else:
                    st.warning("Uyarı: 'id' sütunu işlenmiş test verisinde bulunamadı.")
                    # Handle case where 'id' is not present
                    submission_df = final_prediction_df[['sales_predicted']].copy()
                    submission_df['sales_predicted'] = submission_df['sales_predicted'].apply(lambda x: max(0.0, x)).round().astype(int)
                    st.dataframe(submission_df.head())


            except Exception as e:
                st.error(f"Bir hata oluştu: {e}")
                st.error("Lütfen yüklediğiniz dosyaların ve formatların doğru olduğundan emin olun.")

    else:
        st.error("Lütfen devam etmek için GEREKLİ TÜM dosyaları (train, test, model) yükleyin.") 