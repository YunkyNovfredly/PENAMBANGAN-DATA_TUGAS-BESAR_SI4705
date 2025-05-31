import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, r2_score, mean_squared_error
import os # Import os module

# Set page config
st.set_page_config(
    page_title="Analisis Model Prediksi Harga",
    page_icon="📊",
    layout="wide"
)

# Load data
@st.cache_data
def load_data():
    try:
        # Use 'retail.csv' as loaded earlier in the notebook
        df = pd.read_csv('retail.csv')
        # Perform basic cleaning/preparation done earlier in the notebook
        df['product'].fillna(df['product'].mode()[0], inplace=True)
        df['brand'].fillna(df['brand'].mode()[0], inplace=True)
        # Handle 'description' if it exists, although it wasn't in the final info()
        if 'description' in df.columns:
             df['description'].fillna(df['description'].mode()[0], inplace=True)
        if 'rating' in df.columns:
            df = df.drop(columns=['rating'])

        # Add 'profit' column as done in EDA
        df['profit'] = df['sale_price'] - df['market_price']
        # Add margin column
        df['margin'] = ((df['sale_price'] - df['market_price']) / df['market_price']) * 100

        return df
    except FileNotFoundError:
        st.error("Error: File 'retail.csv' not found.")
        return None
    except Exception as e:
        st.error(f"Error loading or preparing data: {e}")
        return None

# Function to load models from disk
@st.cache_resource # Use cache_resource for models
def load_models():
    models = {}
    # List model files you expect
    model_files = {
        'Linear Regression': 'linear_regression_model.pkl',
        'Decision Tree': 'decision_tree_model.pkl',
        'KMeans Clustering': 'kmeans_model.pkl'
    }

    for model_name, file_path in model_files.items():
        if os.path.exists(file_path):
            try:
                with open(file_path, 'rb') as f:
                    models[model_name] = pickle.load(f)
                #st.sidebar.success(f"{model_name} loaded successfully!") # Optional: show success in sidebar
            except Exception as e:
                st.sidebar.error(f"Error loading {model_name} from {file_path}: {e}")
        else:
            st.sidebar.warning(f"Model file not found: {file_path}")

    return models

# Custom prediction function untuk Decision Tree
def predict_sale_price(model, market_prices_df):
    # Ensure input is a DataFrame with 'market_price' column
    if not isinstance(market_prices_df, pd.DataFrame) or 'market_price' not in market_prices_df.columns:
        raise ValueError("Input to predict_sale_price must be a DataFrame with a 'market_price' column.")

    predictions = model.predict(market_prices_df)
    market_prices_array = market_prices_df['market_price'].values
    # Adjust predictions to ensure sale_price >= 1.1 * market_price
    # Apply adjustment only if model is DecisionTreeRegressor, as LR was not trained with this constraint
    if isinstance(model, DecisionTreeRegressor):
        adjusted_predictions = np.maximum(predictions, market_prices_array * 1.1)
        return adjusted_predictions
    else: # For Linear Regression, just return the prediction
        return predictions


def perform_clustering(df):
    if df is None:
        return None # Return None instead of df, None
    try:
        # Menggunakan StandardScaler untuk menstandarisasi data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df[['market_price', 'sale_price']])

        # Menggunakan KMeans untuk clustering (using optimal k=3 from notebook)
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10) # Added n_init
        df['cluster'] = kmeans.fit_predict(scaled_data)

        # Menghitung silhouette score
        silhouette_avg = silhouette_score(scaled_data, df['cluster'])
        return df, silhouette_avg # Return df and score

    except Exception as e:
        st.error(f"Error during clustering: {e}")
        return df, None # Return df and None for score


def main():
    st.title("Dashboard Analisis Model Prediksi Harga")

    # Load data
    df = load_data()
    if df is None:
        return # Stop if data loading failed

    # Load all available models
    models = load_models()

    # Perform clustering and add cluster column to df (only if KMeans model is loaded)
    kmeans_model = models.get('KMeans Clustering')
    silhouette_avg = None
    if kmeans_model:
         # Use the *loaded* KMeans model to assign clusters to the DataFrame
         try:
             scaler = StandardScaler()
             scaled_data = scaler.fit_transform(df[['market_price', 'sale_price']])
             df['cluster'] = kmeans_model.predict(scaled_data) # Use predict with loaded model
             silhouette_avg = silhouette_score(scaled_data, df['cluster'])
         except Exception as e:
             st.error(f"Error applying loaded KMeans model: {e}")
             df['cluster'] = None # Set cluster to None if error occurs
    else:
        st.warning("KMeans model not loaded. Clustering analysis might be limited.")
        df['cluster'] = None # Ensure cluster column exists, even if None

    # Sidebar for model selection
    st.sidebar.title("Pilih Model")
    model_options = list(models.keys())
    if not model_options:
        st.sidebar.warning("Tidak ada model yang berhasil dimuat.")
        selected_model_name = None
        model = None
    else:
        selected_model_name = st.sidebar.selectbox("Pilih Model yang akan Dianalisis:", model_options)
        model = models[selected_model_name]


    # Conditional display based on whether a model is selected
    if model is not None:
        st.sidebar.write(f"Model Terpilih: **{selected_model_name}**")

        # Tampilkan tabs untuk berbagai analisis
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Model Info", "Feature Analysis", "Prediksi Harga", "Visualisasi Model", "Analisis Data"])

        with tab1:
            st.header(f"Informasi Model: {selected_model_name}")

            if isinstance(model, DecisionTreeRegressor):
                st.write("Tipe Model: Decision Tree Regressor")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Kedalaman Pohon", model.get_depth())
                with col2:
                    st.metric("Jumlah Leaf Nodes", model.get_n_leaves())

                # Feature importance - Check if the model was trained with features
                if hasattr(model, 'feature_importances_') and model.feature_importances_.size > 0:
                    st.subheader("Feature Importance")
                    # Assuming 'market_price' was the only feature used for these models
                    feature_names = ['market_price'] # Or get from model if available
                    importance = model.feature_importances_
                    importance_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': importance
                    })
                    st.dataframe(importance_df)
                else:
                    st.info("Feature importance not available for this model.")


            elif isinstance(model, LinearRegression):
                st.write("Tipe Model: Linear Regression")
                if hasattr(model, 'coef_') and hasattr(model, 'intercept_'):
                    st.write("Koefisien:", model.coef_[0])
                    st.write("Intercept:", model.intercept_)
                else:
                     st.info("Coefficients and Intercept not available for this model.")


            elif isinstance(model, KMeans):
                st.write("Tipe Model: KMeans Clustering")
                st.write("Jumlah Cluster:", model.n_clusters)
                if silhouette_avg is not None:
                    st.write("Silhouette Score:", silhouette_avg)
                else:
                     st.warning("Silhouette Score could not be calculated.")

            else:
                 st.warning("Informasi spesifik tidak tersedia untuk tipe model ini.")


        with tab2:
            st.header("Analisis Feature")

            # Scatter plot harga pasar vs harga jual
            st.subheader("Hubungan Harga Pasar vs Harga Jual")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(data=df, x='market_price', y='sale_price', alpha=0.5)
            plt.title("Scatter Plot Harga Pasar vs Harga Jual")
            plt.xlabel("Harga Pasar")
            plt.ylabel("Harga Jual")
            st.pyplot(fig)

            # Distribusi margin
            st.subheader("Analisis Margin")
            # Ensure 'margin' column exists from load_data
            if 'margin' in df.columns:
                 fig, ax = plt.subplots(figsize=(10, 6))
                 # Filter out infinite values that can result from market_price = 0
                 margin_data = df['margin'].replace([np.inf, -np.inf], np.nan).dropna()
                 sns.histplot(data=margin_data, bins=50)
                 plt.title("Distribusi Margin Harga")
                 plt.xlabel("Margin (%)")
                 plt.ylabel("Frekuensi")
                 st.pyplot(fig)

                 col1, col2 = st.columns(2)
                 with col1:
                     st.metric("Rata-rata Margin", f"{margin_data.mean():.2f}%")
                 with col2:
                     st.metric("Median Margin", f"{margin_data.median():.2f}%")
            else:
                 st.warning("Margin column not available in data.")


        with tab3:
            st.header("Prediksi Harga")

            if isinstance(model, (LinearRegression, DecisionTreeRegressor)):
                # Input untuk prediksi
                col1, col2 = st.columns(2)
                with col1:
                    input_price = st.number_input("Masukkan Harga Pasar:", min_value=0.01, value=100.0, format="%.2f") # Min value > 0 to avoid / 0

                if st.button("Prediksi"):
                    try:
                        # Pass input_price as a DataFrame for predict_sale_price
                        prediction = predict_sale_price(model, pd.DataFrame({'market_price': [input_price]}))[0]
                        margin = ((prediction - input_price) / input_price) * 100 if input_price != 0 else float('inf')

                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Prediksi Harga Jual", f"${prediction:.2f}")
                        with col2:
                             st.metric("Margin Prediksi", f"{margin:.2f}%" if margin != float('inf') else "N/A")

                    except Exception as e:
                        st.error(f"Error saat melakukan prediksi: {e}")

                # Tampilkan contoh prediksi untuk beberapa harga
                st.subheader("Contoh Prediksi untuk Berbagai Harga")
                sample_prices = pd.DataFrame({'market_price': [100, 500, 1000, 2000, 5000]})
                # Use the custom prediction function for both models for consistency and margin logic
                predictions = predict_sale_price(model, sample_prices)


                results = []
                # Ensure predictions is an iterable (like a list or numpy array)
                if isinstance(predictions, np.ndarray):
                    predictions_list = predictions.flatten()
                else:
                    predictions_list = predictions # Assume it's already list-like


                for market_price, pred in zip(sample_prices['market_price'], predictions_list):
                    # Calculate margin only if market_price is not zero to avoid division by zero
                    margin = ((pred - market_price) / market_price) * 100 if market_price != 0 else float('inf')
                    results.append({
                        'Harga Pasar': f"${market_price:.2f}",
                        'Prediksi Harga Jual': f"${pred:.2f}",
                        'Margin': f"{margin:.1f}%" if margin != float('inf') else "N/A"
                    })
                st.table(pd.DataFrame(results))
            else:
                st.info("Model yang dipilih bukan tipe regresi (Linear Regression atau Decision Tree) sehingga prediksi harga tidak didukung.")


        with tab4:
            st.header("Visualisasi Model")

            # st.write(f"Tipe model yang dimuat: {type(model)}") # Optional debug log

            if isinstance(model, DecisionTreeRegressor):
                st.subheader("Visualisasi Pohon Keputusan")
                # Decision Tree plot can be large, handle figure creation
                fig, ax = plt.subplots(figsize=(20, 10))
                plot_tree(model, feature_names=['market_price'], # Assuming 'market_price' is the feature name
                         filled=True, rounded=True, fontsize=10)
                plt.title('Visualisasi Pohon Keputusan')
                st.pyplot(fig)
                plt.close(fig) # Close figure to free memory


                # Plot prediksi vs aktual
                st.subheader("Prediksi vs Aktual (Decision Tree)")
                X_data = df[['market_price']] # Use the market_price column from the data
                y_pred = model.predict(X_data) # Predict using the loaded model

                fig, ax = plt.subplots(figsize=(10, 6))
                plt.scatter(df['market_price'], df['sale_price'], alpha=0.5, label='Aktual')
                plt.scatter(df['market_price'], y_pred, alpha=0.5, label='Prediksi')
                plt.legend()
                plt.title('Perbandingan Harga Aktual vs Prediksi (Decision Tree)')
                plt.xlabel('Harga Pasar')
                plt.ylabel('Harga Jual')
                 # Calculate R²
                try:
                    # Need to calculate R2 against the *original* sale price from df
                    r2 = r2_score(df['sale_price'], y_pred)
                    st.write(f"R² Score (vs actual sale_price): {r2:.4f}")
                except Exception as e:
                    st.warning(f"Could not calculate R²: {e}")

                st.pyplot(fig)
                plt.close(fig) # Close figure


            elif isinstance(model, KMeans):
                st.subheader("Visualisasi Hasil Clustering KMeans")

                # Check if cluster column exists and clustering was successful
                if 'cluster' in df.columns and df['cluster'] is not None:
                    # Plot hasil clustering
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.scatterplot(data=df, x='market_price', y='sale_price', hue='cluster', palette='viridis', alpha=0.5, ax=ax) # Pass ax
                    # Optionally plot centroids if available
                    # Centroids would need to be loaded or recalculated based on clustered data
                    if hasattr(model, 'cluster_centers_'):
                        scaler = StandardScaler() # Need scaler to inverse transform centroids
                        # Refit scaler on the data used for clustering in the notebook
                        # This is tricky - ideally save the scaler too or refit carefully
                        # For simplicity, let's assume we can refit on df for visualization purposes
                        try:
                           X_clustering_viz = df[['market_price', 'sale_price']]
                           scaler_viz = StandardScaler()
                           scaler_viz.fit(X_clustering_viz) # Fit on the data
                           centroids_original = scaler_viz.inverse_transform(model.cluster_centers_)
                           plt.scatter(centroids_original[:, 0], centroids_original[:, 1],
                                       c='red', marker='X', s=200, linewidths=3, label='Centroids')
                        except Exception as e:
                           st.warning(f"Could not visualize centroids: {e}")


                    plt.title('Hasil Clustering KMeans')
                    plt.xlabel('Harga Pasar')
                    plt.ylabel('Harga Jual')
                    plt.legend()
                    st.pyplot(fig)
                    plt.close(fig) # Close figure

                else:
                    st.warning("Clustering results not available in the DataFrame.")


                # Visualisasi Elbow Method (Still requires re-running KMeans for the plot)
                st.subheader("Visualisasi Elbow Method")
                inertias = []
                silhouette_scores_plot = []
                k_range = range(2, 11)

                # Re-run K-Means for the plot data (using df, not scaled_data for plotting consistency)
                # This part is less efficient as it re-runs the model, but necessary for the plot
                try:
                    X_plot_clustering = df[['market_price', 'sale_price']].dropna() # Handle potential NaNs
                    if not X_plot_clustering.empty:
                        scaler_plot = StandardScaler()
                        X_scaled_plot = scaler_plot.fit_transform(X_plot_clustering)

                        for k in k_range:
                            kmeans_plot = KMeans(n_clusters=k, random_state=42, n_init=10) # Added n_init
                            kmeans_plot.fit(X_scaled_plot)
                            inertias.append(kmeans_plot.inertia_)
                            # Check if silhouette_score is valid (requires >= 2 clusters and > 1 sample)
                            if X_scaled_plot.shape[0] > 1 and k >= 2:
                                score = silhouette_score(X_scaled_plot, kmeans_plot.labels_)
                                silhouette_scores_plot.append(score)
                            else:
                                silhouette_scores_plot.append(0) # Append 0 or NaN if not valid

                        # Plot Elbow Method
                        fig, ax = plt.subplots(figsize=(10, 6))
                        plt.plot(k_range, inertias, 'bx-')
                        plt.xlabel('Jumlah Cluster (k)')
                        plt.ylabel('Inertia')
                        plt.title('Elbow Method untuk Optimal k')
                        st.pyplot(fig)
                        plt.close(fig) # Close figure

                        # Plot Silhouette Score
                        st.subheader("Visualisasi Silhouette Score")
                        fig, ax = plt.subplots(figsize=(10, 6))
                        # Plot only for k >= 2 where score is valid
                        plt.plot(k_range[1:], silhouette_scores_plot[1:], 'rx-')
                        plt.xlabel('Jumlah Cluster (k)')
                        plt.ylabel('Silhouette Score')
                        plt.title('Silhouette Score untuk Optimal k')
                        st.pyplot(fig)
                        plt.close(fig) # Close figure

                    else:
                         st.warning("Not enough data for Elbow/Silhouette plots.")

                except Exception as e:
                    st.error(f"Error generating Elbow/Silhouette plots: {e}")


            elif isinstance(model, LinearRegression):
                st.subheader("Model Linear Regression")
                if hasattr(model, 'coef_') and hasattr(model, 'intercept_'):
                     st.write(f"Persamaan Regresi: Sale Price = {model.coef_[0]:.2f} * Market Price + {model.intercept_:.2f}")

                # Plot prediksi vs aktual untuk Linear Regression
                st.subheader("Prediksi vs Aktual (Linear Regression)")
                X_data = df[['market_price']] # Use the market_price column from the data
                y_pred = model.predict(X_data) # Predict using the loaded model

                fig, ax = plt.subplots(figsize=(10, 6))
                plt.scatter(df['market_price'], df['sale_price'], alpha=0.5, label='Aktual')
                plt.scatter(df['market_price'], y_pred, alpha=0.5, label='Prediksi', color='orange')
                plt.plot(df['market_price'], y_pred, color='red', label='Garis Regresi')  # Garis regresi
                plt.legend()
                plt.title('Perbandingan Harga Aktual vs Prediksi (Linear Regression)')
                plt.xlabel('Harga Pasar')
                plt.ylabel('Harga Jual')

                # Calculate R² and RMSE
                try:
                    r2 = r2_score(df['sale_price'], y_pred)
                    mse = mean_squared_error(df['sale_price'], y_pred)
                    rmse = np.sqrt(mse)
                    st.write(f"R² Score (vs actual sale_price): {r2:.4f}")
                    st.write(f"RMSE (vs actual sale_price): {rmse:.2f}")

                     # Add equation and R² to plot
                    equation = f'y = {model.coef_[0]:.2f}x + {model.intercept_:.2f}'
                    r2_text = f'R² = {r2:.4f}'
                    # Use fig.transfrom_axes or ax.transAxes
                    plt.text(0.05, 0.95, equation + '\n' + r2_text,
                            transform=ax.transAxes, # Use ax.transAxes
                            bbox=dict(facecolor='white', alpha=0.8))

                except Exception as e:
                    st.warning(f"Could not calculate R² or RMSE: {e}")


                st.pyplot(fig)
                plt.close(fig) # Close figure


            else:
                st.warning("Model yang dipilih tidak didukung untuk visualisasi model spesifik (Linear Regression, Decision Tree, atau KMeans).")

        with tab5:
            st.header("Analisis Data")
            st.markdown("""
            **Business Understanding**
            Sebuah toko online dengan produk kebutuhan rumah yang terdapat beberapa barang, dari data tersebut kami ingin menganalisis data produk untuk memahami performa produk.
            """)
            st.markdown("""
           | Nama Kolom (Column Name) | Deskripsi (Description)                                                                 |
|--------------------------|-----------------------------------------------------------------------------------------|
| index                    | Nomor urut unik yang mengidentifikasi setiap entri produk.                               |
| product                  | Nama atau judul yang digunakan untuk menampilkan produk.                                  |
| category                 | Klasifikasi utama di mana produk tersebut terdaftar.                                      |
| sub_category             | Pengelompokan lebih spesifik di dalam kategori utama produk.                             |
| brand                    | Nama produsen atau merek dagang yang terkait dengan produk.                               |
| sales_price              | Nilai moneter aktual produk saat ditawarkan untuk dijual pada platform.                   |
| market_price             | Estimasi nilai produk di pasaran umum.                                                   |
| type                     | Varian atau klasifikasi spesifik dari produk.                                            |
| rating                   | Skor evaluasi atau umpan balik kuantitatif dari konsumen mengenai kualitas produk.          |
            """)
            st.subheader("Statistical Summary (Numerical)")
            # Ensure df is not None before accessing describe
            if df is not None:
                st.dataframe(df.select_dtypes(include=np.number).describe()) # Display numerical stats
            else:
                 st.info("Data not loaded, cannot show statistical summary.")


            st.subheader("Missing Values")
            # Ensure df is not None from load_data
            if df is not None:
                missing_info = df.isnull().sum().reset_index()
                missing_info.columns = ['Column', 'Missing Count']
                missing_info['Missing Percentage (%)'] = (missing_info['Missing Count'] / len(df)) * 100
                st.dataframe(missing_info)
            else:
                st.info("Data not loaded, cannot show missing value info.")

            st.subheader("Value Counts (Categorical)")
            # Ensure df is not None
            if df is not None:
                 categorical_cols = df.select_dtypes(include='object').columns
                 if not categorical_cols.empty:
                     for col in categorical_cols:
                         st.write(f"**{col}**")
                         st.dataframe(df[col].value_counts().head(10)) # Show top 10
                 else:
                     st.info("No categorical columns found.")
            else:
                 st.info("Data not loaded, cannot show value counts.")


    else:
        st.info("Tidak ada model yang berhasil dimuat. Pastikan file .pkl model tersedia di direktori yang sama.")

if __name__ == "__main__":
    main()
