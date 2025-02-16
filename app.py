import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import logging
from statsmodels.tsa.api import VAR, VARMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from prophet import Prophet
import plotly.express as px
import plotly.graph_objects as go
from pmdarima import auto_arima
from io import StringIO
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel, ExpSineSquared


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Function to load CSV file
def load_csv(file):
    try:
        df = pd.read_csv(file)
        logging.info("File loaded successfully.")
        return df
    except Exception as e:
        logging.error(f"Error loading file: {e}")
        return None

def clean_demand_column(df):
    try:
        # Interpolate missing values in the 'Demand' column
        df['Demand'] = pd.to_numeric(df['Demand'], errors='coerce')
        df['Demand'] = df['Demand'].interpolate(method='linear')
        
        # Interpolate missing values in additional columns
        for col in df.columns:
            if col.endswith('_mapped'):
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].interpolate(method='linear')
        
        return df
    except Exception as e:
        logging.error(f"Error cleaning columns: {e}")
        return None

# Function to normalize date formats
def normalize_dates(df, date_col):
    try:
        # Make a copy of the date column to avoid modifying the original during attempts
        date_series = df[date_col].copy()
        
        # Case 1: Handle YYYYMMDD format (both integer and string)
        if date_series.dtype == 'int64' or date_series.astype(str).str.match(r'^\d{8}$').all():
            df[date_col] = pd.to_datetime(date_series.astype(str), format='%Y%m%d')
            
        # Case 2: Handle DD-MM-YYYY or DD/MM/YYYY format
        elif date_series.astype(str).str.match(r'^\d{1,2}[-/]\d{1,2}[-/]\d{4}$').all():
            df[date_col] = pd.to_datetime(date_series, format='%d-%m-%Y', errors='coerce')
            # If above fails, try with forward slash
            if df[date_col].isnull().any():
                df[date_col] = pd.to_datetime(date_series, format='%d/%m/%Y', errors='coerce')
                
        # Case 3: Handle YYYY-MM-DD or YYYY/MM/DD format
        elif date_series.astype(str).str.match(r'^\d{4}[-/]\d{1,2}[-/]\d{1,2}$').all():
            df[date_col] = pd.to_datetime(date_series, format='%Y-%m-%d', errors='coerce')
            # If above fails, try with forward slash
            if df[date_col].isnull().any():
                df[date_col] = pd.to_datetime(date_series, format='%Y/%m/%d', errors='coerce')
                
        # Case 4: Handle MMM DD, YYYY format (e.g., 'Jan 01, 2023')
        elif date_series.astype(str).str.match(r'^[A-Za-z]{3}\s+\d{1,2},\s+\d{4}$').all():
            df[date_col] = pd.to_datetime(date_series, format='%b %d, %Y', errors='coerce')
                
        # Case 5: Handle MM-DD-YYYY or MM/DD/YYYY format
        elif date_series.astype(str).str.match(r'^\d{1,2}[-/]\d{1,2}[-/]\d{4}$').all():
            df[date_col] = pd.to_datetime(date_series, format='%m-%d-%Y', errors='coerce')
            # If above fails, try with forward slash
            if df[date_col].isnull().any():
                df[date_col] = pd.to_datetime(date_series, format='%m/%d/%Y', errors='coerce')
        
        # Case 6: As a last resort, try pandas' general date parsing
        else:
            df[date_col] = pd.to_datetime(date_series, errors='coerce')
        
        # Check if any dates couldn't be parsed
        if df[date_col].isnull().any():
            null_dates = date_series[df[date_col].isnull()].unique()
            raise ValueError(f"Could not parse the following dates: {null_dates}")
            
        # Sort the dataframe by date
        df = df.sort_values(by=date_col)
        
        logging.info(f"Dates successfully normalized to format: {df[date_col].dtype}")
        logging.info(f"Date range: {df[date_col].min()} to {df[date_col].max()}")
        
        return df
    
    except Exception as e:
        logging.error(f"Error normalizing dates: {e}")
        return None

# Function to map columns
def map_columns(df, date_col, demand_col, additional_cols=None):
    try:
        df.rename(columns={date_col: 'Date', demand_col: 'Demand'}, inplace=True)
        if additional_cols:
            for col in additional_cols:
                df.rename(columns={col: col + '_mapped'}, inplace=True)
        df = normalize_dates(df, 'Date')
        df = clean_demand_column(df)
        logging.info("Columns successfully mapped!")
        return df
    except Exception as e:
        logging.error(f"Error mapping columns: {e}")
        return None

# Function to infer frequency
def infer_frequency(df, date_col='Date'):
    """
    Infer the frequency of the time series data.
    """
    try:
        # Calculate the most common difference between consecutive dates
        date_diffs = pd.Series(df[date_col].sort_values().diff().dt.days.value_counts())
        most_common_diff = date_diffs.index[0]
        
        # Map the difference to a frequency string
        freq_map = {
            1: 'D',    # Daily
            7: 'W',    # Weekly
            30: 'M',   # Monthly (approximate)
            31: 'M',   # Monthly (approximate)
            28: 'M',   # Monthly (approximate)
            90: 'Q',   # Quarterly (approximate)
            91: 'Q',   # Quarterly (approximate)
            92: 'Q',   # Quarterly (approximate)
            365: 'Y',  # Yearly
            366: 'Y'   # Yearly (leap year)
        }
        
        freq = freq_map.get(most_common_diff, 'D')
        logging.info(f"Inferred frequency: {freq} (most common diff: {most_common_diff} days)")
        return freq
        
    except Exception as e:
        logging.error(f"Error inferring frequency: {e}")
        return 'D'  # Default to daily frequency

# Function for EDA
def perform_eda(df):
    st.subheader("Exploratory Data Analysis (EDA)")
    
    # Display basic statistics
    st.write("### Data Summary")
    st.write(df.describe())
    
    # Check for missing values
    st.write("### Missing Values")
    st.write(df.isnull().sum())
    
    # Plot demand over time
    st.write("### Demand Trend")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['Date'], df['Demand'], label='Historical Demand')
    ax.set_xlabel('Date')
    ax.set_ylabel('Demand')
    ax.set_title('Demand Trend')
    ax.legend()
    st.pyplot(fig)
    
    # Seasonality and trend decomposition
    st.write("### Seasonality and Trend Decomposition")
    decomposition = seasonal_decompose(df.set_index('Date')['Demand'], model='additive', period=30)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))
    decomposition.trend.plot(ax=ax1, title='Trend')
    decomposition.seasonal.plot(ax=ax2, title='Seasonality')
    decomposition.resid.plot(ax=ax3, title='Residuals')
    plt.tight_layout()
    st.pyplot(fig)
    
    # Autocorrelation and Partial Autocorrelation
    st.write("### Autocorrelation and Partial Autocorrelation")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    plot_acf(df['Demand'], lags=30, ax=ax1)
    plot_pacf(df['Demand'], lags=30, ax=ax2)
    plt.tight_layout()
    st.pyplot(fig)

# Function to detect seasonality
def detect_seasonality(df, max_lags=50):
    st.write("### Seasonality Detection")
    
    # Calculate ACF and PACF
    acf_values = acf(df['Demand'], nlags=max_lags)
    pacf_values = pacf(df['Demand'], nlags=max_lags)
    
    # Find potential seasonal periods from ACF
    potential_periods = []
    threshold = 0.2  # Correlation threshold
    for i in range(1, len(acf_values)):
        if acf_values[i] > threshold:
            potential_periods.append(i)
    
    # Plot ACF and PACF
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    plot_acf(df['Demand'], lags=max_lags, ax=ax1)
    plot_pacf(df['Demand'], lags=max_lags, ax=ax2)
    plt.tight_layout()
    st.pyplot(fig)
    
    # Perform seasonal decomposition with different potential periods
    best_period = None
    min_residual_variance = float('inf')
    
    # Try different seasonal periods
    for period in [7, 12, 24, 30, 52]:  # Common business periods
        try:
            decomposition = seasonal_decompose(
                df.set_index('Date')['Demand'],
                period=period,
                model='additive'
            )
            residual_variance = np.var(decomposition.resid.dropna())
            
            # Plot decomposition for each period
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 12))
            decomposition.observed.plot(ax=ax1, title=f'Observed (Period={period})')
            decomposition.trend.plot(ax=ax2, title='Trend')
            decomposition.seasonal.plot(ax=ax3, title='Seasonal')
            decomposition.resid.plot(ax=ax4, title='Residual')
            plt.tight_layout()
            st.pyplot(fig)
            
            # Update best period if this decomposition has lower residual variance
            if residual_variance < min_residual_variance:
                min_residual_variance = residual_variance
                best_period = period
                
        except Exception as e:
            logging.warning(f"Failed to decompose with period {period}: {str(e)}")
            continue
    
    # Perform frequency domain analysis
    try:
        from scipy import signal
        f, Pxx = signal.periodogram(df['Demand'].fillna(method='ffill'))
        dominant_periods = 1/f[signal.find_peaks(Pxx)[0]]
        dominant_periods = dominant_periods[dominant_periods < len(df)/2]
        st.write("### Dominant Periods from Spectral Analysis")
        st.write(f"Detected periods: {[round(p, 1) for p in dominant_periods]}")
    except Exception as e:
        logging.warning(f"Spectral analysis failed: {str(e)}")
    
    # Infer frequency from date index
    date_freq = infer_frequency(df)
    freq_based_period = {
        'D': 7,    # Daily -> Weekly seasonality
        'W': 52,   # Weekly -> Yearly seasonality
        'M': 12,   # Monthly -> Yearly seasonality
        'Q': 4,    # Quarterly -> Yearly seasonality
        'Y': 1     # Yearly -> No seasonality
    }.get(date_freq, 7)  # Default to weekly if unknown
    
    # Combine all information to make final decision
    if best_period is None:
        best_period = freq_based_period
    
    st.write(f"""
    ### Seasonality Analysis Results:
    - Best detected period: {best_period}
    - Data frequency: {date_freq}
    - Potential periods from ACF: {potential_periods}
    - Frequency-based suggestion: {freq_based_period}
    """)
    
    return best_period

# Function to calculate metrics
def calculate_metrics(y_true, y_pred):
    epsilon = 1e-6 # Prevents division by 0

    if y_true is None or y_pred is None:
        logging.warning("Invalid input: y_true or y_pred is None.")
        return None, None, None

    if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
        logging.warning("Invalid predictions: NaN or Inf values detected.")
        return None, None, None

    try:
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true-y_pred)/(y_true+epsilon))) * 100
        mae = mean_absolute_error(y_true, y_pred)
        return rmse, mape, mae
    except Exception as e:
        logging.error(f"Error calculating metrics: {e}")
        return None, None, None

# Function to auto-tune SARIMA
def auto_tune_sarima(train, seasonal_period):
    model = auto_arima(
        train['Demand'],
        seasonal=True,
        m=seasonal_period,
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore"
    )
    return model


def encode_categorical_columns(df, categorical_columns):
    try:
        df_encoded = df.copy()
        for col in categorical_columns:
            unique_categories = sorted(df_encoded[col].dropna().unique())
            df_encoded[col] = pd.Categorical(df_encoded[col], categories=unique_categories)
        return df_encoded
    except Exception as e:
        logging.error(f"Error encoding categorical columns: {e}")
        return df


def create_features(df, lags=3):
    try:
        df = df.copy()

        #creating lags
        for lag in range(1, lags+1):
            df[f'lag_{lag}'] = df['Demand'].shift(lag).astype(float)
        
        #Roll_Mean
        df['rolling_mean_3'] = df['Demand'].rolling(3).mean().astype(float)
        df['rolling_std_3'] = df['Demand'].rolling(3).std().astype(float)
        
        
        # Add date-based categorical features
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['day_of_month'] = df.index.day

        logging.info(f"Final DataFrame Shape: {df.shape}")
        logging.info(f"Final DataFrame Columns: {df.columns}")

         # Drop rows with NaN values (introduced by lags and rolling stats)
        df = df.dropna()
        return df
    except Exception as e:
        logging.error(f"Feature creation error: {e}",exec_info = True)
        return None
 
def xgboost_model(train, val, test):
    try:
        logging.info("Starting XGBoost forecast")
        full_df = pd.concat([train, val, test])
        full_df = create_features(full_df)
 
        # One-hot encode categorical columns
        categorical_cols = full_df.select_dtypes(include=['category', 'object']).columns
        if not categorical_cols.empty:
            full_df = pd.get_dummies(full_df, columns=categorical_cols)

        # Split data
        train_len = len(train)
        val_len = len(val)
        
        X_train = full_df.iloc[:train_len].drop(columns=['Demand'])
        y_train = full_df.iloc[:train_len]['Demand']
        
        X_val = full_df.iloc[train_len:train_len+val_len].drop(columns=['Demand'])
        y_val = full_df.iloc[train_len:train_len+val_len]['Demand']
        
        X_test = full_df.iloc[train_len+val_len:].drop(columns=['Demand'])
        y_test = full_df.iloc[train_len+val_len:]['Demand']

        # Train model
        model = XGBRegressor(
            n_estimators=500,
            learning_rate=0.01,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            early_stopping_rounds=20,
            eval_metric='mae'
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
 
        # Prepare future dates
        freq = pd.infer_freq(test.index) or 'D'
        future_dates = pd.date_range(
            start=test.index[-1] + pd.Timedelta(days=1), 
            periods=30, 
            freq=freq
        )

        # Initialize future dataframe with the same columns as X_train
        future_df = pd.DataFrame(index=future_dates, columns=X_train.columns)
        future_df = future_df.fillna(0)  # Fill with zeros initially
        predictions = []  # Store predictions here
 
        # Update time-based features for future dates
        for i, date in enumerate(future_dates):
            current_features = future_df.iloc[[i]].copy()
            
            if i >= 3:  # For lag features
                current_features['lag_1'] = predictions[i-1]
                current_features['lag_2'] = predictions[i-2]
                current_features['lag_3'] = predictions[i-3]
            else:  # Use last known values for initial predictions
                current_features['lag_1'] = X_test.iloc[-1]['lag_1']
                current_features['lag_2'] = X_test.iloc[-1]['lag_2']
                current_features['lag_3'] = X_test.iloc[-1]['lag_3']
            
            # Update categorical features (one-hot encoded columns)
            if 'day_of_week' in X_train.columns:
                current_features['day_of_week'] = date.dayofweek
            if 'month' in X_train.columns:
                current_features['month'] = date.month
            if 'day_of_month' in X_train.columns:
                current_features['day_of_month'] = date.day
            
            # Make prediction
            pred = model.predict(current_features[model.feature_names_in_])[0]
            predictions.append(pred)
            
            # Update future_df with the new features
            future_df.iloc[i] = current_features.iloc[0]
 
        return {
            'train': calculate_metrics(y_train, model.predict(X_train)),
            'val': calculate_metrics(y_val, model.predict(X_val)),
            'test': calculate_metrics(y_test, model.predict(X_test))
        }, predictions
    
    except Exception as e:
        logging.error(f"XGBoost Error: {str(e)}", exc_info=True)
        return f"XGBoost Failed: {str(e)}", None


def lstm_model(train, val, test, n_steps=3):
    try:
        # Create sequences
        def create_sequences(data, n_steps):
            X, y = [], []
            for i in range(len(data)-n_steps):
                X.append(data[i:i+n_steps])
                y.append(data[i+n_steps])
            return np.array(X), np.array(y)
        
        # Scale data
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train[['Demand']])
        val_scaled = scaler.transform(val[['Demand']])
        test_scaled = scaler.transform(test[['Demand']])
        
        # Create sequences
        X_train, y_train = create_sequences(train_scaled, n_steps)
        X_val, y_val = create_sequences(val_scaled, n_steps)
        X_test, y_test = create_sequences(test_scaled, n_steps)
        
        # Build model
        model = Sequential([
            LSTM(50, activation='relu', input_shape=(n_steps, 1)),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        
        # Train model
        early_stop = EarlyStopping(monitor='val_loss', patience=5)
        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=32,
            callbacks=[early_stop],
            verbose=0
        )
        
        # Generate forecasts
        future_forecast = []
        current_batch = test_scaled[-n_steps:]
        
        for _ in range(30):
            current_pred = model.predict(current_batch.reshape(1, n_steps, 1))[0][0]
            future_forecast.append(current_pred)
            current_batch = np.append(current_batch[1:], current_pred)
        
        return {
            'train': calculate_metrics(y_train, model.predict(X_train).flatten()),
            'val': calculate_metrics(y_val, model.predict(X_val).flatten()),
            'test': calculate_metrics(y_test, model.predict(X_test).flatten())
        }, scaler.inverse_transform(np.array(future_forecast).reshape(-1, 1)).flatten().tolist()
    except Exception as e:
        logging.error(f"LSTM Error: {str(e)}")
        return f"LSTM Failed: {str(e)}", None

def random_forest_model(train, val, test):
    try:
        logging.info("Starting Random Forest forecast")
        full_df = pd.concat([train, val, test])
        full_df = create_features(full_df)
 
        # One-hot encode categorical columns
        categorical_cols = full_df.select_dtypes(include=['category', 'object']).columns
        if not categorical_cols.empty:
            full_df = pd.get_dummies(full_df, columns=categorical_cols)

        # Split data
        train_len = len(train)
        val_len = len(val)
        
        X_train = full_df.iloc[:train_len].drop(columns=['Demand'])
        y_train = full_df.iloc[:train_len]['Demand']
        
        X_val = full_df.iloc[train_len:train_len+val_len].drop(columns=['Demand'])
        y_val = full_df.iloc[train_len:train_len+val_len]['Demand']
        
        X_test = full_df.iloc[train_len+val_len:].drop(columns=['Demand'])
        y_test = full_df.iloc[train_len+val_len:]['Demand']

        # Train model with enhanced parameters
        model = RandomForestRegressor(
            n_estimators=500,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1  # Use all available cores
        )
        model.fit(X_train, y_train)
 
        # Prepare future dates
        freq = pd.infer_freq(test.index) or 'D'
        future_dates = pd.date_range(
            start=test.index[-1] + pd.Timedelta(days=1), 
            periods=30, 
            freq=freq
        )

        # Initialize future dataframe with the same columns as X_train
        future_df = pd.DataFrame(index=future_dates, columns=X_train.columns)
        future_df = future_df.fillna(0)  # Fill with zeros initially
        predictions = []  # Store predictions here
 
        # Update time-based features for future dates
        for i, date in enumerate(future_dates):
            current_features = future_df.iloc[[i]].copy()
            
            if i >= 3:  # For lag features
                current_features['lag_1'] = predictions[i-1]
                current_features['lag_2'] = predictions[i-2]
                current_features['lag_3'] = predictions[i-3]
            else:  # Use last known values for initial predictions
                current_features['lag_1'] = X_test.iloc[-1]['lag_1']
                current_features['lag_2'] = X_test.iloc[-1]['lag_2']
                current_features['lag_3'] = X_test.iloc[-1]['lag_3']
            
            # Update time-based features
            if 'day_of_week' in X_train.columns:
                current_features['day_of_week'] = date.dayofweek
            if 'month' in X_train.columns:
                current_features['month'] = date.month
            if 'day_of_month' in X_train.columns:
                current_features['day_of_month'] = date.day
            if 'week_of_year' in X_train.columns:
                current_features['week_of_year'] = date.isocalendar()[1]
            if 'quarter' in X_train.columns:
                current_features['quarter'] = date.quarter
            
            # Make prediction
            pred = model.predict(current_features[X_train.columns])[0]
            predictions.append(pred)
            
            # Update future_df with the new features
            future_df.iloc[i] = current_features.iloc[0]
 
        # Calculate feature importances
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Log feature importances
        logging.info("Random Forest Feature Importances:")
        logging.info(feature_importance)

        return {
            'train': calculate_metrics(y_train, model.predict(X_train)),
            'val': calculate_metrics(y_val, model.predict(X_val)),
            'test': calculate_metrics(y_test, model.predict(X_test)),
            'feature_importance': feature_importance
        }, predictions
    
    except Exception as e:
        logging.error(f"Random Forest Error: {str(e)}", exc_info=True)
        return f"Random Forest Failed: {str(e)}", None

def gaussian_process_model(train, val, test):
    try:
        logging.info("Starting Gaussian Process forecast")
        full_df = pd.concat([train, val, test])
        full_df = create_features(full_df)
 
        # One-hot encode categorical columns
        categorical_cols = full_df.select_dtypes(include=['category', 'object']).columns
        if not categorical_cols.empty:
            full_df = pd.get_dummies(full_df, columns=categorical_cols)

        # Split data
        train_len = len(train)
        val_len = len(val)
        
        X_train = full_df.iloc[:train_len].drop(columns=['Demand'])
        y_train = full_df.iloc[:train_len]['Demand']
        
        X_val = full_df.iloc[train_len:train_len+val_len].drop(columns=['Demand'])
        y_val = full_df.iloc[train_len:train_len+val_len]['Demand']
        
        X_test = full_df.iloc[train_len+val_len:].drop(columns=['Demand'])
        y_test = full_df.iloc[train_len+val_len:]['Demand']

        # Scale features and target with positive minimum bound
        scaler_X = StandardScaler()
        
        # Use custom scaler for target to ensure non-negative predictions
        y_min = y_train.min()
        y_max = y_train.max()
        y_range = y_max - y_min
        
        def scale_y(y):
            return (y - y_min) / y_range
        
        def inverse_scale_y(y_scaled):
            return y_scaled * y_range + y_min
        
        X_train_scaled = scaler_X.fit_transform(X_train)
        y_train_scaled = scale_y(y_train)
        
        X_val_scaled = scaler_X.transform(X_val)
        X_test_scaled = scaler_X.transform(X_test)

        # Define the kernel with appropriate length scales
        kernel = (
            ConstantKernel(1.0) * RBF(length_scale=1.0) +  # For smooth trends
            ExpSineSquared(length_scale=1.0, periodicity=1.0) +  # For periodic patterns
            WhiteKernel(noise_level=0.1)  # For noise
        )

        # Train model
        model = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=10,
            random_state=42,
            normalize_y=False  # We're handling normalization ourselves
        )
        model.fit(X_train_scaled, y_train_scaled)
 
        # Prepare future dates
        freq = pd.infer_freq(test.index) or 'D'
        future_dates = pd.date_range(
            start=test.index[-1] + pd.Timedelta(days=1), 
            periods=30, 
            freq=freq
        )

        # Initialize future dataframe with the same columns as X_train
        future_df = pd.DataFrame(index=future_dates, columns=X_train.columns)
        future_df = future_df.fillna(0)  # Fill with zeros initially
        predictions = []  # Store predictions here
        uncertainties = []  # Store prediction uncertainties
 
        # Update time-based features for future dates
        for i, date in enumerate(future_dates):
            current_features = future_df.iloc[[i]].copy()
            
            if i >= 3:  # For lag features
                current_features['lag_1'] = predictions[i-1]
                current_features['lag_2'] = predictions[i-2]
                current_features['lag_3'] = predictions[i-3]
            else:  # Use last known values for initial predictions
                current_features['lag_1'] = X_test.iloc[-1]['lag_1']
                current_features['lag_2'] = X_test.iloc[-1]['lag_2']
                current_features['lag_3'] = X_test.iloc[-1]['lag_3']
            
            # Update time-based features
            if 'day_of_week' in X_train.columns:
                current_features['day_of_week'] = date.dayofweek
            if 'month' in X_train.columns:
                current_features['month'] = date.month
            if 'day_of_month' in X_train.columns:
                current_features['day_of_month'] = date.day
            if 'week_of_year' in X_train.columns:
                current_features['week_of_year'] = date.isocalendar()[1]
            if 'quarter' in X_train.columns:
                current_features['quarter'] = date.quarter
            
            # Scale features
            current_features_scaled = scaler_X.transform(current_features)
            
            # Make prediction with uncertainty
            pred_scaled, std = model.predict(current_features_scaled, return_std=True)
            
            # Inverse transform and ensure non-negative predictions
            pred = max(0, inverse_scale_y(pred_scaled[0]))  # Ensure non-negative
            uncertainty = std[0] * y_range  # Scale uncertainty back
            
            predictions.append(pred)
            uncertainties.append(uncertainty)
            
            # Update future_df with the new features
            future_df.iloc[i] = current_features.iloc[0]

        # Calculate metrics for training, validation, and test sets
        train_pred = np.maximum(0, inverse_scale_y(
            model.predict(scaler_X.transform(X_train))
        ))
        val_pred = np.maximum(0, inverse_scale_y(
            model.predict(scaler_X.transform(X_val))
        ))
        test_pred = np.maximum(0, inverse_scale_y(
            model.predict(scaler_X.transform(X_test))
        ))

        return {
            'train': calculate_metrics(y_train, train_pred),
            'val': calculate_metrics(y_val, val_pred),
            'test': calculate_metrics(y_test, test_pred),
            'uncertainties': uncertainties
        }, predictions
    
    except Exception as e:
        logging.error(f"Gaussian Process Error: {str(e)}", exc_info=True)
        return f"Gaussian Process Failed: {str(e)}", None

# Function to forecast using various models
def forecast_models(df, selected_models, additional_cols=None, item_col=None):
    # Create a copy of the dataframe to preserve the original
    df_copy = df.copy()
    
    if additional_cols is None:
        additional_cols = []
    
    # Store the dates before setting index
    dates = df_copy['Date'].copy()
    
    df_copy.set_index('Date', inplace=True)
    df_copy.sort_index(inplace=True)
    results = {}
    future_forecasts = {}

    # Ensure 'Demand' column is of type float64
    df_copy['Demand'] = df_copy['Demand'].astype('float64')

    # Automatically detect seasonality
    seasonal_period = detect_seasonality(df_copy.reset_index())

    # Split data into train, val, and test
    train_val, test = train_test_split(df_copy, test_size=0.2, shuffle=False)
    train, val = train_test_split(train_val, test_size=0.25, shuffle=False)

    # Identify categorical columns
    categorical_columns = df_copy.select_dtypes(include=['object']).columns.tolist()

    # Encode categorical columns
    train_encoded = encode_categorical_columns(train.reset_index(), categorical_columns)
    val_encoded = encode_categorical_columns(val.reset_index(), categorical_columns)
    test_encoded = encode_categorical_columns(test.reset_index(), categorical_columns)


    # AR Model
    if 'AR' in selected_models:
        try:
            model = AutoReg(df_copy['Demand'], lags=2).fit()
            # Training performance
            train_forecast = model.predict(start=0, end=len(df_copy)-1)
            train_rmse, train_mape, train_mae = calculate_metrics(df_copy['Demand'].values, train_forecast)
            # Validation performance
            val_forecast = model.predict(start=len(df_copy), end=len(df_copy)+len(df_copy)-1)
            val_rmse, val_mape, val_mae = calculate_metrics(df_copy['Demand'].values, val_forecast)
            # Test performance
            test_forecast = model.predict(start=len(df_copy), end=len(df_copy)+len(df_copy)-1)
            test_rmse, test_mape, test_mae = calculate_metrics(df_copy['Demand'].values, test_forecast)
            results['AR'] = {
                'train': (train_rmse, train_mape, train_mae),
                'val': (val_rmse, val_mape, val_mae),
                'test': (test_rmse, test_mape, test_mae)
            }
            # Future forecast
            future_forecast = model.predict(start=len(df_copy), end=len(df_copy)+29)
            future_forecasts['AR'] = future_forecast.tolist()
        except Exception as e:
            results['AR'] = str(e)

    # ARMA Model
    if 'ARMA' in selected_models:
        try:
            model = ARIMA(df_copy['Demand'], order=(2, 0, 1)).fit()
            # Training performance
            train_forecast = model.predict(start=0, end=len(df_copy)-1)
            train_rmse, train_mape, train_mae = calculate_metrics(df_copy['Demand'].values, train_forecast)
            # Validation performance
            val_forecast = model.predict(start=len(df_copy), end=len(df_copy)+len(df_copy)-1)
            val_rmse, val_mape, val_mae = calculate_metrics(df_copy['Demand'].values, val_forecast)
            # Test performance
            test_forecast = model.predict(start=len(df_copy), end=len(df_copy)+len(df_copy)-1)
            test_rmse, test_mape, test_mae = calculate_metrics(df_copy['Demand'].values, test_forecast)
            results['ARMA'] = {
                'train': (train_rmse, train_mape, train_mae),
                'val': (val_rmse, val_mape, val_mae),
                'test': (test_rmse, test_mape, test_mae)
            }
            # Future forecast
            future_forecast = model.predict(start=len(df_copy), end=len(df_copy)+29)
            future_forecasts['ARMA'] = future_forecast.tolist()
        except Exception as e:
            results['ARMA'] = str(e)

    # SARIMA Model
    if 'SARIMA' in selected_models:
        try:
        # Build the SARIMA model using auto_arima without unsupported parameters
            model = auto_tune_sarima(df_copy, seasonal_period)
        
        # Training performance: Predict in-sample values for the training period
            train_forecast = model.predict_in_sample()
            train_rmse, train_mape, train_mae = calculate_metrics(df_copy['Demand'].values, train_forecast)
        
        # Validation performance: Forecast for the validation period
            val_forecast = model.predict(n_periods=len(df_copy))
            val_rmse, val_mape, val_mae = calculate_metrics(df_copy['Demand'].values, val_forecast)
        
        # Test performance: Forecast for the test period
            test_forecast = model.predict(n_periods=len(df_copy))
            test_rmse, test_mape, test_mae = calculate_metrics(df_copy['Demand'].values, test_forecast)
        
        # Store the computed metrics
            results['SARIMA'] = {
                'train': (train_rmse, train_mape, train_mae),
                'val': (val_rmse, val_mape, val_mae),
                'test': (test_rmse, test_mape, test_mae)
            }
        
        # Future forecast: Generate forecasts for the next 30 time periods
            future_forecast = model.predict(n_periods=30)
            future_forecasts['SARIMA'] = future_forecast.tolist()
        
        
        except Exception as e:
            results['SARIMA'] = str(e)


    if 'VAR' in selected_models and len(additional_cols) > 0:
        try:
         # Prepare multivariate training data
            train_vars = df_copy[['Demand'] + [col + '_mapped' for col in additional_cols]]
            model = VAR(train_vars)
            model_fitted = model.fit()
        
        # Training performance
            train_forecast = model_fitted.forecast(train_vars.values[-model_fitted.k_ar:], steps=len(df_copy))
            train_rmse, train_mape, train_mae = calculate_metrics(df_copy['Demand'].values, train_forecast[:, 0])

        # Validation performance
            val_forecast = model_fitted.forecast(train_vars.values[-model_fitted.k_ar:], steps=len(df_copy))
            val_rmse, val_mape, val_mae = calculate_metrics(df_copy['Demand'].values, val_forecast[:, 0])
        
        # Test performance
            test_forecast = model_fitted.forecast(train_vars.values[-model_fitted.k_ar:], steps=len(df_copy))
            test_rmse, test_mape, test_mae = calculate_metrics(df_copy['Demand'].values, test_forecast[:, 0])
        
            results['VAR'] = {
                'train': (train_rmse, train_mape, train_mae),
                'val': (val_rmse, val_mape, val_mae),
                'test': (test_rmse, test_mape, test_mae)
            }
        
        # Future forecast
            future_forecast = model_fitted.forecast(train_vars.values[-model_fitted.k_ar:], steps=30)
            future_forecasts['VAR'] = future_forecast[:, 0].tolist()
        except Exception as e:
            results['VAR'] = str(e)

    if 'VARMAX' in selected_models and len(additional_cols) > 0:
        try:
        # Prepare multivariate training data
            train_vars = df_copy[['Demand'] + [col + '_mapped' for col in additional_cols]]
            model = VARMAX(train_vars, order=(1, 1)).fit(disp=False)
        
        # Training performance
            train_forecast = model.forecast(steps=len(df_copy))
            train_rmse, train_mape, train_mae = calculate_metrics(df_copy['Demand'].values, train_forecast['Demand'])

        # Validation performance
            val_forecast = model.forecast(steps=len(df_copy))
            val_rmse, val_mape, val_mae = calculate_metrics(df_copy['Demand'].values, val_forecast['Demand'])
        
        # Test performance
            test_forecast = model.forecast(steps=len(df_copy))
            test_rmse, test_mape, test_mae = calculate_metrics(df_copy['Demand'].values, test_forecast['Demand'])
        
            results['VARMAX'] = {
                'train': (train_rmse, train_mape, train_mae),
                'val': (val_rmse, val_mape, val_mae),
                'test': (test_rmse, test_mape, test_mae)
            }
        
        # Future forecast
            future_forecast = model.forecast(steps=30)
            future_forecasts['VARMAX'] = future_forecast['Demand'].tolist()
        except Exception as e:
            results['VARMAX'] = str(e)

    # Simple Exponential Smoothing (SES)
    if 'SES' in selected_models:
        try:
            model = SimpleExpSmoothing(df_copy['Demand']).fit()
            # Training performance
            train_forecast = model.forecast(steps=len(df_copy))
            train_rmse, train_mape, train_mae = calculate_metrics(df_copy['Demand'].values, train_forecast)
            # Validation performance
            val_forecast = model.forecast(steps=len(df_copy))
            val_rmse, val_mape, val_mae = calculate_metrics(df_copy['Demand'].values, val_forecast)
            # Test performance
            test_forecast = model.forecast(steps=len(df_copy))
            test_rmse, test_mape, test_mae = calculate_metrics(df_copy['Demand'].values, test_forecast)
            results['SES'] = {
                'train': (train_rmse, train_mape, train_mae),
                'val': (val_rmse, val_mape, val_mae),
                'test': (test_rmse, test_mape, test_mae)
            }
            # Future forecast
            future_forecast = model.forecast(steps=30)
            future_forecasts['SES'] = future_forecast.tolist()
        except Exception as e:
            results['SES'] = str(e)

    # Holt-Winters Exponential Smoothing (HWES)
    if 'HWES' in selected_models:
        try:
            model = ExponentialSmoothing(
                df_copy['Demand'],
                seasonal='add',
                seasonal_periods=seasonal_period
            ).fit()
            # Training performance
            train_forecast = model.forecast(steps=len(df_copy))
            train_rmse, train_mape, train_mae = calculate_metrics(df_copy['Demand'].values, train_forecast)
            # Validation performance
            val_forecast = model.forecast(steps=len(df_copy))
            val_rmse, val_mape, val_mae = calculate_metrics(df_copy['Demand'].values, val_forecast)
            # Test performance
            test_forecast = model.forecast(steps=len(df_copy))
            test_rmse, test_mape, test_mae = calculate_metrics(df_copy['Demand'].values, test_forecast)
            results['HWES'] = {
                'train': (train_rmse, train_mape, train_mae),
                'val': (val_rmse, val_mape, val_mae),
                'test': (test_rmse, test_mape, test_mae)
            }
            # Future forecast
            future_forecast = model.forecast(steps=30)
            future_forecasts['HWES'] = future_forecast.tolist()
        except Exception as e:
            results['HWES'] = str(e)

    # Prophet Model
    if 'Prophet' in selected_models:
        try:
            prophet_df = df_copy.reset_index().rename(columns={'Date': 'ds', 'Demand': 'y'})
            model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
            model.fit(prophet_df)
            # Training performance
            train_forecast = model.predict(prophet_df)['yhat']
            train_rmse, train_mape, train_mae = calculate_metrics(df_copy['Demand'].values, train_forecast)
            # Validation performance
            val_df = df_copy.reset_index().rename(columns={'Date': 'ds'})
            val_forecast = model.predict(val_df)['yhat']
            val_rmse, val_mape, val_mae = calculate_metrics(df_copy['Demand'].values, val_forecast)
            # Test performance
            test_forecast = model.predict(df_copy.reset_index().rename(columns={'Date': 'ds'}))['yhat']
            test_rmse, test_mape, test_mae = calculate_metrics(df_copy['Demand'].values, test_forecast)
            results['Prophet'] = {
                'train': (train_rmse, train_mape, train_mae),
                'val': (val_rmse, val_mape, val_mae),
                'test': (test_rmse, test_mape, test_mae)
            }
            # Future forecast
            future = model.make_future_dataframe(periods=30)
            future_forecast = model.predict(future)['yhat'][-30:]
            future_forecasts['Prophet'] = future_forecast.tolist()
        except Exception as e:
            results['Prophet'] = str(e)

    # XGBoost
    if 'XGBoost' in selected_models:
        try:
            # Ensure proper data types
            train = train.copy()
            val = val.copy()
            test = test.copy()
            # Convert demand to float
            train['Demand'] = train['Demand'].astype(float)
            val['Demand'] = val['Demand'].astype(float)
            test['Demand'] = test['Demand'].astype(float)
            # Run XGBoost
            xgb_metrics, xgb_forecast = xgboost_model(train, val, test)
            results['XGBoost'] = xgb_metrics
            future_forecasts['XGBoost'] = xgb_forecast
        
        except Exception as e:
            results['XGBoost'] = str(e)

    # Random Forest
    if 'Random Forest' in selected_models:
        try:
            rf_metrics, rf_forecast = random_forest_model(train, val, test)
            if isinstance(rf_metrics, dict):
                results['Random Forest'] = rf_metrics
                future_forecasts['Random Forest'] = rf_forecast
                
                # Display feature importance if available
                if 'feature_importance' in rf_metrics:
                    st.write("### Random Forest Feature Importance")
                    st.dataframe(rf_metrics['feature_importance'])
            else:
                results['Random Forest'] = str(rf_metrics)
        except Exception as e:
            results['Random Forest'] = str(e)

    #LSTM 
    if 'LSTM' in selected_models:
        try:
            lstm_metrics, lstm_forecast = lstm_model(train, val, test)
            results['LSTM'] = lstm_metrics
            future_forecasts['LSTM'] = lstm_forecast
        except Exception as e:
            results['LSTM'] = str(e)

    # Gaussian Process
    if 'Gaussian Process' in selected_models:
        try:
            gp_metrics, gp_forecast = gaussian_process_model(train, val, test)
            if isinstance(gp_metrics, dict):
                results['Gaussian Process'] = gp_metrics
                future_forecasts['Gaussian Process'] = gp_forecast
                
                # Display uncertainties if available
                if 'uncertainties' in gp_metrics:
                    # Calculate future dates for visualization
                    last_date = test.index[-1]
                    freq = pd.infer_freq(test.index) or 'D'
                    future_dates = pd.date_range(
                        start=last_date + pd.Timedelta(days=1),
                        periods=30,
                        freq=freq
                    )
                    
                    st.write("### Gaussian Process Prediction Uncertainties")
                    uncertainty_df = pd.DataFrame({
                        'Date': future_dates,
                        'Prediction': gp_forecast,
                        'Uncertainty': gp_metrics['uncertainties']
                    })
                    st.dataframe(uncertainty_df)
                    
                    # Plot predictions with uncertainty bands
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=uncertainty_df['Date'],
                        y=uncertainty_df['Prediction'],
                        mode='lines',
                        name='Prediction',
                        line=dict(color='blue')
                    ))
                    fig.add_trace(go.Scatter(
                        x=uncertainty_df['Date'],
                        y=uncertainty_df['Prediction'] + 2*uncertainty_df['Uncertainty'],
                        mode='lines',
                        name='Upper Bound (95%)',
                        line=dict(width=0),
                        showlegend=False
                    ))
                    fig.add_trace(go.Scatter(
                        x=uncertainty_df['Date'],
                        y=uncertainty_df['Prediction'] - 2*uncertainty_df['Uncertainty'],
                        mode='lines',
                        name='Lower Bound (95%)',
                        fill='tonexty',
                        line=dict(width=0)
                    ))
                    fig.update_layout(
                        title='Gaussian Process Forecast with Uncertainty',
                        xaxis_title='Date',
                        yaxis_title='Demand',
                        hovermode='x'
                    )
                    st.plotly_chart(fig)
            else:
                results['Gaussian Process'] = str(gp_metrics)
        except Exception as e:
            results['Gaussian Process'] = str(e)

    return results, future_forecasts, dates

def main():
    st.title("Advanced Demand Forecasting Engine")
    st.write("Upload your CSV file and configure the settings to forecast demand.")

    # File upload
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"], key="file_uploader_1")
    if uploaded_file is not None:
        df = load_csv(uploaded_file)
        if df is not None:
            # Display all columns in the dataset
            st.subheader("Columns in the Dataset")
            st.write(df.columns.tolist())

            # Omit columns (if needed)
            st.subheader("Omit Columns")
            omit_cols = st.multiselect("Select columns to omit:", df.columns)
            if omit_cols:
                df = df.drop(omit_cols, axis=1)
                st.write("Updated Columns:", df.columns.tolist())

            # --- MAP COLUMNS SECTION ---
            # First, ask the user to select the date column,
            # so we know which column to exclude from numeric conversion.
            st.subheader("Map Columns")
            date_col = st.selectbox("Select the date column:", df.columns)
            demand_col = st.selectbox("Select the demand column:", df.columns)
            additional_cols = st.multiselect(
                "Select additional time-dependent variables for VAR/VARMAX:",
                df.columns 
            )

            # Convert non-date columns to numeric if possible (excluding the date column)
            for col in df.columns:
                if col != date_col and df[col].dtype == 'object':
                    try:
                        df[col] = pd.to_numeric(df[col])
                    except Exception as e:
                        df[col] = df[col].astype(str)

            # Map (rename) columns and normalize the date column using your helper function
            df = map_columns(df, date_col, demand_col, additional_cols)
            # --- END MAP COLUMNS SECTION ---

            if df is not None:
                # Ensure the 'Date' column is a datetime object
                if not pd.api.types.is_datetime64_any_dtype(df['Date']):
                    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

                # Forecasting type selection
                forecast_type = st.radio(
                    "Select Forecasting Type:", 
                    ("Overall", "Item-wise", "Store-Item Combination")
                )
                if forecast_type == "Item-wise":
                    item_col = st.selectbox("Select the item column:", df.columns)
                    df[item_col] = df[item_col].astype(str)
                    unique_items = df[item_col].unique()
                elif forecast_type == "Store-Item Combination":
                    item_col = st.selectbox("Select the item column:", df.columns)
                    store_col = st.selectbox("Select the store column:", df.columns)
                    df[item_col] = df[item_col].astype(str)
                    df[store_col] = df[store_col].astype(str)
                    # Create store-item combinations
                    df['store_item'] = df[store_col] + " - " + df[item_col]
                    unique_combinations = df['store_item'].unique()

                perform_eda(df)

                # Model selection
                st.subheader("Model Selection")
                selected_models = st.multiselect(
                    "Select models to run:",
                    ["AR", "ARMA", "SARIMA", "VAR", "VARMAX", "SES", "HWES", 
                     "Prophet", "XGBoost", "LSTM", "Random Forest", "Gaussian Process"],
                    default=[]
                )

                if st.button("Run Forecast"):
                    with st.spinner("Running models..."):
                        if forecast_type == "Item-wise":
                            results = {}
                            future_forecasts = {}
                            validation_data = {}
                            progress_bar = st.progress(0)
                            for i, item in enumerate(unique_items):
                                try:
                                    item_df = df[df[item_col] == item]
                                    if len(item_df) > 0:
                                        train_val, test = train_test_split(item_df, test_size=0.2, shuffle=False)
                                        train, val = train_test_split(train_val, test_size=0.25, shuffle=False)
                                        validation_data[item] = val
                                        item_results, item_future_forecasts, dates = forecast_models(
                                            item_df, selected_models, additional_cols
                                        )
                                        results[item] = item_results
                                        future_forecasts[item] = item_future_forecasts
                                    else:
                                        logging.warning(f"No data available for item {item}.")
                                except Exception as e:
                                    st.error(f"Error processing item {item}: {e}")
                                progress_bar.progress((i + 1) / len(unique_items))
                        elif forecast_type == "Store-Item Combination":
                            results = {}
                            future_forecasts = {}
                            validation_data = {}
                            progress_bar = st.progress(0)
                            
                            for i, combination in enumerate(unique_combinations):
                                try:
                                    combo_df = df[df['store_item'] == combination]
                                    if len(combo_df) > 0:
                                        train_val, test = train_test_split(combo_df, test_size=0.2, shuffle=False)
                                        train, val = train_test_split(train_val, test_size=0.25, shuffle=False)
                                        validation_data[combination] = val
                                        
                                        # Run forecasting models
                                        combo_results, combo_future_forecasts, dates = forecast_models(
                                            combo_df, selected_models, additional_cols
                                        )
                                        results[combination] = combo_results
                                        future_forecasts[combination] = combo_future_forecasts
                                    else:
                                        logging.warning(f"No data available for combination {combination}.")
                                except Exception as e:
                                    st.error(f"Error processing combination {combination}: {e}")
                                progress_bar.progress((i + 1) / len(unique_combinations))
                        else:
                            train_val, test = train_test_split(df, test_size=0.2, shuffle=False)
                            train, val = train_test_split(train_val, test_size=0.25, shuffle=False)
                            validation_data = val
                            results, future_forecasts, dates = forecast_models(df, selected_models, additional_cols)
                    
                    # Get the last date from the dates Series instead of df
                    last_date = pd.to_datetime(dates).max()
                    freq = infer_frequency(df, date_col='Date')
                    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30, freq=freq)
                    future_dates = pd.to_datetime(future_dates)

                    # Display results
                    st.subheader("Forecasting Results")
                    if forecast_type == "Item-wise":
                        selected_item = st.selectbox("Select an item to view detailed forecasts:", unique_items)
                        if selected_item:
                            st.write(f"### Detailed Forecast for {selected_item}")
                            st.write(results[selected_item])
                            st.write(future_forecasts[selected_item])
                    elif forecast_type == "Store-Item Combination":
                        selected_combination = st.selectbox(
                            "Select a store-item combination to view detailed forecasts:", 
                            unique_combinations
                        )
                        if selected_combination:
                            st.write(f"### Detailed Forecast for {selected_combination}")
                            st.write(results[selected_combination])
                            st.write(future_forecasts[selected_combination])

                            # Add store-item specific visualizations
                            store, item = selected_combination.split(" - ")
                            st.write(f"### Historical Demand for {item} at {store}")
                            combo_df = df[df['store_item'] == selected_combination]
                            
                            # Plot historical demand
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=combo_df['Date'],
                                y=combo_df['Demand'],
                                name='Historical Demand',
                                line=dict(color='blue')
                            ))
                            fig.update_layout(
                                title=f'Demand Pattern for {item} at {store}',
                                xaxis_title='Date',
                                yaxis_title='Demand',
                                hovermode='x'
                            )
                            st.plotly_chart(fig)
                    else:
                        for model, metrics in results.items():
                            if isinstance(metrics, dict):
                                st.write(f"**{model}**")
                                
                                # Training metrics
                                if 'train' in metrics and None not in metrics['train']:
                                    st.write(f"- Training: RMSE={metrics['train'][0]:.2f}, MAPE={metrics['train'][1]:.2f}, MAE={metrics['train'][2]:.2f}")
                                
                                # Validation metrics (if available)
                                if 'val' in metrics and None not in metrics.get('val', [None]*3):
                                    st.write(f"- Validation: RMSE={metrics['val'][0]:.2f}, MAPE={metrics['val'][1]:.2f}, MAE={metrics['val'][2]:.2f}")
                                elif 'val' in metrics:
                                    st.write("- Validation: Metrics unavailable")
                                
                                # Test metrics
                                if 'test' in metrics and None not in metrics['test']:
                                    st.write(f"- Test: RMSE={metrics['test'][0]:.2f}, MAPE={metrics['test'][1]:.2f}, MAE={metrics['test'][2]:.2f}")
                            else:
                                st.write(f"{model}: Error - {metrics}")
                    
                    # Display forecasts in a table
                    st.subheader("Forecasted Demand (Tabular Form)")
                    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30, freq=freq)
                    future_dates = pd.to_datetime(future_dates)  # Ensure future_dates is a proper DatetimeIndex

                    if forecast_type == "Item-wise":
                        all_forecasts = pd.concat([
                            pd.DataFrame({
                                'Item': item,
                                'Date': future_dates.strftime('%Y-%m-%d'),
                                **{model: forecast for model, forecast in future_forecasts[item].items()}
                            }) for item in unique_items
                        ])
                        st.dataframe(all_forecasts)
                    elif forecast_type == "Store-Item Combination":
                        all_forecasts = pd.concat([
                            pd.DataFrame({
                                'Store-Item': combination,
                                'Date': future_dates.strftime('%Y-%m-%d'),
                                **{model: forecast for model, forecast in future_forecasts[combination].items()}
                            }) for combination in unique_combinations
                        ])
                        st.dataframe(all_forecasts)
                    else:
                        valid_models = [model for model, forecast in future_forecasts.items() 
                        if isinstance(forecast, list) and len(forecast) == 30]

                        forecast_df = pd.DataFrame({
                            'Date': future_dates.strftime('%Y-%m-%d'),
                            **{model: forecast for model, forecast in future_forecasts.items()}
                        })
                        st.dataframe(forecast_df)

                    # Export forecasts
                    st.subheader("Export Forecasts")
                    if forecast_type == "Item-wise":
                        csv = all_forecasts.to_csv(index=False)
                        st.download_button(
                            label="Download All Item Forecasts as CSV",
                            data=csv,
                            file_name='all_item_forecasts.csv',
                            mime='text/csv'
                        )
                    elif forecast_type == "Store-Item Combination":
                        csv = all_forecasts.to_csv(index=False)
                        st.download_button(
                            label="Download Forecasts as CSV",
                            data=csv,
                            file_name='forecasts.csv',
                            mime='text/csv'
                        )

                    # Export validation data
                    st.subheader("Export Validation Data")
                    if forecast_type == "Item-wise":
                        validation_df = pd.concat([validation_data[item].reset_index(drop=True) for item in unique_items])
                    elif forecast_type == "Store-Item Combination":
                        validation_df = pd.concat([validation_data[combination].reset_index(drop=True) for combination in unique_combinations])
                    else:
                        validation_df = validation_data.reset_index(drop=True)
                    
                    csv = validation_df.to_csv(index=False)
                    st.download_button(
                        label="Download Validation Data as CSV",
                        data=csv,
                        file_name='validation_data.csv',
                        mime='text/csv'
                    )

# Run the app
if __name__ == "__main__":
    main()