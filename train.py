import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Bidirectional, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
import joblib
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
tf.random.set_seed(42)

class PowerPredictor:
    def __init__(self, lookback=24):
        self.lookback = lookback
        self.feature_scaler = StandardScaler()  # For features
        self.target_scaler = MinMaxScaler()     # Separate scaler for target
        self.lstm_model = None
        self.gan_generator = None
        self.feature_cols = []
        
    def load_and_preprocess(self, filepath):
        print("Loading data...")
        df = pd.read_csv(filepath)
        df.columns = [col.strip().replace('"', '').replace(' ', '_') for col in df.columns]
        df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
        df.drop(['Date', 'Time'], axis=1, inplace=True)
        df = df.sort_values('Datetime').reset_index(drop=True)
        df.rename(columns={'POWER_(KW)': 'PowerConsumption_Zone1'}, inplace=True)

        # Time features
        df['hour'] = df['Datetime'].dt.hour
        df['day'] = df['Datetime'].dt.day
        df['month'] = df['Datetime'].dt.month
        df['dayofweek'] = df['Datetime'].dt.dayofweek
        df['quarter'] = df['Datetime'].dt.quarter
        df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)

        # Cyclic features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)

        # Rolling and lag features
        df['power_roll_mean_24'] = df['PowerConsumption_Zone1'].rolling(24, min_periods=1).mean()
        df['power_roll_std_24'] = df['PowerConsumption_Zone1'].rolling(24, min_periods=1).std().fillna(0)
        df['power_roll_max_24'] = df['PowerConsumption_Zone1'].rolling(24, min_periods=1).max()
        df['power_roll_min_24'] = df['PowerConsumption_Zone1'].rolling(24, min_periods=1).min()
        
        # Multiple lag features
        df['power_lag_1'] = df['PowerConsumption_Zone1'].shift(1).fillna(df['PowerConsumption_Zone1'].iloc[0])
        df['power_lag_24'] = df['PowerConsumption_Zone1'].shift(24).fillna(df['PowerConsumption_Zone1'].iloc[0])
        df['power_lag_168'] = df['PowerConsumption_Zone1'].shift(168).fillna(df['PowerConsumption_Zone1'].iloc[0])  # week
        
        # Difference features
        df['power_diff_1'] = df['PowerConsumption_Zone1'].diff().fillna(0)
        df['power_diff_24'] = df['PowerConsumption_Zone1'].diff(24).fillna(0)

        self.feature_cols = [
            'hour', 'day', 'month', 'dayofweek', 'quarter', 'is_weekend',
            'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'day_sin', 'day_cos',
            'power_roll_mean_24', 'power_roll_std_24', 'power_roll_max_24', 'power_roll_min_24',
            'power_lag_1', 'power_lag_24', 'power_lag_168',
            'power_diff_1', 'power_diff_24'
        ]
        
        print("✅ Data successfully loaded and preprocessed!")
        print(f"Total samples: {len(df)}")
        print(f"Features: {len(self.feature_cols)}")
        return df

    def create_sequences(self, features, target):
        """Create sequences with separate feature and target arrays"""
        X, y = [], []
        for i in range(len(features) - self.lookback):
            X.append(features[i:i+self.lookback])
            y.append(target[i+self.lookback])
        return np.array(X), np.array(y)
    
    def build_lstm(self, input_shape):
        """Improved LSTM architecture"""
        model = Sequential([
            Bidirectional(LSTM(128, return_sequences=True), input_shape=input_shape),
            Dropout(0.3),
            BatchNormalization(),
            Bidirectional(LSTM(64, return_sequences=True)),
            Dropout(0.3),
            BatchNormalization(),
            LSTM(32),
            Dropout(0.2),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer=Adam(0.001), loss='huber', metrics=['mae', 'mse'])
        return model

    def build_gan_generator(self, input_dim, output_dim=1):
        """Generator that takes LSTM features + noise"""
        model = Sequential([
            Dense(128, activation='relu', input_dim=input_dim),
            BatchNormalization(),
            Dropout(0.3),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dense(64, activation='relu'),
            Dense(output_dim, activation='linear')
        ])
        return model

    def build_gan_discriminator(self, input_dim):
        """Discriminator for real vs generated predictions"""
        # Input dim should match concatenated features + prediction
        model = Sequential([
            Dense(128, activation='relu', input_dim=input_dim),
            Dropout(0.4),
            Dense(64, activation='relu'),
            Dropout(0.4),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        return model

    def train_gan(self, X_train_lstm, y_train, y_pred_lstm, epochs=150, batch_size=64):
        """Train GAN to refine LSTM predictions"""
        print("\n🔥 Training GAN for prediction refinement...")
        
        # Prepare inputs: concatenate last LSTM hidden state representation with prediction
        lstm_features = X_train_lstm[:, -1, :]  # Last timestep features (batch, features)
        
        # Input to generator: features + lstm prediction
        input_dim = lstm_features.shape[1] + 1  # features + lstm_pred
        
        # Input to discriminator: features + lstm prediction + target/generated prediction
        discriminator_input_dim = lstm_features.shape[1] + 2  # features + lstm_pred + target
        
        generator = self.build_gan_generator(input_dim)
        discriminator = self.build_gan_discriminator(discriminator_input_dim)
        discriminator.compile(optimizer=Adam(0.0001, beta_1=0.5), 
                            loss='binary_crossentropy', 
                            metrics=['accuracy'])

        # Combined GAN model
        from tensorflow.keras.layers import Concatenate
        discriminator.trainable = False
        gan_input = Input(shape=(input_dim,))
        refined_pred = generator(gan_input)
        # Discriminator sees: original input (features + lstm_pred) + refined prediction
        combined = Concatenate()([gan_input, refined_pred])
        gan_output = discriminator(combined)
        gan = Model(gan_input, gan_output)
        gan.compile(optimizer=Adam(0.0001, beta_1=0.5), loss='binary_crossentropy')

        # Training loop
        best_gen_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(epochs):
            # Train discriminator
            idx = np.random.randint(0, len(y_train), batch_size)
            
            # Real samples: features + lstm_pred + actual target
            real_features = lstm_features[idx]
            real_lstm_preds = y_pred_lstm[idx].reshape(-1, 1)
            real_targets = y_train[idx].reshape(-1, 1)
            
            # Combine for discriminator input
            real_combined = np.concatenate([real_features, real_lstm_preds, real_targets], axis=1)
            
            # Generate fake predictions
            gen_input = np.concatenate([real_features, real_lstm_preds], axis=1)
            fake_preds = generator.predict(gen_input, verbose=0)
            fake_combined = np.concatenate([real_features, real_lstm_preds, fake_preds], axis=1)
            
            # Train discriminator on real and fake
            d_loss_real = discriminator.train_on_batch(real_combined, np.ones((batch_size, 1)) * 0.9)
            d_loss_fake = discriminator.train_on_batch(fake_combined, np.zeros((batch_size, 1)))
            
            # Train generator (wants discriminator to think fake predictions are real)
            g_loss = gan.train_on_batch(gen_input, np.ones((batch_size, 1)))
            
            if epoch % 20 == 0:
                # Extract loss values (train_on_batch returns [loss, metrics...])
                d_real = d_loss_real[0] if isinstance(d_loss_real, list) else d_loss_real
                d_fake = d_loss_fake[0] if isinstance(d_loss_fake, list) else d_loss_fake
                g = g_loss[0] if isinstance(g_loss, list) else g_loss
                print(f"Epoch {epoch}/{epochs} - D_loss: {(d_real + d_fake)/2:.4f}, G_loss: {g:.4f}")
            
            # Early stopping for GAN
            g_loss_val = g_loss[0] if isinstance(g_loss, list) else g_loss
            if g_loss_val < best_gen_loss:
                best_gen_loss = g_loss_val
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping GAN at epoch {epoch}")
                break

        self.gan_generator = generator
        print("✅ GAN training complete!")

    def train(self, df):
        print("\n" + "="*50)
        print("Starting Model Training Pipeline")
        print("="*50)
        
        # Separate features and target
        feature_data = df[self.feature_cols].values
        target_data = df['PowerConsumption_Zone1'].values.reshape(-1, 1)
        
        # Scale separately
        scaled_features = self.feature_scaler.fit_transform(feature_data)
        scaled_target = self.target_scaler.fit_transform(target_data).flatten()
        
        # Create sequences
        X, y = self.create_sequences(scaled_features, scaled_target)
        
        # Split data
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        print(f"\n📊 Data Split:")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Test samples: {len(X_test)}")
        print(f"  Sequence length: {self.lookback}")
        print(f"  Number of features: {X_train.shape[2]}")

        # Train LSTM
        print("\n🚀 Training LSTM Model...")
        self.lstm_model = self.build_lstm((X_train.shape[1], X_train.shape[2]))
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6)
        ]
        
        history = self.lstm_model.fit(
            X_train, y_train, 
            validation_data=(X_test, y_test),
            epochs=100, 
            batch_size=32, 
            callbacks=callbacks,
            verbose=1
        )

        # Get LSTM predictions
        print("\n📈 Evaluating LSTM Model...")
        y_pred_train_scaled = self.lstm_model.predict(X_train, verbose=0).flatten()
        y_pred_test_scaled = self.lstm_model.predict(X_test, verbose=0).flatten()
        
        # Calculate LSTM-only metrics
        mse_lstm = mean_squared_error(y_test, y_pred_test_scaled)
        mae_lstm = mean_absolute_error(y_test, y_pred_test_scaled)
        r2_lstm = r2_score(y_test, y_pred_test_scaled)
        rmse_lstm = np.sqrt(mse_lstm)
        
        print("\n=== LSTM-Only Performance (Scaled) ===")
        print(f"MSE:  {mse_lstm:.4f}")
        print(f"RMSE: {rmse_lstm:.4f}")
        print(f"MAE:  {mae_lstm:.4f}")
        print(f"R²:   {r2_lstm:.4f}")

        # Train GAN for refinement
        self.train_gan(X_train, y_train, y_pred_train_scaled, epochs=150)

        # GAN refinement on test set
        print("\n🎯 Applying GAN Refinement...")
        lstm_features_test = X_test[:, -1, :]
        gan_input_test = np.concatenate([lstm_features_test, y_pred_test_scaled.reshape(-1, 1)], axis=1)
        gan_corrections = self.gan_generator.predict(gan_input_test, verbose=0).flatten()
        
        # Refined predictions (blend LSTM + GAN with smaller weight for stability)
        y_pred_refined_scaled = y_pred_test_scaled * 0.7 + gan_corrections * 0.3

        # Calculate refined metrics
        mse_refined = mean_squared_error(y_test, y_pred_refined_scaled)
        mae_refined = mean_absolute_error(y_test, y_pred_refined_scaled)
        r2_refined = r2_score(y_test, y_pred_refined_scaled)
        rmse_refined = np.sqrt(mse_refined)

        # Inverse transform for actual scale metrics
        y_test_actual = self.target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        y_pred_refined_actual = self.target_scaler.inverse_transform(y_pred_refined_scaled.reshape(-1, 1)).flatten()
        
        mse_actual = mean_squared_error(y_test_actual, y_pred_refined_actual)
        mae_actual = mean_absolute_error(y_test_actual, y_pred_refined_actual)
        r2_actual = r2_score(y_test_actual, y_pred_refined_actual)
        rmse_actual = np.sqrt(mse_actual)

        print("\n" + "="*50)
        print("=== Final Model Performance (LSTM + GAN) ===")
        print("="*50)
        print("\nScaled Metrics:")
        print(f"  MSE:  {mse_refined:.4f}")
        print(f"  RMSE: {rmse_refined:.4f}")
        print(f"  MAE:  {mae_refined:.4f}")
        print(f"  R²:   {r2_refined:.4f}")
        print("\nActual Scale Metrics:")
        print(f"  MSE:  {mse_actual:.4f}")
        print(f"  RMSE: {rmse_actual:.4f}")
        print(f"  MAE:  {mae_actual:.4f}")
        print(f"  R²:   {r2_actual:.4f}")
        print("="*50)

        # Save metrics
        metrics = {
            "LSTM_MSE": mse_lstm,
            "LSTM_MAE": mae_lstm,
            "LSTM_R2": r2_lstm,
            "LSTM_RMSE": rmse_lstm,
            "Refined_MSE": mse_refined,
            "Refined_MAE": mae_refined,
            "Refined_R2": r2_refined,
            "Refined_RMSE": rmse_refined,
            "Actual_MSE": mse_actual,
            "Actual_MAE": mae_actual,
            "Actual_R2": r2_actual,
            "Actual_RMSE": rmse_actual
        }
        pd.DataFrame([metrics]).to_csv("metrics.csv", index=False)
        
        # Save processed data with predictions
        df_results = df.iloc[self.lookback:].copy()
        df_results = df_results.iloc[split:].reset_index(drop=True)
        df_results['LSTM_Prediction'] = y_pred_refined_actual
        df_results['Actual_Power'] = y_test_actual
        df_results.to_csv("processed_data.csv", index=False)
        
        return df

    def save_models(self):
        """Save all models and scalers"""
        self.lstm_model.save("lstm_model.h5")
        self.gan_generator.save("gan_generator.h5")
        joblib.dump(self.feature_scaler, "feature_scaler.pkl")
        joblib.dump(self.target_scaler, "target_scaler.pkl")
        joblib.dump(self.feature_cols, "feature_cols.pkl")
        joblib.dump(self.lookback, "lookback.pkl")
        print("\n✅ All models and scalers saved successfully!")

if __name__ == "__main__":
    print("="*50)
    print("Power Consumption Prediction - LSTM + GAN")
    print("="*50)
    
    predictor = PowerPredictor(lookback=24)
    df = predictor.load_and_preprocess("power_data.csv")
    df = predictor.train(df)
    predictor.save_models()
    
    print("\n" + "="*50)
    print("✅ Training Complete!")
    print("="*50)
    print("\nNext steps:")
    print("  1. Check 'metrics.csv' for detailed performance metrics")
    print("  2. Run visualization script to see predictions")
    print("="*50)