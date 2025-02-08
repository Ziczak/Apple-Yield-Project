# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_percentage_error
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, LeakyReLU, Dropout, Attention, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import matplotlib.pyplot as plt

# 1. Load the data
data = pd.read_csv('/Users/zhitanwu/DATAPROJECT/DataScience_Project_1/Merged_Data (PA).csv')

# 2. Preprocess the data
data = data.drop(columns=['NAME', 'Station', 'Latitude', 'Longitude', 'Elevation', 'FMTM', 'PGTM'], errors='ignore')
data = data.dropna()
data['DATE'] = pd.to_datetime(data['DATE'], dayfirst=True)
data['day_of_year'] = data['DATE'].dt.dayofyear
data['year'] = data['DATE'].dt.year

# Clean the 'Value' column
data['Value'] = data['Value'].replace(',', '', regex=True)
data['Value'] = data['Value'].str.extract(r'(\d+\.?\d*)')
data['Value'] = pd.to_numeric(data['Value'], errors='coerce')
data = data.dropna(subset=['Value'])

# Encode categorical columns
encoder = LabelEncoder()
data['state_encoded'] = encoder.fit_transform(data['State'])

# Add cyclical features for day_of_year
data['day_of_year_sin'] = np.sin(2 * np.pi * data['day_of_year'] / 365)
data['day_of_year_cos'] = np.cos(2 * np.pi * data['day_of_year'] / 365)

# 3. Group by year to create yearly samples
data = data.sort_values(by=['year', 'day_of_year'])
X_list = []
y_list = []

for year, group in data.groupby('year'):
    group = group.sort_values('day_of_year')
    X_year = group[['day_of_year_sin', 'day_of_year_cos', 'state_encoded', 'PRCP', 'TMAX', 'TMIN', 'AWND']].values
    y_year = group['Value'].mean()  # Aggregate target value for the year
    X_list.append(X_year)
    y_list.append(y_year)

# Trim sequences to a fixed length
MAX_SEQ_LEN = 365
X_trimmed = [x[-MAX_SEQ_LEN:] if len(x) > MAX_SEQ_LEN else x for x in X_list]
max_len = max(len(x) for x in X_trimmed)
X_padded = np.array([np.pad(x, ((0, max_len - len(x)), (0, 0)), mode='constant', constant_values=0) for x in X_trimmed])
y = np.array(y_list, dtype=np.float32)

# 4. Scaling the data
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X_padded.reshape(-1, X_padded.shape[2])).reshape(X_padded.shape)

scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

# 5. Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, shuffle=False)

# 6. Cross-validation using TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
fold_results = []

for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
    X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
    y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

    # Build the LSTM model with attention
    def build_model_with_attention(input_shape):
        inputs = Input(shape=input_shape)
        lstm_out = LSTM(150, return_sequences=True)(inputs)
        lstm_out = LeakyReLU(alpha=0.01)(lstm_out)
        lstm_out = Dropout(0.2)(lstm_out)

        lstm_out = LSTM(100, return_sequences=True)(lstm_out)
        lstm_out = LeakyReLU(alpha=0.01)(lstm_out)
        lstm_out = Dropout(0.2)(lstm_out)

        lstm_out = LSTM(50, return_sequences=True)(lstm_out)
        lstm_out = LeakyReLU(alpha=0.01)(lstm_out)

        attention_output = Attention()([lstm_out, lstm_out])
        combined = Concatenate()([lstm_out[:, -1, :], attention_output[:, -1, :]])

        outputs = Dense(1, kernel_regularizer='l2')(combined)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=1e-3), loss=Huber(delta=1.0))
        return model

    # Train the model
    model = build_model_with_attention(input_shape=(X_train_fold.shape[1], X_train_fold.shape[2]))

    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(
        X_train_fold, y_train_fold,
        validation_data=(X_val_fold, y_val_fold),
        epochs=40,
        batch_size=25,
        callbacks=[lr_scheduler, early_stopping],
        verbose=1
    )

    val_loss = model.evaluate(X_val_fold, y_val_fold, verbose=0)
    fold_results.append(val_loss)
    print(f"Validation Loss for Fold {fold + 1}: {val_loss:.4f}")

# 7. Final Average Validation Loss
average_loss = np.mean(fold_results)
print(f"Average Validation Loss Across Folds: {average_loss:.4f}")

# 8. Final training on all data
final_model = build_model_with_attention(input_shape=(X_train.shape[1], X_train.shape[2]))
final_model.fit(X_train, y_train, epochs=40, batch_size=25, verbose=1)

# 9. Test set evaluation
y_test_pred_scaled = final_model.predict(X_test)
y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled)

# 10. Calculate Test Accuracy (MAPE)
mape_test = mean_absolute_percentage_error(scaler_y.inverse_transform(y_test), y_test_pred) * 100
test_accuracy = 100 - mape_test
print(f"Test Accuracy: {test_accuracy:.2f}%")

# 11. Predictions and plotting results
y_pred_scaled = final_model.predict(X_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled)

# 12. Future predictions (for the next 5 years)
n_future_years = 5
last_year = 2023
future_years = np.arange(last_year + 1, last_year + 1 + n_future_years)

last_known_data = X_scaled[-1].copy()
future_y_pred_scaled = []

for i in range(n_future_years):
    predicted_value_scaled = final_model.predict(np.array([last_known_data]))[0, 0]
    future_y_pred_scaled.append(predicted_value_scaled)
    last_known_data[:, 0] += 1
    last_known_data[:, -1] = predicted_value_scaled

future_y_pred = scaler_y.inverse_transform(np.array(future_y_pred_scaled).reshape(-1, 1))

# 13. Plot actual vs predicted vs future predictions
plt.figure(figsize=(12, 7))
unique_years = np.arange(2007, 2024)
plt.plot(unique_years, y, label='Actual', color='blue')
plt.plot(unique_years, y_pred.flatten(), label='Predicted', linestyle='--', color='red')
plt.plot([unique_years[-1]] + list(future_years),
         np.concatenate(([y_pred[-1].flatten()[0]], future_y_pred.flatten())),
         label='Future Predicted', linestyle=':', color='green')
plt.title('LSTM Predictions with Attention Layer vs Actual vs Future')
plt.xlabel('Year')
plt.ylabel('Yield (Raw Dataset Values)')
plt.xticks(np.arange(2007, 2029, 2))
plt.legend()
plt.grid()
plt.show()