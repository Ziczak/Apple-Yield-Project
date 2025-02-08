import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_percentage_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, LeakyReLU, Dropout
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

data['Value'] = data['Value'].replace(',', '', regex=True)
data['Value'] = data['Value'].str.extract(r'(\d+\.?\d*)')
data['Value'] = pd.to_numeric(data['Value'], errors='coerce')
data = data.dropna(subset=['Value'])

encoder = LabelEncoder()
data['state_encoded'] = encoder.fit_transform(data['State'])

data['day_of_year_sin'] = np.sin(2 * np.pi * data['day_of_year'] / 365)
data['day_of_year_cos'] = np.cos(2 * np.pi * data['day_of_year'] / 365)

# 3. Group by year
data = data.sort_values(by=['year', 'day_of_year'])
X_list = []
y_list = []

for year, group in data.groupby('year'):
    group = group.sort_values('day_of_year')
    X_year = group[['day_of_year_sin', 'day_of_year_cos', 'state_encoded', 'PRCP', 'TMAX', 'TMIN', 'AWND']].values
    y_year = group['Value'].mean()
    X_list.append(X_year)
    y_list.append(y_year)

MAX_SEQ_LEN = 365
X_trimmed = [x[-MAX_SEQ_LEN:] if len(x) > MAX_SEQ_LEN else x for x in X_list]
max_len = max(len(x) for x in X_trimmed)
X_padded = np.array([np.pad(x, ((0, max_len - len(x)), (0, 0)), mode='constant', constant_values=0) for x in X_trimmed])
y = np.array(y_list, dtype=np.float32)

scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X_padded.reshape(-1, X_padded.shape[2])).reshape(X_padded.shape)

scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

# 4. Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, shuffle=False)

# 5. Cross-validation using TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
fold_results = []

for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
    X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
    y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

    model = Sequential([
        GRU(150, return_sequences=True, input_shape=(X_train_fold.shape[1], X_train_fold.shape[2])),
        LeakyReLU(alpha=0.01),
        Dropout(0.2),
        GRU(100, return_sequences=True),
        LeakyReLU(alpha=0.01),
        Dropout(0.2),
        GRU(50),
        LeakyReLU(alpha=0.01),
        Dense(1, kernel_regularizer='l2')
    ])

    model.compile(optimizer=Adam(learning_rate=1e-3), loss=Huber(delta=1.0))

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

average_loss = np.mean(fold_results)
print(f"Average Validation Loss Across Folds: {average_loss:.4f}")

final_model = Sequential([
    GRU(150, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    LeakyReLU(alpha=0.01),
    Dropout(0.2),
    GRU(100, return_sequences=True),
    LeakyReLU(alpha=0.01),
    Dropout(0.2),
    GRU(50),
    LeakyReLU(alpha=0.01),
    Dense(1, kernel_regularizer='l2')
])

final_model.compile(optimizer=Adam(learning_rate=1e-3), loss=Huber(delta=1.0))
final_history = final_model.fit(X_train, y_train, epochs=40, batch_size=25, verbose=1)

# 6. Test set evaluation
y_test_pred_scaled = final_model.predict(X_test)
y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled)

mape_test = mean_absolute_percentage_error(scaler_y.inverse_transform(y_test), y_test_pred) * 100
test_accuracy = 100 - mape_test
print(f"Test Accuracy: {test_accuracy:.2f}%")