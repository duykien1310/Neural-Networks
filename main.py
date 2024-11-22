# loan_approval_classifier.py

import pandas as pd
import numpy as np
import os
import sys
import argparse
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

def load_data(train_path, test_path):
    """
    Tải dữ liệu từ các file CSV.

    Args:
        train_path (str): Đường dẫn đến file train.csv
        test_path (str): Đường dẫn đến file test.csv

    Returns:
        train_data (pd.DataFrame): Dữ liệu huấn luyện
        test_data (pd.DataFrame): Dữ liệu kiểm tra
    """
    train_data = pd.read_csv('train.csv')
    test_data = pd.read_csv('test.csv')
    return train_data, test_data

def preprocess_data(train_data, test_data, target_column):
    """
    Xử lý dữ liệu bao gồm xử lý giá trị thiếu, mã hóa biến phân loại và chuẩn hóa.

    Args:
        train_data (pd.DataFrame): Dữ liệu huấn luyện
        test_data (pd.DataFrame): Dữ liệu kiểm tra
        target_column (str): Tên cột mục tiêu

    Returns:
        X_train (np.array): Đặc trưng huấn luyện đã xử lý
        y_train (np.array): Nhãn huấn luyện
        X_test (np.array): Đặc trưng kiểm tra đã xử lý
        y_test (np.array): Nhãn kiểm tra
    """
    # Xử lý giá trị thiếu
    numerical_features = train_data.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = train_data.select_dtypes(include=['object']).columns.tolist()

    # Giả sử biến mục tiêu là cột đầu tiên hoặc bạn có thể điều chỉnh lại
    if target_column not in train_data.columns:
        raise ValueError(f"Cột mục tiêu '{target_column}' không tồn tại trong dữ liệu huấn luyện.")
    
    numerical_features.remove(target_column)

    for feature in numerical_features:
        train_data[feature].fillna(train_data[feature].mean(), inplace=True)
        test_data[feature].fillna(test_data[feature].mean(), inplace=True)

    for feature in categorical_features:
        if feature != target_column:
            train_data[feature].fillna(train_data[feature].mode()[0], inplace=True)
            test_data[feature].fillna(test_data[feature].mode()[0], inplace=True)

    # Mã hóa biến phân loại
    label_encoders = {}
    for feature in categorical_features:
        if feature != target_column:
            le = LabelEncoder()
            train_data[feature] = le.fit_transform(train_data[feature])
            test_data[feature] = le.transform(test_data[feature])
            label_encoders[feature] = le

    # Mã hóa biến mục tiêu nếu là phân loại nhị phân
    if train_data[target_column].dtype == 'object':
        le_target = LabelEncoder()
        train_data[target_column] = le_target.fit_transform(train_data[target_column])
        test_data[target_column] = le_target.transform(test_data[target_column])
    else:
        le_target = None

    # Tách biến mục tiêu và đặc trưng
    X_train = train_data.drop(target_column, axis=1).values
    y_train = train_data[target_column].values

    X_test = test_data.drop(target_column, axis=1).values
    y_test = test_data[target_column].values

    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, y_train, X_test, y_test, label_encoders, le_target

def build_model(input_dim):
    """
    Xây dựng mô hình Neural Network.

    Args:
        input_dim (int): Số lượng đặc trưng đầu vào

    Returns:
        model (tf.keras.Model): Mô hình đã xây dựng
    """
    model = Sequential()
    model.add(Dense(128, input_dim=input_dim, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

def plot_history(history):
    """
    Vẽ đồ thị lịch sử huấn luyện của mô hình.

    Args:
        history (tf.keras.callbacks.History): Lịch sử huấn luyện
    """
    # Độ chính xác
    plt.figure(figsize=(14,5))
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Độ Chính Xác Huấn Luyện và Xác Thực')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Mất mát
    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Mất Mát Huấn Luyện và Xác Thực')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

def main(train_path, test_path, target_column):
    # Tải dữ liệu
    print("Đang tải dữ liệu...")
    train_data, test_data = load_data(train_path, test_path)

    # Xử lý dữ liệu
    print("Đang xử lý dữ liệu...")
    X_train, y_train, X_test, y_test, label_encoders, le_target = preprocess_data(train_data, test_data, target_column)

    # Xây dựng mô hình
    print("Đang xây dựng mô hình Neural Network...")
    model = build_model(input_dim=X_train.shape[1])
    model.summary()

    # Huấn luyện mô hình với Early Stopping
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    print("Đang huấn luyện mô hình...")
    history = model.fit(X_train, y_train,
                        epochs=100,
                        batch_size=32,
                        validation_split=0.2,
                        callbacks=[early_stop],
                        verbose=1)

    # Vẽ lịch sử huấn luyện
    plot_history(history)

    # Dự đoán trên tập kiểm tra
    print("Đang dự đoán trên tập kiểm tra...")
    y_pred_prob = model.predict(X_test).flatten()
    y_pred = (y_pred_prob > 0.5).astype(int)

    # Đánh giá mô hình
    print("Đang đánh giá mô hình...")
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, target_names=['Not Approved', 'Approved'])

    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(cr)

    # Hiển thị Confusion Matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Not Approved', 'Approved'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Loan Approval Classification using Neural Networks')
    parser.add_argument('--train', type=str, default='train.csv', help='Path to train.csv')
    parser.add_argument('--test', type=str, default='test.csv', help='Path to test.csv')
    parser.add_argument('--target', type=str, default='loan_status', help='Name of the target column')

    # Use parse_known_args to ignore unknown arguments
    args, unknown = parser.parse_known_args()

    if not os.path.exists(args.train):
        print(f"Không tìm thấy file train: {args.train}")
        sys.exit(1)
    if not os.path.exists(args.test):
        print(f"Không tìm thấy file test: {args.test}")
        sys.exit(1)

    main(args.train, args.test, args.target)