import os
import pickle
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier # <--- THƯ VIỆN ĐÚNG
from sklearn.preprocessing import StandardScaler
from nnCode.FeaturesExtractor import FeaturesExtractor 

warnings.filterwarnings("ignore")

class ModelsTrainer:
    def __init__(self, females_files_path, males_files_path):
        self.females_training_path = females_files_path
        self.males_training_path   = males_files_path
        self.features_extractor    = FeaturesExtractor()

    def get_file_paths(self, females_training_path, males_training_path):
        females = [os.path.join(females_training_path, f) for f in os.listdir(females_training_path) if f.endswith('.wav')]
        males   = [os.path.join(males_training_path, f) for f in os.listdir(males_training_path) if f.endswith('.wav')]
        return females, males

    def extract_stat_features(self, file_path):
        """
        Trích xuất đặc trưng thống kê (Mean & Std) để tạo input cố định cho MLP.
        Output: Vector 1 chiều (ví dụ: 13 mean + 13 std = 26 chiều)
        """
        try:
            # Lấy toàn bộ frame MFCC: shape (T, n_mfcc)
            vector = self.features_extractor.extract_features(file_path)
            
            # Tính Mean và Std dọc theo trục thời gian
            mean_vec = np.mean(vector, axis=0)
            std_vec  = np.std(vector, axis=0)
            
            # Nối lại thành 1 vector duy nhất
            stat_vector = np.concatenate((mean_vec, std_vec))
            return stat_vector
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return None

    def process(self):
        females, males = self.get_file_paths(self.females_training_path, self.males_training_path)
        
        X_data = []
        y_labels = [] # 0: Female, 1: Male

        print("Extracting features for Neural Network...")
        
        # 1. Prepare Female Data
        for f in females:
            vec = self.extract_stat_features(f)
            if vec is not None:
                X_data.append(vec)
                y_labels.append(0) # Label 0 cho Nữ

        # 2. Prepare Male Data
        for m in males:
            vec = self.extract_stat_features(m)
            if vec is not None:
                X_data.append(vec)
                y_labels.append(1) # Label 1 cho Nam

        # Convert to numpy arrays
        X_data = np.array(X_data)
        y_labels = np.array(y_labels)

        print(f"Data shape: {X_data.shape}, Labels shape: {y_labels.shape}")

        # 3. Train MLP Classifier (Neural Network)
        print("Training MLP Classifier...")
        
        # Cấu hình đúng như trong LaTeX mô tả
        mlp = MLPClassifier(
            hidden_layer_sizes=(256, 128), # 2 lớp ẩn
            activation='relu',             # Hàm kích hoạt ReLU
            solver='adam',                 # Tối ưu Adam
            alpha=0.0001,
            batch_size='auto',
            learning_rate='constant',
            learning_rate_init=0.001,
            max_iter=500,                  # Số vòng lặp
            random_state=42,
            verbose=True                   # Hiện quá trình train
        )

        # Scale dữ liệu (Rất quan trọng với Neural Net)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_data)

        # Huấn luyện
        mlp.fit(X_scaled, y_labels)

        # 4. Vẽ biểu đồ Loss curve
        plt.figure(figsize=(8, 5))
        plt.plot(mlp.loss_curve_, label='Loss')
        plt.title('Neural Network Training Loss')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()

        # 5. Lưu mô hình (Lưu cả scaler để dùng lúc test)
        self.save_model(mlp, scaler, "gender_mlp")

    def save_model(self, model, scaler, name):
        base_path = os.path.dirname(os.path.abspath(__file__))
        models_path = os.path.join(base_path, "models")
        if not os.path.exists(models_path):
            os.makedirs(models_path)
            
        filename = os.path.join(models_path, name + ".nn")
        
        # Lưu thành dictionary chứa cả model và scaler
        save_dict = {'model': model, 'scaler': scaler}
        
        with open(filename, 'wb') as f:
            pickle.dump(save_dict, f)
        print(f"SAVED MODEL: {filename}")

if __name__== "__main__":
    if os.path.exists("TrainingData/females") and os.path.exists("TrainingData/males"):
        trainer = ModelsTrainer("TrainingData/females", "TrainingData/males")
        trainer.process()
    else:
        print("Error: 'TrainingData' directory not found.")