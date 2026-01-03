import os
import pickle
import warnings
import numpy as np
import matplotlib.pyplot as plt
from nnCode.FeaturesExtractor import FeaturesExtractor 

warnings.filterwarnings("ignore")

class GenderIdentifier:

    def __init__(self, females_files_path, males_files_path, model_path):
        self.females_training_path = females_files_path
        self.males_training_path   = males_files_path
        self.error                 = 0
        self.total_sample          = 0
        self.features_extractor    = FeaturesExtractor()
        
        # Metrics tracking
        self.true_positive_female = 0   
        self.true_positive_male = 0     
        self.false_positive_female = 0  
        self.false_positive_male = 0    
        self.false_negative_female = 0  
        self.false_negative_male = 0    
        
        # --- NEW: Data storage for Plotting (Giống GMM) ---
        # Lưu tuple (prob_female, prob_male) cho các mẫu thực tế là Nữ
        self.score_history_female = [] 
        # Lưu tuple (prob_female, prob_male) cho các mẫu thực tế là Nam
        self.score_history_male = []
        
        # 1. LOAD MODEL & SCALER
        print(f"Loading Neural Network model from: {model_path}...")
        try:
            with open(model_path, 'rb') as f:
                saved_data = pickle.load(f)
            
            self.model = saved_data['model']
            self.scaler = saved_data['scaler']
            print("Model loaded successfully.")
            
        except Exception as e:
            print(f"CRITICAL ERROR loading model: {e}")
            exit(1)

    def extract_stat_features(self, file_path):
        try:
            vector = self.features_extractor.extract_features(file_path)
            mean_vec = np.mean(vector, axis=0)
            std_vec  = np.std(vector, axis=0)
            stat_vector = np.concatenate((mean_vec, std_vec))
            return stat_vector.reshape(1, -1) 
        except Exception as e:
            return None

    def process(self):
        files = self.get_file_paths(self.females_training_path, self.males_training_path)
        
        for file in files:
            self.total_sample += 1
            filename = os.path.basename(file)

            try: 
                # 1. Trích xuất đặc trưng
                vector = self.extract_stat_features(file)
                
                if vector is not None:
                    # 2. Scale dữ liệu
                    vector_scaled = self.scaler.transform(vector)
                    
                    # 3. Dự đoán
                    # predict_proba trả về [[prob_female, prob_male]] (vì label 0=Female, 1=Male)
                    probs = self.model.predict_proba(vector_scaled)[0]
                    prob_female = probs[0]
                    prob_male   = probs[1]
                    
                    # Quyết định: Ai cao hơn người đó thắng
                    if prob_male > prob_female:
                        winner = "male"
                    else:
                        winner = "female"
                    
                    # 4. Kiểm tra kết quả thực tế (Expectation)
                    if "female" in file.lower():
                        expected_gender = "female"
                        # Lưu điểm số vào lịch sử Nữ để vẽ đồ thị
                        self.score_history_female.append((prob_female, prob_male))
                    elif "male" in file.lower():
                        expected_gender = "male"
                        # Lưu điểm số vào lịch sử Nam để vẽ đồ thị
                        self.score_history_male.append((prob_female, prob_male))
                    else:
                        expected_gender = "unknown"
                    
                    # 5. Cập nhật Metrics
                    self.update_metrics(expected_gender, winner)
                    confidence = max(prob_female, prob_male)

                    print("%10s %6s %1s" % ("+ EXPECTATION",":", expected_gender))
                    print("%10s %3s %1s" % ("+ IDENTIFICATION", ":", winner))
                    print(f"   Probs -> Female: {prob_female:.4f} | Male: {prob_male:.4f}")

                    if winner != expected_gender: 
                        self.error += 1
                else:
                    print(f"Skipping {filename}: Unable to extract features")
                    self.total_sample -= 1

            except Exception as e:
                self.total_sample -= 1
                print(f"Error processing {filename}: {e}")
            print("----------------------------------------------------")

        # Final Statistics & Plot
        if self.total_sample > 0:
            accuracy = (float(self.total_sample - self.error) / float(self.total_sample)) * 100
            print("*** Accuracy = " + str(round(accuracy, 3)) + "% ***")
            self.print_statistics()
            self.process_plot() # <--- Gọi hàm vẽ đồ thị mới

    def update_metrics(self, expected, predicted):
        if expected == "female" and predicted == "female":
            self.true_positive_female += 1
        elif expected == "female" and predicted == "male":
            self.false_negative_female += 1 
            self.false_positive_male += 1    
        elif expected == "male" and predicted == "male":
            self.true_positive_male += 1
        elif expected == "male" and predicted == "female":
            self.false_negative_male += 1    
            self.false_positive_female += 1  

    def get_file_paths(self, females_training_path, males_training_path):
        females = [os.path.join(females_training_path, f) for f in os.listdir(females_training_path) if f.endswith(".wav")]
        males   = [os.path.join(males_training_path, f) for f in os.listdir(males_training_path) if f.endswith(".wav")]
        return females + males

    def calculate_metrics(self):
        metrics = {}
        # Female metrics
        tp_f, fp_f, fn_f = self.true_positive_female, self.false_positive_female, self.false_negative_female
        prec_f = tp_f / (tp_f + fp_f) if (tp_f + fp_f) > 0 else 0
        rec_f  = tp_f / (tp_f + fn_f) if (tp_f + fn_f) > 0 else 0
        f1_f   = 2*(prec_f*rec_f)/(prec_f+rec_f) if (prec_f+rec_f) > 0 else 0
        metrics['female'] = {'precision': prec_f, 'recall': rec_f, 'f1_score': f1_f}
        
        # Male metrics
        tp_m, fp_m, fn_m = self.true_positive_male, self.false_positive_male, self.false_negative_male
        prec_m = tp_m / (tp_m + fp_m) if (tp_m + fp_m) > 0 else 0
        rec_m  = tp_m / (tp_m + fn_m) if (tp_m + fn_m) > 0 else 0
        f1_m   = 2*(prec_m*rec_m)/(prec_m+rec_m) if (prec_m+rec_m) > 0 else 0
        metrics['male'] = {'precision': prec_m, 'recall': rec_m, 'f1_score': f1_m}
        
        # Overall
        acc = (self.total_sample - self.error) / self.total_sample if self.total_sample > 0 else 0
        metrics['accuracy'] = acc
        
        # Macro Avg
        metrics['macro_avg'] = {
            'precision': (prec_f + prec_m)/2,
            'recall': (rec_f + rec_m)/2,
            'f1_score': (f1_f + f1_m)/2
        }
        return metrics

    def print_statistics(self):
        metrics = self.calculate_metrics()
        print("\n" + "="*70)
        print("PERFORMANCE REPORT (NEURAL NETWORK)")
        print("="*70)
        print(f"Overall Accuracy: {metrics['accuracy']*100:.2f}%")
        # (Giữ nguyên phần print cũ của bạn...)

    def process_plot(self):
        """Create comprehensive visualization similar to GMM style"""
        
        if self.total_sample == 0:
            return
            
        metrics = self.calculate_metrics()
        
        # Tạo khung hình lớn
        fig = plt.figure(figsize=(15, 10))
        fig.suptitle('Neural Network Gender Recognition Results', fontsize=16)
        
        # --- 1. Confusion Matrix ---
        ax1 = plt.subplot(2, 3, 1)
        confusion_matrix = np.array([
            [self.true_positive_female, self.false_negative_female],
            [self.false_positive_female, self.true_positive_male]
        ])
        im = ax1.imshow(confusion_matrix, cmap='Blues', aspect='auto')
        ax1.set_xticks([0, 1]); ax1.set_yticks([0, 1])
        ax1.set_xticklabels(['Pred Female', 'Pred Male'])
        ax1.set_yticklabels(['Act Female', 'Act Male'])
        ax1.set_title('Confusion Matrix')
        
        for i in range(2):
            for j in range(2):
                ax1.text(j, i, confusion_matrix[i, j],
                         ha="center", va="center", color="black", fontsize=14, weight='bold')
        plt.colorbar(im, ax=ax1)
        
        # --- 2. Accuracy Bar Chart ---
        ax2 = plt.subplot(2, 3, 2)
        ax2.bar(['Correct', 'Incorrect'], 
                [self.total_sample - self.error, self.error], 
                color=['#2ecc71', '#e74c3c'])
        ax2.set_title(f'Overall Accuracy: {round(metrics["accuracy"]*100, 2)}%')
        ax2.set_ylabel('Number of Samples')
        for i, v in enumerate([self.total_sample - self.error, self.error]):
            ax2.text(i, v + 0.5, str(v), ha='center', va='bottom', fontweight='bold')
        
        # --- 3. Precision, Recall, F1 Comparison ---
        ax3 = plt.subplot(2, 3, 3)
        categories = ['Precision', 'Recall', 'F1-Score']
        female_vals = [metrics['female']['precision'], metrics['female']['recall'], metrics['female']['f1_score']]
        male_vals = [metrics['male']['precision'], metrics['male']['recall'], metrics['male']['f1_score']]
        
        x = np.arange(len(categories))
        width = 0.35
        ax3.bar(x - width/2, female_vals, width, label='Female', color='#e91e63')
        ax3.bar(x + width/2, male_vals, width, label='Male', color='#2196f3')
        ax3.set_title('Metrics by Gender')
        ax3.set_xticks(x); ax3.set_xticklabels(categories)
        ax3.legend(loc='lower right')
        ax3.set_ylim([0, 1.1])
        
        # --- 4. Per-class Accuracy ---
        ax4 = plt.subplot(2, 3, 4)
        denom_f = self.true_positive_female + self.false_negative_female
        denom_m = self.true_positive_male + self.false_negative_male
        
        female_acc = (self.true_positive_female / denom_f) * 100 if denom_f > 0 else 0
        male_acc   = (self.true_positive_male / denom_m) * 100 if denom_m > 0 else 0
        
        bars = ax4.bar(['Female', 'Male'], [female_acc, male_acc], color=['#e91e63', '#2196f3'])
        ax4.set_title('Per-class Accuracy')
        ax4.set_ylabel('Accuracy (%)')
        ax4.set_ylim([0, 110])
        for bar, v in zip(bars, [female_acc, male_acc]):
            ax4.text(bar.get_x() + bar.get_width()/2, v + 2, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # --- 5. Score Distribution (Scatter Plot - GIỐNG GMM) ---
        ax5 = plt.subplot(2, 3, 5)
        
        # Chuyển list thành numpy array để vẽ
        if self.score_history_female:
            f_data = np.array(self.score_history_female) # Cột 0 là Prob_Fem, Cột 1 là Prob_Male
            ax5.scatter(f_data[:, 0], f_data[:, 1], 
                       alpha=0.6, label='Actual Female', color='#e91e63', s=40)
            
        if self.score_history_male:
            m_data = np.array(self.score_history_male)
            ax5.scatter(m_data[:, 0], m_data[:, 1], 
                       alpha=0.6, label='Actual Male', color='#2196f3', s=40)
        
        # Vẽ đường phân cách (Decision Boundary)
        # Với NN softmax, biên quyết định là đường y = x (hoặc Prob_Female = 0.5)
        ax5.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Equal Probability')
        
        ax5.set_xlabel('Probability Female')
        ax5.set_ylabel('Probability Male')
        ax5.set_title('Probability Distribution\n(Separation of Classes)')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        # Lưu ý: Vì Prob_Fem + Prob_Male = 1, các điểm sẽ nằm trên đường chéo y = 1-x.
        
        # --- 6. Summary Statistics Text ---
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        
        summary_text = f"""
        SUMMARY STATISTICS
        ════════════════════════════
        
        Total Samples:    {self.total_sample}
        Correct:          {self.total_sample - self.error}
        Incorrect:        {self.error}
        
        OVERALL ACCURACY: {metrics['accuracy']*100:.2f}%
        
        Actual Females:   {denom_f}
        Actual Males:     {denom_m}
        
        Macro Precision:  {metrics['macro_avg']['precision']*100:.2f}%
        Macro Recall:     {metrics['macro_avg']['recall']*100:.2f}%
        Macro F1-Score:   {metrics['macro_avg']['f1_score']*100:.2f}%
        """
        
        ax6.text(0.05, 0.5, summary_text, fontsize=11, verticalalignment='center',
                 fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

if __name__== "__main__":
    # ĐƯỜNG DẪN: Phải trỏ đúng vào file .nn được tạo bởi ModelsTrainer
    model_file_path = "models/gender_mlp.nn" 
    
    if os.path.exists("TestingData/females") and os.path.exists("TestingData/males") and os.path.exists(model_file_path):
        identifier = GenderIdentifier(
            "TestingData/females", 
            "TestingData/males", 
            model_file_path
        )
        identifier.process()
    else:
        print("Error: Testing Data or Model file not found.")