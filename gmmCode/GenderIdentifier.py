import os
import pickle
import warnings
import numpy as np
from gmmCode.FeaturesExtractor import FeaturesExtractor

warnings.filterwarnings("ignore")

class GenderIdentifier:

    def __init__(self, females_files_path, males_files_path, females_model_path, males_model_path):
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
        
        # Score tracking for analysis
        self.female_scores = []
        self.male_scores = []
        
        # Load models
        self.females_gmm = pickle.load(open(females_model_path, 'rb'))
        self.males_gmm   = pickle.load(open(males_model_path, 'rb'))

    def process(self):
        files = self.get_file_paths(self.females_training_path, self.males_training_path)
        
        print(f"Processing {len(files)} files...")

        for file in files:
            self.total_sample += 1
            print("%10s %8s %1s" % ("--> TESTING", ":", os.path.basename(file)))

            vector = self.features_extractor.extract_features(file)
            
            # Check if vector extraction failed (empty)
            if vector.size == 0:
                print("    [Skipping: Empty features vector]")
                continue

            winner, female_score, male_score = self.identify_gender(vector)
            
            # Get expected gender from folder structure
            parent_folder = os.path.basename(os.path.dirname(file))
            if "female" in parent_folder.lower():
                expected_gender = "female"
            else:
                expected_gender = "male"

            print("%10s %6s %1s" % ("+ EXPECTATION",":", expected_gender))
            print("%10s %3s %1s" %  ("+ IDENTIFICATION", ":", winner))

            # Update confusion matrix metrics
            self.update_metrics(expected_gender, winner)
            
            # Store scores for analysis
            if expected_gender == "female":
                self.female_scores.append((female_score, male_score))
            else:
                self.male_scores.append((female_score, male_score))

            if winner != expected_gender: 
                self.error += 1
                print(f"    [MISMATCH] Expected: {expected_gender}, Got: {winner}")
            print("----------------------------------------------------")

        if self.total_sample > 0:
            # Print all metrics
            self.print_statistics()
            self.process_plot()
        else:
            print("No samples processed.")

    def update_metrics(self, expected, predicted):
        """Update confusion matrix metrics"""
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
        # Get file paths and filter for .wav files only
        females = [ os.path.join(females_training_path, f) for f in os.listdir(females_training_path) if f.endswith('.wav') ]
        males   = [ os.path.join(males_training_path, f) for f in os.listdir(males_training_path) if f.endswith('.wav') ]
        files   = females + males
        return files

    def identify_gender(self, vector):
        # score() returns the average log-likelihood of the samples
        is_female_scores = self.females_gmm.score(vector)
        is_male_scores = self.males_gmm.score(vector)

        print("%10s %5s %1s" % ("+ FEMALE SCORE",":", str(round(is_female_scores, 3))))
        print("%10s %7s %1s" % ("+ MALE SCORE", ":", str(round(is_male_scores, 3))))

        if is_male_scores > is_female_scores: 
            winner = "male"
        else:                                 
            winner = "female"
        
        return winner, is_female_scores, is_male_scores
    
    def calculate_metrics(self):
        """Calculate precision, recall, F1-score for each gender"""
        metrics = {}
        
        # Female metrics
        if (self.true_positive_female + self.false_positive_female) > 0:
            precision_female = self.true_positive_female / (self.true_positive_female + self.false_positive_female)
        else:
            precision_female = 0
            
        if (self.true_positive_female + self.false_negative_female) > 0:
            recall_female = self.true_positive_female / (self.true_positive_female + self.false_negative_female)
        else:
            recall_female = 0
            
        if (precision_female + recall_female) > 0:
            f1_female = 2 * (precision_female * recall_female) / (precision_female + recall_female)
        else:
            f1_female = 0
        
        metrics['female'] = {
            'precision': precision_female,
            'recall': recall_female,
            'f1_score': f1_female
        }
        
        # Male metrics
        if (self.true_positive_male + self.false_positive_male) > 0:
            precision_male = self.true_positive_male / (self.true_positive_male + self.false_positive_male)
        else:
            precision_male = 0
            
        if (self.true_positive_male + self.false_negative_male) > 0:
            recall_male = self.true_positive_male / (self.true_positive_male + self.false_negative_male)
        else:
            recall_male = 0
            
        if (precision_male + recall_male) > 0:
            f1_male = 2 * (precision_male * recall_male) / (precision_male + recall_male)
        else:
            f1_male = 0
        
        metrics['male'] = {
            'precision': precision_male,
            'recall': recall_male,
            'f1_score': f1_male
        }
        
        # Overall accuracy
        metrics['accuracy'] = (self.total_sample - self.error) / self.total_sample if self.total_sample > 0 else 0
        
        # Macro-averaged metrics
        metrics['macro_avg'] = {
            'precision': (precision_female + precision_male) / 2,
            'recall': (recall_female + recall_male) / 2,
            'f1_score': (f1_female + f1_male) / 2
        }
        
        return metrics
    
    def print_statistics(self):
        """Print comprehensive statistics"""
        if self.total_sample == 0:
            print("No samples to analyze.")
            return
            
        print("\n" + "="*60)
        print("DETAILED PERFORMANCE METRICS")
        print("="*60)
        
        # Basic statistics
        accuracy = (self.total_sample - self.error) / self.total_sample * 100
        print(f"\n Overall Accuracy: {round(accuracy, 2)}%")
        print(f" Correct predictions: {self.total_sample - self.error}/{self.total_sample}")
        print(f" Incorrect predictions: {self.error}/{self.total_sample}")
        
        # Confusion Matrix
        print("\n" + "-"*60)
        print("CONFUSION MATRIX")
        print("-"*60)
        print(f"{'':20} {'Predicted Female':>18} {'Predicted Male':>18}")
        print(f"{'Actual Female':20} {self.true_positive_female:>18} {self.false_negative_female:>18}")
        print(f"{'Actual Male':20} {self.false_positive_female:>18} {self.true_positive_male:>18}")
        
        # Detailed metrics
        metrics = self.calculate_metrics()
        
        print("\n" + "-"*60)
        print("PER-CLASS METRICS")
        print("-"*60)
        
        for gender in ['female', 'male']:
            print(f"\n{gender.upper()}:")
            print(f"  Precision: {metrics[gender]['precision']:.4f} ({metrics[gender]['precision']*100:.2f}%)")
            print(f"  Recall:    {metrics[gender]['recall']:.4f} ({metrics[gender]['recall']*100:.2f}%)")
            print(f"  F1-Score:  {metrics[gender]['f1_score']:.4f} ({metrics[gender]['f1_score']*100:.2f}%)")
        
        print("\n" + "-"*60)
        print("MACRO-AVERAGED METRICS")
        print("-"*60)
        print(f"  Precision: {metrics['macro_avg']['precision']:.4f} ({metrics['macro_avg']['precision']*100:.2f}%)")
        print(f"  Recall:    {metrics['macro_avg']['recall']:.4f} ({metrics['macro_avg']['recall']*100:.2f}%)")
        print(f"  F1-Score:  {metrics['macro_avg']['f1_score']:.4f} ({metrics['macro_avg']['f1_score']*100:.2f}%)")
        
        # Score statistics
        if self.female_scores and self.male_scores:
            print("\n" + "-"*60)
            print("SCORE STATISTICS")
            print("-"*60)
            
            female_scores_array = np.array(self.female_scores)
            male_scores_array = np.array(self.male_scores)
            
            print(f"\nFemale samples (n={len(self.female_scores)}):")
            print(f"  Avg female score: {np.mean(female_scores_array[:, 0]):.3f}")
            print(f"  Avg male score:   {np.mean(female_scores_array[:, 1]):.3f}")
            
            print(f"\nMale samples (n={len(self.male_scores)}):")
            print(f"  Avg female score: {np.mean(male_scores_array[:, 0]):.3f}")
            print(f"  Avg male score:   {np.mean(male_scores_array[:, 1]):.3f}")
        
        print("\n" + "="*60 + "\n")
            
    def process_plot(self):
        """Create comprehensive visualization"""
        import matplotlib.pyplot as plt
        
        if self.total_sample == 0:
            return
            
        metrics = self.calculate_metrics()
        
        fig = plt.figure(figsize=(15, 10))
        
        # 1. Confusion Matrix
        ax1 = plt.subplot(2, 3, 1)
        confusion_matrix = np.array([
            [self.true_positive_female, self.false_negative_female],
            [self.false_positive_female, self.true_positive_male]
        ])
        im = ax1.imshow(confusion_matrix, cmap='Blues', aspect='auto')
        ax1.set_xticks([0, 1])
        ax1.set_yticks([0, 1])
        ax1.set_xticklabels(['Pred Female', 'Pred Male'])
        ax1.set_yticklabels(['Actual Female', 'Actual Male'])
        ax1.set_title('Confusion Matrix')
        
        # Add text annotations
        for i in range(2):
            for j in range(2):
                text = ax1.text(j, i, confusion_matrix[i, j],
                              ha="center", va="center", color="black", fontsize=14, weight='bold')
        plt.colorbar(im, ax=ax1)
        
        # 2. Accuracy Bar Chart
        ax2 = plt.subplot(2, 3, 2)
        ax2.bar(['Correct', 'Incorrect'], 
                [self.total_sample - self.error, self.error], 
                color=['#2ecc71', '#e74c3c'])
        ax2.set_title(f'Overall Accuracy: {round(metrics["accuracy"]*100, 2)}%')
        ax2.set_ylabel('Number of Samples')
        for i, v in enumerate([self.total_sample - self.error, self.error]):
            ax2.text(i, v + 0.5, str(v), ha='center', va='bottom', fontweight='bold')
        
        # 3. Precision, Recall, F1 Comparison
        ax3 = plt.subplot(2, 3, 3)
        categories = ['Precision', 'Recall', 'F1-Score']
        female_vals = [metrics['female']['precision'], metrics['female']['recall'], metrics['female']['f1_score']]
        male_vals = [metrics['male']['precision'], metrics['male']['recall'], metrics['male']['f1_score']]
        
        x = np.arange(len(categories))
        width = 0.35
        
        ax3.bar(x - width/2, female_vals, width, label='Female', color='#e91e63')
        ax3.bar(x + width/2, male_vals, width, label='Male', color='#2196f3')
        ax3.set_ylabel('Score')
        ax3.set_title('Metrics by Gender')
        ax3.set_xticks(x)
        ax3.set_xticklabels(categories)
        ax3.legend()
        ax3.set_ylim([0, 1.1])
        
        # 4. Per-class Accuracy
        ax4 = plt.subplot(2, 3, 4)
        female_acc = self.true_positive_female / (self.true_positive_female + self.false_negative_female) if (self.true_positive_female + self.false_negative_female) > 0 else 0
        male_acc = self.true_positive_male / (self.true_positive_male + self.false_negative_male) if (self.true_positive_male + self.false_negative_male) > 0 else 0
        
        ax4.bar(['Female', 'Male'], [female_acc * 100, male_acc * 100], color=['#e91e63', '#2196f3'])
        ax4.set_title('Per-class Accuracy')
        ax4.set_ylabel('Accuracy (%)')
        ax4.set_ylim([0, 110])
        for i, v in enumerate([female_acc * 100, male_acc * 100]):
            ax4.text(i, v + 2, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 5. Score Distribution
        if self.female_scores and self.male_scores:
            ax5 = plt.subplot(2, 3, 5)
            female_scores_array = np.array(self.female_scores)
            male_scores_array = np.array(self.male_scores)
            
            ax5.scatter(female_scores_array[:, 0], female_scores_array[:, 1], 
                       alpha=0.6, label='Female samples', color='#e91e63', s=50)
            ax5.scatter(male_scores_array[:, 0], male_scores_array[:, 1], 
                       alpha=0.6, label='Male samples', color='#2196f3', s=50)
            ax5.plot([min(ax5.get_xlim()[0], ax5.get_ylim()[0]), 
                     max(ax5.get_xlim()[1], ax5.get_ylim()[1])],
                    [min(ax5.get_xlim()[0], ax5.get_ylim()[0]), 
                     max(ax5.get_xlim()[1], ax5.get_ylim()[1])], 
                    'k--', alpha=0.3, label='Decision boundary')
            ax5.set_xlabel('Female GMM Score')
            ax5.set_ylabel('Male GMM Score')
            ax5.set_title('Score Distribution')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        
        # 6. Summary Statistics
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        
        summary_text = f"""
        SUMMARY STATISTICS
        ══════════════════════════════
        
        Total Samples: {self.total_sample}
        Correct: {self.total_sample - self.error}
        Incorrect: {self.error}
        
        Overall Accuracy: {metrics['accuracy']*100:.2f}%
        
        Female Samples: {self.true_positive_female + self.false_negative_female}
        Male Samples: {self.true_positive_male + self.false_negative_male}
        
        Macro Avg Precision: {metrics['macro_avg']['precision']*100:.2f}%
        Macro Avg Recall: {metrics['macro_avg']['recall']*100:.2f}%
        Macro Avg F1: {metrics['macro_avg']['f1_score']*100:.2f}%
        """
        
        ax6.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
                fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout()
        plt.show()

if __name__== "__main__":
    # Ensure paths exist before running
    if os.path.exists("TestingData/females"):
        gender_identifier = GenderIdentifier("TestingData/females", "TestingData/males", "females.gmm", "males.gmm")
        gender_identifier.process()
    else:
        print("Error: Testing Data not found.")