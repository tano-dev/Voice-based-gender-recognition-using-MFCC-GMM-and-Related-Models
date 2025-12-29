import os
import pickle
import warnings
import numpy as np
from svmCode.FeaturesExtractor import FeaturesExtractor
from hmmlearn import hmm
import subprocess
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_curve, auc

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
        
        # SVM-specific metrics
        self.decision_scores = []  # SVM decision function scores
        self.probabilities = []    # Predicted probabilities
        self.true_labels = []      # Actual labels
        self.predicted_labels = [] # Predicted labels
        
        # Component tracking (from GMM means)
        self.component_counts = []  # Number of components per sample

        # NaN statistics
        self.nan_removal_stats = {
            'original_count': 0,
            'removed_count': 0,
            'kept_count': 0
        }
        
        # Load features (Note: these are numpy arrays of stacked means, not actual GMM objects)
        self.females_gmm = pickle.load(open(females_model_path, 'rb'))
        self.males_gmm   = pickle.load(open(males_model_path, 'rb'))
        
        # Stack features for SVM training
        self.X_train = np.vstack((self.females_gmm, self.males_gmm))
        self.y_train = np.hstack(( -1 * np.ones(self.females_gmm.shape[0]), np.ones(self.males_gmm.shape[0])))
        
        # --- FIX 1: REMOVE NaNs BEFORE TRAINING ---
        # Create a mask to identify rows that do NOT have NaN values
        # Track NaN removal
        self.nan_removal_stats['original_count'] = len(self.X_train)
        mask = ~np.isnan(self.X_train).any(axis=1)
        self.nan_removal_stats['removed_count'] = len(mask) - mask.sum()

        # Keep only the valid rows
        self.X_train = self.X_train[mask]
        self.y_train = self.y_train[mask]
        self.nan_removal_stats['kept_count'] = len(self.X_train)
       
        print(f"Training data: {self.X_train.shape}")
        print(f"  - Original samples: {self.nan_removal_stats['original_count']}")
        print(f"  - Removed (NaN): {self.nan_removal_stats['removed_count']}")
        print(f"  - Clean samples: {self.nan_removal_stats['kept_count']}")
        # ------------------------------------------

        print(f"Training SVM with {len(self.X_train)} clean samples (removed {self.nan_removal_stats['removed_count']} NaN rows)...")
        self.clf = SVC(kernel = 'rbf', probability=True)
        self.clf.fit(self.X_train, self.y_train)
        
        # Evaluate on training data
        train_score = self.clf.score(self.X_train, self.y_train)
        print(f"SVM Training Accuracy: {train_score*100:.2f}%")
        print(f"Support Vectors: {self.clf.n_support_}")

    def process(self):
        files = self.get_file_paths(self.females_training_path, self.males_training_path)
        
        for file in files:
            self.total_sample += 1
            filename = os.path.basename(file)
            print("%10s %8s %1s" % ("--> TESTING", ":", os.path.basename(file)))

            try: 
                # extract MFCC & delta MFCC features from audio
                vector = self.features_extractor.extract_features(file)
                
                spk_gmm = hmm.GaussianHMM(n_components=2)      
                spk_gmm.fit(vector)
                spk_vec = spk_gmm.means_
                
                # Track component count
                self.component_counts.append(len(spk_vec))

                # Predict gender
                prediction_result = self.clf.predict(self.spk_vec)
                probabilities = self.clf.predict_proba(spk_vec)
                decision_scores = self.clf.decision_function(spk_vec)

                # Check prediction (Summing results in case of multiple vectors, though usually it's 1 row)
                if np.sum(prediction_result) > 0: 
                    sc = 1
                else: 
                    sc = -1
                    
                genders = {-1: "female", 1: "male"}
                winner = genders[sc]
                
                # --- FIX 2: ROBUST PATH CHECKING (Fixes Windows/Linux split issue) ---
                if "female" in file.lower():
                    expected_gender = "female"
                    true_label = -1
                elif "male" in file.lower():
                    expected_gender = "male"
                    true_label = 1
                else:
                    expected_gender = "unknown"
                    true_label = 0
                # ------------------------------------------
                # Store for metrics (only if valid label)
                if true_label != 0:
                    self.true_labels.append(true_label)
                    self.predicted_labels.append(sc)
                    
                    # Store probabilities and decision scores
                    # Take mean across components
                    mean_prob = np.mean(probabilities, axis=0)
                    mean_decision = np.mean(decision_scores)
                    
                    self.probabilities.append(mean_prob)
                    self.decision_scores.append(mean_decision)
                
                # Update metrics
                self.update_metrics(expected_gender, winner)
                # ---------------------------------------------------------------------
                
                print("%10s %6s %1s" % ("+ EXPECTATION",":", expected_gender))
                print("%10s %3s %1s" %  ("+ IDENTIFICATION", ":", winner))
                print(f"   Components: {len(spk_vec)}")
                print(f"   Decision score: {np.mean(decision_scores):.3f}")
                print(f"   Probability: Female={np.mean(probabilities[:, 0]):.3f}, Male={np.mean(probabilities[:, 1]):.3f}")

                if winner != expected_gender: 
                    self.error += 1
                print("----------------------------------------------------")

            except Exception as e:
                print(f"Error processing {os.path.basename(file)}: {e}")
                self.total_sample -= 1

        # Final accuracy
        if self.total_sample > 0:
            accuracy     = ( float(self.total_sample - self.error) / float(self.total_sample) ) * 100
            accuracy_msg = "*** Accuracy = " + str(round(accuracy, 3)) + "% ***"
            print(accuracy_msg)
        else:
            print("No samples processed.")
        
        
        # Print comprehensive statistics
        self.print_statistics()
        self.process_plot()


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
        females = [ os.path.join(females_training_path, f) for f in os.listdir(females_training_path) ]
        males   = [ os.path.join(males_training_path, f) for f in os.listdir(males_training_path) ]
        files   = females + males
        return files
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
            print("No samples processed successfully.")
            return
            
        print("\n" + "="*70)
        print("COMPREHENSIVE PERFORMANCE REPORT")
        print("="*70)
        
        # Basic statistics
        accuracy = (self.total_sample - self.error) / self.total_sample * 100
        print(f"\n OVERALL PERFORMANCE")
        print(f"   Accuracy: {round(accuracy, 2)}%")
        print(f"   Correct: {self.total_sample - self.error}/{self.total_sample}")
        print(f"   Incorrect: {self.error}/{self.total_sample}")
        
        # Confusion Matrix
        print("\n" + "-"*70)
        print("CONFUSION MATRIX")
        print("-"*70)
        print(f"{'':20} {'Predicted Female':>20} {'Predicted Male':>20}")
        print(f"{'Actual Female':20} {self.true_positive_female:>20} {self.false_negative_female:>20}")
        print(f"{'Actual Male':20} {self.false_positive_female:>20} {self.true_positive_male:>20}")
        
        # Detailed metrics
        metrics = self.calculate_metrics()
        
        print("\n" + "-"*70)
        print("PER-CLASS METRICS")
        print("-"*70)
        
        for gender in ['female', 'male']:
            print(f"\n{gender.upper()}:")
            print(f"  Precision: {metrics[gender]['precision']:.4f} ({metrics[gender]['precision']*100:.2f}%)")
            print(f"  Recall:    {metrics[gender]['recall']:.4f} ({metrics[gender]['recall']*100:.2f}%)")
            print(f"  F1-Score:  {metrics[gender]['f1_score']:.4f} ({metrics[gender]['f1_score']*100:.2f}%)")
        
        print("\n" + "-"*70)
        print("MACRO-AVERAGED METRICS")
        print("-"*70)
        print(f"  Precision: {metrics['macro_avg']['precision']:.4f} ({metrics['macro_avg']['precision']*100:.2f}%)")
        print(f"  Recall:    {metrics['macro_avg']['recall']:.4f} ({metrics['macro_avg']['recall']*100:.2f}%)")
        print(f"  F1-Score:  {metrics['macro_avg']['f1_score']:.4f} ({metrics['macro_avg']['f1_score']*100:.2f}%)")
        
        # SVM Model Information
        print("\n" + "-"*70)
        print("SVM MODEL INFORMATION")
        print("-"*70)
        print(f"\n  Kernel: {self.clf.kernel}")
        print(f"  Total Support Vectors: {sum(self.clf.n_support_)}")
        print(f"  Support Vectors (Female): {self.clf.n_support_[0]}")
        print(f"  Support Vectors (Male): {self.clf.n_support_[1]}")
        print(f"  Gamma: {self.clf.gamma}")
        print(f"  C (regularization): {self.clf.C}")
        
        # Training data statistics
        print("\n" + "-"*70)
        print("TRAINING DATA STATISTICS")
        print("-"*70)
        print(f"\n  Original samples: {self.nan_removal_stats['original_count']}")
        print(f"  Removed (NaN): {self.nan_removal_stats['removed_count']}")
        print(f"  Clean samples used: {self.nan_removal_stats['kept_count']}")
        if self.nan_removal_stats['original_count'] > 0:
            removal_pct = (self.nan_removal_stats['removed_count'] / self.nan_removal_stats['original_count']) * 100
            print(f"  Removal rate: {removal_pct:.1f}%")
        
        # Decision scores statistics
        if self.decision_scores:
            print("\n" + "-"*70)
            print("SVM DECISION SCORES STATISTICS")
            print("-"*70)
            
            decision_scores_array = np.array(self.decision_scores)
            true_labels_array = np.array(self.true_labels)
            
            female_scores = decision_scores_array[true_labels_array == -1]
            male_scores = decision_scores_array[true_labels_array == 1]
            
            print(f"\nFemale samples (n={len(female_scores)}):")
            if len(female_scores) > 0:
                print(f"  Mean decision score: {np.mean(female_scores):.3f}")
                print(f"  Std decision score:  {np.std(female_scores):.3f}")
                print(f"  Min/Max: {np.min(female_scores):.3f} / {np.max(female_scores):.3f}")
            
            print(f"\nMale samples (n={len(male_scores)}):")
            if len(male_scores) > 0:
                print(f"  Mean decision score: {np.mean(male_scores):.3f}")
                print(f"  Std decision score:  {np.std(male_scores):.3f}")
                print(f"  Min/Max: {np.min(male_scores):.3f} / {np.max(male_scores):.3f}")
        
        # Probability statistics
        if self.probabilities:
            print("\n" + "-"*70)
            print("PREDICTION PROBABILITY STATISTICS")
            print("-"*70)
            
            probs_array = np.array(self.probabilities)
            true_labels_array = np.array(self.true_labels)
            
            female_probs = probs_array[true_labels_array == -1]
            male_probs = probs_array[true_labels_array == 1]
            
            print(f"\nFemale samples:")
            if len(female_probs) > 0:
                print(f"  Avg P(Female): {np.mean(female_probs[:, 0]):.3f}")
                print(f"  Avg P(Male):   {np.mean(female_probs[:, 1]):.3f}")
            
            print(f"\nMale samples:")
            if len(male_probs) > 0:
                print(f"  Avg P(Female): {np.mean(male_probs[:, 0]):.3f}")
                print(f"  Avg P(Male):   {np.mean(male_probs[:, 1]):.3f}")
        
        # Component statistics
        if self.component_counts:
            print("\n" + "-"*70)
            print("GMM COMPONENT STATISTICS")
            print("-"*70)
            
            print(f"\n  Average components per sample: {np.mean(self.component_counts):.1f}")
            print(f"  Min/Max components: {np.min(self.component_counts)} / {np.max(self.component_counts)}")
        

        print("\n" + "="*70 + "\n")

    def process_plot(self):
        """Create comprehensive visualization"""
        import matplotlib.pyplot as plt
        
        if self.total_sample == 0:
            return
            
        metrics = self.calculate_metrics()
        
        fig = plt.figure(figsize=(18, 12))
        
        # 1. Confusion Matrix
        ax1 = plt.subplot(3, 4, 1)
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
        
        for i in range(2):
            for j in range(2):
                ax1.text(j, i, confusion_matrix[i, j],
                        ha="center", va="center", color="black", fontsize=14, weight='bold')
        plt.colorbar(im, ax=ax1)
        
        # 2. Accuracy Bar Chart
        ax2 = plt.subplot(3, 4, 2)
        ax2.bar(['Correct', 'Incorrect'], 
                [self.total_sample - self.error, self.error], 
                color=['#2ecc71', '#e74c3c'])
        ax2.set_title(f'Overall Accuracy: {round(metrics["accuracy"]*100, 2)}%')
        ax2.set_ylabel('Number of Samples')
        for i, v in enumerate([self.total_sample - self.error, self.error]):
            ax2.text(i, v + 0.5, str(v), ha='center', va='bottom', fontweight='bold')
        
        # 3. Metrics Comparison
        ax3 = plt.subplot(3, 4, 3)
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
        
        # 4. SVM Decision Scores Distribution
        if self.decision_scores and self.true_labels:
            ax4 = plt.subplot(3, 4, 4)
            decision_scores_array = np.array(self.decision_scores)
            true_labels_array = np.array(self.true_labels)
            
            female_scores = decision_scores_array[true_labels_array == -1]
            male_scores = decision_scores_array[true_labels_array == 1]
            
            ax4.hist(female_scores, bins=15, alpha=0.6, label='Female samples', 
                    color='#e91e63', edgecolor='black')
            ax4.hist(male_scores, bins=15, alpha=0.6, label='Male samples', 
                    color='#2196f3', edgecolor='black')
            ax4.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Decision boundary')
            ax4.set_xlabel('SVM Decision Score')
            ax4.set_ylabel('Frequency')
            ax4.set_title('Decision Score Distribution')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        # 5. Decision Scores Scatter Plot
        if self.decision_scores and self.true_labels:
            ax5 = plt.subplot(3, 4, 5)
            decision_scores_array = np.array(self.decision_scores)
            true_labels_array = np.array(self.true_labels)
            
            female_idx = true_labels_array == -1
            male_idx = true_labels_array == 1
            
            ax5.scatter(np.where(female_idx)[0], decision_scores_array[female_idx],
                       alpha=0.6, color='#e91e63', s=60, label='Female', edgecolors='black', linewidth=0.5)
            ax5.scatter(np.where(male_idx)[0], decision_scores_array[male_idx],
                       alpha=0.6, color='#2196f3', s=60, label='Male', edgecolors='black', linewidth=0.5)
            ax5.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Decision boundary')
            ax5.set_xlabel('Sample Index')
            ax5.set_ylabel('Decision Score')
            ax5.set_title('Decision Scores by Sample')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        
        # 6. Probability Distribution
        if self.probabilities and self.true_labels:
            ax6 = plt.subplot(3, 4, 6)
            probs_array = np.array(self.probabilities)
            true_labels_array = np.array(self.true_labels)
            
            female_probs = probs_array[true_labels_array == -1]
            male_probs = probs_array[true_labels_array == 1]
            
            if len(female_probs) > 0:
                ax6.scatter(female_probs[:, 0], female_probs[:, 1], 
                           alpha=0.6, color='#e91e63', s=60, label='Female samples',
                           edgecolors='black', linewidth=0.5)
            if len(male_probs) > 0:
                ax6.scatter(male_probs[:, 0], male_probs[:, 1], 
                           alpha=0.6, color='#2196f3', s=60, label='Male samples',
                           edgecolors='black', linewidth=0.5)
            
            ax6.plot([0, 1], [1, 0], 'k--', alpha=0.5, linewidth=2, label='Decision boundary')
            ax6.set_xlabel('P(Female)')
            ax6.set_ylabel('P(Male)')
            ax6.set_title('Prediction Probabilities')
            ax6.set_xlim([-0.05, 1.05])
            ax6.set_ylim([-0.05, 1.05])
            ax6.legend()
            ax6.grid(True, alpha=0.3)
        
        # 7. Support Vectors Distribution
        ax7 = plt.subplot(3, 4, 7)
        sv_labels = ['Female SV', 'Male SV']
        sv_counts = self.clf.n_support_
        colors_sv = ['#e91e63', '#2196f3']
        bars = ax7.bar(sv_labels, sv_counts, color=colors_sv)
        ax7.set_ylabel('Count')
        ax7.set_title(f'Support Vectors\nTotal: {sum(sv_counts)}')
        for bar, count in zip(bars, sv_counts):
            ax7.text(bar.get_x() + bar.get_width()/2, count + 0.5,
                    str(count), ha='center', va='bottom', fontweight='bold')
        ax7.grid(True, alpha=0.3, axis='y')
        
 
        
        # 8. Per-class Accuracy
        ax8 = plt.subplot(3, 4, 8)
        female_acc = self.true_positive_female / (self.true_positive_female + self.false_negative_female) if (self.true_positive_female + self.false_negative_female) > 0 else 0
        male_acc = self.true_positive_male / (self.true_positive_male + self.false_negative_male) if (self.true_positive_male + self.false_negative_male) > 0 else 0
        
        bars = ax8.bar(['Female', 'Male'], [female_acc * 100, male_acc * 100], 
                      color=['#e91e63', '#2196f3'])
        ax8.set_title('Per-class Accuracy')
        ax8.set_ylabel('Accuracy (%)')
        ax8.set_ylim([0, 110])
        
        for bar, v in zip(bars, [female_acc * 100, male_acc * 100]):
            ax8.text(bar.get_x() + bar.get_width()/2, v + 2,
                    f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')
        ax8.grid(True, alpha=0.3, axis='y')
        
        # 9. Training Data Quality
        ax9 = plt.subplot(3, 4, 9)
        quality_labels = ['Clean', 'NaN Removed']
        quality_counts = [self.nan_removal_stats['kept_count'], 
                         self.nan_removal_stats['removed_count']]
        colors_quality = ['#2ecc71', '#e74c3c']
        bars = ax9.bar(quality_labels, quality_counts, color=colors_quality)
        ax9.set_ylabel('Count')
        ax9.set_title('Training Data Quality')
        for bar, count in zip(bars, quality_counts):
            if count > 0:
                ax9.text(bar.get_x() + bar.get_width()/2, count + 0.5,
                         str(count), ha='center', va='bottom', fontweight='bold')
        ax9.grid(True, alpha=0.3, axis='y')
        
        # 10. ROC Curve
        if self.probabilities and self.true_labels:
            ax10 = plt.subplot(3, 4, 10)
            probs_array = np.array(self.probabilities)
            true_labels_array = np.array(self.true_labels)
            
            # Convert labels to binary (0 for female, 1 for male)
            binary_labels = (true_labels_array == 1).astype(int)
            male_probs = probs_array[:, 1]  # Probability of male class
            
            fpr, tpr, thresholds = roc_curve(binary_labels, male_probs)
            roc_auc = auc(fpr, tpr)
            
            ax10.plot(fpr, tpr, color='#2196f3', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
            ax10.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--',label='Random')
            ax10.set_xlim([0.0, 1.0])
            ax10.set_ylim([0.0, 1.05])
            ax10.set_xlabel('False Positive Rate')
            ax10.set_ylabel('True Positive Rate')
            ax10.set_title('ROC Curve')
            ax10.legend(loc="lower right")
            ax10.grid(True, alpha=0.3)
        
        # 11. Summary Statistics
        ax11 = plt.subplot(3, 4, 11)
        ax11.axis('off')
        
        avg_decision_score = np.mean(np.abs(self.decision_scores)) if self.decision_scores else 0
        
        summary_text = f"""
        SUMMARY STATISTICS
        ═══════════════════════════════════
        
        Total Samples:        {self.total_sample}
        Correct:              {self.total_sample - self.error}
        Incorrect:            {self.error}
        
        Overall Accuracy:     {metrics['accuracy']*100:.2f}%
        
        Female Samples:       {self.true_positive_female + self.false_negative_female}
        Male Samples:         {self.true_positive_male + self.false_negative_male}
        
        Macro Avg Precision:  {metrics['macro_avg']['precision']*100:.2f}%
        Macro Avg Recall:     {metrics['macro_avg']['recall']*100:.2f}%
        Macro Avg F1:         {metrics['macro_avg']['f1_score']*100:.2f}%
        
        SVM Kernel:           {self.clf.kernel}
        Total Support Vectors: {sum(self.clf.n_support_)}
        
        Training Samples:     {self.nan_removal_stats['kept_count']}
        NaN Removed:          {self.nan_removal_stats['removed_count']}
        
        Avg Decision Score:   {avg_decision_score:.3f}
        """
        
        ax11.text(0.1, 0.5, summary_text, fontsize=8, verticalalignment='center',
                fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout()
        plt.show()




if __name__== "__main__":
    # Ensure these paths point to your actual .svm (pickle) files
    gender_identifier = GenderIdentifier("TestingData/females", "TestingData/males", "svmCode/females.svm", "svmCode/males.svm")
    gender_identifier.process()