import os
import pickle
import warnings
import numpy as np
import matplotlib.pyplot as plt
from hmmlearn import hmm
from nnCode.FeaturesExtractor import FeaturesExtractor
import tensorflow as tf # Replaces generic keras import for better compatibility
from tensorflow import keras
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
        
        # Voting statistics
        self.vote_distributions = {
            'female_samples': [],  # (female_votes, male_votes) for actual female samples
            'male_samples': []     # (female_votes, male_votes) for actual male samples
        }
        
        # Confidence tracking (vote margin)
        self.confidence_scores = []
        
        # Component-level predictions tracking
        self.component_predictions = {
            'female': [],  # List of 16-element arrays for female samples
            'male': []     # List of 16-element arrays for male samples
        }
        
        
        # NN training history
        self.nn_training_history = None

        # 1. LOAD MODELS
        print("Loading GMM models...")
        self.females_gmm = pickle.load(open(females_model_path, 'rb'))
        self.males_gmm   = pickle.load(open(males_model_path, 'rb'))
        
        # 2. PREPARE DATA FOR NEURAL NETWORK
        # The Neural Network cannot read the GMM object directly. 
        # We must extract the 'means_' (the 16 vectors representing the voice).
        
        # Shape of means_: (16, 39) -> 16 components, 39 features each
        female_vectors = self.females_gmm.means_
        male_vectors   = self.males_gmm.means_
        
        # Stack them to create training data
        self.X_train = np.vstack((female_vectors, male_vectors))
        
        # Create labels: 0 for Female, 1 for Male
        # We have 16 female vectors and 16 male vectors
        self.y_train = np.hstack((np.zeros(len(female_vectors)), np.ones(len(male_vectors))))
        
        print(f"NN Training Data Shape: {self.X_train.shape}") # Should be (32, 39)
        
        # 3. DEFINE & TRAIN KERAS MODEL
        # We train the NN to classify these specific vectors
        print("Training Neural Network...")
        self.model = keras.Sequential()
        self.model.add(keras.layers.Dense(39, input_dim=39, activation='relu'))
        self.model.add(keras.layers.Dense(13, activation='relu'))
        self.model.add(keras.layers.Dense(2, activation='softmax')) # Softmax is better for categorical
        
        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy', # Matches integer labels (0, 1)
                           metrics=['accuracy'])
        # Store training history                   
        self.nn_training_history =self.model.fit(self.X_train, self.y_train, epochs=50, verbose=0)
        print("Neural Network trained successfully.")
        
    def process(self):
        files = self.get_file_paths(self.females_training_path, self.males_training_path)
        
        for file in files:
            self.total_sample += 1
            filename = os.path.basename(file)
            # print("%10s %8s %1s" % ("--> TESTING", ":", filename))

            try: 
                # Extract features from audio
                vector = self.features_extractor.extract_features(file)
                
                # OPTIMIZATION: Downsample to speed up GMM fitting
                # vector = vector[::5]

                # Fit a temporary GMM to this single file to get its "Supervectors"
                spk_gmm = hmm.GaussianHMM(
                    n_components=16, 
                    covariance_type='diag', 
                    n_iter=20,
                    min_covar=0.01  # Stability fix
                )
                
                spk_gmm.fit(vector)
                
                # Get the means of this file
                spk_vec = spk_gmm.means_
                
                # Predict gender for EACH of the 16 components
                # Note: predict_classes is removed in newer Keras, using argmax instead
                predictions = np.argmax(self.model.predict(spk_vec, verbose=0), axis=-1)
                
                # COUNT VOTES
                # 0 = Female, 1 = Male
                female_votes = np.sum(predictions == 0)
                male_votes   = np.sum(predictions == 1)
                
                if male_votes > female_votes:
                    sc = 1
                    winner = "male"
                else:
                    sc = 0
                    winner = "female"
                
                # Check Expectation (Robust Logic for Windows Paths)
                if "female" in file.lower():
                    expected_gender = "female"
                elif "male" in file.lower():
                    expected_gender = "male"
                else:
                    expected_gender = "unknown"
                
                # Store component predictions for analysis
                if expected_gender == "female":
                    self.component_predictions['female'].append(predictions)
                    self.vote_distributions['female_samples'].append((female_votes, male_votes))
                else:
                    self.component_predictions['male'].append(predictions)
                    self.vote_distributions['male_samples'].append((female_votes, male_votes))
                
                total_votes = female_votes + male_votes
                confidence = abs(female_votes - male_votes) / total_votes if total_votes > 0 else 0
                self.confidence_scores.append(confidence)
                
                # Update metrics
                self.update_metrics(expected_gender, winner)

                print("%10s %6s %1s" % ("+ EXPECTATION",":", expected_gender))
                print("%10s %3s %1s" % ("+ IDENTIFICATION", ":", winner))
                print(f"   Votes -> Female: {female_votes} | Male: {male_votes}")
                print(f"   Confidence: {confidence*100:.1f}%")

                if winner != expected_gender: 
                    self.error += 1

                
            except Exception as e:
                self.total_sample -= 1
                print(f"Error processing {filename}: {e}")
            print("----------------------------------------------------")

        # Final Accuracy
        if self.total_sample > 0:
            accuracy     = (float(self.total_sample - self.error) / float(self.total_sample)) * 100
            accuracy_msg = "*** Accuracy = " + str(round(accuracy, 3)) + "% ***"
            print(accuracy_msg)
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
        females = [os.path.join(females_training_path, f) for f in os.listdir(females_training_path) if f.endswith(".wav")]
        males   = [os.path.join(males_training_path, f) for f in os.listdir(males_training_path) if f.endswith(".wav")]
        return females + males

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
        print(f"\nOVERALL PERFORMANCE")
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
        
        # Neural Network Training Performance
        if self.nn_training_history:
            print("\n" + "-"*70)
            print("NEURAL NETWORK TRAINING PERFORMANCE")
            print("-"*70)
            
            final_train_acc = self.nn_training_history.history['accuracy'][-1]
            final_train_loss = self.nn_training_history.history['loss'][-1]
            
            print(f"\n  Final Training Accuracy:  {final_train_acc*100:.2f}%")
            print(f"  Final Training Loss:      {final_train_loss:.4f}")
            
            if 'val_accuracy' in self.nn_training_history.history:
                final_val_acc = self.nn_training_history.history['val_accuracy'][-1]
                final_val_loss = self.nn_training_history.history['val_loss'][-1]
                print(f"  Final Validation Accuracy: {final_val_acc*100:.2f}%")
                print(f"  Final Validation Loss:     {final_val_loss:.4f}")
        
        # Voting Statistics
        if self.vote_distributions['female_samples'] or self.vote_distributions['male_samples']:
            print("\n" + "-"*70)
            print("VOTING STATISTICS (16 components per sample)")
            print("-"*70)
            
            if self.vote_distributions['female_samples']:
                female_votes_array = np.array(self.vote_distributions['female_samples'])
                print(f"\nFemale samples (n={len(self.vote_distributions['female_samples'])}):")
                print(f"  Avg female votes:  {np.mean(female_votes_array[:, 0]):.2f} / 16")
                print(f"  Avg male votes:    {np.mean(female_votes_array[:, 1]):.2f} / 16")
                print(f"  Avg vote margin:   {np.mean(female_votes_array[:, 0] - female_votes_array[:, 1]):.2f}")
            
            if self.vote_distributions['male_samples']:
                male_votes_array = np.array(self.vote_distributions['male_samples'])
                print(f"\nMale samples (n={len(self.vote_distributions['male_samples'])}):")
                print(f"  Avg female votes:  {np.mean(male_votes_array[:, 0]):.2f} / 16")
                print(f"  Avg male votes:    {np.mean(male_votes_array[:, 1]):.2f} / 16")
                print(f"  Avg vote margin:   {np.mean(male_votes_array[:, 1] - male_votes_array[:, 0]):.2f}")
        
        # Confidence Statistics
        if self.confidence_scores:
            print("\n" + "-"*70)
            print("CONFIDENCE STATISTICS")
            print("-"*70)
            
            print(f"\n  Average confidence:  {np.mean(self.confidence_scores)*100:.2f}%")
            print(f"  Median confidence:   {np.median(self.confidence_scores)*100:.2f}%")
            print(f"  Min confidence:      {np.min(self.confidence_scores)*100:.2f}%")
            print(f"  Max confidence:      {np.max(self.confidence_scores)*100:.2f}%")
            print(f"  Std deviation:       {np.std(self.confidence_scores)*100:.2f}%")
        
        # Component-level Analysis
        if self.component_predictions['female'] or self.component_predictions['male']:
            print("\n" + "-"*70)
            print("COMPONENT-LEVEL PREDICTION ANALYSIS")
            print("-"*70)
            
            if self.component_predictions['female']:
                female_comp = np.array(self.component_predictions['female'])
                female_agreement = np.mean(female_comp == 0) * 100
                print(f"\nFemale samples:")
                print(f"  Component agreement: {female_agreement:.2f}% predicted as female")
                print(f"  Most consistent sample: {np.max([np.sum(pred == 0) for pred in female_comp])} / 16 components")
                print(f"  Least consistent sample: {np.min([np.sum(pred == 0) for pred in female_comp])} / 16 components")
            
            if self.component_predictions['male']:
                male_comp = np.array(self.component_predictions['male'])
                male_agreement = np.mean(male_comp == 1) * 100
                print(f"\nMale samples:")
                print(f"  Component agreement: {male_agreement:.2f}% predicted as male")
                print(f"  Most consistent sample: {np.max([np.sum(pred == 1) for pred in male_comp])} / 16 components")
                print(f"  Least consistent sample: {np.min([np.sum(pred == 1) for pred in male_comp])} / 16 components")
        
        
        print("\n" + "="*70 + "\n")

    def process_plot(self):
        """Create comprehensive visualization"""        
        if self.total_sample == 0:
            return
            
        metrics = self.calculate_metrics()
        
        fig = plt.figure(figsize=(16, 16))
        
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
        
        # 4. Neural Network Training History
        if self.nn_training_history:
            ax4 = plt.subplot(3, 4, 4)
            epochs = range(1, len(self.nn_training_history.history['accuracy']) + 1)
            ax4.plot(epochs, self.nn_training_history.history['accuracy'], 'b-', label='Training Acc', linewidth=2)
            if 'val_accuracy' in self.nn_training_history.history:
                ax4.plot(epochs, self.nn_training_history.history['val_accuracy'], 'r-', label='Validation Acc', linewidth=2)
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Accuracy')
            ax4.set_title('NN Training History')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        # 5. Vote Distribution (Female Samples)
        if self.vote_distributions['female_samples']:
            ax5 = plt.subplot(3, 4, 5)
            female_votes_array = np.array(self.vote_distributions['female_samples'])
            ax5.scatter(female_votes_array[:, 0], female_votes_array[:, 1], 
                       alpha=0.6, color='#e91e63', s=80, edgecolors='black', linewidth=0.5)
            ax5.plot([0, 16], [16, 0], 'k--', alpha=0.5, linewidth=2, label='Decision boundary')
            ax5.set_xlabel('Female Votes (out of 16)')
            ax5.set_ylabel('Male Votes (out of 16)')
            ax5.set_title('Vote Distribution - Female Samples')
            ax5.set_xlim([-0.5, 16.5])
            ax5.set_ylim([-0.5, 16.5])
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        
        # 6. Vote Distribution (Male Samples)
        if self.vote_distributions['male_samples']:
            ax6 = plt.subplot(3, 4, 6)
            male_votes_array = np.array(self.vote_distributions['male_samples'])
            ax6.scatter(male_votes_array[:, 0], male_votes_array[:, 1], 
                       alpha=0.6, color='#2196f3', s=80, edgecolors='black', linewidth=0.5)
            ax6.plot([0, 16], [16, 0], 'k--', alpha=0.5, linewidth=2, label='Decision boundary')
            ax6.set_xlabel('Female Votes (out of 16)')
            ax6.set_ylabel('Male Votes (out of 16)')
            ax6.set_title('Vote Distribution - Male Samples')
            ax6.set_xlim([-0.5, 16.5])
            ax6.set_ylim([-0.5, 16.5])
            ax6.legend()
            ax6.grid(True, alpha=0.3)
        
        # 7. Confidence Distribution
        if self.confidence_scores:
            ax7 = plt.subplot(3, 4, 7)
            ax7.hist(np.array(self.confidence_scores) * 100, bins=20, 
                    color='#9b59b6', edgecolor='black', alpha=0.7)
            ax7.axvline(x=np.mean(self.confidence_scores)*100, color='red', 
                       linestyle='--', linewidth=2, label=f'Mean: {np.mean(self.confidence_scores)*100:.1f}%')
            ax7.set_xlabel('Confidence (%)')
            ax7.set_ylabel('Frequency')
            ax7.set_title('Prediction Confidence Distribution')
            ax7.legend()
            ax7.grid(True, alpha=0.3)
        

        
        # 8. Vote Margin Distribution
        if self.vote_distributions['female_samples'] or self.vote_distributions['male_samples']:
            ax8 = plt.subplot(3, 4, 8)
            
            if self.vote_distributions['female_samples']:
                female_votes_array = np.array(self.vote_distributions['female_samples'])
                female_margins = female_votes_array[:, 0] - female_votes_array[:, 1]
                ax8.hist(female_margins, bins=15, alpha=0.6, label='Female samples', 
                        color='#e91e63', edgecolor='black')
            
            if self.vote_distributions['male_samples']:
                male_votes_array = np.array(self.vote_distributions['male_samples'])
                male_margins = male_votes_array[:, 1] - male_votes_array[:, 0]
                ax8.hist(male_margins, bins=15, alpha=0.6, label='Male samples', 
                        color='#2196f3', edgecolor='black')
            
            ax8.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Decision threshold')
            ax8.set_xlabel('Vote Margin (positive = correct)')
            ax8.set_ylabel('Frequency')
            ax8.set_title('Vote Margin Distribution')
            ax8.legend()
            ax8.grid(True, alpha=0.3)
        
        # 9. Component Agreement Rate
        if self.component_predictions['female'] or self.component_predictions['male']:
            ax9 = plt.subplot(3, 4, 9)
            
            agreement_rates = []
            labels = []
            colors = []
            
            if self.component_predictions['female']:
                female_comp = np.array(self.component_predictions['female'])
                female_agreement = np.mean(female_comp == 0) * 100
                agreement_rates.append(female_agreement)
                labels.append('Female')
                colors.append('#e91e63')
            
            if self.component_predictions['male']:
                male_comp = np.array(self.component_predictions['male'])
                male_agreement = np.mean(male_comp == 1) * 100
                agreement_rates.append(male_agreement)
                labels.append('Male')
                colors.append('#2196f3')
            
            bars = ax9.bar(labels, agreement_rates, color=colors)
            ax9.set_ylabel('Agreement Rate (%)')
            ax9.set_title('Component-Level Agreement')
            ax9.set_ylim([0, 110])
            
            for bar, rate in zip(bars, agreement_rates):
                ax9.text(bar.get_x() + bar.get_width()/2, rate + 2,
                         f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
            ax9.grid(True, alpha=0.3, axis='y')
        
        # 10. Per-class Accuracy
        ax10 = plt.subplot(3, 4, 10)
        female_acc = self.true_positive_female / (self.true_positive_female + self.false_negative_female) if (self.true_positive_female + self.false_negative_female) > 0 else 0
        male_acc = self.true_positive_male / (self.true_positive_male + self.false_negative_male) if (self.true_positive_male + self.false_negative_male) > 0 else 0
        
        bars = ax10.bar(['Female', 'Male'], [female_acc * 100, male_acc * 100], 
                       color=['#e91e63', '#2196f3'])
        ax10.set_title('Per-class Accuracy')
        ax10.set_ylabel('Accuracy (%)')
        ax10.set_ylim([0, 110])
        
        for bar, v in zip(bars, [female_acc * 100, male_acc * 100]):
            ax10.text(bar.get_x() + bar.get_width()/2, v + 2,
                     f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')
        ax10.grid(True, alpha=0.3, axis='y')
        
        # 11. Summary Statistics
        ax11 = plt.subplot(3, 4, 11)
        ax11.axis('off')
        
        avg_confidence = np.mean(self.confidence_scores) * 100 if self.confidence_scores else 0
        
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
        
        Avg Confidence:       {avg_confidence:.1f}%
        
        NN Components:        16 per sample
        NN Training Acc:      {self.nn_training_history.history['accuracy'][-1]*100:.1f}%
        """
        
        ax11.text(0.1, 0.5, summary_text, fontsize=8, verticalalignment='center',
                fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout()
        plt.show()
            
if __name__== "__main__":
    # Ensure these point to the .hmm files you saved with ModelsTrainer
    gender_identifier = GenderIdentifier(
        "TestingData/females", 
        "TestingData/males", 
        "models/females.hmm", 
        "models/males.hmm"
    )
    gender_identifier.process()