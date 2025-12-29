import os
import pickle
import warnings
import numpy as np
from hmmCode.FeaturesExtractor import FeaturesExtractor
from hmmlearn import hmm

warnings.filterwarnings("ignore")

import pydub
import librosa
import soundfile as sf
import numpy as np
import speech_recognition as sr
from pydub import AudioSegment
from pydub.silence import split_on_silence, detect_nonsilent


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
        # Score tracking
        self.female_scores = []  # (female_score, male_score, ubm_score) for female samples
        self.male_scores = []    # (female_score, male_score, ubm_score) for male samples
        
        
        # Silence removal statistics
        self.silence_stats = {
            'original_durations': [],
            'trimmed_durations': [],
            'silence_percentages': []
        }
        # load models
        self.females_gmm = pickle.load(open(females_model_path, 'rb'))
        self.males_gmm   = pickle.load(open(males_model_path, 'rb'))
        self.ubm         = pickle.load(open("ubm.hmm", 'rb'))
        
        
        
    def ffmpeg_silence_eliminator(self, input_path, output_path):
        """
        Eliminate silence using librosa (No FFmpeg required).
        """
        try:
            # 1. Load the audio file
            # sr=None preserves original sampling rate
            y, sr = librosa.load(input_path, sr=None)
            
            # Calculate original duration for logging
            orig_duration = librosa.get_duration(y=y, sr=sr)

            # 2. Trim silence (top_db=36 matches your old ffmpeg -36dB setting)
            y_trimmed, _ = librosa.effects.trim(y, top_db=36)
            
            # Calculate new duration
            new_duration = librosa.get_duration(y=y_trimmed, sr=sr)

            # Track silence statistics
            silence_removed = orig_duration - new_duration
            silence_percentage = (silence_removed / orig_duration) * 100 if orig_duration > 0 else 0
            
            self.silence_stats['original_durations'].append(orig_duration)
            self.silence_stats['trimmed_durations'].append(new_duration)
            self.silence_stats['silence_percentages'].append(silence_percentage)

            # 3. Print info (Replacing your os.popen/sed logic)
            print("%-32s %-7s %-50s" % ("ORIGINAL SAMPLE DURATION", ":", float(orig_duration)))
            print("%-23s %-7s %-50s" % ("SILENCE FILTERED SAMPLE DURATION", ":", float(new_duration)))
            print("%-22s %-7s %-50s" % ("SILENCE REMOVED (seconds)", ":", float(silence_removed)))
            print("%-24s %-7s %-50s" % ("SILENCE REMOVED (%)", ":", str(round(silence_percentage, 3)) + "%"))

            # 4. Save the processed file to disk
            # We save as PCM_16 to ensure compatibility with standard WAV readers
            sf.write(output_path, y_trimmed, sr, subtype='PCM_16')
            
            # Return values (Your process() method ignores these, but we return them to match the old signature)
            return y_trimmed, new_duration

        except Exception as e:
            print(f"Error processing {input_path}: {e}")
            return None, 0
    
        
        
    def process(self):
        files = self.get_file_paths(self.females_training_path, self.males_training_path)
        # read the test directory and get the list of test audio files
        for file in files:
            self.total_sample += 1
            print("%10s %8s %1s" % ("--> TESTING", ":", os.path.basename(file)))

            self.ffmpeg_silence_eliminator(file, file.split('.')[0] + "_without_silence.wav")
        
            # extract MFCC & delta MFCC features from audio
            try: 
                vector = self.features_extractor.extract_features(file.split('.')[0] + "_without_silence.wav")
                winner = self.identify_gender(vector)
                # OLD (Broken on Windows):
                # expected_gender = file.split("/")[1][:-1]

                # NEW (Robust):
                # 1. Get the folder name (e.g., "females" or "males")
                folder_name = os.path.basename(os.path.dirname(file))

                # 2. Remove the last 's' to turn "females" into "female"
                expected_gender = folder_name[:-1]

                # Identify gender and get scores
                winner, female_score, male_score, ubm_score = self.identify_gender(vector)
                
                # Store scores
                if expected_gender == "female":
                    self.female_scores.append((female_score, male_score, ubm_score))
                else:
                    self.male_scores.append((female_score, male_score, ubm_score))
                
                # Update metrics
                self.update_metrics(expected_gender, winner)

                print("%10s %6s %1s" % ("+ EXPECTATION",":", expected_gender))
                print("%10s %3s %1s" %  ("+ IDENTIFICATION", ":", winner))

                if winner != expected_gender: self.error += 1
                print("----------------------------------------------------")

    
            except :
                self.total_sample -= 1  # Don't count failed samples 
                pass
            os.remove(file.split('.')[0] + "_without_silence.wav")
            
        accuracy     = ( float(self.total_sample - self.error) / float(self.total_sample) ) * 100
        accuracy_msg = "*** Accuracy = " + str(round(accuracy, 3)) + "% ***"
        print(accuracy_msg)  
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
        # get file paths
        females = [ os.path.join(females_training_path, f) for f in os.listdir(females_training_path) ]
        males   = [ os.path.join(males_training_path, f) for f in os.listdir(males_training_path) ]
        files   = females + males
        return files

    def identify_gender(self, vector):
        ubm_score = self.ubm.score(vector)
        
        # USE SUBTRACTION (Standard Log-Likelihood Ratio)
        is_female_log_likelihood = self.females_gmm.score(vector) - ubm_score
        is_male_log_likelihood   = self.males_gmm.score(vector)   - ubm_score

        print("%10s %5s %1s" % ("+ UBM SCORE",":", str(round(ubm_score, 3))))
        print("%10s %5s %1s" % ("+ FEMALE SCORE",":", str(round(is_female_log_likelihood, 3))))
        print("%10s %7s %1s" % ("+ MALE SCORE", ":", str(round(is_male_log_likelihood,3))))

        if is_male_log_likelihood > is_female_log_likelihood: winner = "male"
        else                                                : winner = "female"
        return winner, is_female_log_likelihood, is_male_log_likelihood, ubm_score
    
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
        
        # HMM Score statistics with UBM
        if self.female_scores and self.male_scores:
            print("\n" + "-"*70)
            print("HMM SCORE STATISTICS (with UBM normalization)")
            print("-"*70)
            
            female_scores_array = np.array(self.female_scores)
            male_scores_array = np.array(self.male_scores)
            
            print(f"\nFemale samples (n={len(self.female_scores)}):")
            print(f"  Avg female score (normalized): {np.mean(female_scores_array[:, 0]):.3f}")
            print(f"  Avg male score (normalized):   {np.mean(female_scores_array[:, 1]):.3f}")
            print(f"  Avg UBM score:                 {np.mean(female_scores_array[:, 2]):.3f}")
            print(f"  Avg score gap:                 {np.mean(female_scores_array[:, 0] - female_scores_array[:, 1]):.3f}")
            
            print(f"\nMale samples (n={len(self.male_scores)}):")
            print(f"  Avg female score (normalized): {np.mean(male_scores_array[:, 0]):.3f}")
            print(f"  Avg male score (normalized):   {np.mean(male_scores_array[:, 1]):.3f}")
            print(f"  Avg UBM score:                 {np.mean(male_scores_array[:, 2]):.3f}")
            print(f"  Avg score gap:                 {np.mean(male_scores_array[:, 1] - male_scores_array[:, 0]):.3f}")
        
        # Silence removal statistics
        if self.silence_stats['original_durations']:
            print("\n" + "-"*70)
            print("SILENCE REMOVAL STATISTICS")
            print("-"*70)
            
            avg_original = np.mean(self.silence_stats['original_durations'])
            avg_trimmed = np.mean(self.silence_stats['trimmed_durations'])
            avg_silence_pct = np.mean(self.silence_stats['silence_percentages'])
            
            print(f"\n  Average original duration:  {avg_original:.2f}s")
            print(f"  Average trimmed duration:   {avg_trimmed:.2f}s")
            print(f"  Average silence removed:    {avg_silence_pct:.1f}%")
            print(f"  Total time saved:           {sum(self.silence_stats['original_durations']) - sum(self.silence_stats['trimmed_durations']):.2f}s")
        

        print("\n" + "="*70 + "\n")

    def process_plot(self):
        """Create comprehensive visualization"""
        import matplotlib.pyplot as plt
        
        if self.total_sample == 0:
            return
            
        metrics = self.calculate_metrics()
        
        fig = plt.figure(figsize=(16, 14))
        
        # 1. Confusion Matrix
        ax1 = plt.subplot(3, 3, 1)
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
        plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

        # 2. Accuracy Bar Chart
        ax2 = plt.subplot(3, 3, 2)
        ax2.bar(['Correct', 'Incorrect'], 
                [self.total_sample - self.error, self.error], 
                color=['#2ecc71', '#e74c3c'])
        ax2.set_title(f'Overall Accuracy: {round(metrics["accuracy"]*100, 2)}%')
        ax2.set_ylabel('Number of Samples')
        for i, v in enumerate([self.total_sample - self.error, self.error]):
            ax2.text(i, v + 0.5, str(v), ha='center', va='bottom', fontweight='bold')
        
        # 3. Precision, Recall, F1 Comparison
        ax3 = plt.subplot(3, 3, 3)
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
        
        # 4. HMM Score Distribution (Female vs Male) with UBM
        if self.female_scores and self.male_scores:
            ax4 = plt.subplot(3, 3, 4)
            female_scores_array = np.array(self.female_scores)
            male_scores_array = np.array(self.male_scores)
            
            ax4.scatter(female_scores_array[:, 0], female_scores_array[:, 1], 
                       alpha=0.6, label='Female samples', color='#e91e63', s=50, edgecolors='black', linewidth=0.5)
            ax4.scatter(male_scores_array[:, 0], male_scores_array[:, 1], 
                       alpha=0.6, label='Male samples', color='#2196f3', s=50, edgecolors='black', linewidth=0.5)
            
            # Decision boundary
            lims = [min(ax4.get_xlim()[0], ax4.get_ylim()[0]), 
                   max(ax4.get_xlim()[1], ax4.get_ylim()[1])]
            ax4.plot(lims, lims, 'k--', alpha=0.5, linewidth=2, label='Decision boundary')
            
            ax4.set_xlabel('Female HMM Score (normalized)')
            ax4.set_ylabel('Male HMM Score (normalized)')
            ax4.set_title('HMM Score Distribution')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        # 5. Score Gap Distribution
        if self.female_scores and self.male_scores:
            ax5 = plt.subplot(3, 3, 5)
            female_scores_array = np.array(self.female_scores)
            male_scores_array = np.array(self.male_scores)
            
            female_gaps = female_scores_array[:, 0] - female_scores_array[:, 1]
            male_gaps = male_scores_array[:, 1] - male_scores_array[:, 0]
            
            ax5.hist(female_gaps, bins=15, alpha=0.6, label='Female samples', color='#e91e63', edgecolor='black')
            ax5.hist(male_gaps, bins=15, alpha=0.6, label='Male samples', color='#2196f3', edgecolor='black')
            ax5.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Decision threshold')
            ax5.set_xlabel('Score Gap (correct - incorrect)')
            ax5.set_ylabel('Frequency')
            ax5.set_title('Score Gap Distribution')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        
        # 6. Silence Removal Statistics
        if self.silence_stats['silence_percentages']:
            ax6 = plt.subplot(3, 3, 6)
            ax6.hist(self.silence_stats['silence_percentages'], bins=20, 
                    color='#9b59b6', edgecolor='black', alpha=0.7)
            ax6.set_xlabel('Silence Removed (%)')
            ax6.set_ylabel('Number of Files')
            ax6.set_title(f'Silence Removal Distribution\nAvg: {np.mean(self.silence_stats["silence_percentages"]):.1f}%')
            ax6.grid(True, alpha=0.3)

        # 7. Per-class Accuracy
        ax7 = plt.subplot(3, 3, 7)
        female_acc = self.true_positive_female / (self.true_positive_female + self.false_negative_female) if (self.true_positive_female + self.false_negative_female) > 0 else 0
        male_acc = self.true_positive_male / (self.true_positive_male + self.false_negative_male) if (self.true_positive_male + self.false_negative_male) > 0 else 0
        
        bars = ax7.bar(['Female', 'Male'], [female_acc * 100, male_acc * 100], color=['#e91e63', '#2196f3'])
        ax7.set_title('Per-class Accuracy')
        ax7.set_ylabel('Accuracy (%)')
        ax7.set_ylim([0, 110])
        for i, (bar, v) in enumerate(zip(bars, [female_acc * 100, male_acc * 100])):
            ax7.text(bar.get_x() + bar.get_width()/2, v + 2, 
                    f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')
        ax7.grid(True, alpha=0.3, axis='y')
        
        # 8. Summary Statistics
        ax8 = plt.subplot(3, 3, 8)
        ax8.axis('off')
        
        avg_silence = np.mean(self.silence_stats['silence_percentages']) if self.silence_stats['silence_percentages'] else 0
        
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
        
        Avg Silence Removed:  {avg_silence:.1f}%
        """
        
        ax8.text(0.1, 0.5, summary_text, fontsize=9, verticalalignment='center',
                fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout()
        plt.show()


# if __name__== "__main__":
#     gender_identifier = GenderIdentifier("TestingData/females", "TestingData/males", "females.hmm", "males.hmm")
#     gender_identifier.process()
