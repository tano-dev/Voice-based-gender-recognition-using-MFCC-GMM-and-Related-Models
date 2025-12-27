import os
import pickle
import warnings
import numpy as np
import matplotlib.pyplot as plt  # Import for graphing
from hmmlearn import hmm
from svmCode.FeaturesExtractor import FeaturesExtractor
import subprocess
from subprocess import Popen, PIPE

warnings.filterwarnings("ignore")

class ModelsTrainer:

    def __init__(self, females_files_path, males_files_path):
        self.females_training_path = females_files_path
        self.males_training_path   = males_files_path
        self.features_extractor    = FeaturesExtractor()

    def process(self):
        females, males = self.get_file_paths(self.females_training_path,
                                             self.males_training_path)
        files = females + males
        
        # We will store the resulting super-vectors here
        features = {"female": np.asarray(()), "male": np.asarray(())}
        
        # We will store training history for a few samples to visualize
        history_samples = {"female": [], "male": []}
        
        # print(f"Starting processing of {len(files)} files...")

        for i, file in enumerate(files):
            # Determine gender robustly
            if "female" in file.lower():
                gender = "female"
            elif "male" in file.lower():
                gender = "male"
            else:
                continue

            # print(f"[{i+1}/{len(files)}] Processing {gender}: {os.path.basename(file)}")

            try: 
                # Extract MFCC features
                vector = self.features_extractor.extract_features(file)
                
                # Train a mini-HMM for this specific file (Super Vector generation)
                # We enable verbose=False to keep console clean, but hmmlearn tracks history automatically
                spk_gmm = hmm.GaussianHMM(n_components=16, n_iter=50)      
                spk_gmm.fit(vector)
                
                # Capture history for the first 5 files of each gender for plotting
                if len(history_samples[gender]) < 5:
                    history_samples[gender].append(spk_gmm.monitor_.history)

                # Extract the "Super Vector" (the flattened means of the HMM)
                spk_vec = spk_gmm.means_.flatten() # Flattening ensures 1D vector per speaker
                
                # Stack vectors
                if features[gender].size == 0:  
                    features[gender] = spk_vec
                else:                                   
                    features[gender] = np.vstack((features[gender], spk_vec))
            
            except Exception as e:
                # print(f"  Error: {e}")
                pass
        
        # --- SHOW GRAPHS ---
        self.visualize_process(history_samples, features)

        # Save models
        self.save_gmm(features["female"], "females")
        self.save_gmm(features["male"],   "males")

    def visualize_process(self, history_samples, features):
        """
        Visualizes the HMM convergence for samples and the final data distribution.
        """
        plt.figure(figsize=(14, 6))

        # 1. Plot Training Convergence (HMM Fitting)
        plt.subplot(1, 2, 1)
        for h in history_samples['female']:
            plt.plot(h, color='red', alpha=0.5, linewidth=1)
        for h in history_samples['male']:
            plt.plot(h, color='blue', alpha=0.5, linewidth=1)
        
        plt.plot([], [], color='red', label='Female Files')
        plt.plot([], [], color='blue', label='Male Files')
        plt.title('HMM Convergence (Sample Files)')
        plt.xlabel('Iterations')
        plt.ylabel('Log-Likelihood')
        plt.legend()
        plt.grid(True)

        # 2. Plot Extracted Super Vectors (First 2 Dimensions)
        # This gives a rough idea if the data is becoming separable for the SVM
        plt.subplot(1, 2, 2)
        
        if features['female'].size > 0:
            # We only plot the first 2 dimensions of the super vector
            plt.scatter(features['female'][:, 0], features['female'][:, 1], 
                        color='red', label='Female', alpha=0.6, s=10)
            
        if features['male'].size > 0:
            plt.scatter(features['male'][:, 0], features['male'][:, 1], 
                        color='blue', label='Male', alpha=0.6, s=10)
            
        plt.title('Extracted Super Vectors (Dim 0 vs Dim 1)')
        plt.xlabel('Feature Dimension 0')
        plt.ylabel('Feature Dimension 1')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    def get_file_paths(self, females_training_path, males_training_path):
        females = [ os.path.join(females_training_path, f) for f in os.listdir(females_training_path) if f.endswith('.wav')]
        males   = [ os.path.join(males_training_path, f) for f in os.listdir(males_training_path) if f.endswith('.wav')]
        return females, males

    def save_gmm(self, gmm, name):
        """ Save data using pickle. """
        # Saves to the current directory for simplicity
        filename = name + ".svm"
        with open(filename, 'wb') as gmm_file:
            pickle.dump(gmm, gmm_file)
        print ("%5s %10s" % ("SAVED", filename))

    # (Optional: ffmpeg_silence_eliminator left out for brevity as it wasn't used in main loop)

if __name__== "__main__":
    if os.path.exists("TrainingData/females") and os.path.exists("TrainingData/males"):
        models_trainer = ModelsTrainer("TrainingData/females", "TrainingData/males")
        models_trainer.process()
    else:
        print("Error: TrainingData directories not found.")