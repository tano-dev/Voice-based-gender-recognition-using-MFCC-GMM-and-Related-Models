import os
import pickle
import warnings
import numpy as np
import matplotlib.pyplot as plt  # Import matplotlib for graphing
from Modules.MFCC_FeaturesExtractor import FeaturesExtractor
from hmmlearn import hmm
import librosa
import soundfile as sf

warnings.filterwarnings("ignore")

class ModelsTrainer:

    def __init__(self, females_files_path, males_files_path):
        self.females_training_path = females_files_path
        self.males_training_path   = males_files_path
        self.features_extractor    = FeaturesExtractor()

    def ffmpeg_silence_eliminator(self, input_path, output_path):
        """
        Eliminate silence using librosa (Pure Python, no ffmpeg required).
        """
        try:
            # 1. Load the audio file (sr=None preserves original sampling rate)
            y, sr = librosa.load(input_path, sr=None)

            # 2. Trim silence (top_db=36 matches standard thresholds)
            y_trimmed, index = librosa.effects.trim(y, top_db=36)

            # 3. Save the processed file
            sf.write(output_path, y_trimmed, sr, subtype='PCM_16')
            
            duration = librosa.get_duration(y=y_trimmed, sr=sr)
            # Optional: Return data
            return y_trimmed, duration

        except Exception as e:
            print(f"Error processing {input_path}: {e}")
            return None, 0

    def process(self):
        females, males = self.get_file_paths(self.females_training_path,
                                             self.males_training_path)
        
        # Collect voice features
        print(f"Extracting features from {len(females)} Female and {len(males)} Male files...")
        female_voice_features = self.collect_features(females)
        male_voice_features   = self.collect_features(males)
        
        # --- MODEL CONFIGURATION ---
        # Note: 'verbose=True' allows you to see progress in the console
        females_gmm = hmm.GaussianHMM(n_components=5, verbose=True, n_iter=100)
        males_gmm   = hmm.GaussianHMM(n_components=5, verbose=True, n_iter=100)
        ubm         = hmm.GaussianHMM(n_components=5, verbose=True, n_iter=100)

        # --- FITTING ---
        print("\nTraining Female HMM...")
        females_gmm.fit(female_voice_features)
        
        print("\nTraining Male HMM...")
        males_gmm.fit(male_voice_features)
        
        print("\nTraining UBM HMM...")
        ubm.fit(np.vstack((female_voice_features, male_voice_features)))
        
        # --- PLOTTING HISTORY ---
        # hmmlearn stores convergence history in .monitor_.history
        self.plot_training_history(
            females_gmm.monitor_.history,
            males_gmm.monitor_.history,
            ubm.monitor_.history
        )

        # --- SAVING MODELS ---
        self.save_gmm(females_gmm, "females")
        self.save_gmm(males_gmm,   "males")
        self.save_gmm(ubm,         "ubm")

    def plot_training_history(self, female_hist, male_hist, ubm_hist):
        """
        Plots the log-likelihood convergence for all models.
        """
        plt.figure(figsize=(15, 5))

        # Plot Female History
        plt.subplot(1, 3, 1)
        plt.plot(female_hist, label="Log-Likelihood", color='blue')
        plt.title("Female HMM Training")
        plt.xlabel("Iterations")
        plt.ylabel("Log-Likelihood")
        plt.grid(True)
        plt.legend()

        # Plot Male History
        plt.subplot(1, 3, 2)
        plt.plot(male_hist, label="Log-Likelihood", color='red')
        plt.title("Male HMM Training")
        plt.xlabel("Iterations")
        plt.grid(True)
        plt.legend()

        # Plot UBM History
        plt.subplot(1, 3, 3)
        plt.plot(ubm_hist, label="Log-Likelihood", color='green')
        plt.title("UBM HMM Training")
        plt.xlabel("Iterations")
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()

    def get_file_paths(self, females_training_path, males_training_path):
        females = [ os.path.join(females_training_path, f) for f in os.listdir(females_training_path) if f.endswith('.wav')]
        males   = [ os.path.join(males_training_path, f) for f in os.listdir(males_training_path) if f.endswith('.wav')]
        return females, males

    def collect_features(self, files):
        features = np.asarray(())
        for file in files:
            print("%5s %10s" % ("PROCESSING", file))
            
            # Temporary file without silence
            temp_file = file.replace('.wav', '_without_silence.wav')
            self.ffmpeg_silence_eliminator(file, temp_file)
        
            try: 
                vector = self.features_extractor.extract_features(temp_file)
                if features.size == 0:  
                    features = vector
                else:                   
                    features = np.vstack((features, vector))          
            except Exception as e:
                # print(f"Error extracting features: {e}")
                pass
            
            # Clean up temp file
            if os.path.exists(temp_file):
                os.remove(temp_file)
                
        return features

    def save_gmm(self, gmm, name):
        filename = name + ".hmm"
        with open(filename, 'wb') as gmm_file:
            pickle.dump(gmm, gmm_file)
        print ("%5s %10s" % ("SAVED", filename))

if __name__== "__main__":
    if os.path.exists("TrainingData/females") and os.path.exists("TrainingData/males"):
        models_trainer = ModelsTrainer("TrainingData/females", "TrainingData/males")
        models_trainer.process()
    else:
        print("Error: TrainingData folders not found.")