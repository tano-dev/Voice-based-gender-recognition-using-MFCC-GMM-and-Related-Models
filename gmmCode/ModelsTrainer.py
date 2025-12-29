import os
import pickle
import warnings
import numpy as np
import matplotlib.pyplot as plt  # Import matplotlib
from sklearn.mixture import GaussianMixture
from gmmCode.FeaturesExtractor import FeaturesExtractor

warnings.filterwarnings("ignore")

class ModelsTrainer:

    def __init__(self, females_files_path, males_files_path):
        self.females_training_path = females_files_path
        self.males_training_path   = males_files_path
        self.features_extractor    = FeaturesExtractor()

    def process(self):
        females, males = self.get_file_paths(self.females_training_path,
                                             self.males_training_path)
        
        # Collect voice features
        print(f"Extracting features for {len(females)} female files and {len(males)} male files...")
        female_voice_features = self.collect_features(females)
        male_voice_features   = self.collect_features(males)
        
        # --- NEW TRAINING LOGIC WITH PLOTTING ---
        
        # Train and capture history
        print("Training Female GMM...")
        females_gmm, female_history = self.train_with_history(female_voice_features, "Female")
        
        print("Training Male GMM...")
        males_gmm, male_history = self.train_with_history(male_voice_features, "Male")
        
        # Plot the training curves
        self.plot_training_history(female_history, male_history)
        
        # Save models
        self.save_gmm(females_gmm, "females")
        self.save_gmm(males_gmm,   "males")

    def train_with_history(self, features, label_name, max_epochs=240):
        """
        Trains a GMM iteratively to capture the log-likelihood history for plotting.
        """
        # Note: warm_start=True is required to keep training state between iterations.
        # n_init must be 1 when using warm_start loop manually.
        gmm = GaussianMixture(
            n_components=16, 
            max_iter=1,             # We iterate manually
            covariance_type='diag', 
            n_init=1, 
            warm_start=True,        # Keep weights from previous iteration
            verbose=0
        )
        
        history = []
        
        # Manual Training Loop
        for epoch in range(max_epochs):
            gmm.fit(features)
            
            # lower_bound_ is the log-likelihood of the best fit of EM
            history.append(gmm.lower_bound_)
            
            # Optional: Print progress every 20 epochs
            if epoch % 20 == 0:
                print(f"  [{label_name}] Epoch {epoch}: Log-Likelihood = {gmm.lower_bound_:.4f}")
            
            # Check for convergence manually (stop if change is tiny)
            if epoch > 1 and abs(history[-1] - history[-2]) < gmm.tol:
                print(f"  [{label_name}] Converged at epoch {epoch}")
                break
                
        return gmm, history

    def plot_training_history(self, female_hist, male_hist):
        """
        Plots the Log-Likelihood over iterations.
        """
        plt.figure(figsize=(12, 5))

        # Plot Female
        plt.subplot(1, 2, 1)
        plt.plot(female_hist, color='red', label='Log-Likelihood')
        plt.title('Female Model Training')
        plt.xlabel('Iterations')
        plt.ylabel('Log-Likelihood (Higher is better)')
        plt.grid(True)
        plt.legend()

        # Plot Male
        plt.subplot(1, 2, 2)
        plt.plot(male_hist, color='blue', label='Log-Likelihood')
        plt.title('Male Model Training')
        plt.xlabel('Iterations')
        plt.ylabel('Log-Likelihood')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        # merge 2 plots into one
        plt.figure(figsize=(8, 6))
        plt.plot(female_hist, color='red', label='Female Model')
        plt.plot(male_hist, color='blue', label='Male Model')
        plt.title('GMM Training Log-Likelihood')
        plt.xlabel('Iterations')
        plt.ylabel('Log-Likelihood')
        plt.grid(True)
        plt.legend()
        plt.show()
        

    def get_file_paths(self, females_training_path, males_training_path):
        females = [ os.path.join(females_training_path, f) for f in os.listdir(females_training_path) if f.endswith('.wav') ]
        males   = [ os.path.join(males_training_path, f) for f in os.listdir(males_training_path) if f.endswith('.wav') ]
        return females, males

    def collect_features(self, files):
        """
        Collect voice features from various speakers of the same gender.
        """
        features_list = []
        
        for file in files:
            print("%5s %10s" % ("PROCESSING", file))
            try:
                vector = self.features_extractor.extract_features(file)
                if vector is not None and vector.size > 0:
                    features_list.append(vector)
            except Exception as e:
                print(f"Error processing {file}: {e}")

        if features_list:
            features = np.vstack(features_list)
            return features
        else:
            return np.array([])

    def save_gmm(self, gmm, name):
        """ Save Gaussian mixture model using pickle. """
        filename = name + ".gmm"
        with open(filename, 'wb') as gmm_file:
            pickle.dump(gmm, gmm_file)
        print ("%5s %10s" % ("SAVED", filename))


if __name__== "__main__":
    if os.path.exists("TrainingData/females") and os.path.exists("TrainingData/males"):
        models_trainer = ModelsTrainer("TrainingData/females", "TrainingData/males")
        models_trainer.process()
    else:
        print("Error: Training Data directories not found.")