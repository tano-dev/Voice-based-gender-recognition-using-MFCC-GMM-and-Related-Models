import os
import pickle
import warnings
import numpy as np
from sklearn.mixture import GaussianMixture
from Code.FeaturesExtractor import FeaturesExtractor

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
        
        # Generate Gaussian Mixture Models
        # FIX: Changed 'n_iter' to 'max_iter' for modern sklearn compatibility
        females_gmm = GaussianMixture(n_components=16, max_iter=200, covariance_type='diag', n_init=3)
        males_gmm   = GaussianMixture(n_components=16, max_iter=200, covariance_type='diag', n_init=3)
        
        # Fit features to models
        print("Training Female GMM...")
        females_gmm.fit(female_voice_features)
        
        print("Training Male GMM...")
        males_gmm.fit(male_voice_features)
        
        # Save models
        self.save_gmm(females_gmm, "females")
        self.save_gmm(males_gmm,   "males")

    def get_file_paths(self, females_training_path, males_training_path):
        females = [ os.path.join(females_training_path, f) for f in os.listdir(females_training_path) if f.endswith('.wav') ]
        males   = [ os.path.join(males_training_path, f) for f in os.listdir(males_training_path) if f.endswith('.wav') ]
        return females, males

    def collect_features(self, files):
        """
        Collect voice features from various speakers of the same gender.
        """
        # Using an empty list is often more efficient than repeated np.vstack
        features_list = []
        
        for file in files:
            print("%5s %10s" % ("PROCESSING", file))
            try:
                # Extract MFCC & delta MFCC features from audio
                vector = self.features_extractor.extract_features(file)
                # Only add valid vectors (check if vector is not empty or None)
                if vector is not None and vector.size > 0:
                    features_list.append(vector)
            except Exception as e:
                print(f"Error processing {file}: {e}")

        # Stack all features at once
        if features_list:
            features = np.vstack(features_list)
            return features
        else:
            return np.array([])

    def save_gmm(self, gmm, name):
        """ Save Gaussian mixture model using pickle. """
        filename = name + ".gmm"
        # Ensure directory exists if needed, or save to specific model folder
        with open(filename, 'wb') as gmm_file:
            pickle.dump(gmm, gmm_file)
        print ("%5s %10s" % ("SAVED", filename))


if __name__== "__main__":
    # Ensure these folders exist before running
    if os.path.exists("TrainingData/females") and os.path.exists("TrainingData/males"):
        models_trainer = ModelsTrainer("TrainingData/females", "TrainingData/males")
        models_trainer.process()
    else:
        print("Error: Training Data directories not found.")