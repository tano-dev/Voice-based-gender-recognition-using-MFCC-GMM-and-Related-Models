import os
import pickle
import warnings
import numpy as np
from Code.FeaturesExtractor import FeaturesExtractor

warnings.filterwarnings("ignore")

class GenderIdentifier:

    def __init__(self, females_files_path, males_files_path, females_model_path, males_model_path):
        self.females_training_path = females_files_path
        self.males_training_path   = males_files_path
        self.error                 = 0
        self.total_sample          = 0
        self.features_extractor    = FeaturesExtractor()
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

            winner = self.identify_gender(vector)
            
            # --- FIX STARTS HERE ---
            # 1. Get the parent directory name (e.g., "females" or "males")
            parent_folder = os.path.basename(os.path.dirname(file))
            
            # 2. Determine expectation based on the folder name
            if "female" in parent_folder.lower():
                expected_gender = "female"
            else:
                expected_gender = "male"
            # --- FIX ENDS HERE ---

            print("%10s %6s %1s" % ("+ EXPECTATION",":", expected_gender))
            print("%10s %3s %1s" %  ("+ IDENTIFICATION", ":", winner))

            if winner != expected_gender: 
                self.error += 1
                print(f"    [MISMATCH] Expected: {expected_gender}, Got: {winner}")
            print("----------------------------------------------------")

        if self.total_sample > 0:
            accuracy     = ( float(self.total_sample - self.error) / float(self.total_sample) ) * 100
            accuracy_msg = "*** Accuracy = " + str(round(accuracy, 3)) + "% ***"
            print(accuracy_msg)
        else:
            print("No samples processed.")

    def get_file_paths(self, females_training_path, males_training_path):
        # Get file paths and filter for .wav files only
        females = [ os.path.join(females_training_path, f) for f in os.listdir(females_training_path) if f.endswith('.wav') ]
        males   = [ os.path.join(males_training_path, f) for f in os.listdir(males_training_path) if f.endswith('.wav') ]
        files   = females + males
        return files

    def identify_gender(self, vector):
        # score() returns the average log-likelihood of the samples
        # female hypothesis scoring
        is_female_scores = self.females_gmm.score(vector)
        
        # male hypothesis scoring
        is_male_scores = self.males_gmm.score(vector)

        print("%10s %5s %1s" % ("+ FEMALE SCORE",":", str(round(is_female_scores, 3))))
        print("%10s %7s %1s" % ("+ MALE SCORE", ":", str(round(is_male_scores, 3))))

        if is_male_scores > is_female_scores: 
            winner = "male"
        else:                                 
            winner = "female"
        return winner

if __name__== "__main__":
    # Ensure paths exist before running
    if os.path.exists("TestingData/females"):
        gender_identifier = GenderIdentifier("TestingData/females", "TestingData/males", "females.gmm", "males.gmm")
        gender_identifier.process()
    else:
        print("Error: Testing Data not found.")