import os
import pickle
import warnings
import numpy as np
from svmCode.FeaturesExtractor import FeaturesExtractor
from hmmlearn import hmm
import subprocess
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

class GenderIdentifier:

    def __init__(self, females_files_path, males_files_path, females_model_path, males_model_path):
        self.females_training_path = females_files_path
        self.males_training_path   = males_files_path
        self.error                 = 0
        self.total_sample          = 0
        self.features_extractor    = FeaturesExtractor()
        
        # Load features (Note: these are numpy arrays of stacked means, not actual GMM objects)
        self.females_gmm = pickle.load(open(females_model_path, 'rb'))
        self.males_gmm   = pickle.load(open(males_model_path, 'rb'))
        
        # Stack features for SVM training
        self.X_train = np.vstack((self.females_gmm, self.males_gmm))
        self.y_train = np.hstack(( -1 * np.ones(self.females_gmm.shape[0]), np.ones(self.males_gmm.shape[0])))
        
        # --- FIX 1: REMOVE NaNs BEFORE TRAINING ---
        # Create a mask to identify rows that do NOT have NaN values
        mask = ~np.isnan(self.X_train).any(axis=1)
        
        # Keep only the valid rows
        self.X_train = self.X_train[mask]
        self.y_train = self.y_train[mask]
        
        print(f"Training SVM with {len(self.X_train)} clean samples (removed {len(mask) - mask.sum()} NaN rows)...")
        # ------------------------------------------

        self.clf = SVC(kernel = 'rbf', probability=True)
        self.clf.fit(self.X_train, self.y_train)
        

    def process(self):
        files = self.get_file_paths(self.females_training_path, self.males_training_path)
        
        for file in files:
            self.total_sample += 1
            print("%10s %8s %1s" % ("--> TESTING", ":", os.path.basename(file)))

            try: 
                # extract MFCC & delta MFCC features from audio
                vector = self.features_extractor.extract_features(file)
                
                spk_gmm = hmm.GaussianHMM(n_components=2)      
                spk_gmm.fit(vector)
                
                self.spk_vec = spk_gmm.means_
                
                # Predict gender
                prediction_result = self.clf.predict(self.spk_vec)
                
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
                elif "male" in file.lower():
                    expected_gender = "male"
                else:
                    expected_gender = "unknown"
                # ---------------------------------------------------------------------
                
                print("%10s %6s %1s" % ("+ EXPECTATION",":", expected_gender))
                print("%10s %3s %1s" %  ("+ IDENTIFICATION", ":", winner))

                if winner != expected_gender: 
                    self.error += 1
                print("----------------------------------------------------")

            except Exception as e:
                print(f"Error processing {os.path.basename(file)}: {e}")
            
            
        if self.total_sample > 0:
            accuracy     = ( float(self.total_sample - self.error) / float(self.total_sample) ) * 100
            accuracy_msg = "*** Accuracy = " + str(round(accuracy, 3)) + "% ***"
            print(accuracy_msg)
            self.process_plot()
        else:
            print("No samples processed.")


    def get_file_paths(self, females_training_path, males_training_path):
        females = [ os.path.join(females_training_path, f) for f in os.listdir(females_training_path) ]
        males   = [ os.path.join(males_training_path, f) for f in os.listdir(males_training_path) ]
        files   = females + males
        return files
    def process_plot(self):
        import matplotlib.pyplot as plt
        
        # Plot training data 
        if self.total_sample > 0:
            accuracy     = ( float(self.total_sample - self.error) / float(self.total_sample) ) * 100
            accuracy_msg = "*** Accuracy = " + str(round(accuracy, 3)) + "% ***"
            print(accuracy_msg)
            plt.figure(figsize=(6,4))
            plt.bar(['Correct', 'Incorrect'], [self.total_sample - self.error, self.error], color=['green', 'red'])
            plt.title('Gender Identification Accuracy: ' + str(round(accuracy, 2)) + '%')
            plt.xlabel('Outcome')
            plt.ylabel('Number of Samples')
            plt.show()
            
if __name__== "__main__":
    # Ensure these paths point to your actual .svm (pickle) files
    gender_identifier = GenderIdentifier("TestingData/females", "TestingData/males", "svmCode/females.svm", "svmCode/males.svm")
    gender_identifier.process()