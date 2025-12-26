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

            # 3. Print info (Replacing your os.popen/sed logic)
            print("%-32s %-7s %-50s" % ("ORIGINAL SAMPLE DURATION", ":", float(orig_duration)))
            print("%-23s %-7s %-50s" % ("SILENCE FILTERED SAMPLE DURATION", ":", float(new_duration)))

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

                print("%10s %6s %1s" % ("+ EXPECTATION",":", expected_gender))
                print("%10s %3s %1s" %  ("+ IDENTIFICATION", ":", winner))

                if winner != expected_gender: self.error += 1
                print("----------------------------------------------------")

    
            except : pass
            os.remove(file.split('.')[0] + "_without_silence.wav")
            
        accuracy     = ( float(self.total_sample - self.error) / float(self.total_sample) ) * 100
        accuracy_msg = "*** Accuracy = " + str(round(accuracy, 3)) + "% ***"
        print(accuracy_msg)  
        


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

        print("%10s %5s %1s" % ("+ FEMALE SCORE",":", str(round(is_female_log_likelihood, 3))))
        print("%10s %7s %1s" % ("+ MALE SCORE", ":", str(round(is_male_log_likelihood,3))))

        if is_male_log_likelihood > is_female_log_likelihood: winner = "male"
        else                                                : winner = "female"
        return winner


if __name__== "__main__":
    gender_identifier = GenderIdentifier("TestingData/females", "TestingData/males", "females.hmm", "males.hmm")
    gender_identifier.process()
