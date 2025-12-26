import os
import pickle
import warnings
import numpy as np
from hmmCode.FeaturesExtractor import FeaturesExtractor
from hmmlearn import hmm


import pydub
import librosa
import soundfile as sf
import numpy as np
import speech_recognition as sr
from pydub import AudioSegment
from pydub.silence import split_on_silence, detect_nonsilent

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
            # 1. Load the audio file
            # sr=None preserves the original sampling rate
            y, sr = librosa.load(input_path, sr=None)

            # 2. Trim silence
            # top_db=36 matches your previous ffmpeg -36dB threshold
            y_trimmed, index = librosa.effects.trim(y, top_db=36)

            # 3. Save the processed file to disk
            # We save as PCM_16 to ensure compatibility with your FeaturesExtractor
            sf.write(output_path, y_trimmed, sr, subtype='PCM_16')
            
            # Optional: Print info to match your old logs
            duration = librosa.get_duration(y=y_trimmed, sr=sr)
            print(f"Processed: {duration:.2f}s kept")
            
            # Return data (though your main loop ignores this return value)
            return y_trimmed, duration

        except Exception as e:
            print(f"Error processing {input_path}: {e}")
            return None, 0
    





    def process(self):
        females, males = self.get_file_paths(self.females_training_path,
                                             self.males_training_path)
        # collect voice features
        female_voice_features = self.collect_features(females)
        male_voice_features   = self.collect_features(males)
        # generate gaussian mixture models
        females_gmm = hmm.GaussianHMM(n_components=3)
        males_gmm   = hmm.GaussianHMM(n_components=3)
        ubm         = hmm.GaussianHMM(n_components=3)
        # fit features to models
        females_gmm.fit(female_voice_features)
        males_gmm.fit(male_voice_features)
        ubm.fit(np.vstack((female_voice_features, male_voice_features)))
        # save models
        self.save_gmm(females_gmm, "females")
        self.save_gmm(males_gmm,   "males")
        self.save_gmm(ubm, "ubm")          # <--- Pass the 'ubm' variable


    def get_file_paths(self, females_training_path, males_training_path):
        # get file paths
        females = [ os.path.join(females_training_path, f) for f in os.listdir(females_training_path) ]
        males   = [ os.path.join(males_training_path, f) for f in os.listdir(males_training_path) ]
        return females, males

    def collect_features(self, files):
        """
    	Collect voice features from various speakers of the same gender.

    	Args:
    	    files (list) : List of voice file paths.

    	Returns:
    	    (array) : Extracted features matrix.
    	"""
        features = np.asarray(())
        # extract features for each speaker
        for file in files:
            print("%5s %10s" % ("PROCESSNG ", file))
            self.ffmpeg_silence_eliminator(file, file.split('.')[0] + "_without_silence.wav")
        
            # extract MFCC & delta MFCC features from audio
            try: 
                vector    = self.features_extractor.extract_features(file.split('.')[0] + "_without_silence.wav")
                # stack the features
                if features.size == 0:  features = vector
                else:                   features = np.vstack((features, vector))           
            except : pass
            os.remove(file.split('.')[0] + "_without_silence.wav")
        return features

    def save_gmm(self, gmm, name):
        """ Save Gaussian mixture model using pickle.

            Args:
                gmm        : Gaussian mixture model.
                name (str) : File name.
        """
        filename = name + ".hmm"
        with open(filename, 'wb') as gmm_file:
            pickle.dump(gmm, gmm_file)
        print ("%5s %10s" % ("SAVING", filename,))


if __name__== "__main__":
    models_trainer = ModelsTrainer("TrainingData/females", "TrainingData/males")
    models_trainer.process()
