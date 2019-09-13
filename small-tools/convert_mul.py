from features import LogMelExtractor
import glob
import os
import numpy as np
import h5py
from multiprocessing.pool import ThreadPool
from utilities import pad_or_trunc
import librosa
import joblib
from functools import partial
feature_extractor = LogMelExtractor(sample_rate=32000,
                                        window_size=2048,
                                        overlap=720,
                                        mel_bins=64)

def covfea(audio):
    try:
        y, sr = librosa.core.load(audio, sr=32000)
        duration = librosa.get_duration(y=y, sr=sr)
        if duration < 11:
            return None
        else:
            data = y[0:320000]
            data /= np.max(np.abs(data))
            feature = feature_extractor.transform(data)
            feature = pad_or_trunc(feature, 240)
            return feature

    except KeyboardInterrupt:
        exit(0)
    except Exception as e:
        print(e)
        pass


if __name__ == '__main__':
    mp = "/mnt/xutan/ry/hjz/classify/pyxctools/sounds"
    hf = h5py.File("/mnt/xutan/ry/hjz/classify/pub_dcase2018_task3/features/logmel/pre4.h5", 'w')
    hf.create_dataset(
        name='feature',
        shape=(0, 240, 64),
        maxshape=(None, 240, 64),
        dtype=np.float32)
    all_audio = glob.glob(os.path.join(mp, "*.mp3"))
    non = 0
    ind = 0
    for i in range(60):
        jobs = [joblib.delayed(covfea)(audio) for audio in all_audio[0+i*1000:1000+i*1000]]
        out = joblib.Parallel(n_jobs=200,verbose = 1)(jobs)

        for f in out:
            if f is None:
               non += 1
            else:
                hf['feature'].resize((ind + 1, 240, 64))
                hf['feature'][ind] = f
                ind += 1
        print(i)
        print("------")
        print(non)
        print("------")
        print(ind)
    hf.close()