import numpy as np
import os
from pathlib import Path 
import soundfile as sf


def main():
    ## create a silent wav 
    # 16000 samples per second
    length = 1
    silent_wav = np.zeros(16000 * length)
    silent_wav = silent_wav.astype(np.int16)
    

    # save the silent wav file
    folder = Path("/ps/project/EmotionalFacialAnimation/emote")
    sf.write(folder / "silent.wav", silent_wav, 16000, subtype='PCM_16')
    


if __name__ == "__main__": 
    main()
