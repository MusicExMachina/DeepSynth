import math
import pyaudio
import numpy as np
import scipy.signal as ss
from math import pi, sin, floor
from fractions import gcd
import matplotlib.pyplot as plt
import functools


###IMPORT FROM GLOBAL?
SAMPLE_RATE=44100


def two_var_synth(freq_list, amp_list, samplerate=SAMPLE_RATE):
    assert len(freq_list) == len(amp_list)
    wav_len = len(freq_list)
    frames = np.zeros(len(freq_list))
    phase_index=0
    
    for i in range(wav_len):
        phase_delta = 2.*np.pi*freq_list[i]/samplerate
        phase_index += phase_delta
        frames[i] = amp_list[i]*np.sin(phase_index)
    
    return frames

class Oscillator:
    def __init__(self,freq,amp,func,samplerate=SAMPLE_RATE):
        """
            The Oscillator Class is a building block of the SubtractiveSynth class
            
            freq:  automation of frequency [float list]
            amp:  automation of amplitude [float list]
            func:  harmonic function of time representing wave osc. type [func/lambda]
            samplerate: integer as how many samples per second
        """
        if len(freq_list) != len(amp_list):
            print("Invalid Frequency/Amplitude Lists! Cannot create Oscillator.")
            return
        self.freq = freq
        self.amp = amp
        self.func = func
        self.samplerate=samplerate
        
    def run(self):
        wav_len = len(self.freq)
        
        #init
        frames = np.zeros(wav_len)
        phase_index=0
        
        #calculate frame value by phase index method
        for i in range(wav_len):
            phase_delta = 2.*np.pi*self.freq[i]/self.samplerate
            phase_index += phase_delta
            frames[i] = self.amp[i]*np.sin(phase_index)
    
    return frames




class SubtractiveSynth:
    def __init__(self,oscillators,comb_op):
        """
            The SubtractiveSynth class builds a Subtractive Synth
            
            oscillators: list of oscillators [oscillators]
            comb_op: combination operation [lambda/func ( x , y )]
        """
        self.oscillators=oscillators
        self.comb_op=comb_op
        
    def run(self):
        """
        Run SubtractiveSynth by comibining values of all oscillators
        """
        all_frames = []
        for osc in self.oscillators:
            all_frames.append(osc.run())
        
        wav_lens = map(len,all_frames)
        if not all(x==wav_lens[0] for x in wav_lens):
            print("All Oscillators must have the same number of frames")
            return
        
        all_frames_arr = np.stack(all_frames)
        
        final_frames = np.zeros(all_frames_arr.shape[1])
        
        for i in range(all_frames_arr.shape[0]):
            final_frames = self.comb_op(all_frames_arr[i],final_frames)
            
        return final_frames
        