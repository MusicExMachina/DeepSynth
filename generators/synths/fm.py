import math
import pyaudio
import numpy as np
import scipy.signal as ss
from math import pi, sin, floor
from fractions import gcd
import matplotlib.pyplot as plt


###IMPORT FROM GLOBAL?
SAMPLE_RATE=44100

class Operator():
    def __init__(self,freq,amp,func,dependents,samplerate=SAMPLE_RATE):
        """
            The Operator Class is a building block of the feed forward graph that the FM synth is built on
            
            freq:  automation of frequency [float list]
            amp:  automation of amplitude [float list]
            func:  harmonic function of time representing wave osc. type [func/lambda]
            dependents:  list of operators [list of Operators]
            samplerate: integer as how many samples per second
        """
        #Check freq/amp lists are the same
        if freq.shape[0]!=amp.shape[0]:
            print("Invalid Frequency/Amplitude Lists! Cannot create Operator.")
            return
        
        self.freq=freq
        self.amp=amp
        self.func=func
        self.dependents=dependents
        self.samplerate=SAMPLE_RATE
    
    def calc_output(self):
        """
            Calculate the output of the current Operator
                -Calculate the output of the dependent Operators
                -Modulate the frequency of the current Operator
                -Run phase-index oscillator calculation, and apply harmonic function and amplitude
        """
        #copy init
        mod_freq = np.copy(self.freq).astype('float64')
        
        #CALCULATE DEPENDANT OPERATORS
        for dep_operator in self.dependents:
            dou = dep_operator.calc_output()
            mod_freq *= 1.+dou
        
        #INIT OUTPUT
        phase_index=0.
        out = np.zeros(len(mod_freq))
        
        #CALC OUTPUT
        for i in range(len(out)):
            phase_delta = 2.*np.pi*(mod_freq[i])/self.samplerate
            phase_index += phase_delta
            out[i] = self.amp[i]*self.func(phase_index)
        
        return out
    
class FMSynth():
    def __init__(self,operators,out):
        """
            The FMSynth class builds a Frequency Modulator Synth
            
            operators: list of all operators involved
            out: list of all operators that modulate output wave.
        """
        self.operators=operators
        self.out=out
        
    def run(self):
        """
        Run FMSynth by calculating output of operators in out
        """
        frames = 1.
        for i in self.out:
            frames *= self.operators[i].calc_output()
        return frames