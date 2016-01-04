#! /usr/bin/python

import numpy as np
from scipy.fftpack import hilbert
import mdp
from scipy.signal import buttord, butter, filtfilt
from scipy.fftpack import hilbert, rfft, irfft
from pylab import detrend_linear			# for bandpass

pi2 = 2.*np.pi
oscillations_per_window = 20



### FFT Filter things

class butter(object):

	def __init__(self, passband, stopband=None, sampling_rate=1, pass_attenuation=3., stop_attenuation=40., verbose=False):
		# passband = [w_low, w_high], w_low = 0. -> lowpass, w_high = np.infty -> highpass

		self.verbose = verbose
		self.filterCoefs(passband, stopband, sampling_rate, pass_attenuation, stop_attenuation)




	def __call__(self, signal):
		return sig.filtfilt(self.b, self.a, signal)



	def filterCoefs(self, passband, stopband, sampling_rate, pass_attenuation, stop_attenuation):		# determines self.b, and self.a
		
		self.sampling_rate = sampling_rate
		self.passband = np.asarray(passband, dtype=float)

		if stopband == None:	self.stopband = np.array([0.5*passband[0], 2.*passband[1]])
		else:			self.stopband = np.asarray(stopband, dtype=float)

		pb, sb = 2.*self.passband/float(self.sampling_rate), 2.*self.stopband/float(self.sampling_rate)

		if pb[0] == 0.:	# lowpass mode
			pb, sb = pb[1], sb[1]
			self.mode = 'low'		# lowpass

		elif pb[1] == np.infty:	# highpass mode
			pb, sb = pb[0], sb[0]
			self.mode = 'high'		# highpass

		else:
			self.mode = 'band'		# bandpass


		if self.verbose:
			print "# filtersetting:", self.mode
			print "# band:", self.passband

		N, Wn = sig.buttord(pb, sb, pass_attenuation, stop_attenuation)
		self.b, self.a = sig.butter(N, Wn, btype=self.mode)



def bandpass(x, sampling_rate, f_min, f_max, verbose=0):
	"""
	xf = bandpass(x, sampling_rate, f_min, f_max)

	Description
	--------------

	Phasen-treue mit der rueckwaerts-vorwaerts methode!
	Function bandpass-filters a signal without roleoff.  The cutoff frequencies,
	f_min and f_max, are sharp.

	Arguements
	--------------
		x: 			input timeseries
		sampling_rate: 			equidistant sampling with sampling frequency sampling_rate
		f_min, f_max:			filter constants for lower and higher frequency
	
	Returns
	--------------
		xf:		the filtered signal
	"""
	x, N = np.asarray(x, dtype=float), len(x)
	t = np.arange(N)/np.float(sampling_rate)
	xn = detrend_linear(x)
	del t

	yn = np.concatenate((xn[::-1], xn))		# backwards forwards array
	f = np.float(sampling_rate)*np.asarray(np.arange(2*N)/2, dtype=int)/float(2*N)
	s = rfft(yn)*(f>f_min)*(f<f_max)			# filtering

	yf = irfft(s)					# backtransformation
	xf = (yf[:N][::-1]+yf[N:])/2.			# phase average

	return xf



def tukey_window(x, **kwargs):
	return tukey(x.size, **kwargs)*x



def tukey(N, alpha=0.1, n=None):
	"""
	Creates Tukey window function
	alpha=0.1	: Fraction of N used for tappering.
	n		: Fixed number of datapoints used for tappering.
	"""
	if n == None:	# default to alpha
		alpha = float(alpha)
		N0, N1 = int(alpha*(N-1)/2), int((N-1)*(1.-alpha/2.))
	else:
		N0, N1 = n, N-n

	win = np.ones((N), float)
	n0, n1 = np.arange(N0), np.arange(N1, N)
	win[:N0] *= 0.5*(1.+np.cos(np.pi*(n0/float(N0)-1.)))
	win[N1:] *= 0.5*(1.+np.cos(np.pi*(n1/float(N0)-2./alpha+1.)))
	return win



### 2 Pi- Unmodding of phase traces


def unmod(phase):
	phase = np.asarray(phase)
	difference = phase[1:]-phase[:-1]	# prepare for fast computation
	plus_one, minus_one = np.zeros((phase.size), int), np.zeros((phase.size), int)

	plus_one[1:] = np.asarray(difference < -np.pi, dtype=int)
	minus_one[1:] = np.asarray(difference > np.pi, dtype=int)

	return phase + pi2 * cumsum(plus_one - minus_one)


### Overlapping of two traces.


def overlap(begin, end, n):
	
	begin, end = np.asarray(begin, dtype=float), np.asarray(end, dtype=float)
	assert begin.size > n and end.size > n

	ret = np.zeros((begin.size+end.size-n), float)

	ret[:begin.size-n] = begin[:-n]
	ret[begin.size:] = end[n:]

	window = 0.5*(1.+np.cos(np.pi*np.arange(1, n+1, 1)/float(n+1)))
	ret[begin.size-n:begin.size] = window*begin[-n:]+window[::-1]*end[:n]

	return ret




def append(signal, segment, oldsize, OVERLAP):

	segment = np.asarray(segment, dtype=float)				# check that array operations posible
	if not segment.size > OVERLAP:	OVERLAP = segment.size

	newsize = oldsize-OVERLAP + segment.size			# compute next end of the series
	assert signal.size >= newsize

	if oldsize == 0:
		signal[:segment.size] = segment
		return segment.size

	window = 0.5*(1.+np.cos(np.pi*np.arange(1, OVERLAP+1, 1)/float(OVERLAP+1)))

	signal[oldsize-OVERLAP:oldsize] = window*signal[oldsize-OVERLAP:oldsize] + window[::-1]*segment[:OVERLAP]
	signal[oldsize:newsize] = segment[OVERLAP:]

	return newsize



def signalAndNoise(x, filt):
	xf = filt(x)
	signalVariance = np.var(xf)
	noiseVariance =  np.var( tukey_window(x-xf, alpha=0.25) )
	return signalVariance, noiseVariance



SMALL = 10**-8
LARGE = 1./SMALL
def normalize(X, Filter, index=None):

	sqSNR = np.sqrt( np.array([ signalAndNoise(X[:, i], Filter) for i in xrange(X.shape[1]) ]) )	# the S and R amplitudes
	stds = sqSNR[:, 1]

	if not index == None:
		reference_amplitude = sqSNR[index, 0] # sqrt( A2(channel[index]) )
		if stds[index] < SMALL:	raise ValueError
		stds = stds/stds[index]
	
	else:
		reference_amplitude = 1.

	for i in xrange(stds.size):
		if stds[i] < SMALL:		# this component is defunct
			stds[i] = LARGE		# ... remove component from analysis
			print "component %i removed from analysis." % (i)


	return X/stds, reference_amplitude



from pylab import plot, show
def _Kosambi_Hilbert_torsion(X, Filter, index=0):
	# X[time, channel] should be X_j(t_i)

	if X.shape[1] == 1: return X[:, 0]

	# declarations and initializations
	X = np.asarray(X, dtype=float)
	channels = range(X.shape[1])
	Y = np.zeros((X.shape[0], 2*X.shape[1]+1), float)			# Y[time,  channel], X, and H(X) with Y[:, 0] as the reference channel.
	Yf = np.zeros(Y.shape, float)						# Filter(ed) version of Y.


	X, reference_amplitude = normalize(X, Filter, index=index)
	Y[:, 0] = X[:, index]							# save the reference channel to vector zero
	channels.pop(index)							# reference channel is treated separately.


	for (c, channel) in enumerate(channels):
		Y[:, 1+2*c] = X[:, channel]
		Y[:, 1+2*c+1] = hilbert(Y[:, channel])


	for i in xrange(Y.shape[1]):
		Yf[:, i] = Filter(Y[:, i])


	pcanode = mdp.nodes.PCANode(svd=True)	# this pcanode is used by the function below. (it's actually some static variable)
	pcanode.execute(Yf)			# get the principle components from Yf
	Proj = pcanode.get_projmatrix()		# ...and their projection matrix.

	if Proj[0, 0] < 0: Proj = -Proj		# ... why do I need to do this?


	KHT_component = np.dot(Y, Proj)[:, 0]		# apply them to Y!!!
	pca_amplitude = np.sqrt(signalAndNoise(KHT_component, Filter)[0])

	return reference_amplitude/pca_amplitude * KHT_component 



def Kosambi_Hilbert_torsion(X, sampling_rate, passband, stopband=None, index=0, moving_window=False, **kwargs):

	if stopband == None:
		def Filter(x):
			return bandpass(x, sampling_rate, passband[0], passband[1])
	else:
		Filter = butter(passband=passband, stopband=stopband, sampling_rate=sampling_rate)


	if moving_window == False:
		avg_signal = _Kosambi_Hilbert_torsion(X=X, Filter=Filter, index=index)


	else:
		if kwargs.has_key('center_frequency'):	expected_period = 1./kwargs['center_frequency']
		else:					expected_period = 2./(passband[0]+passband[1])

		WINDOW = int(oscillations_per_window * expected_period*sampling_rate)	# datapoints cointaining approx. 20 oscillations
		OVERLAP = WINDOW/2							# should be half the window size
		STEP = WINDOW-OVERLAP							# step of the moving window

		STEPS = int((X.shape[0]-WINDOW)/STEP)	# number of steps
		print 'STEPS', STEPS

		if STEPS < 1: return _Kosambi_Hilbert_torsion(X=X, Filter=Filter, index=index)	# whole signal

		avg_signal = np.zeros((X.shape[0]), float)
		
		oldsize = 0
		for s in xrange(0, X.shape[0]-WINDOW, STEP):
			sig_s = _Kosambi_Hilbert_torsion(X=X[s:s+WINDOW], Filter=Filter, index=index)
			oldsize = overlap.append(avg_signal, sig_s, oldsize, OVERLAP)

		sig_final = _Kosambi_Hilbert_torsion(X=X[s+STEP:], Filter=Filter, index=index)
		oldsize = overlap.append(avg_signal, sig_final, oldsize, OVERLAP)
		
	return avg_signal



def Kosambi_Hilbert_phase(X, sampling_rate, passband=None, index=0, moving_window=False):

	y = Kosambi_Hilbert_torsion(X, sampling_rate, passband=passband, index=index, moving_window=moving_window)
	Hy = hilbert(y)

	radius = np.sqrt(y**2+Hy**2)
	phase = np.arctan2(y, Hy)
	phi_u = unmod(phase)

	if phi_u[-1]-phi_u[0] < 0.:	# if phase shrinks, reverse it.
		phase = -phase

	phase = np.mod(phase, pi2)

	return phase, radius



_kht = _Kosambi_Hilbert_torsion	# Shorter name for the function.
kht = Kosambi_Hilbert_torsion		# Shorter name for the function.
khp = Kosambi_Hilbert_phase		# Shorter name for the function.



if __name__ == '__main__':
	import pylab as pl

	sampling_rate = 200
	SAMPLES = 5*sampling_rate	# 
	t = np.arange(SAMPLES)/float(sampling_rate)
	f = 10. # Hz
	fA = f/15.
	w = pi2*f	# T = 1 sec.
	wA = pi2*fA	# T = 1 sec.
	noise =	pl.randn(SAMPLES)

	# This is the global phase dynamics
	phase = t*w + 0.5/np.sqrt(sampling_rate) * np.cumsum(noise)

	# These are the channels
	CHANNELS = 100
	X = np.zeros((SAMPLES, CHANNELS), dtype=float)
	for c in xrange(CHANNELS):
		A = (1.+pl.sin(wA * t + pi2 * pl.rand()))/2.
		X[:, c] = A * pl.sin(phase + pi2 * pl.rand()) + 0.4 * pl.randn(SAMPLES)

	pl.subplot(111)
	for c in xrange(10):
		pl.plot(t, X[:, c] - c*2.5, 'k-', lw=1.)
	
	y = kht(X, sampling_rate=sampling_rate, passband=[f-2., f+2], moving_window=False)

	pl.plot(t, y- (c+1) * 2.5, 'r-', lw=1.)
	
	pl.tight_layout()
	pl.show()
