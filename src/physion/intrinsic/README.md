# Intrinsic Imaging

## Data acquisition and stimulation

- Strategy of [Kalatsky and Stryker, Neuron (2003)](https://doi.org/10.1016/s0896-6273(03)00286-1): using periodic stimulation (repetitions) and Fourier analysis to catch weak evoked responses.
- A flickering moving bar (N.B. straight on the screen, unlike Juavinett et al., 2016 or Zhuang et al., 2017)
- The protocol keeps on adding data. Stop whenever you are happy with the obtained maps.

## Retinotopic maps

Building retinotopic maps using the strategy . From a recording `movie` that contains `Nrepeat` repetitions of the flickering bar in one irection:

```
spectrum = np.fft.fft(movie, axis=0)

# generate power map
power_map = np.abs(spectrum)[Nrepeat,:,:]

#generate phase movie
phase_map = -np.angle(spectrum)[Nrepeat,:,:] % (2.*np.pi)
```

## Segmentation of Visual Areas

The code comes from from [a script from Jun Zhuang](https://github.com/zhuangjun1981/NeuroAnalysisTools/blob/master/NeuroAnalysisTools/RetinotopicMapping.py).

## Material and Methods




