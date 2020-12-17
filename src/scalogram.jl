module scalogram
    using PyCall

```
Function that uses Pythons' PyWavelets package to compute the scalogram (or spectrogram) of a phase measurement.

Inputs:
        phi::Vector{Float64}  -- the vector of measured phases
        sweepRange::Float64   -- the frequency sweep range of the reflectometer (GHz)
        sweepTime::Float64    -- the frequency sweep time of the reflectometer (seconds)
        df::Float64           -- the Δf between each frequency measurement (GHz)
        
Optional inputs:
        cmor::Int64           -- parameter controlling the width of the wavelet. Default is 5.


Output:
        Wxx   -- the 2-D array of spectrogram values
        freq2 -- the spectrogram frequency values (MHz)
```
    function WxxCalc(phi::Vector{Float64},sweepRange::Float64,sweepTime::Float64, df::Float64; cmor::Int64=5)
        fs=1/(1e6/((sweepRange)/sweepTime))
        s0=50
        smax=400
        wave=string("cmor", string(cmor), "-1")
        pywt = pyimport("pywt")
        Wxx,freq2=pywt.cwt(phi,scales,wave,sampling_period=1)
        freq2 = freq2/(df*1e9)
        Wxx = abs.(Wxx)

        Wxx, freq2
    end

```
Function that calculates the maximum spectrogram frequency from a spectrogram.

Input:
        Wxx::Array{Float64}        -- the 2-D array of spectrogram values from WxxCalc
        freq::Vector{Float64}      -- vector of reflectometer frequency meausurements (GHz)
        datafreq::Vector{Float64}  -- the spectrogram frequency values from WxxCalc

Optional Inputs:
         maxDelay::Float64   -- The maximum spectrogram frequency(group delay) to be considered ().
         minDelay::Float64=0 -- The minimum spectrogram frequency (group delay) to be considered ().
```
    function max_calc(Wxx::Array{Float64},freq::Vector{Float64},datafreq::Vector{Float64}; maxDelay::Float64=1e32, minDelay::Float64=0)
        local maxfreq = length(datafreq)
        local minfreq = 1
        for j in 1:1:length(datafreq)
            if datafreq[j] < maxDelay
                maxfreq=j
                break
            end
        end
        for j in 1:1:length(datafreq)
            if datafreq[j] < minDelay
                minfreq = j
                break
            end
        end
        maxvec = zeros(length(freq))
        for k in 1:1:length(freq)
            step = findall(x->x==maximum(Wxx[maxfreq:minfreq,k]),Wxx[maxfreq:minfreq,k])[1]
            maxvec[k] = datafreq[maxfreq+step]
        end
        maxvec
    end
end
