module scalogram
    using PyCall
    function WxxCalc(phi::Vector{Float64},cmor::Int64,sweepRange::Float64,sweepTime::Float64, df::Float64)
        fs=1/(1e6/((sweepRange)/sweepTime))
        s0=50
        smax=400
        wave=string("cmor", string(cmor), "-1")
        pywt = pyimport("pywt")
        Wxx,freq2=pywt.cwt(phi,scales,wave,sampling_period=1)
        freq2 = freq2/df
        Wxx = abs.(Wxx)

        Wxx, freq2
    end

    function max_calc(Wxx::Array{Float64},freq::Vector{Float64},datafreq::Vector{Float64}, maxDelay::Float64, minDelay::Float64)
        local maxfreq
        local minfreq
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
