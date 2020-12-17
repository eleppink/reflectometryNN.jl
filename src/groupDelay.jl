include("radCalcs.jl")
using QuadGK

function dphidw_omode(freq::Vector{Float64}, coeffs_fit::Vector{Float64},calibration::Vector{Float64})
    ophi = Array{Float64}(undef,length(freq))
    rad = radCalc.OMode(freq,coeffs_fit)
    for j in 1:1:length(freq)
        ophi[j] = calibration[j] / 3e8 * (freq[j]*2*pi*1e9) + 2*(freq[j]*2*pi*1e9) / 3e8 * quadgk(x -> refractive_omode(x;coeffs = coeffs_fit, omega = freq[j]*2*pi*1e9),0,(rad[j]),rtol=1e-12)[1]
    end
    ophiw = Array{Float64}(undef,length(freq))
    dw = (freq[2]-freq[1])*2*pi*1e9
    ophiw[1] = (ophi[2]-ophi[1])/dw
    ophiw[length(freq)] = (ophi[length(freq)]-ophi[length(freq)-1])/dw
    for j in 2:1:length(ophiw)-1
        ophiw[j] = (ophi[j+1]-ophi[j-1])/(2*dw)
    end

    ophiw
end

function dphidw_xmode(freq::Vector{Float64}, coeffs_fit::Vector{Float64},Bmag::Float64, startingRad::Float64, calibration::Vector{Float64},cutoff::String)
    ophi = Array{Float64}(undef,length(freq))
    if cutoff=="left"
        rad = radCalc.XModeL(freq,coeffs_fit)
    elseif cutoff=="right"
        rad = radCalc.XModeR(freq,coeffs_fit)
    end
    for j in 1:1:length(freq)
        ophi[j] = calibration[j] / 3e8 * (freq[j]*2*pi*1e9) + 2*(freq[j]*2*pi*1e9) / 3e8 * quadgk(x -> refractive_xmode(x;coeffs = coeffs_fit, omega = freq[j]*2*pi*1e9, b0 = Bmag, r0 = startingRad),0,(rad[j]),rtol=1e-12)[1]
    end
    ophiw = Array{Float64}(undef,length(freq))
    dw = (freq[2]-freq[1])*2*pi*1e9
    ophiw[1] = (ophi[2]-ophi[1])/dw
    ophiw[length(freq)] = (ophi[length(freq)]-ophi[length(freq)-1])/dw
    for j in 2:1:length(ophiw)-1
        ophiw[j] = (ophi[j+1]-ophi[j-1])/(2*dw)
    end

    ophiw
end

function refractive_omode(x::Float64; coeffs::Vector{Float64}, omega::Float64)
    q = 1.6022e-19
    me = 9.109e-31
    e0 = 8.854e-12
    density = 1e19.*(coeffs[1].*(x).^5 .+coeffs[2].*(x).^4 .+coeffs[3].*(x).^3 .+coeffs[4].*(x).^2 .+coeffs[5].*(x).+coeffs[6])
    wpe2 = (q)^2 * density / (e0*me)
    sqrt(max(0.0,1.0 .-wpe2./omega^2))
end

function refractive_xmode(x::Float64; coeffs::Vector{Float64}, omega::Float64, b0::Float64, r0::Float64)
    q = 1.6022e-19
    me = 9.109e-31
    e0 = 8.854e-12
    density = 1e19.*(coeffs[1].*(x).^5 .+coeffs[2].*(x).^4 .+coeffs[3].*(x).^3 .+coeffs[4].*(x).^2 .+coeffs[5].*(x).+coeffs[6])
    wpe2 = (q)^2 * density / (e0*me)
    wce = b0 / (r0+x) * q / me
    sqrt(max(0.0,1.0 .-wpe2./omega^2 * (omega^2-wpe2)/(omega^2-(wpe2+wce^2))))
end
