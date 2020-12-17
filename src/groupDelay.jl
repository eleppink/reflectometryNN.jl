include("radCalcs.jl")
using QuadGK

```
Function for calculating the OMode group delay.

Inputs:
        freq::Vector{Float64}         -- Measured frequencies of the reflectometer (GHz)
        coeffs_fit::Vector{Float64}   -- The 6 fit coefficients for the 5th order polynomial for plasma density in units of 1e19 (per cubic meter)
                                            n(r) = 1e19*(coeffs_fit[1]*r^5+coeffs_fit[2]*r^4+coeffs_fit[3]*r^3+coeffs_fit[4]*r^2+coeffs_fit[5]*r^1+coeffs_fit[6])
                                            where r=radius into the plasma from the plasma facing wall.
        calibration::Vector{Float64}  -- Calibration length in meters (non-plasma group delay)
        rad::Vector{Float64}          -- Radial location of cutoffs are each frequency in freq (meters from first wall).


Output:
        ophiw   -- the group delay at each frequency in freq
```
function dphidw_omode(freq::Vector{Float64}, coeffs_fit::Vector{Float64},calibration::Vector{Float64},rad::Vector{Float64})
    ophi = Array{Float64}(undef,length(freq))
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



```
Function for calculating the XMode group delay.

Inputs:
        freq::Vector{Float64}         -- Measured frequencies of the reflectometer (GHz)
        coeffs_fit::Vector{Float64}   -- The 6 fit coefficients for the 5th order polynomial for plasma density in units of 1e19 (per cubic meter)
                                            n(r) = 1e19*(coeffs_fit[1]*r^5+coeffs_fit[2]*r^4+coeffs_fit[3]*r^3+coeffs_fit[4]*r^2+coeffs_fit[5]*r^1+coeffs_fit[6])
                                            where r=radius into the plasma from the plasma facing wall.
        calibration::Vector{Float64}  -- Calibration length in meters (non-plasma group delay)
        rad::Vector{Float64}          -- Radial location of cutoffs are each frequency in freq (meters from first wall).
        Bmag::Float64                 -- The magnitude of the magnetic field at major radius startingRad (Tesla)
        startingRad::Float64          -- The major radial measurement location of the magnetic field (meters).

Output:
        ophiw -- the group delay at each frequency in freq
```
function dphidw_xmode(freq::Vector{Float64}, coeffs_fit::Vector{Float64}, calibration::Vector{Float64}, Bmag::Float64, startingRad::Float64,rad::Vector{Float64})
    ophi = Array{Float64}(undef,length(freq))
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


```
Function for calculating the OMode index of refraction

Inputs:
        x::Float64                    -- the radial location in meters from the first wall
        coeffs_fit::Vector{Float64}   -- The 6 fit coefficients for the 5th order polynomial for plasma density in units of 1e19 (per cubic meter)
                                            n(r) = 1e19*(coeffs_fit[1]*r^5+coeffs_fit[2]*r^4+coeffs_fit[3]*r^3+coeffs_fit[4]*r^2+coeffs_fit[5]*r^1+coeffs_fit[6])
                                            where r=radius into the plasma from the plasma facing wall.
         omega::Float64               -- the excited frequency (radians per second)

Output:
        -- the OMode index of refraction
```
function refractive_omode(x::Float64; coeffs::Vector{Float64}, omega::Float64)
    q = 1.6022e-19
    me = 9.109e-31
    e0 = 8.854e-12
    density = 1e19.*(coeffs[1].*(x).^5 .+coeffs[2].*(x).^4 .+coeffs[3].*(x).^3 .+coeffs[4].*(x).^2 .+coeffs[5].*(x).+coeffs[6])
    wpe2 = (q)^2 * density / (e0*me)
    sqrt(max(0.0,1.0 .-wpe2./omega^2))
end


```
Function for calculating the XMode index of refraction

Inputs:
        x::Float64                    -- the radial location in meters from the first wall
        coeffs_fit::Vector{Float64}   -- The 6 fit coefficients for the 5th order polynomial for plasma density in units of 1e19 (per cubic meter)
                                            n(r) = 1e19*(coeffs_fit[1]*r^5+coeffs_fit[2]*r^4+coeffs_fit[3]*r^3+coeffs_fit[4]*r^2+coeffs_fit[5]*r^1+coeffs_fit[6])
                                            where r=radius into the plasma from the plasma facing wall.
         omega::Float64               -- the excited frequency (radians per second)
         b0::Float64                 -- The magnitude of the magnetic field at major radius startingRad (Tesla)
         r0::Float64          -- The major radial measurement location of the magnetic field (meters).

Output:
        -- the XMode index of refraction
```
function refractive_xmode(x::Float64; coeffs::Vector{Float64}, omega::Float64, b0::Float64, r0::Float64)
    q = 1.6022e-19
    me = 9.109e-31
    e0 = 8.854e-12
    density = 1e19.*(coeffs[1].*(x).^5 .+coeffs[2].*(x).^4 .+coeffs[3].*(x).^3 .+coeffs[4].*(x).^2 .+coeffs[5].*(x).+coeffs[6])
    wpe2 = (q)^2 * density / (e0*me)
    wce = b0 / (r0+x) * q / me
    sqrt(max(0.0,1.0 .-wpe2./omega^2 * (omega^2-wpe2)/(omega^2-(wpe2+wce^2))))
end
