module radCalc

"""
Function that calculates the radial cutoff location for the O-Mode cutoff, given a density profile.

Inputs:
        freq                        -- Vector of frequencies (GHz)
        coeffs_fit::Array{Float64}  -- The 6 fit coefficients for the 5th order polynomial for plasma density in units of 1e19 (per cubic meter)
                                            n(r) = 1e19*(coeffs_fit[1]*r^5+coeffs_fit[2]*r^4+coeffs_fit[3]*r^3+coeffs_fit[4]*r^2+coeffs_fit[5]*r^1+coeffs_fit[6])
                                            where r=radius into the plasma from the plasma facing wall.

Output:
        rvec -- radial location of each cutoff w.r.t the first wall (meters).
"""

    function OMode(freq,coeffs_fit::Array{Float64})
        omega = freq .* 2.0 .* pi .* 1e9
        q = 1.6022e-19
        me = 9.109e-31
        e0 = 8.854e-12
        rvec = zeros(length(omega))
        dvec = zeros(length(omega))
        rtest = 0
        wpe = 0
        for j in 1:1:length(rvec)
            while wpe < omega[j]
                rtest = rtest + 0.00001
                density = 1e19*(coeffs_fit[1]*(rtest)^5+coeffs_fit[2]*(rtest)^4+coeffs_fit[3]*(rtest)^3+coeffs_fit[4]*(rtest)^2+coeffs_fit[5]*(rtest)+coeffs_fit[6])
                wpe = sqrt(max(0,(q)^2*density/(e0*me)))
            end
            rvec[j]=rtest
            dvec[j]=density
        end

        rvec,dvec
    end



"""
Function that calculates the radial cutoff location for the X-Mode R cutoff, given a density profile.

Inputs:
        freq::Vector{Float64}       -- Vector of frequencies (GHz)
        coeffs_fit::Array{Float64}  -- The 6 fit coefficients for the 5th order polynomial for plasma density in units of 1e19 (per cubic meter)
                                            n(r) = 1e19*(coeffs_fit[1]*r^5+coeffs_fit[2]*r^4+coeffs_fit[3]*r^3+coeffs_fit[4]*r^2+coeffs_fit[5]*r^1+coeffs_fit[6])
                                            where r=radius into the plasma from the plasma facing wall.
        Bmag::Number                -- the magnitude of the magnetic field at location r0 (Tesla).
        r0::Number                  -- the major-radial location of the magnetic field magnitude (meters).

Output:
        rvec -- radial location of each cutoff w.r.t the first wall (meters).
"""
    function XModeR(freq,coeffs_fit::Array{Float64}, Bmag::Number, r0::Number)
        omega = freq .* 2.0 .* pi .* 1e9
        q = 1.6022e-19
        me = 9.109e-31
        e0 = 8.854e-12
        rvec = zeros(length(omega))
        dvec = zeros(length(omega))
        rtest = 0
        wcutoff = 0
        for j in 1:1:length(rvec)
            while wcutoff < omega[j]
                rtest = rtest + 0.00001
                wce = BMag / (rtest+r0) * q / me
                density = 1e19*(coeffs_fit[1]*(rtest)^5+coeffs_fit[2]*(rtest)^4+coeffs_fit[3]*(rtest)^3+coeffs_fit[4]*(rtest)^2+coeffs_fit[5]*(rtest)+coeffs_fit[6])
                wpe = sqrt(max(0,(q)^2*density/(e0*me)))
                wcutoff = 0.5*(wce+sqrt(wce^2+4*wpe^2))
            end
            rvec[j]=rtest
            dvec[j]=density
        end

        rvec,dvec
    end




"""
Function that calculates the radial cutoff location for the X-Mode L cutoff, given a density profile.

Inputs:
        freq::Vector{Float64}       -- Vector of frequencies (GHz)
        coeffs_fit::Vector{Float64} -- The 6 fit coefficients for the 5th order polynomial for plasma density in units of 1e19 (per cubic meter)
                                            n(r) = 1e19*(coeffs_fit[1]*r^5+coeffs_fit[2]*r^4+coeffs_fit[3]*r^3+coeffs_fit[4]*r^2+coeffs_fit[5]*r^1+coeffs_fit[6])
                                            where r=radius into the plasma from the plasma facing wall.
        Bmag::Number                -- the magnitude of the magnetic field at location r0 (Tesla).
        r0::Number                  -- the major-radial location of the magnetic field magnitude (meters).

Output:
        rvec -- radial location of each cutoff w.r.t the first wall (meters).
"""
    function XModeL(freq,coeffs_fit::Vector{Float64}, BMag::Number, r0::Number)
        omega = freq .* 2.0 .* pi .* 1e9
        q = 1.6022e-19
        me = 9.109e-31
        e0 = 8.854e-12
        rvec = zeros(length(omega))
        dvec = zeros(length(omega))
        rtest = 0
        wcutoff = 0
        for j in 1:1:length(rvec)
            while wcutoff < omega[j]
                rtest = rtest + 0.00001
                wce = BMag / (rtest+r0) * q / me
                density = 1e19*(coeffs_fit[1]*(rtest)^5+coeffs_fit[2]*(rtest)^4+coeffs_fit[3]*(rtest)^3+coeffs_fit[4]*(rtest)^2+coeffs_fit[5]*(rtest)+coeffs_fit[6])
                wpe = sqrt(max(0,(q)^2*density/(e0*me)))
                wcutoff = 0.5*(-wce+sqrt(wce^2+4*wpe^2))
            end
            rvec[j]=rtest
            dvec[j]=density
        end

        rvec,dvec
    end

end
