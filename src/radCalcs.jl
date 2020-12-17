module radCalc

    function OMode(freq::Vector{Float64},coeffs_fit::Vector{Float64})
        omega = freq .* 2.0 .* pi .* 1e9
        q = 1.6022e-19
        me = 9.109e-31
        e0 = 8.854e-12
        rvec = zeros(length(omega))
        rtest = 0
        wpe = 0
        for j in 1:1:length(rvec)
            while wpe < omega[j]
                rtest = rtest + 0.00001
                density = 1e19*(coeffs_fit[1]*(rtest)^5+coeffs_fit[2]*(rtest)^4+coeffs_fit[3]*(rtest)^3+coeffs_fit[4]*(rtest)^2+coeffs_fit[5]*(rtest)+coeffs_fit[6])
                wpe = sqrt(max(0,(q)^2*density/(e0*me)))
            end
            rvec[j]=rtest
        end

        rvec
    end

    function XModeR(freq::Vector{Float64},coeffs_fit::Vector{Float64})
        omega = freq .* 2.0 .* pi .* 1e9
        q = 1.6022e-19
        me = 9.109e-31
        e0 = 8.854e-12
        rvec = zeros(length(omega))
        rtest = 0
        wcutoff = 0
        for j in 1:1:length(rvec)
            while wcutoff < omega[j]
                rtest = rtest + 0.00001
                wce = Bmag / (rtest+r0) * q / me
                density = 1e19*(coeffs_fit[1]*(rtest)^5+coeffs_fit[2]*(rtest)^4+coeffs_fit[3]*(rtest)^3+coeffs_fit[4]*(rtest)^2+coeffs_fit[5]*(rtest)+coeffs_fit[6])
                wpe = sqrt(max(0,(q)^2*density/(e0*me)))
                wcutoff = 0.5*(wce+sqrt(wce^2+4*wpe^2))
            end
            rvec[j]=rtest
        end

        rvec
    end

    function XModeL(freq::Vector{Float64},coeffs_fit::Vector{Float64}, BMag::Float64, r0::Float64)
        omega = freq .* 2.0 .* pi .* 1e9
        q = 1.6022e-19
        me = 9.109e-31
        e0 = 8.854e-12
        rvec = zeros(length(omega))
        rtest = 0
        wcutoff = 0
        for j in 1:1:length(rvec)
            while wcutoff < omega[j]
                rtest = rtest + 0.00001
                wce = Bmag / (rtest+r0) * q / me
                density = 1e19*(coeffs_fit[1]*(rtest)^5+coeffs_fit[2]*(rtest)^4+coeffs_fit[3]*(rtest)^3+coeffs_fit[4]*(rtest)^2+coeffs_fit[5]*(rtest)+coeffs_fit[6])
                wpe = sqrt(max(0,(q)^2*density/(e0*me)))
                wcutoff = 0.5*(-wce+sqrt(wce^2+4*wpe^2))
            end
            rvec[j]=rtest
        end

        rvec
    end

end
