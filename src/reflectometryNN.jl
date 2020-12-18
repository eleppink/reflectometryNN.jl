module reflectometryNN

    using Flux
    using BSON: @save
    include("groupDelay.jl")
    mutable struct trainingData
        data
        density
        freqs
        measurement_max
        normalizationRad
        XMode
        B0
        R0
    end

    struct NNOutput
        NN
        normalizationRad::Float64
    end



"""
Function to create neural network training data.

Inputs:
        freq                         -- Vector of frequencies of the reflectometry measurements (GHz)
        dif_fits::Int64              -- The number of different profiles in a single batch (batch size)
        coeffs_fit::Array{Float64}   -- The 6 fit coefficients for the 5th order polynomial for plasma density in units of 1e19 (per cubic meter)
                                            n(r) = 1e19*(coeffs_fit[1]*r^5+coeffs_fit[2]*r^4+coeffs_fit[3]*r^3+coeffs_fit[4]*r^2+coeffs_fit[5]*r^1+coeffs_fit[6])
                                            where r=radius into the plasma from the plasma facing wall.
        normalizationRad::Number     -- The output radius normalization factor, that helps with training. Make output order 1-10.
        calibration::Array{Float64}  -- Vector of calibrated group delay length (non-plasma length) at each frequency, for a given reflectometry measurement. Length in METERS.

Optional inputs:
        trainingSpread::Float64      -- The variation in the profiles, fraction of each coefficient. Default value is 0.5.
        XMode::Bool                  -- A boolean if XMode is the desired profile (true or false). Default is false (OMode).
        Xcutoff::String              -- The type of XMode cutoff ("LEFT" or "RIGHT").
        Bmag::Number                 -- The magnetic field magnitude measured at major-radius R0 (Tesla)
        R0::Number                   -- The major-radial location of the magnetic field measurement (meters)

    e.g. for the O-Mode HFS reflectometer on DIII-D
    makeTrainingData(range(6, 27, length = 200), 1000, ,[2.181e5, -2.901e4, 1252, -16.14, 2.381, 0.03], 5e7; trainingSpread = 0.5)


Output is a trainingData struct that includes:
        trainingData.data             -- the simulated training profiles
        trainingData.freqs            -- the frequency vector used for the measurements (GHz)
        trainingData.measurement_max  -- the normalization used for each group delay in the batch
        trainingData.normalizationRad -- the output radius normalization inputted into the function
        trainingData.Xmode            -- Boolean that says if XMode was enabled
        trainingData.B0               -- Magnitude of magnetic field for XMode (Tesla)
        trainingData.R0               -- Major-radial location of magnetic field measurement for XMode (meters)
"""
    function makeTrainingData(freq,dif_fits::Int64, coeffs_fit::Array{Float64}, normalizationRad::Number, calibration::Array{Float64}; trainingSpread::Float64 = 0.5, XMode::Bool = false, Xcutoff::String="LEFT", Bmag::Number=0, R0::Number=0)
        if XMode
            data = (Array{Float64}(undef,length(freq)+1,dif_fits),Array{Float64}(undef,length(freq),dif_fits))
        else
            data = (Array{Float64}(undef,length(freq),dif_fits),Array{Float64}(undef,length(freq),dif_fits))
        end
        measurement_max = Array{Float64}(undef,(dif_fits))
        radius = Array{Float64}(undef,length(freq))
        density = Array{Float64}(undef,length(freq))
        for j in 1:1:dif_fits
            monocheck=1
            local coeffs
            while monocheck==1
                coeffs = [coeffs_fit[1]+trainingSpread*rand()*(-1)^rand(1:2), coeffs_fit[2]+trainingSpread*rand()*(-1)^rand(1:2), coeffs_fit[3]+trainingSpread*rand()*(-1)^rand(1:2),
                                coeffs_fit[4]+trainingSpread*rand()*(-1)^rand(1:2), coeffs_fit[5]+trainingSpread*rand()*(-1)^rand(1:2), coeffs_fit[6]+trainingSpread*rand()*(-1)^rand(1:2)]
                if XMode
                    if Xcutoff=="LEFT"
                        radius,density = radCalc.XModeL(freq,coeffs,Bmag,R0)
                    elseif Xcutoff == "RIGHT"
                        radius,density = radCalc.XModeR(freq,coeffs,Bmag,R0)
                    end
                else
                    radius,density = radCalc.OMode(freq,coeffs)
                end
                monocheck=0
                for j in 2:1:length(radius)
                    if (radius[j] - radius[j-1]) < 0
                        monocheck=1
                        break
                    end
                end
            end
            if XMode
                measurement = dphidw_xmode(freq,coeffs,calibration,Bmag,R0,radius)
                data[1][1,j] = Bmag
                data[1][2:end,j] = measurement/maximum(measurement)
                measurement_max[j] = maximum(measurement)
                data[2][:,j] = radius/maximum(measurement)/normalizationRad
            else
                measurement = dphidw_omode(freq,coeffs,calibration,radius)
                data[1][:,j] = measurement/maximum(measurement)
                measurement_max[j] = maximum(measurement)
                data[2][:,j] = radius/maximum(measurement)/normalizationRad
            end
        end

        if XMode
            data_formated = Flux.Data.DataLoader(data;batchsize=dif_fits,shuffle=false,partial=true)
            output = trainingData(data_formated, density, freq, measurement_max, normalizationRad, XMode, Bmag, R0)
        else
            data_formated = Flux.Data.DataLoader(data;batchsize=dif_fits,shuffle=false,partial=true)
            output = trainingData(data_formated, density, freq, measurement_max, normalizationRad, XMode, NaN, NaN)
        end

        output
    end



"""
Function to create the neural network object

Inputs:
        dataNN:trainingData  -- a trainingData struct from the makeTrainingData() function
        epochs_num::Int64    -- the number of epochs you wish to train the neural network for

Optional inputs:
        actication           -- the activation function used in the neural network. Default is sigmoid.
        optimizer            -- the optimizer (descent algorithm) used in the neural network. Default is NADAM
        learningrate         -- the initial optimizer learning rate. Default is 1e-5
        learningdecay        -- the exponential factor of decay of the learning rate. Decay is 1e-4 per epoch.
        neurons              -- the number of neurons. Default is 200.
        layers               -- the number of layers in the neural network (2 or 3). Default is 2.

Output is a NNOutput struct, with form:
        NNOutput.NN               -- the neural network object.
        NNoutput.normalizationRad -- the output radius normalization used for this neural network object

"""
    function NeuralNet(dataNN::trainingData, epochs_num::Int64; activation = swish, optimizer = RMSProp, learningrate::Float64 = 1e-4, learningdecay::Float64 = 1e-4, neurons::Int64=200, layers::Int64=2, min_learningrate::Float64=1e-6,print_epoch::Number = 1000)
        local loss_check = 1
        local loss_count = 0
        if layers==2
            if dataNN.XMode
                NN = Chain(Dense(length(dataNN.freqs)+1,neurons, activation ),
                        Dense(neurons,length(dataNN.freqs)))
            else
                NN = Chain(Dense(length(dataNN.freqs),neurons, activation ),
                        Dense(neurons,length(dataNN.freqs)))
            end

        else
            if dataNN.XMode
                NN = Chain(Dense(length(dataNN.freqs)+1,neurons, activation ),
                        Dense(neurons, neurons, activation),
                        Dense(neurons,length(dataNN.freqs)))
            else
                NN = Chain(Dense(length(dataNN.freqs),neurons, activation ),
                        Dense(neurons, neurons, activation),
                        Dense(neurons,length(dataNN.freqs)))
            end
        end
        function losss(x,y)
            loss_check = sum(abs2,NN(x)-(y))
        end
        for j in 1:1:epochs_num
                if j % print_epoch == 0
                    loss_count += 1
                    print(string(loss_count), string(" / "), string(round(epochs_num/print_epoch)), string(":  "), string(loss_check), string("\n"))
                end
                descent_param = max(min_learningrate,learningrate*exp(-1*j*learningdecay))
                Flux.train!(losss,Flux.params(NN),dataNN.data,optimizer(descent_param))
        end
       NNOutput(NN, dataNN.normalizationRad)
    end


"""
A function that simply saves an object (typically used for the NNOutput object to be used later).

Inputs:
        filename::String  -- the filename of the object, saved in current directory
        object            -- the object to be saved, typically the NNOutput object for this use case
"""
    function save(filename::String, object)
        @save filename object
    end

end
