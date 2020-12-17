module reflectometryNN
    using Flux
    using BSON: @save
    include("radCalcs.jl")

    mutable struct trainingData
        data
        measurement_max
        normalizationRad
    end

    struct NNOutput
        NN
        normalizationRad::Float64
    end

    function makeTrainingData(freq::Vector{Float64},dif_fits::Float64, coeffs_fit::Vector{Float64}, normalizationRad::Float64; trainingSpread::Float64 = 0.5)
        data = (Array{Float64}(undef,length(freq),dif_fits),Array{Float64}(undef,length(freq),dif_fits))
        measurement_max = Array{Float64}(undef,(dif_fits))
        for j in 1:1:dif_fits
            monocheck=1
            local coeffs
            while monocheck==1
                coeffs = [coeffs_fit[1]+trainingSpread*rand()*(-1)^rand(1:2), coeffs_fit[2]+trainingSpread*rand()*(-1)^rand(1:2), coeffs_fit[3]+trainingSpread*rand()*(-1)^rand(1:2),
                                coeffs_fit[4]+trainingSpread*rand()*(-1)^rand(1:2), coeffs_fit[5]+trainingSpread*rand()*(-1)^rand(1:2), coeffs_fit[6]+trainingSpread*rand()*(-1)^rand(1:2)]
                data_proposal = radCalc.Omode(freq,coeffs)
                monocheck=0
                for j in 2:1:length(data_proposal)
                    if (data_proposal[j] - data_proposal[j-1]) < 0
                        monocheck=1
                        break
                    end
                end
            end
            radius = radCalc.Omode(freq,coeffs)
            measurement= dphidw(coeffs)
            measurement_max[j] = maximum(measurement)
            data[1][:,j] = measurement/maximum(measurement)
            data[2][:,j] = radius/maximum(measurement)/normalizationRad
        end
        data_formated = Flux.Data.DataLoader(data;batchsize=dif_fits,shuffle=false,partial=true)

        trainingData(data_formated , measurement_max, normalizationRad)
    end


    function NeuralNet(dataNN::trainingData, freqs::Vector{Float64}, epochs_num::Int64; activation = sigmoid, optimizer = NADAM, learningrate = 1e-5, learningdecay = 1e-4, neurons=200, layers=2)
        local loss_check = 1
        if layers==2
            NN = Chain(Dense(length(freqs),neurons, activation ),
                        Dense(neurons,length(freqs)))
        else
            NN = Chain(Dense(length(freqs),neurons, activation ),
                        Dense(neurons, neurons, activation),
                        Dense(neurons,length(freqs)))
        end
        function losss(x,y)
            loss_check = sum(abs2,NN(x)-(y))
        end
        timeTake = @elapsed begin
            Threads.@threads for j in 1:1:epochs_num
                descent_param = learningrate*exp(-1*j*learningdecay)
                Flux.train!(losss,Flux.params(NN),dataNN.data,optimizer(descent_param))#NADAM(descent_param[1], (0.9, 0.8)))
            end
        end
       NNOutput(NN, dataNN.normalizationRad)
    end


    function save(filename::String, object)
        @save filename object
    end

end
