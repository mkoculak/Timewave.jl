#= 
Welch periodogram for real-valued arrays, where each column is a separate signal.
Only onesided spectrum is returned. However, it allows for more customization then
default DSP.jl version, with alternative options of aggregation, detrending of segments,
and facilitation of memory-conscious multithreading.
Based on DSP.jl internals with parts of implmentation translated from SciPy.
=#
function welch(data::AbstractArray{<:Real}, fs, nperseg, noverlap, nfft, window, scaling, )
    nSamples, nChannels = size(data)

    nfft >= nperseg || error("nfft must be >= nperseg")

    if typeof(window) <: Function
        win = window(nperseg)
    elseif typeof(window) <: Vector{<:AbstractFloat}
        length(window) != nperseg && error("Window must be of length nperseg.")
        win = window
    else
        error("Unknown window type. Window can be a function or a vector of length nperseg.")
    end

    if scaling == "density"
        scale = 1. / (fs * sum(abs2, win))
    elseif scaling == "spectrum"
        scale = 1. / sum(win)^2
    end

    segmentIdx = collect(1:(nperseg-noverlap):(nSamples-nperseg+1))

    output = zeros(nfft>>1+1, nChannels, length(segmentIdx))

    sigtemp = zeros(nfft)
    tmp = zeros(ComplexF64, nfft >> 1+1)
    plan = DSP.FFTW.plan_rfft(sigtemp)

    for chan in 1:nChannels
        seg = 1
        @views for segment in segmentIdx
            mmean = mean(data[segment:segment+nperseg-1, chan])
            sigtemp[1:nperseg] .= (data[segment:segment+nperseg-1, chan] .- mmean) .* win
            mul!(tmp, plan, sigtemp)
            output[:, chan, seg] .= abs2.(tmp) .* scale
            seg +=1
        end
    end

    # Correct for pairing when onesided and density
    if scaling == "density"
        if nfft % 2 == 0
            @views output[2:end,:,:] .*= 2.
        else
            @views output[2:end-1,:,:] .*= 2.
        end
    end

    return output
end