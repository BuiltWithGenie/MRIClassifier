using Images: channelview, Gray, imresize, load
using Flux: Adam, Losses, setup, sigmoid, update!, withgradient
using Printf: @printf
using Flux: BatchNorm, Chain, Conv, Dense, MaxPool, relu, SamePad
using MLUtils: flatten
using Random: seed!

const DATAPATH = "public/"
const YESPATH = joinpath(DATAPATH, "yes")
const NOPATH = joinpath(DATAPATH, "no")
const IMAGE_SIZE = (80, 80)
const FRAC_TEST = 0.2

filt_macos(x) = x != ".DS_Store"

function load_and_process_images(dir)
    files = filter(filt_macos, readdir(dir))
    map(files) do file
        imresize(float.(channelview(Gray.(load(joinpath(dir, file))))), IMAGE_SIZE)
    end
end

function split_data(data)
    ntest = round(Int, length(data) * FRAC_TEST)
    data[1:ntest], data[ntest+1:end]
end

format_data = x -> cat(x...; dims=ndims(x[1]) + 2)

function create_model()
    seed!(0)
    filter_size, downsample_factor, num_downsamples = (3, 3), (2, 2), 3
    num_channels = [1, 32, 64, 128]
    num_channels = [1, 8, 16, 32]

    Chain(
        Conv(filter_size, num_channels[1] => num_channels[2], relu; pad=SamePad()),
        MaxPool(downsample_factor), BatchNorm(num_channels[2]),
        Conv(filter_size, num_channels[2] => num_channels[3], relu; pad=SamePad()),
        MaxPool(downsample_factor), BatchNorm(num_channels[3]),
        Conv(filter_size, num_channels[3] => num_channels[4], relu; pad=SamePad()),
        MaxPool(downsample_factor), flatten,
        Dense(num_channels[4] * prod(IMAGE_SIZE) ÷ prod(downsample_factor)^num_downsamples => 1), vec
    )
end

function get_layer_outputs(model, input)
    outputs = []
    x = input
    for layer in model.layers
        x = layer(x)
        push!(outputs, x)
    end
    outputs
end

classify(model, input) = sigmoid(model(input)) .> 0.5

compute_accuracy(model, input, labels) = count(classify(model, input) .== labels) / length(labels)

function train_model(model, train_data, train_labels, test_data, test_labels)
    opt_state = setup(Adam(), model)
    log = []

    nepochs = 100
    for epoch = 1:nepochs
        # Compute the loss function and its gradient
        (loss, grads) = withgradient(model) do m
            Losses.logitbinarycrossentropy(m(train_data), train_labels)
        end
        # Update the neural network weights
        update!(opt_state, model, grads[1])
        # Compute the accuracy of the model on the training data and the test data
        train_acc = compute_accuracy(model, train_data, train_labels)
        test_acc = compute_accuracy(model, test_data, test_labels)
        # Store results in the log
        push!(log, (; epoch, loss, train_acc, test_acc))
        # Print status updates
        @printf("Epoch %3d: loss = %0.4f, train_accuracy = %0.4f, test_accuracy = %0.4f\n", log[end]...)
    end
    log
end

