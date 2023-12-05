module App
using GenieFramework, PlotlyBase, Random, ImageIO, FileIO, Colors, Statistics, FixedPointNumbers, JLD2
include("utils.jl")
#= model = create_model() =#
@load "public/model.jld2" model
@load "public/plotdata.jld2" epochs train_accs test_accs
all_yes, all_no = load_and_process_images(YESPATH), load_and_process_images(NOPATH)
img_yes = readdir(YESPATH)[1:10]
img_no = readdir(NOPATH)[1:10]

@app begin
    @in training = false
    @out log_str = "Epoch 1: Train acc: 0.6138613861386139, Test acc: 0.6078431372549019 \n Epoch 2: Train acc: 0.6138613861386139, Test acc: 0.6078431372549019"
    @out train_acc = round(train_accs[end], digits=4)
    @out test_acc = round(test_accs[end],digits=4)
    @out trace = [
                  scatter(x=epochs, y=train_accs, mode="lines+markers", name="Train"),
                  scatter(x=epochs, y=test_accs, mode="lines+markers", name="Test")
                 ]
    @out layout = PlotlyBase.Layout(title="Train and test error")
    @out images = []
    @in clicked_img = ""
    @out image_layers = []
    @out animation_url = "/animation.gif"
    @out img_yes = img_yes
    @out img_no = img_no
    @out label = "???"
    @out ground_truth = "???"
    @out classification_error = false


    @onbutton training begin
        if !(get(ENV, "GENIE_ENV", "") == "prod")
        test_yes, train_yes = split_data(all_yes)
        test_no, train_no = split_data(all_no)

        # Format and combine data
        test_data, train_data = format_data([test_yes; test_no]), format_data([train_yes; train_no])
        test_labels, train_labels = [trues(length(test_yes)); falses(length(test_no))], [trues(length(train_yes)); falses(length(train_no))]

        # Create model and train
        model = create_model()
        train_log = train_model(model, train_data, train_labels, test_data, test_labels)
        train_acc = train_log[end].train_acc
        test_acc = train_log[end].test_acc
        # Update the plot traces with the new training log
        epochs = 1:length(train_log)
        train_accs = getindex.(train_log, :train_acc)
        test_accs = getindex.(train_log, :test_acc)
        log_str = string(join(map(x -> "Epoch $(x.epoch): Train acc: $(x.train_acc), Test acc: $(x.test_acc)", train_log), "<br>")...)
        @save "public/plotdata.jld2" epochs train_accs test_accs
        @save "public/model.jld2" model
        trace = [
                 scatter(x=epochs, y=train_accs, mode="lines+markers", name="Train"),
                 scatter(x=epochs, y=test_accs, mode="lines+markers", name="Test")
                ]
    end
    end

    @onchange clicked_img begin
        @show clicked_img
        img = ones(IMAGE_SIZE[1],IMAGE_SIZE[2],1,1)
        #= img = load("public/"*clicked_img) =#
        #= img[:,:,1,1] = images[clicked_img] =#
        img[:,:,1,1] = imresize(float.(channelview(Gray.(load("public/"*clicked_img)))), IMAGE_SIZE)
        label = classify(model, img)[1] ? "Tumor" : "No tumor"
        ground_truth = occursin("yes",clicked_img) ? "Tumor" : "No tumor"
        classification_error = label != ground_truth
        image_layers = get_layer_outputs(model, img)
        L = 10
        frames = zeros(RGB{N0f8},IMAGE_SIZE[1],IMAGE_SIZE[2],L)
        for l in 1:L
            normalized_layer = abs.(image_layers[l][:,:,1,1] ./ maximum(image_layers[l][:,:,1,1]) )
            px = size(normalized_layer,1)
            @show px
            if px > IMAGE_SIZE[1]
                break
            end
            rgb_img = map(x -> RGB(x, x, x), normalized_layer)
            save("public/img_layer_$l.png", rgb_img)
            # place the layer output in a centered frame
            start_x = round(Int, (IMAGE_SIZE[1] - px) / 2) + 1
            end_x = start_x + px - 1
            frames[start_x:end_x,start_x:end_x,l] = rgb_img
        end
        save("public/animation.gif", frames; fps=1)
        animation_url = ""
        sleep(0.1)
        animation_url = "/animation.gif?v=$(Base.time())"
    end
end

function ui()
    [
     h1(class="my-4 text-center", "Brain tumor classifier"),
     h2("Training"),
     Html.div(style="display:flex", 
              [
               Html.div(style="display:block", [
                                                card([
                                                h4("Results"),
                                                     plot(:trace, layout=:layout, style="height:400px;width:800px"),
                                                                                     p("Training accuracy: {{train_acc}}"),
                                                                                     p("Test accuracy: {{test_acc}}"),
                                                    ]),
                                                card(style="display:flex;", [
                                                                                     btn("Train model", @click(:training), loading=:training,color="primary"),
                                                                                     btn("Download model", href="/model.jld2", color="primary", class="q-ml-md"),
                                                                                     btn("Download data", href="/archive.zip", color="primary", class="q-ml-md"),
                                                                                    ]),
                                            card( expansionitem(label="Training log", dense=true, var"dense-toggle"=true, var"expand-separator"=true, var"header-class"="bg-grey-1", p("{{log_str}}"))
                                                    )
                                               ]),
               card([
                    h4("Detecting tumors in an MRI scan"),
                    p("This app trains a convolutional neural network on a dataset of MRI brain scans to detect tumors. The training data consists of 253 images of brains with tumors and 98 images of brains without tumors."),
                    p("To train the network, click the TRAIN button. This button only works when running the app locally in development mode."),
                    p("To test the trained classifier, click one of the images below. You'll see the output of each layer of the network as the image is processed. The final layer output is used to classify the image as having a tumor or not."),
                     h4("Network diagram"),
                     img(style="width:600px;height:200px;",src="/diagram.png"),
                     p("This diagram excludes pooling and normalization layers")])

              ]),
     h2("Testing"),
     Html.div(style="display:flex",
              [
     Html.div(style="width:50%;",
              [
               card([
                                        h4("Images with tumor"),
                                        scrollarea(style=" height:190px",
                                                   [
                                                    Html.div(style="display:flex",
                                                             [
                                                              card(style="background:orange;padding:5px", @recur("img in img_yes"),imageview(style="width:120px;height:120px;cursor:pointer",var":src"="'/yes/'+img", @on(:click,"clicked_img = '/yes/'+img"))),
                                                             ]),
                                                   ])
                                       ]),
    card([
                            h4("Images without tumor"),
                            scrollarea(style="height:190px",
                                       [
                                        Html.div(style="display:flex",
                                                 [
                                                  card(style="background:green;padding:5px",@recur("img in img_no"),imageview(style="width:120px;height:120px;cursor:pointer",var":src"="'/no/'+img", @on(:click,"clicked_img = '/no/'+img"))),
                                                 ]),
                                       ])
                           ]),
    ]),
    Html.div(style="width:50%;height:100%;display:flex", 
             [
                        card([
                              h4("Network propagation"),imageview(style="width:350px;height:350px;",src=:animation_url)]),
                        card([
                              h4("Classification result:"), 
                              "{{label}}",
                              h4("Ground truth:"), 
                              "{{ground_truth}}",
                              """<br/><q-badge style="margin-top:50px" color="red" v-if="classification_error">$(icon(size="15px", "warning")) Classification error!</q-badge>""",

                             ])
             ])
   ]),
   ]
end

@page("/", "app.jl.html")
      end
