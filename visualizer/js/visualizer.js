
//import {morph, opponent as op} from "/js/opponent.js";
// called when the client connects
function onConnect() {
  // Once a connection has been made, make a subscription and send a message.
  console.log("Connected to MQTT broker");
  client.subscribe("game/level");
  client.subscribe("player2/score"); // player 2 is ai
  client.subscribe("player1/score"); // player 1 is human
  client.subscribe("ai/activation");
  client.subscribe("depth/feed");
}

// called when the client loses its connection
function onConnectionLost(responseObject) {
  if (responseObject.errorCode !== 0) {
    console.log("onConnectionLost:"+responseObject.errorMessage);
  }
}

// // called when a message arrives
// function onMessageArrived(message) {
//     if(message.destinationName === "ai/activation") {
//         // Save the raw string instead of parsing because we may skip this
//         // frame if we get a new one before the next render - no need to waste cycles
//         last_activations = message.payloadString;
//     } else if(message.destinationName === "game/level") {
//         level = JSON.parse(message.payloadString)["level"];
//         model_initialized = false;
//         init_model(level);
//         console.log("next level");
//         //console.log(op);
//         //morph(op, morphTargets.Frustrated, 1)
//     }
// }
function myMethod(message) {
    //console.log(message.destinationName)
    if(message.destinationName === "game/level") {
        level = JSON.parse(message.payloadString)["level"];
        levelg = level // levelg is global 
        model_initialized = false;
        //init_model(level);
        //console.log("Changing to level:")
        //console.log(level);
        ai_score = 0;
        player_score = 0;
        morphAllZero()
        morphOp("Smug", 0.5)
        // clearInterval(rightInterval0);
        // clearInterval(rightInterval1);
        // clearInterval(leftInterval0);
        // clearInterval(leftInterval1);

        const info_ctx = infoCanvas.getContext("2d");

        switch(level) {
            case 0:
                level = 1
                //levelg = 1
                //console.log('level 0. level set to:')
                //console.log(level)
                info_step = 40;
                // render_info(info_ctx, 0, MAIN_INFO, ADDITIONAL_INFO)
                
                init_model(level);
                break;
            case 1:
                console.log('level 1')
                morphOp("Happy",0.0)
                setTimeout(smugZero, 0)
                right10();
                leftZero();
                setTimeout(rightZero, 1000)
                setTimeout(left10, 2000)
                // rightInterval1 = setInterval(right10, 2910)
                // rightInterval0 = setInterval(rightZero, 3650)
                // leftInterval1 = setInterval(left10, 2750)
                // leftInterval0 = setInterval(leftZero, 3550)

                info_step = 200;
                // render_info(info_ctx, 1, MAIN_INFO, ADDITIONAL_INFO)
                init_model(level);

                break;
            case 2:
                setTimeout(smug07, 500)
                morphOp("Happy",0.2)
                setTimeout(smugZero, 5000)
                //setTimeout(mad10, 5000)
                // rightInterval1 = setInterval(right10, 3709)
                // rightInterval0 = setInterval(rightZero, 4550)
                // leftInterval1 = setInterval(left10, 3610)
                // leftInterval0 = setInterval(leftZero, 4200)
                
                left10();
                rightZero();
                setTimeout(leftZero, 700)
                setTimeout(left10, 2000)

                info_step = 200;
                render_info(info_ctx, 2, MAIN_INFO, ADDITIONAL_INFO)
                
                init_model(level);
                break;
            case 3:
                setTimeout(smug09, 5000)
                morphOp("Happy",0.4)
                morphOp("Mad", 0.3)
                //setTimeout(smugZero, 5000)
                // //setTimeout(mad10, 5000)
                // rightInterval1 = setInterval(right10, 5505)
                // //rightInterval0 = setInterval(rightZero, 5600)
                // rightInterval1 = setInterval(left10, 4910)
                //rightInterval0 = setInterval(leftZero, 6250)
                
                right10();
                leftZero();
                setTimeout(rightZero, 1500)
                setTimeout(left10, 2500)
                setTimeout(leftZero, 3500)
                setTimeout(right10, 4000)

                info_step = 200;
                // render_info(info_ctx, 3, MAIN_INFO, ADDITIONAL_INFO)
                init_model(level);
                break;
        }

    } else if(message.destinationName === "ai/activation") {
        last_activations = message.payloadString;
        //morphOp("Sad",0.5)
    } else if(message.destinationName === "player2/score") {
        newScore = JSON.parse(message.payloadString)["score"]
        //console.log("score received")
        if (ai_score !== newScore) {
            //console.log("ai scored")
            ai_score = newScore;
            switch(levelg) {
                case 1:
                    morphOp("Happy", 1.0)
                    setTimeout(happyZero, 1000)
                    left10();
                    rightZero();
                    setTimeout(leftZero, 500)
                    setTimeout(right10, 2000)
                    break;
                case 2:
                    morphOp("Smug", 1.0)
                    setTimeout(smug07, 1000)
                    right10();
                    leftZero();
                    setTimeout(rightZero, 700)
                    setTimeout(left10, 2500)
                    break;
                case 3:
                    morphOp("Smug", 1.0)
                    setTimeout(smug09, 1000)
                    morphOp("Happy", 0.5)
                    setTimeout(happyZero, 1000)
                    right10();
                    leftZero();
                    setTimeout(rightZero, 2200)
                    setTimeout(left10, 3400)
                    break;
            }
        }
    } else if(message.destinationName === "player1/score") {
        newScore = JSON.parse(message.payloadString)["score"]
        //console.log("score received")
        if (player_score !== newScore) {
            //console.log("player scored")
            player_score = newScore;
            switch(levelg) {
                case 1:
                    morphOp("Sad", 1.0)
                    morphOp("Smug", 0.0)
                    setTimeout(sadZero, 1000)
                    break;
                case 2:
                    morphOp("Confused", 1.0)
                    morphOp("Smug", 0.0)
                    setTimeout(confusedZero, 1000)
                    setTimeout(smug07, 1000)
                    break;
                case 3:
                    morphOp("Confused", 1.0)
                    morphOp("Mad", 0.8)
                    morphOp("Smug", 0.0)
                    setTimeout(confusedZero, 1000)
                    setTimeout(mad03, 1000)
                    setTimeout(smug09, 1000)
                    break;
            }
        }
    } else if(message.destinationName === "depth/feed") {
        //console.log('received depth image')
        depthFeedStr = (JSON.parse(message.payloadString)["feed"])
        //console.log(depthFeedStr)
        //decoded = atob(rawfeed)

        
    } 
}

function smugZero() {
    morphOp("Smug", 0.0)
}
function confusedZero() {
    morphOp("Confused", 0.0)
}
function madZero() {
    morphOp("Mad", 0.0)
}
function mad03() {
    morphOp("Mad", 0.3)
}

function smug04() {
    morphOp("Smug", 0.4)
}
function smug07() {
    morphOp("Smug", 0.7)
}
function smug09() {
    morphOp("Smug", 0.9)
}
function mad10() {
    morphOp("Mad", 0.5)
    morphOp("Sad", 0.2)
}

function happyZero() {
    morphOp("Happy", 0.0)
}
function sadZero() {
    morphOp("Sad", 0.0)
}

function morphAllZero() {
    morphOp("Happy",0.2)
    morphOp("Frustrated",0.0)
    morphOp("Mad",0.0)
    morphOp("Smug",0.0)
    morphOp("Sad",0.0)
    morphOp("Right",0.0)
    morphOp("Left",0.0)
    morphOp("Confused",0.0)
}

function right10() {
    morphOp("Right", Math.random())
}
function rightZero() {
    morphOp("Right", 0.0)
}
function left10() {
    morphOp("Left", Math.random())
}
function leftZero() {
    morphOp("Left", 0.0)
}


function render_game(ctx, frame, image_upscale = 2) {
    // For rendering the image of the Pong game environment

    frame = scale(frame, 255);
    image_upscale = canvas.width / 225
    frameCanvas.width = frame_width
    frameCanvas.height = frame_height

    // Get the canvas
    const frame_ctx = frameCanvas.getContext("2d");
    const imageData = frame_ctx.getImageData(0, 0, frame_width, frame_height);
    const frameData = imageData.data;
    
    for(let i = 0; i < frame.length; i++) {
        idx = i * 4;
        // convert our -1, 0 and 1 values into a full RGB image
        frameData[idx] = 200-frame[i]; // Red
        frameData[idx+1] = 200-frame[i]; // Green
        frameData[idx+2] = 200-frame[i]; // Blue
        frameData[idx+3] = 255; // Alpha
    }

    // Draw the image
    frame_ctx.putImageData(imageData, 0, 0);
    ctx.drawImage(frameCanvas, img_x, img_y, img_w, img_h); 

}
function render_depth_feed(ctx, image_upscale = 3.6) {
    var image = new Image();
    image.src = 'data:image/jpg;base64,' + depthFeedStr 
    // We received the image from the depth camera via MQTT

    // You have to use image.onload so that you dont try to use the image before it exists
    image.onload = function() {
        // scale image_upscale to fit to the left of the pong screen
        // the left edge of yellow box / this image width
        image_upscale = (img_x - (0.1*img_w)) /image.width;
        ctx.drawImage(image, 0, (img_y + (.5 * img_h)) -( 0.5 * image.height*image_upscale), image.width * image_upscale, image.height * image_upscale)

        // Text of labels that say "YOU" and 'AI INPUT" below the depth image and Pong image
        ctx.textAlign = "center";
        ctx.font = FEED_LABELS_FONT;
        ctx.fillStyle = "#333333"
        ctx.fillText("^ YOU ^", image.width * image_upscale * 0.5, img_y + (1.1*img_h));
        ctx.fillText("^ AI INPUT ^", img_x + 0.5*img_w, img_y + (1.1*img_h));
        ctx.font = "22px monospace";
        ctx.fillText("^ NEURAL NETWORK - AI 'THINKING' ^", 0.5*canvas_width, img_y - (0.08*img_h));
        ctx.fillText("^ AI DECISION ^", 0.5*canvas_width, OUTPUT_LAYER_Y * canvas_height * 1.7);
    }
    

}

function render_weight_image(ctx, hl_activations, image_upscale = 5) {
    // Get the neuron with the strongest activity
    const top_neuron = argmax(hl_activations);
    console.log('INSIDE RENDER WEIGHT IMAGE')
    // Select its weight with respect to each input pixel
    let frame = get_weight_map(hidden_weights, top_neuron)
    max_weight = max(frame)
    frame = scale(frame, 127/max_weight);
    frame = add(frame, 127)
    //Render game frame
    const frame_width = 160; // Base state dimension
    const frame_height = 192; 

    weightImageCanvas.width = frame_width
    weightImageCanvas.height = frame_height

    const frame_ctx = weightImageCanvas.getContext("2d");
    const imageData = frame_ctx.getImageData(0, 0, frame_width, frame_height);
    const frameData = imageData.data;
    for(let i = 0; i < frame.length; i++) {
        idx = i * 4;
        frameData[idx] = frame[i]; // Red
        frameData[idx+1] = frame[i]; // Green
        frameData[idx+2] = frame[i]; // Blue
        frameData[idx+3] = Math.max((frame[i] - 64), 0); // Alpha
    }

    frame_ctx.putImageData(imageData, 0, 0);

    img_w = frame_width * image_upscale;
    img_h = frame_height * image_upscale;
    img_x = (canvas_width * 3/4) - (img_w/2) - (img_w/10);
    img_y = canvas_height - (img_h) - (img_h/10);

    ctx.drawImage(weightImageCanvas, img_x, img_y, img_w, img_h); 
}

function render_tick(ctx, render_frame, state_frame, hl_activations, ol_activations, d_ctx) {
    // Clean slate before redraw
    ctx.clearRect(0, 0, canvas_width, canvas_height); 

    // Update rendered probabilities
    percent_prob = scale(ol_activations, 100)
    up_prob = parseFloat(percent_prob[0]).toFixed(2)+"%"
    down_prob = parseFloat(percent_prob[1]).toFixed(2)+"%"
    none_prob = parseFloat(percent_prob[2]).toFixed(2)+"%"

    if (percent_prob[0] > percent_prob[1] & percent_prob[0] > percent_prob[2]) {labelChosen = 0}
    else if (percent_prob[1] > percent_prob[2]) {labelChosen = 1}
    else {labelChosen = 2}


    if (levelg == 1) {
        // Draws a yellow rectangle to highlight the game view camera
        ctx.beginPath();
        ctx.strokeStyle = "yellow";
        ctx.lineWidth = 15;
        ctx.fillStyle = "yellow";
        ctx.strokeRect(img_x - (0.05 * img_w), img_y - (0.05 * img_h), 1.1 * img_w, 1.1 * img_h);
    } else if (levelg == 2) {
        // Draws a yellow rectangle to highlight the nodes of the neural network
        ctx.beginPath();
        ctx.strokeStyle = "yellow";
        ctx.lineWidth = 15;
        ctx.strokeRect(0, (img_y - (0.2 * img_h)) - (VERTICAL_SPREAD) - (VERTICAL_SPREAD * 1.1), canvas_width, 2 * VERTICAL_SPREAD * 1.1);
    } else if (levelg == 3) {
        // Draws a yellow rectangle to highlight the output of the neural network
        ctx.beginPath();
        ctx.strokeStyle = "yellow";
        ctx.lineWidth = 15;
        ctx.strokeRect(canvas_width * (1/5), OUTPUT_LAYER_Y * canvas_height /6, canvas_width * (3/5), canvas_height * OUTPUT_LAYER_Y * 1.2);
    }
    let t = timer("render_game");
    // Render game frame

    render_depth_feed(d_ctx)
    render_game(ctx, render_frame);
    t.stop()

    t = timer("render_weight_image");
    // Render game frame
    t.stop()
    /*************************** */
    // Render image weights
    t = timer("image_activations");
    const image_activations = is_firing(state_frame);
    const hw_activations = is_weight_active(hidden_weights, image_activations, copy(significant_hw));
    t.stop()

    t = timer("render_hidden_weights");
    render_weights(ctx, pixel_pos, hidden_pos, rescaled_hw, hw_activations, 1)
    t.stop()
    /******************************************** */
    // Re-compute hidden activations for rendering
    t = timer("hl_activations");
    ow_activations = is_weight_active(output_weights, hl_activations, copy(significant_ow))
    t.stop()

    // Render hidden weights
    t = timer("render_output_weights");
    render_weights(ctx, hidden_pos, out_pos, render_rescale(output_weights), ow_activations, 2)
    t.stop()
    /********************************************* */
    // Re-compute middle activations for rendering
    t = timer("il_activations");
    iw_activations = is_weight_active(hidden_weights, hl_activations, copy(significant_hw))
    t.stop()

    // Render middle weights
    t = timer("render_middle_weights");
    render_weights(ctx, hidden_pos, hidden_pos, rescaled_hw, iw_activations, 3)
    t.stop()
    /************************************************** */

    t = timer("render_layers");
     // render node circles
    render_layer(ctx, render_rescale(hidden_biases, 1), 0,
         canvas_width, (img_y - (0.2*img_h)) - (VERTICAL_SPREAD), NEURON_SIZE, hl_activations)
    render_layer(ctx, render_rescale(output_biases, 1), 0,
        canvas_width, OUTPUT_LAYER_Y * canvas_height, NEURON_SIZE, null, OUTPUT_LABELS, ol_activations)
    t.stop()

    // Create dynamic labels for inference confidence
    up_pos = out_pos[0]
    down_pos = out_pos[1]
    none_pos = out_pos[2]
    ctx.font = TITLE_FONT;
    ctx.textAlign = "center";
    
    // These lines show the actual probability value of the output.
    // This was too busy for exhibit and didn't add value.
    // ctx.fillText(up_prob, up_pos[0], up_pos[1]-40);
    // ctx.fillText(down_prob, down_pos[0], down_pos[1]-40);
    // ctx.fillText(none_prob, none_pos[0], none_pos[1]-40);
}
function render_loop() {
    const info_ctx = infoCanvas.getContext("2d");
    if (last_activations && last_activations != last_rendered_activations) {
        const ctx = canvas.getContext("2d");
        const d_ctx = d_canvas.getContext("2d");

        const [state_frame, hl_activations, ol_activations] = JSON.parse(last_activations);
        render_tick(ctx, state_frame, state_frame, hl_activations, ol_activations, d_ctx);
        last_rendered_activations = last_activations;
    }

    // Slide and fade in the info label for our highlighted section. 
    // A new section gets highlighted per level.
    if (info_step > 0) {
        info_ctx.globalAlpha = 1 - (info_step / 200)
        render_info(info_ctx, levelg, MAIN_INFO, ADDITIONAL_INFO, info_step/2)
        info_step = info_step - 1;
    }

    // This moves the opponent face to make room for info labels
    // It gradually moves the face instead of jumping. The movement
    // will hopefully help draw eyes to the info.
    newOpPos = opPosArr[levelg];
    if (Math.abs(oldOpPos - newOpPos) <= 0.01) {
        oldOpPos = newOpPos;
        emptyAnimateFunction(oldOpPos);
    } else if (oldOpPos != newOpPos && newOpPos > oldOpPos) {
        oldOpPos = oldOpPos + 0.005;
        emptyAnimateFunction(oldOpPos);
    } else if (oldOpPos != newOpPos) {
        oldOpPos = oldOpPos - 0.005;
        emptyAnimateFunction(oldOpPos);
    }
    
    // Break out of render loop if we receive a new model to render
    if(model_initialized) {
        requestAnimationFrame(render_loop);
    }
}

function init_model(level) {
    /*
    This initialization runs any time we receive a new model structure.
    It renders its weights and precomputes any convenient values for later use in realtime rendering.
    */
    if(!initialized) {
        // Busy wait if we receive a new model before the normal init is run.
        console.log("!initialized")
        init_model(structure);
    } else {
        const ctx = canvas.getContext("2d");
        structure = null;
        // Switch between the three models (1 per level)
        switch(level) {
            case 1:
                structure = easy_model;
                break;
            case 2:
                structure = medium_model;
                break;
            case 3:
                structure = hard_model;
                break;
        }
        // Parse out the structure
        hidden_weights = structure[0];
        hidden_biases = structure[1];
        output_weights = structure[2];
        output_biases = structure[3];

        // Store image-related rendering measurements
        const pixel_dims = [frame_width, frame_height]
        img_coords = [img_x, img_y]
        img_size = [img_w, img_h]
        top_corner = [img_x, img_y]
        pixel_step = img_size[1] / pixel_dims[1]

        // Calculate positions of individual pixels, for use as a "layer" to connect edges
        pixel_pos = []
        for(let y = 0; y < pixel_dims[1]; y++) {
            for(let x = 0; x < pixel_dims[0]; x++) {
                x_pos = (x * pixel_step) + top_corner[0] + (pixel_step / 2)
                y_pos = (y * pixel_step) + top_corner[1] + (pixel_step / 2)
                pixel_pos.push([x_pos, y_pos])
            }
        }

        rescaled_hw = render_rescale(hidden_weights);

        // Save the "significant" weights: we only render the 2% most important hidden weights and
        // the 30% most important output weights so the screen doesn't get crowded
        significant_hw = is_significant(hidden_weights, 0.02);
        significant_ow = is_significant(output_weights, 0.3)

        // Determine appropriate base size for a neuron based on minimum allowable padding for a specific layer
        NEURON_SIZE = (canvas_height - (hidden_biases.length * MIN_PADDING)) / (hidden_biases.length)
        NEURON_SIZE = 0.8;

        // Render neuron nodes, saving calculated positions for weight rendering
        hidden_pos = render_layer(ctx, render_rescale(hidden_biases, 1), 0,
            canvas_width, (img_y - (0.2*img_h)) - (VERTICAL_SPREAD), NEURON_SIZE)

        out_pos = render_layer(ctx, render_rescale(output_biases, 1), 0,
            canvas_width, OUTPUT_LAYER_Y * canvas_height, NEURON_SIZE, null, OUTPUT_LABELS)

        model_initialized = true;
        requestAnimationFrame(render_loop);
    }
}

function init() {
    /*
    Basic housekeeping initialization. Makes sure the canvases we need
    exist, and do any precomputing or sizing that aren't model dependent.
    */
    // Create a client instance
    client = new Paho.MQTT.Client("localhost", 1883, "visualizer_module");

    // set callback handlers
    client.onConnectionLost = onConnectionLost;
    client.onMessageArrived = myMethod; // onMessageArrived;

    // connect the client
    client.connect({onSuccess:onConnect});

    // Canvas for the nodes and output of neural network
    canvas = document.getElementById("visualizer");
    const ctx = canvas.getContext("2d");

    // canvas for our depth camera image
    d_canvas = document.getElementById("depth");
    const d_ctx = d_canvas.getContext("2d");

    // canvas for our info labels for highlighted section
    infoCanvas = document.getElementById("info");
    const info_ctx = infoCanvas.getContext("2d");
    
    // Positioning the canvases
    canvas.width = document.body.clientWidth*(.6) -(document.body.clientWidth/60); 
    canvas.height = document.body.clientHeight - 2*(document.body.clientHeight/60);
    
    canvas.style.left = (document.body.clientWidth/60)+'px';
    canvas.style.top = (document.body.clientWidth/120) + 'px';
    canvas.style.position = 'absolute';
    console.log("canvas.left is ")
    console.log(canvas.left)

    image_upscale = canvas.width / 225
    img_w = frame_width * image_upscale;
    img_h = frame_height * image_upscale;
    console.log("image_upscale:")
    console.log(image_upscale);

    // a factor for spreading the nodes of the neural network
    VERTICAL_SPREAD = (document.body.clientHeight/8)
    console.log("VERTICAL_SPREAD:")
    console.log(VERTICAL_SPREAD);

    // Positioning the depth image canvas
    d_canvas.width = canvas.width; 
    d_canvas.height = canvas.height; 
    d_canvas.style.left = canvas.style.left;
    d_canvas.style.top = canvas.style.top; 
    d_canvas.style.position = 'absolute';

    d_canvas_width = d_canvas.width
    d_canvas_height = d_canvas.height
    
    // Positioning the depth image canvas
    infoCanvas.width = document.body.clientWidth - canvas.width;
    infoCanvas.height = canvas.height;
    
    infoCanvas.style.left = (document.body.clientWidth/60) + canvas.width + 'px';
    infoCanvas.style.top = canvas.style.top;
    infoCanvas.style.position = 'absolute';

    infoCanvas_width = infoCanvas.width
    infoCanvas_height = infoCanvas.height

    // Save canvas dimensions
    canvas_width = canvas.width;
    canvas_height = canvas.height;

    img_x = (canvas_width * 3/4) - (img_w/2) - (img_w/10);
    img_y = canvas_height - (img_h) - (img_h/10);
    
    // We will update this with game state pixels and embed it on the visualizer canvas
    frameCanvas = document.createElement('canvas');
    // This one will hold a model weight image overlay to see what the network is picking up on
    weightImageCanvas = document.createElement('canvas');

    // A listener to appropriately resize our canvases when the window size changes
    window.addEventListener('resize', onWindowResizeV, false);

    initialized = true;
}
 

function onWindowResizeV() {
    // Function to resize our canvases when the window resizes. 
    // This is also triggered when you go full screen, as that changes the size a little

    console.log("ON WINDOW RESIZE")
    canvas.width = document.body.clientWidth*(.6) -(document.body.clientWidth/60); 
    canvas.height = document.body.clientHeight - 2*(document.body.clientHeight/60);
    
    canvas_width = canvas.width;
    canvas_height = canvas.height;

    canvas.style.left = (document.body.clientWidth/60)+'px'; 
    canvas.style.top = (document.body.clientWidth/120) + 'px';
    
    image_upscale = canvas.width / 225
    img_w = frame_width * image_upscale;
    img_h = frame_height * image_upscale;
    img_x = (canvas_width * 3/4) - (img_w/2) - (img_w/10);
    img_y = canvas_height - (img_h) - (img_h/10);
    
    d_canvas.width = canvas.width;
    d_canvas.height = canvas.height;
    d_canvas.style.left = canvas.style.left;
    d_canvas.style.top = canvas.style.top;
    
    d_canvas_width = d_canvas.width
    d_canvas_height = d_canvas.height

    VERTICAL_SPREAD = (document.body.clientHeight/8)
    
    infoCanvas.width = document.body.clientWidth - canvas.width;
    infoCanvas.height = canvas.height;
    
    infoCanvas.style.left = (document.body.clientWidth/60) + canvas.width + 'px'
    infoCanvas.style.top = canvas.style.top;
    infoCanvas.style.position = 'absolute';

    infoCanvas_width = infoCanvas.width
    infoCanvas_height = infoCanvas.height
    
}

// This function gets set from opponent.js
// We have to do this since opponent.js is a module, visualizer.js is not. 
function morphOp(value){
    console.log("empty morphOp function")};

// This function gets set from opponent.js
function emptyAnimateFunction(value){
    console.log("empty emptyAnimateFunction function")};

// Intentionally global. Canvas is for base drawing. Frame canvas holds game frame images.
var canvas = null;
var d_canvas = null;
var frameCanvas = null;
var weightImageCanvas = null;
var infoCanvas = null;

// Track initialization status
var initialized = false;
var model_initialized = false;

// Structure data
var hidden_weights = null;
var hidden_biases = null;
var output_weights = null;
var output_biases = null;

// Canvas rendering locations for nodes
var pixel_pos = null;
var hidden_pos = null;
var out_pos = null;

// Weights that are important enough to render
var significant_hw = null;
var significant_ow = null;

// Precompute normalized hidden weights for nicer rendering
var rescaled_hw = null;

// Rendering config
var NEURON_SIZE = null;
var MIN_PADDING = 3.8; // was 3. Used at line 212
var HIDDEN_LAYER_Y = 0.475; // no longer used
var OUTPUT_LAYER_Y = 0.08; // was 0.1
var OUTPUT_LABELS = ["LEFT", "RIGHT", "NONE"] // the 3 options for the AI, labeled at the top of the canvas

// The text of the informational labels. A heading big label and some extra details
var MAIN_INFO = ["<- what the AI sees", "<- the AI's neural network\n'thinking'", "<- the AI deciding to go\nleft, right, or stay still"]
var ADDITIONAL_INFO = ["The AI sees a flat version of the game\nThe pixels in this image activate the AI's\nNeural Network", "Each circle is a node in the Neural Network\nThey are like neurons in the human brain\nThe blue lit nodes are activated", "Which nodes are activated determines the chosen action\nIf many nodes with strong connections to 'Left'\nare activated, the AI goes left"]

var image_upscale = 4;
var frame_width = 192 / 2; // Base state dimension, scaled down by two
var frame_height = 160 / 2;
var img_w = frame_width * image_upscale;
var img_h = frame_height * image_upscale;
var img_x = null; // These are computed elsewhere based on canvas dimensions
var img_y = null;
var canvas_width = null;
var canvas_height = null;
var d_canvas_width = null;
var d_canvas_height = null;
var infoCanvas_width = null;
var infoCanvas_height = null;

var last_activations = null;
var last_rendered_activations = null;

let rightInterval1;
let rightInterval0;
let leftInterval1;
let leftInterval0;

let textInterval;

var levelg = 1;
var ai_score = 0;
var player_score = 0;

var labelChosen = 0;

// Stored positions that the opponent face goes to as levels change
var oldOpPos = -0.7;
var opPosArr = [0.1, 0.1, -0.71, -0.71]
var newOpPos = -0.7;
var oldVisW = 0.6;
var visWArr = [0.3, 0.6, 0.6, 0.6]
var newVisW = 0.6;
var visResizeCounter = 1;

// A value that we use for moving the info labels in
var info_step = 40;

var depthFeedStr = "";

window.onload = init