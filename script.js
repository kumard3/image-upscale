// Function to load the ONNX model
async function loadModel() {
  console.log("Loading ONNX model...");
  const session = await ort.InferenceSession.create(
    "./model/Real-ESRGAN_x2plus.onnx"
  );
  console.log("Model loaded.");
  return session;
}

// Function to handle image upload
document.getElementById("image-upload").addEventListener("change", (event) => {
  const file = event.target.files[0];
  if (file) {
    const reader = new FileReader();
    reader.onload = function (e) {
      const img = document.getElementById("original-img");
      img.src = e.target.result;
    };
    reader.readAsDataURL(file);
  }
});

// Function to upscale the image using ONNX model
document.getElementById("upscale-btn").addEventListener("click", async () => {
  const imgElement = document.getElementById("original-img");

  if (!imgElement.src) {
    alert("Please upload an image first.");
    return;
  }

  // Load the model
  const model = await loadModel();

  // Convert the image to a tensor
  const img = new Image();
  img.src = imgElement.src;
  img.onload = async () => {
    const inputTensor = preprocessImage(img);

    // Run the model
    const output = await model.run({ input: inputTensor });

    // Post-process and display the upscaled image
    const upscaledImage = postprocessImage(output.output);
    displayUpscaledImage(upscaledImage);
  };
});

// Preprocess the image for the model
function preprocessImage(img) {
  const canvas = document.createElement("canvas");
  const context = canvas.getContext("2d");

  // Set image dimensions for the model input
  canvas.width = 64;
  canvas.height = 64;
  context.drawImage(img, 0, 0, canvas.width, canvas.height);

  // Get image data and normalize pixel values between 0 and 1
  const imageData = context.getImageData(0, 0, canvas.width, canvas.height);
  const { data } = imageData;

  const input = new Float32Array(1 * 3 * canvas.width * canvas.height);
  let idx = 0;

  for (let i = 0; i < data.length; i += 4) {
    input[idx++] = data[i] / 255.0; // R
    input[idx++] = data[i + 1] / 255.0; // G
    input[idx++] = data[i + 2] / 255.0; // B
  }

  // Return as a tensor with shape [1, 3, height, width]
  return new ort.Tensor("float32", input, [1, 3, canvas.height, canvas.width]);
}

// Post-process the output tensor back into an image
function postprocessImage(outputTensor) {
  const [batch, channels, height, width] = outputTensor.dims;
  const output = outputTensor.data;

  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d");

  const imageData = context.createImageData(width, height);

  let idx = 0;
  for (let i = 0; i < width * height; i++) {
    imageData.data[i * 4] = output[idx++] * 255; // R
    imageData.data[i * 4 + 1] = output[idx++] * 255; // G
    imageData.data[i * 4 + 2] = output[idx++] * 255; // B
    imageData.data[i * 4 + 3] = 255; // A (opaque)
  }

  context.putImageData(imageData, 0, 0);

  return canvas;
}

// Display the upscaled image
function displayUpscaledImage(canvas) {
  const outputCanvas = document.getElementById("upscaled-img");
  const context = outputCanvas.getContext("2d");

  // Resize the canvas to match the upscaled image
  outputCanvas.width = canvas.width;
  outputCanvas.height = canvas.height;

  // Draw the upscaled image on the canvas
  context.drawImage(canvas, 0, 0);
}
