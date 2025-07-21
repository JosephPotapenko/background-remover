const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
let model;

// Load ONNX model
async function loadModel() {
  model = await ort.InferenceSession.create('./u2net.onnx');
  console.log("Model loaded");
}
loadModel();

document.getElementById('imageInput').addEventListener('change', async function (e) {
  const file = e.target.files[0];
  const img = new Image();
  img.onload = async () => {
    canvas.width = img.width;
    canvas.height = img.height;
    ctx.drawImage(img, 0, 0);
    const imageData = ctx.getImageData(0, 0, img.width, img.height);

    // Preprocess image to 320x320 RGB Float32
    const resized = await resizeImage(img, 320, 320);
    const inputTensor = imageToTensor(resized);

    const feeds = { 'input': inputTensor };
    const results = await model.run(feeds);
    const mask = results.output.data;

    const processed = applyMask(imageData, mask, img.width, img.height);
    ctx.putImageData(processed, 0, 0);
  };
  img.src = URL.createObjectURL(file);
});

function imageToTensor(img) {
  const canvasTmp = document.createElement('canvas');
  canvasTmp.width = 320;
  canvasTmp.height = 320;
  const ctxTmp = canvasTmp.getContext('2d');
  ctxTmp.drawImage(img, 0, 0, 320, 320);

  const imgData = ctxTmp.getImageData(0, 0, 320, 320).data;
  const float32 = new Float32Array(1 * 3 * 320 * 320);
  for (let i = 0; i < 320 * 320; i++) {
    float32[i] = imgData[i * 4] / 255.0;
    float32[i + 320 * 320] = imgData[i * 4 + 1] / 255.0;
    float32[i + 2 * 320 * 320] = imgData[i * 4 + 2] / 255.0;
  }
  return new ort.Tensor('float32', float32, [1, 3, 320, 320]);
}

function applyMask(imageData, mask, width, height) {
  const output = ctx.createImageData(width, height);
  const scaleX = 320 / width;
  const scaleY = 320 / height;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = (y * width + x) * 4;
      const mx = Math.floor(x * scaleX);
      const my = Math.floor(y * scaleY);
      const maskValue = mask[my * 320 + mx];

      output.data[i] = imageData.data[i];
      output.data[i + 1] = imageData.data[i + 1];
      output.data[i + 2] = imageData.data[i + 2];
      output.data[i + 3] = Math.floor(maskValue * 255); // apply alpha
    }
  }
  return output;
}

function resizeImage(img, width, height) {
  return new Promise(resolve => {
    const tmpCanvas = document.createElement('canvas');
    tmpCanvas.width = width;
    tmpCanvas.height = height;
    const tmpCtx = tmpCanvas.getContext('2d');
    tmpCtx.drawImage(img, 0, 0, width, height);
    const image = new Image();
    image.onload = () => resolve(image);
    image.src = tmpCanvas.toDataURL();
  });
}

function download() {
  const link = document.createElement("a");
  link.download = "background_removed.png";
  link.href = canvas.toDataURL("image/png");
  link.click();
}
