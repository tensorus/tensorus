const path = require('path');
const { uploadTensor, loadTensor } = require('./index');

const filePath = process.argv[2];
if (!filePath) {
  console.error("Usage: node test.js <path-to-tensor-json-file>");
  process.exit(1);
}

const destPath = uploadTensor(filePath);
console.log("Uploaded file stored at:", destPath);

const tensorData = loadTensor(path.basename(filePath));
console.log("Loaded tensor data:", tensorData);
