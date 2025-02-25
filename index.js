const fs = require('fs');
const path = require('path');

const storageDir = path.join(__dirname, 'storage');
if (!fs.existsSync(storageDir)) {
  fs.mkdirSync(storageDir);
}

/**
 * Uploads a file by copying it into the storage directory.
 * @param {string} filePath - The path of the file to upload.
 * @returns {string} - The destination path where the file is stored.
 */
function uploadTensor(filePath) {
  const fileName = path.basename(filePath);
  const destPath = path.join(storageDir, fileName);
  fs.copyFileSync(filePath, destPath);
  return destPath;
}

/**
 * Loads tensor data from a file stored in the storage directory.
 * Assumes the file contains JSON-encoded tensor data.
 * @param {string} fileName - The name of the file in the storage directory.
 * @returns {any} - The parsed tensor data.
 */
function loadTensor(fileName) {
  const filePath = path.join(storageDir, fileName);
  const data = fs.readFileSync(filePath, 'utf8');
  return JSON.parse(data);
}

module.exports = { uploadTensor, loadTensor };
