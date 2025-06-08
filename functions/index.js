const functions = require('firebase-functions');
const { spawn } = require('child_process');
const path = require('path');

exports.app = functions.https.onRequest((request, response) => {
  // Spawn Python process
  const process = spawn('python', ['../run.py'], {
    cwd: path.join(__dirname),
  });

  let dataString = '';

  process.stdout.on('data', (data) => {
    dataString += data.toString();
  });

  process.stderr.on('data', (data) => {
    console.error(`Error: ${data}`);
  });

  process.on('close', (code) => {
    response.send(dataString);
  });
}); 