/**
 * Import function triggers from their respective submodules:
 *
 * const {onCall} = require("firebase-functions/v2/https");
 * const {onDocumentWritten} = require("firebase-functions/v2/firestore");
 *
 * See a full list of supported triggers at https://firebase.google.com/docs/functions
 */

const {onRequest} = require("firebase-functions/v2/https");
const logger = require("firebase-functions/logger");
const {spawn} = require("child_process");
const path = require("path");

// Create and deploy your first functions
// https://firebase.google.com/docs/functions/get-started

exports.app = onRequest((request, response) => {
  // Log the request
  logger.info("Processing request", {url: request.url});

  // Spawn Python process
  const process = spawn("python", ["../run.py"], {
    cwd: path.join(__dirname),
  });

  let dataString = "";

  process.stdout.on("data", (data) => {
    dataString += data.toString();
    logger.info("Python output:", {data: data.toString()});
  });

  process.stderr.on("data", (data) => {
    logger.error("Python error:", {error: data.toString()});
  });

  process.on("close", (code) => {
    logger.info("Python process closed", {code});
    response.send(dataString);
  });
});
