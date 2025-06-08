// Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app";
import { getAnalytics } from "firebase/analytics";

// Your web app's Firebase configuration
const firebaseConfig = {
  apiKey: "AIzaSyAps0-1yspdOrAJyOJuZ5wKU2zlLzwz-LY",
  authDomain: "init-12295.firebaseapp.com",
  projectId: "init-12295",
  storageBucket: "init-12295.firebasestorage.app",
  messagingSenderId: "50130683900",
  appId: "1:50130683900:web:96a2becd1d7b24927e1479",
  measurementId: "G-2DLJ39N7RV"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const analytics = getAnalytics(app);

export { app, analytics }; 