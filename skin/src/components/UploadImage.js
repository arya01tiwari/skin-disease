import React, { useState } from "react";
import "./UploadImage.css";

const UploadImage = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewImage, setPreviewImage] = useState(null);
  const [prediction, setPrediction] = useState("");
  const [confidence, setConfidence] = useState("");
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    setSelectedFile(file);
    setPreviewImage(URL.createObjectURL(file));
    setPrediction("");
    setConfidence("");
  };

  const handleAnalyze = async () => {
    if (!selectedFile) return;

    const formData = new FormData();
    formData.append("file", selectedFile);
    setIsAnalyzing(true);

    try {
      const response = await fetch("http://localhost:5000/predict", {
        method: "POST",
        body: formData,
      });

      const result = await response.json();
      setPrediction(result.prediction);
      setConfidence(result.confidence.toFixed(2));
    } catch (error) {
      console.error("Error uploading file:", error);
      setPrediction("Error during prediction");
      setConfidence("");
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="upload-container">
      <h2>Skin Disease Detection</h2>

      <input type="file" accept="image/*" onChange={handleFileChange} />

      {previewImage && (
        <div className="preview">
          <img src={previewImage} alt="Preview" />
        </div>
      )}

      <div className="buttons">
        <button onClick={handleAnalyze} disabled={!selectedFile || isAnalyzing}>
          {isAnalyzing ? "Analyzing..." : "Analyze"}
        </button>
      </div>

      {prediction && (
        <div className="result">
          <h3>Prediction: {prediction}</h3>
          <p>Confidence: {confidence}%</p>
        </div>
      )}
    </div>
  );
};

export default UploadImage;
