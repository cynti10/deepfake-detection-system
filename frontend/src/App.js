import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [type, setType] = useState('image'); // image, video, audio
  const [errorMsg, setErrorMsg] = useState('');

  const handleUpload = async () => {
    setErrorMsg('');
    setResult(null);
    if (!file) {
      setErrorMsg('Please select a file first.');
      return;
    }
    const formData = new FormData();
    formData.append('file', file);

    try {
      // Backend URL (WSL localhost mapped to Windows)
      const endpoint = `http://localhost:5000/detect-${type}`;
      const response = await axios.post(endpoint, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setResult(response.data);
      setErrorMsg('');
    } catch (error) {
      console.error(error);
      const apiMsg = error?.response?.data?.error || 'Error detecting file. Ensure the corresponding model is loaded on the backend.';
      setErrorMsg(apiMsg);
    }
  };

  return (
    <div className="App" style={{ padding: '50px' }}>
      <h1>Deepfake Detection System</h1>
      
      <div className="controls">
        <select onChange={(e) => setType(e.target.value)} value={type}>
          <option value="image">Image Detection</option>
          <option value="video">Video Detection</option>
          <option value="audio">Audio Detection</option>
        </select>
        
        <br /><br />
        <input type="file" onChange={(e) => setFile(e.target.files[0])} />
        <br /><br />
        <button onClick={handleUpload} style={{ padding: '10px 20px' }}>
          Analyze
        </button>
      </div>

      {result && (
        <div style={{ marginTop: '20px', border: '1px solid #ccc', padding: '20px' }}>
          <h2>Result: {result.result}</h2>
          <p>Confidence: {(result.confidence * 100).toFixed(2)}%</p>
        </div>
      )}

      {errorMsg && (
        <div style={{ marginTop: '20px', color: 'red' }}>
          <strong>{errorMsg}</strong>
        </div>
      )}
    </div>
  );
}

export default App;