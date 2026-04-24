import React, { useEffect, useMemo, useRef, useState } from 'react';
import axios from 'axios';
import { BrowserRouter, Link, Route, Routes, useLocation } from 'react-router-dom';
import './App.css';

const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:5000';

const DETECTION_OPTIONS = [
  {
    key: 'image',
    title: 'Image Detection',
    description: 'Upload a face image and classify whether it is real or deepfake.',
    route: '/image',
    endpoint: '/detect-image',
    accept: 'image/*'
  },
  {
    key: 'audio',
    title: 'Audio Detection',
    description: 'Upload a voice clip and detect synthetic or manipulated speech.',
    route: '/audio',
    endpoint: '/detect-audio',
    accept: 'audio/*,.wav,.mp3,.m4a,.flac'
  },
  {
    key: 'video',
    title: 'Video Detection',
    description: 'Upload a video and run temporal deepfake detection with confidence scoring.',
    route: '/video',
    endpoint: '/detect-video',
    accept: 'video/*'
  }
];

function App() {
  return (
    <BrowserRouter>
      <div className="app-shell">
        <AnimatedBackground />
        <Navigation />
        <main className="main-content">
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/image" element={<DetectionPage mode="image" />} />
            <Route path="/audio" element={<DetectionPage mode="audio" />} />
            <Route path="/video" element={<DetectionPage mode="video" />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}

function Navigation() {
  const location = useLocation();
  const showRouteLinks = location.pathname !== '/';

  return (
    <header className="top-nav glass">
      <Link className="brand" to="/">
        <span className="brand-badge" />
        <span className="brand-wordmark">Arti-fact</span>
      </Link>
      {showRouteLinks ? (
        <nav className="route-links">
          {DETECTION_OPTIONS.map((option) => (
            <Link
              key={option.key}
              to={option.route}
              className={`route-link ${location.pathname === option.route ? 'active' : ''}`}
            >
              {option.title}
            </Link>
          ))}
        </nav>
      ) : null}
    </header>
  );
}

function HomePage() {
  return (
    <section className="home-wrapper">
      <div className="hero glass">
        <h1>Choose your detection mode</h1>
        <p>
          Select a detector, upload media, and receive an instant prediction with confidence score.
          Image, audio, and video routes are connected to backend models.
        </p>
      </div>

      <div className="mode-grid">
        {DETECTION_OPTIONS.map((option) => (
          <Link to={option.route} className="mode-card glass" key={option.key}>
            <div className="card-shine" />
            <h3>{option.title}</h3>
            <p>{option.description}</p>
            <span className="mode-cta">Open</span>
          </Link>
        ))}
      </div>
    </section>
  );
}

function DetectionPage({ mode }) {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState('');
  const [result, setResult] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef(null);

  const option = useMemo(() => DETECTION_OPTIONS.find((item) => item.key === mode), [mode]);

  useEffect(() => {
    setFile(null);
    setErrorMsg('');
    setResult(null);
    setIsDragging(false);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  }, [mode]);

  const isFileAccepted = (selectedFile) => {
    const kind = option?.key;
    const mime = (selectedFile?.type || '').toLowerCase();
    const name = (selectedFile?.name || '').toLowerCase();

    if (kind === 'image') {
      return mime.startsWith('image/') || /\.(jpg|jpeg|png|webp|bmp)$/i.test(name);
    }
    if (kind === 'audio') {
      return mime.startsWith('audio/') || /\.(wav|mp3|m4a|flac|ogg)$/i.test(name);
    }
    if (kind === 'video') {
      return mime.startsWith('video/') || /\.(mp4|mov|avi|mkv|webm)$/i.test(name);
    }
    return false;
  };

  const handleFileSelection = (selectedFile) => {
    if (!selectedFile) {
      setFile(null);
      return;
    }

    if (!isFileAccepted(selectedFile)) {
      const prettyMode = option?.title || 'selected detection mode';
      const warning = `Invalid file type for ${prettyMode}. Please upload a matching file format.`;
      window.alert(warning);
      setErrorMsg(warning);
      setFile(null);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
      return;
    }

    setErrorMsg('');
    setFile(selectedFile);
  };

  const onAnalyze = async () => {
    if (!file) {
      setErrorMsg('Please choose a file before analyzing.');
      return;
    }

    setLoading(true);
    setErrorMsg('');
    setResult(null);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await axios.post(`${API_BASE_URL}${option.endpoint}`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });

      setResult(response.data);
    } catch (error) {
      const apiMsg =
        error?.response?.data?.error ||
        'Detection failed. Verify backend server is running and models are loaded.';
      setErrorMsg(apiMsg);
    } finally {
      setLoading(false);
    }
  };

  return (
    <section className="detector-page">
      <div className="detector-panel glass">
        <h2>{option.title}</h2>
        <p>{option.description}</p>

        <label className="uploader">
          <input
            ref={fileInputRef}
            className="file-input-hidden"
            type="file"
            accept={option.accept}
            onChange={(event) => handleFileSelection(event.target.files?.[0] || null)}
          />
          <div
            className={`drop-zone ${isDragging ? 'drag-active' : ''}`}
            role="button"
            tabIndex={0}
            onClick={() => fileInputRef.current?.click()}
            onKeyDown={(event) => {
              if (event.key === 'Enter' || event.key === ' ') {
                event.preventDefault();
                fileInputRef.current?.click();
              }
            }}
            onDragOver={(event) => {
              event.preventDefault();
              setIsDragging(true);
            }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={(event) => {
              event.preventDefault();
              setIsDragging(false);
              handleFileSelection(event.dataTransfer?.files?.[0] || null);
            }}
          >
            <p className="drop-zone-title">Drop your {mode} file here</p>
            <p className="drop-zone-subtitle">or</p>
            <button
              type="button"
              className="browse-btn"
              onClick={(event) => {
                event.stopPropagation();
                fileInputRef.current?.click();
              }}
            >
              Choose File
            </button>
            <span>{file ? file.name : `No ${mode} file selected yet`}</span>
          </div>
        </label>

        <button className="primary-btn" onClick={onAnalyze} disabled={loading}>
          {loading ? 'Analyzing...' : 'Analyze'}
        </button>

        {errorMsg ? <p className="error-text">{errorMsg}</p> : null}

        {result ? <ResultCard result={result} /> : null}
      </div>
    </section>
  );
}

function ResultCard({ result }) {
  const confidence = Number(result?.confidence || 0);
  const percent = Math.max(0, Math.min(100, confidence * 100));
  const isFake = String(result?.result || '').toUpperCase() === 'FAKE';

  return (
    <div className="result-card">
      <div className={`verdict ${isFake ? 'fake' : 'real'}`}>{isFake ? 'DEEPFAKE' : 'AUTHENTIC'}</div>
      <p className="result-label">Model output: {result?.result}</p>
      <div className="confidence-track">
        <div className="confidence-fill" style={{ width: `${percent}%` }} />
      </div>
      <p className="confidence-text">Confidence score: {percent.toFixed(2)}%</p>
    </div>
  );
}

function AnimatedBackground() {
  return (
    <div className="background-layer" aria-hidden="true">
      <span className="orb orb-a" />
      <span className="orb orb-b" />
      <span className="orb orb-c" />
      <span className="grid-fade" />
    </div>
  );
}

export default App;