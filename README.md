# 🏎️ F1 Predictor (Full-Stack)

AI-powered Grand Prix prediction web app combining machine learning with interactive data visualization. Built with a modern full-stack architecture: FastAPI + LightGBM backend and React + TailwindCSS frontend.

## 🔥 Key Features

* **Race Selection**: Choose from 2021–2025 seasons and all available races
* **AI Predictions**: Predict Grand Prix winners using real F1 data and ML models
* **Confidence Scores**: Visualize podium predictions with win probabilities
* **Team Insights**: Displays team logos, team colors, and driver-team affiliations
* **Feature Analysis**: Shows key features that contributed to predictions
* **Responsive UI**: Modern design, optimized for both desktop and mobile

---

## 🧠 Machine Learning Model

The backend uses a trained LightGBM model (Gradient Boosted Decision Trees) for race winner prediction.

### Features Used

* **Qualifying Time**
* **Grid Position**
* **Team Consistency Score**
* **Pit Strategy**: Avg stint length, total pit stops
* **Race Metadata**: Total laps, derived features

### Techniques

* Feature engineering from FastF1 data (qualifying, pit stops, compounds, weather)
* Labeling race winners from historical data
* Normalized probability outputs
* Robust error handling and fallback to mock predictions

The model is trained using historical F1 data (2021–2024), leveraging FastF1 for rich telemetry and strategy details.

---

## 🛠️ Tech Stack

### Frontend

* **React 19**
* **TailwindCSS** (modern, utility-first styling)
* **Vite** (blazing fast dev experience)
* **Axios** (API communication)

### Backend

* **FastAPI** (Python web framework)
* **LightGBM** (machine learning model)
* **FastF1** (race telemetry and timing data)
* **Pandas/Numpy/Joblib** (data processing & serialization)

---

## 🧩 Project Structure

```bash
f1-predictor/
├── backend/               # FastAPI ML backend
│   ├── main.py            # App entrypoint
│   ├── routes/            # API routes
│   ├── services/          # FastF1 logic
│   ├── models/            # ML model logic
│   ├── utils/             # Preprocessing
│   └── requirements.txt
├── frontend/              # React + Tailwind client
│   ├── components/        # Header, RacePicker, PredictionCard, etc.
│   ├── pages/
│   ├── api/
│   └── index.css
├── README.md              # You're here
└── .gitignore
```

---

## 🚀 Getting Started

### Prerequisites

* Node.js 18+
* Python 3.10+

### Backend Setup

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

Visit the app at: `http://localhost:5173`

---

## 🔗 API Endpoints

* `GET /api/v1/predict/{season}/{round}` – Race prediction
* `GET /api/v1/features/{season}/{round}` – Engineered feature matrix
* `GET /api/v1/seasons/{season}/races` – Available races
* `GET /health` – Backend health
* `GET /api/v1/model/info` – Model metadata

---

## ✨ Styling

* Clean, modern layout with TailwindCSS
* F1-themed color palette
* Team branding with logos and primary team colors
* Prediction cards and tables with hover animations

---

## 🧪 Development

### Scripts

```bash
npm run dev         # Start frontend
npm run build       # Build frontend for prod
uvicorn main:app    # Run FastAPI server
```

### Extending

* Add new features in `frontend/src/components/`
* Extend model input/output in `backend/models/predictor.py`
* Add new preprocessing logic in `utils/`

---

## 🐛 Troubleshooting

* Backend 500 Errors? → Check model file or LightGBM installation
* Frontend race list empty? → Backend not connected or CORS error
* Predictions fail? → Race might not have real data; fallback to mock

---

## 🤝 Contributing

1. Fork the repo and clone it
2. Use consistent Tailwind utility classes
3. Handle API errors gracefully
4. Write readable, modular components
5. Keep backend and frontend interfaces in sync

---

## 📄 License

MIT

---

## 📸 Screenshots

> Add `public/screenshots/` and embed them here with markdown!

---

## 🙌 Credits

* [FastF1](https://theoehrly.github.io/Fast-F1/) for F1 data
* [LightGBM](https://lightgbm.readthedocs.io/) for model training
* [React](https://react.dev/), [FastAPI](https://fastapi.tiangolo.com/), [TailwindCSS](https://tailwindcss.com/)
