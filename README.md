# 🏎️ F1 Predictor (Full-Stack)

AI-powered Grand Prix prediction web app combining machine learning with interactive data visualization. Built with a modern full-stack architecture: **FastAPI + LightGBM backend** and **React + TailwindCSS frontend**.

---

## 🔥 Key Features

- **Race Selection**: Choose from 2021–2025 seasons and all available races  
- **AI Predictions**: Predict Grand Prix winners using real F1 data and ML models  
- **Confidence Scores**: Visualize podium predictions with win probabilities  
- **Team Insights**: Displays team logos, team colors, and driver-team affiliations  
- **Feature Analysis**: Shows key features that contributed to predictions  
- **Responsive UI**: Modern design, optimized for both desktop and mobile  
- **Error Handling**: Graceful fallback with mock predictions when real data is missing  

---

## 🧠 Machine Learning Model

The backend uses a trained **LightGBM** model for binary classification of race winners.

### Features Used
- `grid_position`
- `qualifying_time`
- `qualifying_performance`
- `grid_position_score`
- `team_consistency`
- `avg_stint_length`
- `total_pit_stops`
- `total_laps`

### Techniques
- Feature engineering from FastF1 data (telemetry, qualifying, strategy)
- Historical labeling of winners (2021–2024)
- LightGBM model with probability normalization
- Mock fallback predictions for unavailable races

---

## 🛠️ Tech Stack

| Layer        | Technology                  |
|--------------|-----------------------------|
| Frontend     | React 19, Vite, TailwindCSS |
| Backend      | FastAPI, Python             |
| ML Model     | LightGBM                    |
| Data Source  | FastF1                      |
| Utilities    | Pandas, NumPy, Joblib       |

---

## 🧩 Project Structure

```
f1-predictor/
├── backend/
│   ├── main.py
│   ├── routes/
│   ├── services/
│   ├── models/
│   ├── utils/
│   └── requirements.txt
├── frontend/
│   ├── components/
│   ├── pages/
│   ├── api/
│   └── index.css
├── model/
│   └── lgbm_model.pkl
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites
- Node.js v18+
- Python 3.10+
- Backend runs at `http://localhost:8000`

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

Visit: `http://localhost:5173`

---

## 🔗 API Endpoints

| Endpoint                              | Purpose                         |
|--------------------------------------|---------------------------------|
| `/api/v1/predict/{season}/{round}`   | Predict winner & podium         |
| `/api/v1/features/{season}/{round}`  | Engineered driver features      |
| `/api/v1/seasons/{season}/races`     | Available races in that season |
| `/api/v1/model/info`                 | Metadata about model features   |
| `/health`                            | Backend connection check        |

---

## ✨ Styling

- Team-based dynamic backgrounds (colors/logos)
- Smooth animations on dropdowns and cards
- TailwindCSS responsive layout
- Shadow + scale transitions on hover/focus

---

## 🧪 Development

```bash
npm run dev        # Start frontend
uvicorn main:app   # Start backend
npm run build      # Build frontend for production
```

---

## 🐛 Troubleshooting

### Common Issues

- **Prediction 500 errors**: LightGBM may not be installed correctly or race data may be missing
- **No data returned**: FastF1 may not support the selected race yet
- **CORS problems**: Check if FastAPI includes CORS middleware
- **Git errors**: If you're facing Git issues, try:
  ```bash
  git status
  git remote -v
  git pull origin main
  git config --global init.defaultBranch main
  ```

---

## 🤝 Contributing

- Keep logic modular and typed  
- Use TailwindCSS for all UI components  
- Test across screen sizes and races  
- Write fallback logic for all async operations  

---

## 📄 License

MIT License

---

## ⚠️ Note

> This is a single-repo full-stack project (frontend + backend + model). If running locally, make sure both frontend and backend are started separately. Some races may not have data yet; fallback predictions will be used.
