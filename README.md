# 👁️ Eye Mouse Control

<div align="center">

*Look. Control.*

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-latest-FF6F00?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active-00e676?style=for-the-badge)

</div>

---

## 🌟 What is this?

Your eyes become your mouse. The webcam detects your iris movement in real time and moves the cursor across the screen — no keyboard, no mouse, no hands. A short blink clicks. Eyes closed for 5 seconds exits the program.

Beyond control, the system also records **how you look** — generating heatmaps, charts and full metrics about your visual behavior.

---

## ✨ Features

```
👁️  Real-time iris tracking
🖱️  Cursor control through gaze
👆  Click by blinking
🌡️  Live heatmap of viewed areas
📍  Fixation and saccade detection
📊  Auto-generated reports (CSV + PNG + JSON)
🔲  3×3 grid with per-zone statistics
⚙️  Custom calibration for maximum precision
```

---

## 🛠️ Tech Stack

| | Library | Role |
|--|---------|------|
| 🎥 | `OpenCV` | Video capture and image processing |
| 🧠 | `MediaPipe` | Face and iris detection (Face Mesh) |
| 🖱️ | `PyAutoGUI` | Mouse control |
| 📐 | `NumPy` | Math and smoothing |
| 📊 | `Matplotlib` | Charts and visual reports |
| 🖼️ | `mss` | Screen capture for heatmap overlay |
| ⌨️ | `keyboard` | Real-time key detection |

---

## 🚀 Getting Started

**1. Clone the repo**
```bash
git clone https://github.com/dahliaadriana5/Eye-mouse-control.git
cd Eye-mouse-control
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run**
```bash
python eyetracking.py
```

---

## ⌨️ Controls

| Key | Action |
|-----|--------|
| `C` | Start custom calibration |
| `H` | Toggle heatmap overlay |
| `F` | Toggle fixation display |
| `ESC` | Pause tracking |
| `S` | Resume tracking |
| `Q` | Exit + save report |
| 👁️ short blink | Mouse click |
| 👁️ eyes closed 5s | Auto exit |

---

## 📁 Project Structure

```
Eye-mouse-control/
│
├── 🐍 eyetracking.py        — main program
├── 📋 requirements.txt      — dependencies
│
└── 📂 eye_report/           — auto-generated after each session
    ├── gaze_*.csv           — all gaze points recorded
    ├── fixations_*.csv      — detected fixations
    ├── metrics_*.json       — full academic metrics
    ├── report_*.png         — visual session report
    └── screen_heatmap_*.jpg — heatmap overlaid on screen
```

---

## 📊 Auto-Generated Report

After every session the program automatically saves:

- 🗺️ **Heatmap** — where you looked the most
- 📍 **Gaze path** — the journey of your eyes over time
- 📈 **Fixations** — how many, how long, in which zones
- ⚡ **Saccades** — rapid movements between fixations
- ⏱️ **TTFF** — time to first fixation per zone
- 🏆 **Dominant zone** — where your attention stayed the longest

---

<div align="center">

Made with 👁️ and lots of ☕ by **Dalia**

[GitHub](https://github.com/dahliaadriana5)

</div>
