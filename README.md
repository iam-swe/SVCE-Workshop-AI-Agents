# 📘 Exam Helper System – Setup Guide

This repository contains the complete code for the Exam Helper System workshop.

Follow the steps below to set up the project on Windows or macOS.

## ⚙️ Prerequisites

- Python 3.11 – 3.13

- Git

- A code editor (VS Code recommended)

Check Python version:

```
python --version
```

### 📦 1. Clone the Repository
```
git clone https://github.com/iam-swe/SVCE-Workshop-AI-Agents.git
cd SVCE-Workshop-AI-Agents
```

### 🚀 2. Install uv (Fast Python package manager)
#### macOS
```
brew install uv
```

OR

#### Windows
```
pip install uv
```

#### Windows (PowerShell)
```
powershell -ExecutionPolicy Bypass -c "irm https://astral.sh/uv/install.ps1 | iex"
```


Restart the terminal after installation.

Verify:

```
uv --version
```


If not recognized:

```
setx PATH "%LOCALAPPDATA%\Programs\uv;%PATH%"
```

Restart terminal again.

### 🐍 3. Use a Compatible Python Version

Some packages may not support very new Python versions.

We pin Python for consistency:

```
uv python install 3.13
uv python pin 3.13
```

### 📁 4. Create Virtual Environment & Install Dependencies

```
uv venv
uv sync
```

### ▶️ 5. Activate the Virtual Environment

#### macOS / Linux
```
source .venv/bin/activate
```

#### Windows – PowerShell
```
.\.venv\Scripts\Activate.ps1
```

If blocked:

```
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

#### Windows – Command Prompt
```
.\.venv\Scripts\activate.bat
``` 

You can also run Python directly without activating:

```
.venv\Scripts\python.exe --version
```

### 🔑 6. Environment Variables Setup

Copy the example env file:

#### macOS
```
cp .env.example .env
```

#### Windows (cmd)
```
copy .env.example .env
```


Fill in the following in the .env file:

```
GOOGLE_API_KEY=your_google_ai_studio_key
FIRECRAWL_API_KEY=your_firecrawl_key
```

#### Get Your API Keys
**🔹 Gemini API Key (Google AI Studio)**

- Go to: https://aistudio.google.com/app/apikey

- Create an API key

- Paste it into .env

**🔹 Firecrawl API Key**

- Go to: https://www.firecrawl.dev

- Generate an API key

- Paste it into .env

### 🧠 7. Running the Exam Helper System

Start the application:
```
python -m app.main
```

### 🖥️ 8. IDE Setup (VS Code Recommended)
To select the virtual environment, press Ctrl/Cmd + Shift + P, select Python: Select Interpreter and pick the one from .venv


## 🛠️ Troubleshooting
❌ uv not found

Restart the terminal or re-add it to PATH.

❌ Python version incompatible

Run:

```
uv python pin 3.13
uv sync
```

❌ PowerShell blocks venv activation

Run:

```
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```