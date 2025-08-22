# Resume AI Telegram Bot

An advanced Telegram bot that generates professional, ATS-optimized PDF resumes using AI models from OpenRouter. The bot analyzes your existing resume against job descriptions and creates perfectly formatted LaTeX-compiled PDFs ready for job applications.

## ✨ Key Features

- 📄 **PDF Resume Generation**: Creates professional LaTeX-compiled PDF resumes
- 🎯 **ATS Optimization**: Optimizes keywords and formatting for Applicant Tracking Systems  
- 🤖 **AI-Powered Analysis**: Uses OpenRouter AI models (Claude, GPT-4, etc.)
- 📋 **Job-Specific Tailoring**: Customizes resumes for specific job descriptions
- 💾 **Persistent Settings**: Remembers your API key and model between sessions
- ✨ **Interactive Menus**: Intuitive button-based navigation
- 🔍 **Smart Model Search**: Real-time AI model discovery and selection
- ✅ **Settings Validation**: Tests configuration with detailed feedback

## 🚀 Latest Features (v3.1)

### PDF Generation System
- **Professional LaTeX Compilation**: Generates publication-quality PDF resumes
- **Multiple Engine Support**: Uses pdfLaTeX with XeLaTeX fallback
- **Error Recovery**: Provides LaTeX code when PDF generation fails
- **Debug Logging**: Comprehensive error reporting for troubleshooting

### Persistent User Settings
- **Session Storage**: API keys and models saved between bot restarts
- **Automatic Backup**: Secure local storage with JSON format
- **Privacy Controls**: Clear settings option for data removal
- **Seamless Experience**: No re-configuration needed

### Enhanced ATS Optimization
- **Keyword Matching**: Aligns resume content with job requirements
- **Structure Optimization**: Reorders sections for maximum relevance
- **Truth Preservation**: Never invents fake experience or achievements
- **Comment Tracking**: Shows exactly what optimizations were made

## 🛠️ Setup

### Prerequisites
- Python 3.8+
- LaTeX distribution (TeX Live, MiKTeX, or MacTeX)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/alwinpaul1/Resume-AI.git
   cd Resume-AI
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install LaTeX (for PDF generation):**
   ```bash
   # macOS
   brew install --cask mactex
   
   # Ubuntu/Debian
   sudo apt-get install texlive-latex-recommended
   
   # Windows
   # Download MiKTeX from https://miktex.org/
   ```

4. **Create a Telegram Bot:**
   - Message [@BotFather](https://t.me/BotFather) on Telegram
   - Use `/newbot` command and follow instructions
   - Save your bot token

5. **Configure environment variables:**
   - Copy `.env.example` to `.env`
   - Add your Telegram bot token:
   ```env
   TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
   ```

### OpenRouter Setup (Done via Bot)
- Sign up at [OpenRouter.ai](https://openrouter.ai/) 
- The bot will guide you through API key setup
- Settings are automatically saved and persist between sessions

## 📱 Usage

### Starting the Bot
```bash
python bot.py
```

### First-Time Setup
1. Find your bot on Telegram and send `/start`
2. Follow the guided setup:
   - **Step 1**: Choose your AI model (Claude, GPT-4, etc.)
   - **Step 2**: Enter your OpenRouter API key
   - **Step 3**: Validate configuration
3. Settings are automatically saved for future use

### Creating Optimized Resumes
1. **Upload Resume**: Send PDF, DOCX, or TXT file
2. **Provide Job Description**: Paste the target job posting
3. **Receive PDF**: Get professional, ATS-optimized PDF resume
4. **Download & Apply**: Ready-to-use resume file

## 🎛️ Interactive Features

### Main Menu
- 📄 **Optimize Resume (PDF)** - Upload resume and generate optimized PDF
- ⚙️ **Settings** - Manage API key and AI model preferences
- 🆘 **Help** - Comprehensive usage guide
- ℹ️ **About** - Bot information and privacy details

### Settings Management
- 🔑 **Set/Update API Key** - Configure OpenRouter credentials
- 🤖 **Select/Change Model** - Browse and select AI models
- ✅ **Validate Settings** - Test configuration with detailed feedback
- 🗑️ **Clear Settings** - Remove stored data (with confirmation)

### Smart Features
- **Model Search**: Type partial names (e.g., "claude", "gpt") to find models
- **Session Persistence**: Settings survive bot restarts
- **Error Recovery**: Fallback options when PDF generation fails
- **Privacy Controls**: Clear data option and transparent storage

## 📄 Supported Input Formats

| Format | Extension | Max Size | Notes |
|--------|-----------|----------|-------|
| PDF | `.pdf` | 20MB | Most common format |
| Word | `.docx` | 20MB | Modern Word documents |
| Text | `.txt` | 20MB | Plain text resumes |

## 🤖 AI Model Options

### Premium Models (Higher Quality)
- **Claude 3.5 Sonnet** - Best for professional writing
- **Claude 3 Opus** - Most capable model
- **GPT-4 Turbo** - Excellent analysis and optimization

### Budget-Friendly Options
- **Claude 3 Haiku** - Fast and cost-effective
- **GPT-3.5 Turbo** - Good quality, lower cost
- **Llama Models** - Open-source alternatives

### Search and Discovery
- Type partial model names to browse options
- Real-time filtering with instant results
- Model descriptions and pricing information

## 🔧 Commands

| Command | Description |
|---------|-------------|
| `/start` | Launch bot with interactive setup |
| `/help` | Display comprehensive help |
| `/analyze` | Quick access to resume optimization |

## 🎯 How It Works

1. **Upload**: Submit your current resume in any supported format
2. **Analyze**: AI extracts and analyzes your experience and skills
3. **Match**: System compares your background with job requirements
4. **Optimize**: Keywords and content are strategically aligned
5. **Generate**: Professional LaTeX code is created and compiled
6. **Deliver**: Receive ATS-optimized PDF ready for applications

## 🔒 Privacy & Security

- **Local Storage**: User settings stored locally, not in cloud
- **Session Only**: Resume content never permanently stored
- **User Control**: Clear settings option available anytime
- **Transparent**: Open about what data is stored and why

## 📈 Version History

- **v3.1** (Current): PDF generation, persistent storage, enhanced ATS optimization
- **v3.0**: LaTeX compilation system, multi-engine support
- **v2.0**: OpenRouter integration, interactive menus, model search
- **v1.0**: Basic resume analysis prototype

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚡ Quick Start Example

```bash
# Clone and setup
git clone https://github.com/alwinpaul1/Resume-AI.git
cd Resume-AI
pip install -r requirements.txt

# Configure bot token in .env
echo "TELEGRAM_BOT_TOKEN=your_token_here" > .env

# Start bot
python bot.py
```

Then on Telegram:
1. Send `/start` to your bot
2. Complete the guided setup
3. Upload your resume and job description  
4. Receive your optimized PDF! 🎉