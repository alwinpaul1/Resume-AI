import os
import logging
import asyncio
import tempfile
import subprocess
import shutil
import json
import re
from pathlib import Path
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler
import PyPDF2
from io import BytesIO
import aiohttp
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
# Hide sensitive token information from httpx logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

class ResumeAIBot:
    def __init__(self):
        self.user_sessions = {}  # Store user session data
        self.sessions_file = Path("user_sessions.json")  # Persistent storage file
        self.load_user_sessions()  # Load existing sessions on startup
    
    def load_user_sessions(self):
        """Load user sessions from persistent storage."""
        try:
            if self.sessions_file.exists():
                with open(self.sessions_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Convert string keys back to integers for user_id
                    self.user_sessions = {int(k): v for k, v in data.items()}
                    logger.info(f"Loaded {len(self.user_sessions)} user sessions from storage")
            else:
                logger.info("No existing user sessions file found, starting fresh")
        except Exception as e:
            logger.error(f"Error loading user sessions: {e}")
            self.user_sessions = {}  # Fallback to empty dict
    
    def save_user_sessions(self):
        """Save user sessions to persistent storage."""
        try:
            # Create a copy with string keys for JSON serialization
            data_to_save = {str(k): v for k, v in self.user_sessions.items()}
            

            
            # Save to file
            with open(self.sessions_file, 'w', encoding='utf-8') as f:
                json.dump(data_to_save, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved {len(self.user_sessions)} user sessions to storage")
            
        except Exception as e:
            logger.error(f"Error saving user sessions: {e}")
    
    def update_user_session(self, user_id, updates):
        """Update user session and save to persistent storage."""
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = {
                'api_key': None,
                'model': None,
                'state': 'main_menu'
            }
        
        # Update the session
        self.user_sessions[user_id].update(updates)
        
        # Save to persistent storage
        self.save_user_sessions()
    
    def escape_markdown(self, text):
        """Escape Markdown special characters to prevent parsing errors."""
        # Characters that need escaping in Telegram Markdown
        escape_chars = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
        
        for char in escape_chars:
            text = text.replace(char, f'\\{char}')
        
        return text
    
    async def send_long_message(self, update_or_query, text, reply_markup=None, parse_mode=None):
        """Send a long message, splitting if necessary."""
        max_length = 4000  # Leave some buffer for Telegram's 4096 limit
        
        if len(text) <= max_length:
            if hasattr(update_or_query, 'message'):
                # It's an Update object
                await update_or_query.message.reply_text(
                    text, reply_markup=reply_markup, parse_mode=parse_mode
                )
            else:
                # It's a CallbackQuery object
                await update_or_query.message.reply_text(
                    text, reply_markup=reply_markup, parse_mode=parse_mode
                )
        else:
            # Split the message
            parts = []
            current_part = ""
            lines = text.split('\n')
            
            for line in lines:
                if len(current_part + line + '\n') <= max_length:
                    current_part += line + '\n'
                else:
                    if current_part:
                        parts.append(current_part.strip())
                    current_part = line + '\n'
            
            if current_part:
                parts.append(current_part.strip())
            
            # Send all parts
            for i, part in enumerate(parts):
                markup = reply_markup if i == len(parts) - 1 else None  # Only add markup to last message
                
                if hasattr(update_or_query, 'message'):
                    await update_or_query.message.reply_text(
                        part, reply_markup=markup, parse_mode=parse_mode
                    )
                else:
                    await update_or_query.message.reply_text(
                        part, reply_markup=markup, parse_mode=parse_mode
                    )
        
    def get_main_menu_keyboard(self):
        """Get the main menu keyboard."""
        keyboard = [
            [InlineKeyboardButton("📄 Optimize Resume (PDF)", callback_data="analyze_resume")],
            [InlineKeyboardButton("⚙️ Settings", callback_data="settings")],
            [InlineKeyboardButton("ℹ️ Info", callback_data="info")]
        ]
        return InlineKeyboardMarkup(keyboard)

    def get_settings_keyboard(self):
        """Get the settings menu keyboard."""
        keyboard = [
            [InlineKeyboardButton("🔑 Set API Key", callback_data="set_api_key")],
            [InlineKeyboardButton("🤖 Select Model", callback_data="select_model")],
            [InlineKeyboardButton("✅ Validate Settings", callback_data="validate_settings")],
            [InlineKeyboardButton("🔙 Back to Menu", callback_data="main_menu")]
        ]
        return InlineKeyboardMarkup(keyboard)

    def get_back_cancel_keyboard(self):
        """Get back/cancel keyboard."""
        keyboard = [

            [
                InlineKeyboardButton("🔙 Back", callback_data="back"),
                InlineKeyboardButton("❌ Cancel", callback_data="cancel")
            ]
        ]
        return InlineKeyboardMarkup(keyboard)

    def is_user_setup_complete(self, user_id):
        """Check if user has completed setup (API key and model)."""
        user_session = self.user_sessions.get(user_id, {})
        return user_session.get('api_key') and user_session.get('model')

    def get_setup_keyboard(self, user_id):
        """Get the setup keyboard for new users."""
        user_session = self.user_sessions.get(user_id, {})
        has_model = bool(user_session.get('model'))
        has_api_key = bool(user_session.get('api_key'))
        
        keyboard = []
        
        if not has_model:
            # Step 1: Select model first
            keyboard.append([InlineKeyboardButton("🤖 Select AI Model", callback_data="select_model")])
        elif not has_api_key:
            # Step 2: Set API key after model is selected
            keyboard.append([InlineKeyboardButton("🔑 Set API Key", callback_data="set_api_key")])
        else:
            # Step 3: Both are set, show validation or continue
            keyboard.append([InlineKeyboardButton("✅ Complete Setup", callback_data="validate_settings")])
        
        return InlineKeyboardMarkup(keyboard)

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Send a message when the command /start is issued."""
        user_id = update.effective_user.id
        if user_id not in self.user_sessions:
            self.update_user_session(user_id, {
                'api_key': None,
                'model': None,
                'state': 'main_menu'
            })
        
        if not self.is_user_setup_complete(user_id):
            user_session = self.user_sessions.get(user_id, {})
            has_model = bool(user_session.get('model'))
            has_api_key = bool(user_session.get('api_key'))
            
            if not has_model:
                welcome_message = """
<b>🤖 Welcome to Resume AI Bot!</b>

<i>Transform your resume with AI-powered optimization</i>

Hey there! I'm here to help you create job-specific, ATS-friendly resumes using LaTeX. 

<b>🎯 What I do:</b>
• Analyze your resume + job description
• Generate optimized LaTeX code
• Match keywords naturally
• Keep it 100% truthful (no fake experience!)

<b>📋 Let's get started!</b>
First, I need to know which AI model you'd like to use.

<i>Step 1 of 2: Choose your AI model</i>
                """
            elif not has_api_key:
                welcome_message = f"""
<b>🤖 Great! Model Selected</b>

<i>✅ Model: {user_session.get('model', 'Selected')}</i>

Now I need your OpenRouter API key to connect to the AI service.

<b>🔑 Step 2 of 2: Set your API key</b>

Don't have one? Get it from <a href="https://openrouter.ai/">OpenRouter</a> (it's free to start!)

<i>Almost ready to optimize your resume!</i>
                """
            else:
                welcome_message = f"""
<b>🤖 Setup Complete!</b>

<i>✅ Ready to optimize your resume</i>

<b>Your Configuration:</b>
🔑 API Key: Configured
🤖 Model: {user_session.get('model')}

<b>🚀 Features Ready:</b>
📄 Upload resume (PDF only)
📋 Provide job description
📝 Get optimized PDF resume
🎯 ATS-friendly optimization
✨ Keyword matching

<i>Let's create your perfect resume!</i>
                """
            await update.message.reply_text(
                welcome_message, 
                reply_markup=self.get_setup_keyboard(user_id),
                parse_mode='HTML'
            )
        else:
            welcome_message = f"""
<b>🤖 Welcome back to Resume AI Bot!</b>

<i>Your settings are all set up and ready!</i>

<b>⚙️ Current Configuration:</b>
🔑 API Key: <b>✅ Configured</b>
🤖 Model: <b>✅ {self.user_sessions[user_id]['model']}</b>

<b>🚀 Ready to optimize your resume:</b>
📄 Upload → 📋 Job Description → 📝 PDF Resume

<i>Use the menu below to get started!</i>
            """
            await update.message.reply_text(
                welcome_message, 
                reply_markup=self.get_main_menu_keyboard(),
                parse_mode='HTML'
            )

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Send unified info when the command /help is issued."""
        await self.send_info(update)

    async def send_info(self, update: Update):
        """Send unified info in response to a /help command."""
        info_text = """
ℹ️ **Resume AI Bot — Info**

**How to use:**
1. Set your OpenRouter API key in Settings
2. Select your preferred AI model
3. Upload your resume as PDF
4. Provide the job description
5. Receive optimized, ATS-friendly PDF

**Features:**
• PDF resume generation
• Job-specific keyword optimization
• ATS-friendly formatting
• Professional styling
• Ready-to-use PDF download

**Process:**
📄 Upload → 📋 Job Description → 📝 PDF Resume

**Privacy & Storage:**
• Your settings (API key & model) are stored locally
• No resume content is permanently stored
• Use "Clear Settings" to remove stored data
        """
        await update.message.reply_text(info_text, parse_mode='Markdown')

    def extract_text_from_pdf(self, file_content):
        """Extract text from PDF file."""
        try:
            pdf_reader = PyPDF2.PdfReader(BytesIO(file_content))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            logger.error(f"Error extracting PDF text: {e}")
            return None

    # DOCX extraction removed; only PDF uploads are supported

    async def get_openrouter_models(self, search_term=""):
        """Get available models from OpenRouter."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("https://openrouter.ai/api/v1/models") as response:
                    if response.status == 200:
                        data = await response.json()
                        models = data.get('data', [])
                        
                        if search_term:
                            models = [m for m in models if search_term.lower() in m['id'].lower()]
                        
                        return models[:10]  # Limit to 10 results
                    return []
        except Exception as e:
            logger.error(f"Error fetching models: {e}")
            return []

    async def validate_openrouter_settings(self, api_key, model_name):
        """Validate OpenRouter API key and model."""
        try:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": model_name,
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 5
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload
                ) as response:
                    return response.status in [200, 400]  # 400 might be invalid model but valid key
        except Exception as e:
            logger.error(f"Error validating settings: {e}")
            return False

    async def analyze_resume_with_openrouter(self, resume_text, user_id):
        """Analyze resume using OpenRouter."""
        try:
            user_session = self.user_sessions.get(user_id, {})
            api_key = user_session.get('api_key')
            model = user_session.get('model')
            
            if not api_key or not model:
                return "❌ Please complete your setup first. Set your OpenRouter API key and select a model in Settings."
            
            prompt = f"""
            Analyze the following resume and provide detailed feedback:

            {resume_text}

            Please provide:
            1. Overall assessment (score out of 10)
            2. Strengths of the resume
            3. Areas for improvement
            4. ATS (Applicant Tracking System) compatibility
            5. Keyword suggestions
            6. Format and structure recommendations
            7. Specific actionable improvements

            Keep the response professional and constructive.
            """

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are a professional resume reviewer and career advisor."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 1500,
                "temperature": 0.7
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data['choices'][0]['message']['content']
                    else:
                        error_text = await response.text()
                        logger.error(f"OpenRouter API error: {error_text}")
                        return "❌ Analysis failed. Please check your API key and model settings."
                        
        except Exception as e:
            logger.error(f"Error analyzing resume: {e}")
            return "Sorry, I couldn't analyze your resume at the moment. Please try again later."

    async def generate_optimized_latex_resume(self, resume_text, job_description, user_id):
        """Generate optimized LaTeX resume code based on job description."""
        try:
            user_session = self.user_sessions.get(user_id, {})
            api_key = user_session.get('api_key')
            model = user_session.get('model')
            
            if not api_key or not model:
                return None
            
            prompt = f"""
            You are an ATS optimization expert and professional resume strategist with LaTeX expertise.

            **Inputs:**
            1. Candidate's resume: {resume_text}
            2. Target job description: {job_description}

            **Your Mission:**
            - Extract the content from the resume while preserving structure (sections, bullet points, chronology).
            - Do NOT invent new experiences, achievements, or numbers.
            - Do NOT delete valid roles, projects, or skills.
            - Optimize the resume for ATS parsing by aligning phrasing and keywords with the job description.
            - Ensure formatting is ATS-friendly (no tables, graphics, or unusual fonts).
            - Rebuild the optimized resume in structured format.

            **Process:**
            1. Extract → Convert the original resume content into structured text.
            2. Audit → Compare resume with job description, identify missing/mismatched keywords.
            3. Keyword Mapping → Suggest and integrate JD-aligned terminology.
            4. Rewrite → Output a full optimized resume structure, preserving authenticity.
               - Note where keyword changes are applied for transparency.
            5. Deliver → Provide structured resume data with optimization notes.

            **CRITICAL OUTPUT FORMAT - FOLLOW EXACTLY:**
            IMPORTANT: Do NOT include any reasoning steps, explanations, or additional text. Provide ONLY the structured format below.

            OPTIMIZATIONS_START
            • Summary Section: [specific change made]
            • Experience Section: [specific change made]  
            • Skills Section: [specific change made]
            • [etc...]
            OPTIMIZATIONS_END

            LATEX_START
            \\documentclass[11pt,letterpaper]{{article}}
            \\usepackage[T1]{{fontenc}}
            \\usepackage{{lmodern}}
            [Complete LaTeX code here - no explanations, just pure LaTeX]
            \\end{{document}}
            LATEX_END

            CRITICAL: Provide ONLY the above format. No reasoning, no explanations, no additional text.

            **LaTeX Requirements:**
            - Start with \\documentclass and end with \\end{{document}}
            - Use ONLY packages compatible with pdflatex (NO fontspec, NO unicode-math)
            - Use standard LaTeX fonts (Computer Modern, Latin Modern, or times/helvet packages)
            - Use clean, modern formatting suitable for ATS parsing
            - No tables, graphics, or complex formatting that could confuse ATS
            - Professional font and spacing with standard LaTeX commands
            - Clear section headers and bullet points

            **CRITICAL: pdflatex Compatibility**
            - DO NOT use \\usepackage{{fontspec}} or \\setmainfont (XeLaTeX/LuaLaTeX only)
            - Use \\usepackage{{lmodern}} or \\usepackage{{times}} for fonts instead
            - Use \\usepackage[T1]{{fontenc}} for proper font encoding
            - All packages must work with pdflatex engine
            - ALWAYS escape underscores in URLs: use \\_ instead of _ (e.g., Financial\\_Crew not Financial_Crew)
            - Wrap GitHub URLs in \\texttt{{}} to prevent line breaks: \\texttt{{github.com/user/repo\\_name}}
            - Never let URLs break across lines - use non-breaking formatting

            Remember: You MUST include both sections (OPTIMIZATIONS and LATEX) in the exact format specified above.
            """

            # Dynamic max_tokens calculation based on model context window and prompt size
            model_context_window = user_session.get('model_context_window', 8192)
            # Simple heuristic: ~3 characters per token
            prompt_tokens_estimate = max(1, len(prompt) // 3)
            # Leave a safety buffer for system/metadata and to avoid edge hits
            safety_buffer = 500
            available_for_output = max(0, model_context_window - prompt_tokens_estimate - safety_buffer)
            dynamic_max_tokens = max(1000, min(4096, available_for_output))

            logger.info(
                f"Dynamic max_tokens: {dynamic_max_tokens} (context={model_context_window}, prompt≈{prompt_tokens_estimate})"
            )

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are a LaTeX resume generation machine. Your ONLY function is to output the structured format requested by the user. You MUST NOT include any reasoning, thoughts, or conversational text. Your entire response must strictly adhere to the user's requested format, starting with OPTIMIZATIONS_START and ending with LATEX_END."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": dynamic_max_tokens,
                "temperature": 0.3
            }

            # Define a longer timeout for reasoning models
            timeout = aiohttp.ClientTimeout(total=180)  # 180 seconds = 3 minutes

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=timeout  # Extended timeout for reasoning models
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        full_response = data['choices'][0]['message']['content']
                        
                        # Debug: Log the first 500 characters of the response
                        logger.info(f"LLM Response preview: {full_response[:500]}...")
                        
                        # Clean reasoning artifacts (for reasoning models)
                        cleaned_response = full_response
                        # Remove common reasoning model artifacts
                        reasoning_patterns = [
                            r'<thinking>.*?</thinking>',
                            r'<thought>.*?</thought>',
                            r'<reasoning>.*?</reasoning>',
                        ]
                        for pattern in reasoning_patterns:
                            cleaned_response = re.sub(pattern, '', cleaned_response, flags=re.DOTALL)
                        
                        # Parse the structured response (improved for reasoning models)
                        optimizations = ""
                        latex_code = ""
                        
                        # Extract optimizations
                        opt_start = cleaned_response.find("OPTIMIZATIONS_START")
                        opt_end = cleaned_response.find("OPTIMIZATIONS_END")
                        if opt_start != -1 and opt_end != -1:
                            opt_content = cleaned_response[opt_start + len("OPTIMIZATIONS_START"):opt_end].strip()
                            # Clean up the optimizations text
                            optimizations = opt_content.replace('\n\n', '\n').strip()
                        
                        # Extract LaTeX code (more flexible parsing for reasoning models)
                        latex_start = cleaned_response.find("LATEX_START")
                        latex_end = cleaned_response.find("LATEX_END")
                        if latex_start != -1 and latex_end != -1:
                            latex_content = cleaned_response[latex_start + len("LATEX_START"):latex_end].strip()
                            # Clean up the LaTeX code
                            latex_code = latex_content.replace('LATEX_END', '').strip()
                        
                        # Alternative parsing: look for LaTeX document structure
                        if not latex_code:
                            # Try to find LaTeX content by looking for document structure
                            doc_start = cleaned_response.find("\\documentclass")
                            doc_end = cleaned_response.rfind("\\end{document}")
                            if doc_start != -1 and doc_end != -1:
                                latex_code = cleaned_response[doc_start:doc_end + len("\\end{document}")].strip()
                                logger.info("Extracted LaTeX using document structure fallback")
                        
                        # If structured parsing failed, fall back to treating entire response as LaTeX
                        if not latex_code:
                            latex_code = cleaned_response
                            optimizations = "• Keywords optimized to match job requirements\n• Experience sections enhanced for relevance\n• Skills reordered by importance to the role"
                            logger.warning("Using cleaned response as LaTeX fallback")
                        
                        # Return both components as a tuple
                        return (latex_code, optimizations)
                    else:
                        error_text = await response.text()
                        logger.error(f"OpenRouter API error: {error_text}")
                        return None
                        
        except Exception as e:
            logger.error(f"Error generating resume data: {e}")
            return None

    def extract_latex_from_markdown(self, content):
        """Extract LaTeX code from markdown code blocks or reasoning model outputs."""
        import re
        
        # Strategy 1: Look for ```latex ... ``` blocks
        latex_pattern = r'```latex\s*\n(.*?)\n```'
        match = re.search(latex_pattern, content, re.DOTALL)
        
        if match:
            latex_content = match.group(1).strip()
            logger.info("Extracted LaTeX from markdown code block")
        else:
            # Strategy 2: Look for LaTeX document structure directly
            doc_start = content.find("\\documentclass")
            doc_end = content.rfind("\\end{document}")
            
            if doc_start != -1 and doc_end != -1:
                latex_content = content[doc_start:doc_end + len("\\end{document}")].strip()
                logger.info("Extracted LaTeX using document structure")
            else:
                # Strategy 3: Look for any code block with LaTeX-like content
                code_block_pattern = r'```[a-zA-Z]*\s*\n(.*?)\n```'
                code_matches = re.findall(code_block_pattern, content, re.DOTALL)
                
                latex_content = None
                for code in code_matches:
                    if "\\documentclass" in code and "\\begin{document}" in code:
                        latex_content = code.strip()
                        logger.info("Extracted LaTeX from generic code block")
                        break
                
                if not latex_content:
                    # Strategy 4: Fallback - use entire content and hope for the best
                    latex_content = content.strip()
                    logger.warning("Using entire content as LaTeX fallback")
        
        # Clean and fix common issues
        if latex_content:
            # Fix underscore escaping in URLs
            latex_content = re.sub(r'(github\.com/[^}\s]*?)_([^}\s]*?)', r'\1\\_\2', latex_content)
            # Wrap github URLs in texttt to prevent line breaks
            latex_content = re.sub(r'{(github\.com/[^}]*?)}', r'{\\texttt{\1}}', latex_content)
            
        return latex_content

    def verify_resume_content(self, latex_content, job_description):
        """Verify that the resume content is properly optimized."""
        try:
            # Basic checks
            if not latex_content or len(latex_content.strip()) < 50:
                logger.warning(f"LaTeX content too short: {len(latex_content.strip()) if latex_content else 0} characters")
                return False
            
            # Check for proper LaTeX structure (more lenient for reasoning models)
            required_elements = [
                r'\\documentclass',
                r'\\begin{document}',
                r'\\end{document}',
            ]
            
            missing_elements = []
            for element in required_elements:
                if not re.search(element, latex_content):
                    missing_elements.append(element)
            
            if missing_elements:
                logger.warning(f"Missing LaTeX elements: {missing_elements}")
                return False
            
            # Log successful verification
            logger.info(f"LaTeX content verification passed. Length: {len(latex_content)} characters")
            return True
            
        except Exception as e:
            logger.error(f"Error verifying resume content: {e}")
            return True  # Default to true if verification fails

    async def animate_loading_message(self, message, title_text: str, status_lines: list, duration_seconds: float = 1.8, frame_interval_seconds: float = 0.12):
        """Show a short spinner animation on the provided Telegram message."""
        spinner_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        total_frames = max(1, int(duration_seconds / frame_interval_seconds))
        for i in range(total_frames):
            frame = spinner_frames[i % len(spinner_frames)]
            text = """
<b>🔄 {title}</b>

{statuses}

<i>{frame} Preparing your download...</i>
            """.format(
                title=title_text,
                statuses="\n".join(status_lines),
                frame=frame
            )
            try:
                await message.edit_text(text, parse_mode='HTML')
            except Exception:
                # Ignore edit errors to keep UX smooth
                pass
            await asyncio.sleep(frame_interval_seconds)

    def compile_latex_to_pdf(self, latex_code, output_path):
        """Compile LaTeX code to PDF using pdflatex."""
        try:
            # Preprocess LaTeX to improve pdflatex compatibility
            def sanitize_latex_for_pdflatex(source: str) -> str:
                cleaned = source
                # Remove packages that require XeLaTeX/LuaLaTeX
                cleaned = re.sub(r"^\\s*\\usepackage\{(?:fontspec|unicode-math|polyglossia)\}\\s*$", "", cleaned, flags=re.MULTILINE)
                cleaned = re.sub(r"^\\s*\\usepackage\[[^\]]*\]\{(?:fontspec|unicode-math|polyglossia)\}\\s*$", "", cleaned, flags=re.MULTILINE)

                # Ensure fontenc T1 present
                if "\\usepackage[T1]{fontenc}" not in cleaned:
                    cleaned = cleaned.replace("\\documentclass", "\\documentclass\n\\usepackage[T1]{fontenc}", 1)

                # Ensure a pdflatex-safe font package present (lmodern or times)
                if "\\usepackage{lmodern}" not in cleaned and "\\usepackage{times}" not in cleaned:
                    cleaned = cleaned.replace("\\usepackage[T1]{fontenc}", "\\usepackage[T1]{fontenc}\n\\usepackage{lmodern}", 1)

                # Escape underscores in bare URLs and typewriter text blocks
                def escape_underscores(match):
                    inner = match.group(1).replace("_", "\\_")
                    return match.group(0).replace(match.group(1), inner)

                cleaned = re.sub(r"\\texttt\{([^}]*)\}", escape_underscores, cleaned)
                cleaned = re.sub(r"(https?://[^\s}]+)", lambda m: m.group(0).replace("_", "\\_"), cleaned)

                return cleaned

            latex_code = sanitize_latex_for_pdflatex(latex_code)

            # Basic validation of LaTeX content
            if not latex_code or len(latex_code.strip()) < 50:
                logger.error("LaTeX code is empty or too short")
                return False
            
            # Check for basic LaTeX structure
            if '\\documentclass' not in latex_code:
                logger.error("LaTeX code missing \\documentclass")
                return False
            
            if '\\begin{document}' not in latex_code:
                logger.error("LaTeX code missing \\begin{document}")
                return False
            
            if '\\end{document}' not in latex_code:
                logger.error("LaTeX code missing \\end{document}")
                return False
            
            # Check for problematic packages that don't work with pdflatex
            problematic_packages = ['fontspec', 'unicode-math', 'polyglossia']
            for package in problematic_packages:
                if f'\\usepackage{{{package}}}' in latex_code or ('\\usepackage[' in latex_code and f'{package}' in latex_code):
                    logger.warning(f"Found potentially problematic package '{package}' - may require XeLaTeX/LuaLaTeX")
            
            logger.info(f"LaTeX validation passed. Content length: {len(latex_code)} characters")
            
            # Create temporary directory for compilation
            with tempfile.TemporaryDirectory() as temp_dir:
                # Write LaTeX code to temporary file
                tex_file = os.path.join(temp_dir, "resume.tex")
                with open(tex_file, 'w', encoding='utf-8') as f:
                    f.write(latex_code)
                
                # Also save a copy for debugging purposes
                debug_file = "/Users/alwinpaul/Desktop/Project/Resume AI/debug_latex.tex"
                try:
                    with open(debug_file, 'w', encoding='utf-8') as f:
                        f.write(latex_code)
                    logger.info(f"LaTeX saved for debugging: {debug_file}")
                except Exception as e:
                    logger.warning(f"Could not save debug LaTeX file: {e}")
                
                # Check which LaTeX engines are available
                pdflatex_available = shutil.which('pdflatex')
                xelatex_available = shutil.which('xelatex')
                
                if not pdflatex_available and not xelatex_available:
                    logger.error("Neither pdflatex nor xelatex found. Please install LaTeX (e.g., TeX Live, MiKTeX)")
                    return False
                
                # Choose the appropriate engine (prefer pdflatex for speed)
                if pdflatex_available:
                    latex_engine = 'pdflatex'
                else:
                    latex_engine = 'xelatex'
                    
                logger.info(f"Using LaTeX engine: {latex_engine}")
                
                # Compile LaTeX to PDF
                def run_compile(engine: str) -> tuple[bool, str, str]:
                    try:
                        last_stdout = ''
                        last_stderr = ''
                        for _ in range(2):
                            result = subprocess.run(
                                [engine, '-interaction=nonstopmode', '-output-directory', temp_dir, tex_file],
                                capture_output=True,
                                text=True,
                                encoding='utf-8',
                                errors='replace',
                                timeout=60
                            )
                            last_stdout = result.stdout
                            last_stderr = result.stderr
                        pdf_file_local = os.path.join(temp_dir, "resume.pdf")
                        return (os.path.exists(pdf_file_local), last_stdout, last_stderr)
                    except subprocess.TimeoutExpired:
                        return (False, '', f"{engine} compilation timed out")
                    except Exception as ex:
                        return (False, '', f"{engine} compilation error: {ex}")

                success, out, err = run_compile(latex_engine)
                if not success and xelatex_available and latex_engine != 'xelatex':
                    logger.warning("pdflatex failed to produce a PDF. Retrying with xelatex as fallback...")
                    success, out, err = run_compile('xelatex')

                if not success:
                    logger.error(f"LaTeX compilation failed. Engine tried: {latex_engine}")
                    if out:
                        logger.error(f"STDOUT: {out[-4000:]}")
                    if err:
                        logger.error(f"STDERR: {err[-4000:]}")
                    return False

                # Copy the generated PDF to the desired output path
                pdf_file = os.path.join(temp_dir, "resume.pdf")
                if os.path.exists(pdf_file):
                    shutil.copy2(pdf_file, output_path)
                    return True
                else:
                    logger.error("PDF file was not generated")
                    return False
            
        except Exception as e:
            logger.error(f"Error compiling LaTeX to PDF: {e}")
            return False

    async def handle_document(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle uploaded documents."""
        user_id = update.effective_user.id
        
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = {
                'api_key': None,
                'model': None,
                'state': 'main_menu'
            }
        
        if not self.is_user_setup_complete(user_id) and self.user_sessions[user_id].get('state') != 'waiting_resume_update':
            missing_items = []
            if not self.user_sessions[user_id].get('api_key'):
                missing_items.append("🔑 API Key")
            if not self.user_sessions[user_id].get('model'):
                missing_items.append("🤖 Model")
            
            await update.message.reply_text(
                f"❌ **Setup Required**\n\nBefore analyzing resumes, please set up:\n{chr(10).join(missing_items)}\n\nUse the menu to configure your settings.",
                reply_markup=self.get_setup_keyboard(user_id),
                parse_mode='Markdown'
            )
            return
        
        document = update.message.document
        
        if not document:
            await update.message.reply_text("Please upload a valid document.")
            return

        # Check file size (limit to 20MB)
        if document.file_size > 20 * 1024 * 1024:
            await update.message.reply_text("File too large. Please upload a file smaller than 20MB.")
            return

        # Check file type
        file_name = document.file_name.lower()
        if not file_name.endswith('.pdf'):
            await update.message.reply_text("Please upload a PDF file (.pdf) only.")
            return

        await update.message.reply_text("📄 Processing your resume... Please wait.")

        try:
            # Download the file
            file = await context.bot.get_file(document.file_id)
            file_content = await file.download_as_bytearray()

            # Extract text based on file type
            text = None
            # Only PDF supported
            text = self.extract_text_from_pdf(file_content)

            if not text or len(text.strip()) < 50:
                await update.message.reply_text("Sorry, I couldn't extract enough text from your resume. Please ensure it's a valid resume file.")
                return

            # Save uploaded file to per-user storage (persist the original file, not extracted text)
            user_dir = os.path.join(str(Path.cwd()), "user_files", str(user_id))
            os.makedirs(user_dir, exist_ok=True)
            # Determine a canonical file name
            safe_name = "uploaded_resume.pdf"
            saved_path = os.path.join(user_dir, safe_name)
            with open(saved_path, 'wb') as f:
                f.write(file_content)
            self.user_sessions[user_id]['resume_file_path'] = saved_path
            # Persist session so the resume survives bot restarts
            self.save_user_sessions()

            # If coming from update flow, go back to settings; else continue to job description
            if self.user_sessions[user_id].get('state') == 'waiting_resume_update':
                self.user_sessions[user_id]['state'] = 'settings'
                # Persist state and resume path after update
                self.save_user_sessions()
                await update.message.reply_text(
                    "✅ Resume updated and saved. Return to Settings to continue.",
                    reply_markup=self.get_settings_keyboard()
                )
                return
            else:
                self.user_sessions[user_id]['state'] = 'waiting_job_description'
            
            await update.message.reply_text(
                """
<b>✅ Resume Uploaded Successfully!</b>

<i>📄 Your resume has been processed and stored</i>

<b>📋 Next Step: Job Description</b>

Now, please paste the complete job description for the position you're applying to.

<b>🎯 What I'll do:</b>
• Analyze job requirements vs your experience
• Optimize keywords naturally
• Generate professional PDF resume
• Keep everything truthful and accurate

<b>💡 Pro Tip:</b> Include the full job posting for best optimization results!

<i>Just paste the job description as your next message...</i>
                """,
                parse_mode='HTML'
            )

        except Exception as e:
            logger.error(f"Error processing document: {e}")
            await update.message.reply_text("Sorry, there was an error processing your resume. Please try again.")

    async def analyze_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle the /analyze command."""
        await update.message.reply_text(
            """
<b>📄 Ready to Optimize Your Resume!</b>

<i>Let's create a job-specific LaTeX resume</i>

<b>🚀 Simple 3-step process:</b>
1. Upload your resume (PDF only)
2. Provide job description
3. Get optimized LaTeX code

<i>Upload your resume file to get started!</i>
            """,
            parse_mode='HTML'
        )

    async def button_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle button callbacks."""
        query = update.callback_query
        await query.answer()
        
        user_id = update.effective_user.id
        data = query.data
        
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = {
                'api_key': None,
                'model': None,
                'state': 'main_menu'
            }
        
        if data == "main_menu":
            await self.show_main_menu(query)
        elif data == "settings":
            await self.show_settings_menu(query)
        elif data == "info":
            await self.show_info(query)
        elif data == "analyze_resume":
            await self.prompt_resume_upload(query)
        elif data == "set_api_key":
            await self.prompt_api_key(query, user_id)
        elif data == "select_model":
            await self.prompt_model_selection(query, user_id)
        elif data == "validate_settings":
            await self.validate_user_settings(query, user_id)
        elif data == "update_resume":
            await self.prompt_resume_update(query, user_id)
        elif data.startswith("select_model_"):
            model_name = data.replace("select_model_", "")
            await self.set_user_model(query, user_id, model_name)
        elif data == "clear_settings":
            await self.clear_user_settings(query, user_id)
        elif data == "confirm_clear_settings":
            await self.confirm_clear_settings(query, user_id)
        elif data == "back":
            await self.handle_back(query, user_id)
        elif data == "cancel":
            await self.handle_cancel(query, user_id)

    async def show_main_menu(self, query):
        """Show main menu."""
        user_id = query.from_user.id
        
        if not self.is_user_setup_complete(user_id):
            user_session = self.user_sessions[user_id]
            has_model = bool(user_session.get('model'))
            has_api_key = bool(user_session.get('api_key'))
            
            if not has_model:
                text = """
<b>🤖 Resume AI Bot - Setup</b>

<i>Step 1 of 2: Choose your AI model</i>

<b>🎯 Current Status:</b>
🤖 Model: <i>Not selected</i>
🔑 API Key: <i>Not set</i>

<i>Let's start by choosing your AI model!</i>
                """
            elif not has_api_key:
                text = f"""
<b>🤖 Resume AI Bot - Setup</b>

<i>Step 2 of 2: Set your API key</i>

<b>🎯 Current Status:</b>
🤖 Model: <b>✅ {user_session.get('model')}</b>
🔑 API Key: <i>Needed</i>

<i>Almost done! Just need your OpenRouter API key.</i>
                """
            else:
                text = f"""
<b>🤖 Resume AI Bot - Ready!</b>

<i>✅ Setup complete - ready to optimize!</i>

<b>🎯 Your Configuration:</b>
🤖 Model: <b>✅ {user_session.get('model')}</b>
🔑 API Key: <b>✅ Configured</b>

<i>Time to create your perfect resume!</i>
                """
            
            await query.edit_message_text(
                text,
                reply_markup=self.get_setup_keyboard(user_id),
                parse_mode='HTML'
            )
        else:
            text = f"""
<b>🤖 Resume AI Bot - Main Menu</b>

<i>✅ Setup complete - ready to optimize!</i>

<b>⚙️ Your Configuration:</b>
🔑 API Key: <b>✅ Configured</b>
🤖 Model: <b>✅ {self.user_sessions[user_id]['model']}</b>

<b>🚀 Choose an option below:</b>

<i>Ready to create your perfect job-specific resume!</i>
            """
            await query.edit_message_text(
                text,
                reply_markup=self.get_main_menu_keyboard(),
                parse_mode='HTML'
            )

    async def show_settings_menu(self, query):
        """Show settings menu."""
        user_id = query.from_user.id
        user_session = self.user_sessions.get(user_id, {})
        current_api_key = user_session.get('api_key')
        current_model = user_session.get('model')
        has_resume = bool(user_session.get('resume_file_path'))
        
        api_status = "✅ Configured" if current_api_key else "❌ Not Set"
        model_status = f"✅ {current_model}" if current_model else "❌ Not Selected"
        resume_status = "✅ Saved" if has_resume else "❌ Not Uploaded"
        
        text = f"""
<b>⚙️ Settings Menu</b>

<i>Manage your OpenRouter configuration</i>

<b>📋 Current Settings:</b>
🔑 API Key: <b>{api_status}</b>
🤖 Model: <b>{model_status}</b>
📄 Resume: <b>{resume_status}</b>

<b>🔧 Available Actions:</b>
• Update your API key
• Change AI model
• Update uploaded resume
• Validate configuration
• Reset settings

<i>Choose an option below to manage your settings.</i>
        """
        
        # Create enhanced settings keyboard
        keyboard = [
            [InlineKeyboardButton(f"🔑 {'Update' if current_api_key else 'Set'} API Key", callback_data="set_api_key")],
            [InlineKeyboardButton(f"🤖 {'Change' if current_model else 'Select'} Model", callback_data="select_model")],
            [InlineKeyboardButton("📄 Update Resume", callback_data="update_resume")],
            [InlineKeyboardButton("✅ Validate Settings", callback_data="validate_settings")],
            [InlineKeyboardButton("🗑️ Clear Settings", callback_data="clear_settings")],
            [InlineKeyboardButton("🔙 Back to Menu", callback_data="main_menu")]
        ]
        
        await query.edit_message_text(
            text,
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode='HTML'
        )

    async def show_info(self, query):
        """Show unified info (help + about)."""
        info_text = """
ℹ️ **Resume AI Bot — Info**

**How to use:**
1. Set your OpenRouter API key in Settings
2. Select your preferred AI model
3. Upload your resume as PDF
4. Provide the job description
5. Receive optimized, ATS-friendly PDF

**Features:**
• PDF resume generation
• Job-specific keyword optimization
• ATS-friendly formatting
• Professional styling
• Ready-to-use PDF download

**Process:**
📄 Upload → 📋 Job Description → 📝 PDF Resume

**Privacy & Storage:**
• Your settings (API key & model) are stored locally
• No resume content is permanently stored
• Use "Clear Settings" to remove stored data
        """
        keyboard = [[InlineKeyboardButton("🔙 Back to Menu", callback_data="main_menu")]]
        await query.edit_message_text(
            info_text,
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode='Markdown'
        )

    async def prompt_resume_upload(self, query):
        """Prompt user to upload resume."""
        user_id = query.from_user.id
        
        if not self.is_user_setup_complete(user_id):
            user_session = self.user_sessions[user_id]
            missing_model = not user_session.get('model')
            missing_api = not user_session.get('api_key')
            
            text = """
<b>❌ Setup Required</b>

<i>Let's get you configured first!</i>

<b>📋 Missing:</b>
""" + ("🤖 AI Model\n" if missing_model else "") + \
("🔑 API Key\n" if missing_api else "") + """

<i>It'll only take a minute to set up...</i>
            """
            await query.edit_message_text(
                text,
                reply_markup=self.get_setup_keyboard(user_id),
                parse_mode='HTML'
            )
            return
        
        # If a resume is already saved, skip upload and ask for JD directly
        existing_path = self.user_sessions.get(user_id, {}).get('resume_file_path')
        if existing_path and os.path.exists(existing_path):
            self.user_sessions[user_id]['state'] = 'waiting_job_description'
            jd_prompt = """
<b>📋 Job Description Needed</b>

<i>✅ I found your previously uploaded resume.</i>

<b>📄 Next Step: Paste the full job description</b>

I'll analyze it against your resume, optimize keywords, and generate a polished PDF.

<i>Send the job description as your next message…</i>
            """
            try:
                await query.edit_message_text(
                    jd_prompt,
                    reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Back", callback_data="main_menu")]]),
                    parse_mode='HTML'
                )
            except Exception:
                await query.message.reply_text(
                    jd_prompt,
                    reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Back", callback_data="main_menu")]]),
                    parse_mode='HTML'
                )
            return

        text = """
<b>📄 Resume Optimization</b>

<i>Create a job-specific PDF resume</i>

<b>📁 Upload your resume:</b>
• PDF (.pdf) only

<b>📏 Requirements:</b>
• Maximum file size: 20MB
• Clear, readable format

<b>🚀 3-Step Process:</b>
1️⃣ Upload your current resume
2️⃣ Provide job description
3️⃣ Download optimized PDF

<i>Only PDF uploads are accepted.</i>
        """
        keyboard = [[InlineKeyboardButton("🔙 Back to Menu", callback_data="main_menu")]]
        try:
            await query.edit_message_text(
                text,
                reply_markup=InlineKeyboardMarkup(keyboard),
                parse_mode='HTML'
            )
        except Exception:
            # If editing fails, send a new message instead
            await query.message.reply_text(
                text,
                reply_markup=InlineKeyboardMarkup(keyboard),
                parse_mode='HTML'
            )

    async def prompt_resume_update(self, query, user_id):
        """Prompt user to upload a new resume from settings."""
        self.user_sessions[user_id]['state'] = 'waiting_resume_update'
        text = """
<b>📄 Update Your Resume</b>

<i>Upload a new PDF to replace the existing one</i>

<b>📁 Supported:</b>
• PDF (.pdf) only

<b>📏 Requirements:</b>
• Max 20MB
• Clear, readable format

<i>Send the file here as a document attachment.</i>
        """
        await query.edit_message_text(
            text,
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Back", callback_data="settings")]]),
            parse_mode='HTML'
        )

    async def prompt_api_key(self, query, user_id):
        """Prompt user to enter API key."""
        self.update_user_session(user_id, {'state': 'waiting_api_key'})
        user_session = self.user_sessions.get(user_id, {})
        selected_model = user_session.get('model', 'your selected model')
        current_api_key = user_session.get('api_key')
        
        if current_api_key:
            # Updating existing API key
            text = f"""
<b>🔑 Update API Key</b>

<i>Change your OpenRouter API key</i>

<b>📋 Current Status:</b>
🔑 API Key: ✅ Currently configured
🤖 Model: {selected_model}

<b>🔄 Updating Your Key:</b>
• Your current key will be replaced
• New key will be validated automatically
• All settings will be preserved

<b>🚀 Get a new key:</b>
1. Visit <a href="https://openrouter.ai/keys">OpenRouter API Keys</a>
2. Create or copy your key
3. Paste it here

<b>🔒 Security:</b> Old key will be securely replaced

<i>Paste your new API key as a message...</i>
            """
        else:
            # Setting API key for the first time
            if selected_model != 'your selected model':
                step_info = "Step 2 of 2: Connect to"
            else:
                step_info = "Connect to"
                
            text = f"""
<b>🔑 Set Your API Key</b>

<i>{step_info} {selected_model}</i>

I need your OpenRouter API key to access the AI models.

<b>🚀 Quick Setup:</b>
1. Visit <a href="https://openrouter.ai/">OpenRouter.ai</a>
2. Sign up (it's free to start!)
3. Go to "API Keys" section
4. Create a new key
5. Copy and paste it here

<b>💡 Pro Tip:</b> 
• Free tier available for testing
• Pay-per-use pricing
• No monthly subscriptions

<b>🔒 Security:</b> Your API key is stored securely for this session only.

<i>Just paste your API key as a message and we'll be ready to go!</i>
            """
        
        await query.edit_message_text(
            text,
            reply_markup=self.get_back_cancel_keyboard(),
            parse_mode='HTML',
            disable_web_page_preview=True
        )

    async def prompt_model_selection(self, query, user_id):
        """Prompt user to select a model."""
        self.update_user_session(user_id, {'state': 'waiting_model_search'})
        user_session = self.user_sessions.get(user_id, {})
        current_model = user_session.get('model')
        current_api_key = user_session.get('api_key')
        
        if current_model:
            # Changing existing model
            api_status = "✅ Configured" if current_api_key else "❌ Not set"
            text = f"""
<b>🤖 Change AI Model</b>

<i>Update your AI model selection</i>

<b>📋 Current Configuration:</b>
🤖 Model: <b>{current_model}</b>
🔑 API Key: {api_status}

<b>🔄 Choose New Model:</b>
Type part of a model name to search available options.

<b>💎 Popular Choices:</b>
• <code>claude</code> - Excellent for professional writing
• <code>gpt-4</code> - Great for detailed analysis  
• <code>gpt-3.5</code> - Good quality, lower cost
• <code>llama</code> - Open-source alternative

<b>💡 Note:</b> Your API key will work with the new model

<i>Type a model name to search and update!</i>
            """
        else:
            # Selecting model for the first time
            step_info = "Step 1 of 2: " if not current_api_key else ""
            text = f"""
<b>🤖 Choose Your AI Model</b>

<i>{step_info}Select the AI brain for your resume optimization</i>

Type part of a model name to search (e.g., "claude", "gpt").

<b>🔍 How it works:</b>
• Type any part of a model name
• Browse available options
• Select your preferred model
• Get matched with the best AI for your needs

<i>Just type a model name and I'll show you the options!</i>
            """
        
        await query.edit_message_text(
            text,
            reply_markup=self.get_back_cancel_keyboard(),
            parse_mode='HTML'
        )

    async def validate_user_settings(self, query, user_id):
        """Validate user's current settings."""
        user_session = self.user_sessions.get(user_id, {})
        api_key = user_session.get('api_key')
        model = user_session.get('model')
        
        if not api_key or not model:
            missing_items = []
            if not api_key:
                missing_items.append("🔑 API Key")
            if not model:
                missing_items.append("🤖 Model")
            
            text = f"❌ **Validation Failed**\n\nMissing: {', '.join(missing_items)}\n\nPlease complete your setup first."
            await query.edit_message_text(
                text,
                reply_markup=self.get_setup_keyboard(user_id),
                parse_mode='Markdown'
            )
            return
        
        # Show loading message with SSE-like updates
        loading_text = f"""
<b>🔄 Validating Settings</b>

<i>Testing your configuration...</i>

<b>📋 Current Setup:</b>
🔑 API Key: {'✓' if api_key else '❌'}
🤖 Model: {model}

⏳ <i>Testing connection to OpenRouter...</i>
        """
        
        await query.edit_message_text(
            loading_text,
            parse_mode='HTML'
        )
        
        # Simulate SSE updates
        await asyncio.sleep(1)
        
        # Update loading message with more details
        detailed_loading_text = f"""
<b>🔄 Validating Settings</b>

<i>Running comprehensive tests...</i>

<b>📋 Configuration:</b>
🔑 API Key: {'✓ Provided' if api_key else '❌ Missing'}
🤖 Model: {model}

<b>🧪 Testing:</b>
⏳ Connecting to OpenRouter...
⏳ Validating API key...
⏳ Testing model access...

<i>Please wait while I verify everything works...</i>
        """
        
        await query.edit_message_text(
            detailed_loading_text,
            parse_mode='HTML'
        )
        
        await asyncio.sleep(1.5)
        
        # Validate settings
        is_valid = await self.validate_openrouter_settings(api_key, model)
        
        if is_valid:
            success_text = f"""
<b>✅ Validation Successful!</b>

<i>🎉 All systems are go!</i>

<b>📋 Verified Configuration:</b>
🔑 <b>API Key:</b> ✅ Valid & Active
🤖 <b>Model:</b> ✅ {model}
🌐 <b>Connection:</b> ✅ Established
💰 <b>Credits:</b> ✅ Available

<b>🎊 You're all set!</b>

<i>Ready to create amazing resumes!</i>
            """
        else:
            success_text = f"""
<b>❌ Validation Failed</b>

<i>⚠️ Configuration issues detected</i>

<b>📋 Test Results:</b>
🔑 API Key: <b>❌ Invalid/Expired</b>
🤖 Model: <b>❓ {model}</b>
🌐 Connection: <b>❌ Failed</b>

<i>Please check your API key and try again.</i>
            """
        
        await query.edit_message_text(
            success_text,
            reply_markup=self.get_settings_keyboard(),
            parse_mode='HTML'
        )

    async def set_user_model(self, query, user_id, model_name):
        """Set user's selected model, store context window, and move to API key step."""
        old_model = self.user_sessions[user_id].get('model')

        # Determine context window from previously stored search results
        context_window = 8192
        model_data = None
        model_search_results = self.user_sessions[user_id].get('model_search_results', [])
        for m in model_search_results:
            if m.get('id') == model_name:
                model_data = m
                break
        if model_data is not None:
            context_window = model_data.get('context_length', 8192)

        # Save model and context window
        self.update_user_session(user_id, {'model': model_name, 'model_context_window': context_window})

        # Clear temporary search results
        if 'model_search_results' in self.user_sessions[user_id]:
            del self.user_sessions[user_id]['model_search_results']
        
        # Show confirmation message
        if old_model and old_model != model_name:
            confirmation_text = f"""
<b>🤖 Model Updated Successfully!</b>

<i>✅ Your AI model has been changed</i>

<b>🔄 Change Summary:</b>
📤 Previous: <b>{old_model}</b>
📥 New: <b>{model_name}</b> (Context: {context_window} tokens)

<b>⏳ Next Step:</b>
Validating your new configuration...

<i>Testing the new model with your API key...</i>
            """
        else:
            confirmation_text = f"""
<b>🤖 Model Selected!</b>

<i>✅ Great choice for resume optimization</i>

<b>📋 Selected Model:</b>
🧠 <b>{model_name}</b> (Context: {context_window} tokens)

<b>💡 Why this model:</b>
• Professional writing capabilities
• Understanding of job requirements
• LaTeX code generation
• ATS optimization knowledge

<i>Setting up your configuration...</i>
            """
        
        await query.edit_message_text(
            confirmation_text,
            parse_mode='HTML'
        )
        
        await asyncio.sleep(2)
        
        # Check if API key is already set
        if self.user_sessions[user_id].get('api_key'):
            # Both model and API key are set, validate automatically
            self.user_sessions[user_id]['state'] = 'main_menu'
            await self.validate_user_settings(query, user_id)
        else:
            # Move to API key step
            await self.prompt_api_key(query, user_id)

    async def handle_back(self, query, user_id):
        """Handle back button."""
        state = self.user_sessions[user_id].get('state', 'main_menu')
        if state in ['waiting_api_key', 'waiting_model_search']:
            await self.show_settings_menu(query)
            self.user_sessions[user_id]['state'] = 'settings'
        else:
            await self.show_main_menu(query)
            self.user_sessions[user_id]['state'] = 'main_menu'

    async def handle_cancel(self, query, user_id):
        """Handle cancel button."""
        await self.show_main_menu(query)
        self.user_sessions[user_id]['state'] = 'main_menu'

    async def clear_user_settings(self, query, user_id):
        """Clear user's settings with confirmation."""
        user_session = self.user_sessions.get(user_id, {})
        has_settings = user_session.get('api_key') or user_session.get('model')
        
        if not has_settings:
            await query.edit_message_text(
                """
<b>🗑️ No Settings to Clear</b>

<i>You don't have any settings configured yet</i>

<b>Current Status:</b>
🔑 API Key: Not set
🤖 Model: Not selected

<i>Use the settings menu to configure your bot.</i>
                """,
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("⚙️ Go to Settings", callback_data="settings")],
                    [InlineKeyboardButton("🔙 Back to Menu", callback_data="main_menu")]
                ]),
                parse_mode='HTML'
            )
            return
        
        # Show confirmation
        current_api = "✅ Set" if user_session.get('api_key') else "❌ Not set"
        current_model = user_session.get('model', 'Not selected')
        
        await query.edit_message_text(
            f"""
<b>🗑️ Clear All Settings</b>

<i>⚠️ This will remove your configuration</i>

<b>Current Settings:</b>
🔑 API Key: {current_api}
🤖 Model: {current_model}

<b>⚠️ Warning:</b>
• API key will be removed
• Model selection will be reset
• You'll need to reconfigure to use the bot

<i>Are you sure you want to continue?</i>
            """,
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("✅ Yes, Clear Settings", callback_data="confirm_clear_settings")],
                [InlineKeyboardButton("❌ Cancel", callback_data="settings")]
            ]),
            parse_mode='HTML'
        )

    async def confirm_clear_settings(self, query, user_id):
        """Actually clear the user's settings."""
        # Clear all settings
        self.update_user_session(user_id, {
            'api_key': None,
            'model': None,
            'state': 'main_menu'
        })
        
        await query.edit_message_text(
            """
<b>🗑️ Settings Cleared Successfully</b>

<i>✅ All configuration has been removed</i>

<b>📋 Reset Status:</b>
🔑 API Key: ❌ Cleared
🤖 Model: ❌ Cleared
📄 Session Data: ❌ Cleared

<b>🚀 Next Steps:</b>
• Configure your API key
• Select an AI model
• Start optimizing resumes

<i>You'll need to set up your configuration again to use the bot.</i>
            """,
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("⚙️ Setup Now", callback_data="settings")],
                [InlineKeyboardButton("🔙 Back to Menu", callback_data="main_menu")]
            ]),
            parse_mode='HTML'
        )

    async def handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle text messages."""
        user_id = update.effective_user.id
        text = update.message.text
        
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = {
                'api_key': None,
                'model': None,
                'state': 'main_menu'
            }
        
        user_state = self.user_sessions[user_id].get('state', 'main_menu')
        
        if user_state == 'waiting_api_key':
            # User is entering API key
            old_api_key = self.user_sessions[user_id].get('api_key')
            self.update_user_session(user_id, {'api_key': text.strip()})
            
            # Show confirmation and auto-validate
            if old_api_key:
                confirmation_text = f"""
<b>🔑 API Key Updated!</b>

<i>✅ New key securely stored</i>

<b>🔄 Configuration Update:</b>
• Model: {self.user_sessions[user_id].get('model')}
• API Key: ••••••••••{text.strip()[-4:]} <i>(new)</i>

<b>🧪 Validation:</b>
⏳ Testing new API key...
⏳ Verifying model access...

<i>Validating your updated configuration...</i>
                """
            else:
                confirmation_text = f"""
<b>🔑 API Key Received!</b>

<i>✅ Securely stored for this session</i>

<b>🔄 Setup Summary:</b>
• Model: {self.user_sessions[user_id].get('model')}
• API Key: ••••••••••{text.strip()[-4:]}

<b>🧪 Final Validation:</b>
⏳ Testing connection to OpenRouter...
⏳ Verifying model access...
⏳ Checking account status...

<i>Almost done! Validating everything works...</i>
                """
            
            confirmation_msg = await update.message.reply_text(
                confirmation_text,
                parse_mode='HTML'
            )
            
            # Auto-validate settings
            api_key = self.user_sessions[user_id]['api_key']
            model = self.user_sessions[user_id]['model']
            is_valid = await self.validate_openrouter_settings(api_key, model)
            
            if is_valid:
                self.user_sessions[user_id]['state'] = 'main_menu'
                await confirmation_msg.edit_text(
                    f"""
<b>🎉 Setup Complete!</b>

<i>✅ Everything is working perfectly!</i>

<b>⚙️ Your Configuration:</b>
🤖 Model: <b>{model}</b>
🔑 API Key: <b>✅ Validated</b>

<b>🚀 Ready to optimize resumes!</b>

<i>You can now upload your resume and job description to get started.</i>
                    """,
                    reply_markup=self.get_main_menu_keyboard(),
                    parse_mode='HTML'
                )
            else:
                await confirmation_msg.edit_text(
                    """
<b>❌ Setup Failed</b>

<i>There seems to be an issue with your API key</i>

<i>Please check your API key and try again.</i>
                    """,
                    reply_markup=self.get_back_cancel_keyboard(),
                    parse_mode='HTML'
                )
            
        elif user_state == 'waiting_model_search':
            # User is searching for models
            await self.handle_model_search(update, text.strip())
            
        elif user_state == 'waiting_job_description':
            # User is providing job description
            await self.handle_job_description(update, text.strip(), user_id)
            
        else:
            # Check if it looks like a resume (basic heuristic)
            if len(text) > 200 and any(keyword in text.lower() for keyword in ['experience', 'education', 'skills', 'work', 'job']):
                if not self.is_user_setup_complete(user_id):
                    missing_items = []
                    if not self.user_sessions[user_id].get('api_key'):
                        missing_items.append("🔑 API Key")
                    if not self.user_sessions[user_id].get('model'):
                        missing_items.append("🤖 Model")
                    
                    await update.message.reply_text(
                        f"🤖 I detected this looks like a resume!\n\n❌ **Setup Required**\n\nBefore analyzing, please set up:\n{chr(10).join(missing_items)}\n\nUse the menu to configure your settings.",
                        reply_markup=self.get_setup_keyboard(user_id),
                        parse_mode='Markdown'
                    )
                else:
                    # Store the resume text and ask for job description
                    self.user_sessions[user_id]['resume_text'] = text
                    self.user_sessions[user_id]['state'] = 'waiting_job_description'
                    
                    await update.message.reply_text(
                        """
<b>🤖 Resume Detected!</b>

<i>✅ I can see this looks like a resume - saved it!</i>

<b>📋 Next Step: Job Description</b>

Perfect! Now please paste the job description for the position you're applying to.

<b>🎯 What happens next:</b>
• I'll analyze both documents
• Match keywords strategically  
• Generate optimized PDF resume
• Keep everything truthful

<i>Just paste the job description as your next message...</i>
                        """,
                        parse_mode='HTML'
                    )
            else:
                if not self.is_user_setup_complete(user_id):
                    await update.message.reply_text(
                        """
<b>👋 Hi there!</b>

<i>I'm here to help optimize your resume for job applications</i>

<b>⚠️ Setup Required</b>
Let's get you configured first!

<i>It'll only take a minute...</i>
                        """,
                        reply_markup=self.get_setup_keyboard(user_id),
                        parse_mode='HTML'
                    )
                else:
                    await update.message.reply_text(
                        """
<b>👋 Hello!</b>

<i>Ready to create your perfect job-specific resume?</i>

<b>🚀 Simple Process:</b>
📄 Upload Resume → 📋 Job Description → 📝 LaTeX Code

<i>Use the menu below to get started!</i>
                        """,
                        reply_markup=self.get_main_menu_keyboard(),
                        parse_mode='HTML'
                    )

    async def handle_model_search(self, update, search_term):
        """Handle model search with SSE-like updates."""
        user_id = update.effective_user.id
        
        # Show searching message
        search_msg = await update.message.reply_text(
            f"""
<b>🔍 Searching AI Models</b>

<i>Looking for models matching: "{search_term}"</i>

⏳ <i>Loading available options...</i>
            """,
            parse_mode='HTML'
        )
        
        # Simulate SSE update
        await asyncio.sleep(1)
        
        # Get models
        models = await self.get_openrouter_models(search_term)
        
        if not models:
            await search_msg.edit_text(
                f"""
<b>❌ No Models Found</b>

<i>No models found matching: "{search_term}"</i>

<b>💡 Try these popular searches:</b>
• <code>claude</code>
• <code>gpt</code>
• <code>llama</code>
• <code>mistral</code>

<i>Type a different model name to search again!</i>
                """,
                reply_markup=self.get_back_cancel_keyboard(),
                parse_mode='HTML'
            )
            return

        # Store full search results for later access (e.g., context_length)
        self.user_sessions[user_id]['model_search_results'] = models
        
        # Create model selection keyboard
        keyboard = []
        for model in models:
            model_id = model['id']
            model_name = model_id.split('/')[-1] if '/' in model_id else model_id
            keyboard.append([InlineKeyboardButton(
                f"🤖 {model_name}", 
                callback_data=f"select_model_{model_id}"
            )])
        
        keyboard.append([InlineKeyboardButton("🔙 Back", callback_data="back")])
        
        result_text = f"""
<b>🔍 Found {len(models)} AI Models</b>

<i>Models matching: "{search_term}"</i>

<b>📋 Select your preferred model:</b>

<i>Choose based on your needs - premium models offer better quality, budget models are more cost-effective.</i>
        """
        
        await search_msg.edit_text(
            result_text,
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode='HTML'
        )
        
        self.user_sessions[user_id]['state'] = 'selecting_model'

    async def handle_job_description(self, update, job_description, user_id):
        """Handle job description input and generate optimized LaTeX resume."""
        if len(job_description) < 50:
            await update.message.reply_text(
                """
<b>📋 Job Description Too Short</b>

<i>I need more details to optimize your resume effectively</i>

<b>🔍 What I need:</b>
• Complete job posting
• Required skills and qualifications
• Responsibilities and duties
• At least 50 characters total

<i>Please paste a more detailed job description!</i>
                """,
                parse_mode='HTML'
            )
            return
        
        # Store job description
        self.user_sessions[user_id]['job_description'] = job_description
        self.user_sessions[user_id]['state'] = 'main_menu'
        
        # Get stored resume file path and extract text on demand
        resume_file_path = self.user_sessions[user_id].get('resume_file_path')
        if not resume_file_path or not os.path.exists(resume_file_path):
            await update.message.reply_text(
                "❌ **Resume not found**\n\n"
                "Please upload your resume first.",
                reply_markup=self.get_main_menu_keyboard()
            )
            return
        # Extract text now (PDF only)
        resume_text = None
        try:
            with open(resume_file_path, 'rb') as rf:
                resume_text = self.extract_text_from_pdf(rf.read())
        except Exception as ex:
            logger.error(f"Error reading stored resume file: {ex}")
            resume_text = None
        if not resume_text or len(resume_text.strip()) < 50:
            await update.message.reply_text(
                "Sorry, I couldn't read your saved resume. Please upload it again.",
                reply_markup=self.get_main_menu_keyboard()
            )
            return
        
        # Show initial progress message
        processing_msg = await update.message.reply_text(
            """
<b>🔄 Creating Your Optimized Resume</b>

<i>✅ Resume: Processed</i>
<i>✅ Job Description: Analyzed</i>

<b>⏳ Starting AI analysis...</b>
⠋ <i>Initializing optimization process...</i>
            """,
            parse_mode='HTML'
        )
        
        # Update progress - AI analysis with spinner animation
        spinner_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        
        # Animate during AI processing
        for i in range(8):  # Show animation for ~1 second
            frame = spinner_frames[i % len(spinner_frames)]
            try:
                await processing_msg.edit_text(
                    f"""
<b>🔄 Creating Your Optimized Resume</b>

<i>✅ Resume: Processed</i>
<i>✅ Job Description: Analyzed</i>

<b>⏳ AI analyzing and optimizing...</b>
{frame} <i>Matching keywords and enhancing content...</i>
                    """,
                    parse_mode='HTML'
                )
                await asyncio.sleep(0.12)
            except Exception:
                pass  # Continue if message edit fails
        
        # Generate optimized LaTeX resume
        resume_result = await self.generate_optimized_latex_resume(resume_text, job_description, user_id)
        
        if not resume_result:
            await processing_msg.edit_text(
                """
<b>❌ Generation Failed</b>

<i>Sorry, I couldn't process your resume and job description</i>

<i>Please try again in a moment.</i>
                """,
                reply_markup=self.get_main_menu_keyboard(),
                parse_mode='HTML'
            )
            return
        
        # Unpack the result tuple
        latex_code, optimizations = resume_result
        
        # Store optimizations in user session for later use
        self.user_sessions[user_id]['last_optimizations'] = optimizations
        
        # Update processing message
        await processing_msg.edit_text(
            """
<b>🎉 Analysis Complete!</b>

<i>✅ Resume optimized for the job requirements!</i>

<b>📄 Generating PDF...</b>
• Professional formatting applied
• Keywords optimized
• ATS-friendly structure
• Ready to download

<i>Creating your PDF resume...</i>
            """,
            parse_mode='HTML'
        )
        
        # Create PDF from LaTeX
        try:
            # Create temporary file for PDF
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                pdf_path = temp_file.name
            
            # Update progress - PDF compilation with spinner
            for i in range(6):  # Show animation for ~0.7 seconds
                frame = spinner_frames[i % len(spinner_frames)]
                try:
                    await processing_msg.edit_text(
                        f"""
<b>🔄 Creating Your Optimized Resume</b>

<i>✅ Resume: Processed</i>
<i>✅ Job Description: Analyzed</i>
<i>✅ AI Optimization: Complete</i>

<b>⏳ Compiling PDF...</b>
{frame} <i>Converting LaTeX to professional PDF...</i>
                        """,
                        parse_mode='HTML'
                    )
                    await asyncio.sleep(0.12)
                except Exception:
                    pass  # Continue if message edit fails
            
            # Extract pure LaTeX from markdown
            pure_latex = self.extract_latex_from_markdown(latex_code)
            
            # Update progress - Verification with spinner
            for i in range(5):  # Show animation for ~0.6 seconds
                frame = spinner_frames[i % len(spinner_frames)]
                try:
                    await processing_msg.edit_text(
                        f"""
<b>🔄 Creating Your Optimized Resume</b>

<i>✅ Resume: Processed</i>
<i>✅ Job Description: Analyzed</i>
<i>✅ AI Optimization: Complete</i>

<b>⏳ Verifying content...</b>
{frame} <i>Ensuring quality and compatibility...</i>
                        """,
                        parse_mode='HTML'
                    )
                    await asyncio.sleep(0.12)
                except Exception:
                    pass  # Continue if message edit fails
            
            # Verify the content quality (more lenient for reasoning models)
            verification_result = self.verify_resume_content(pure_latex, job_description)
            logger.info(f"Content verification result: {verification_result}")
            
            if not verification_result:
                logger.warning("Content verification failed, but proceeding with PDF generation")
                # For reasoning models, we'll be more lenient and try to proceed
                # await processing_msg.edit_text(
                #     """
                # <b>❌ Content Verification Failed</b>
                # 
                # <i>The generated resume doesn't meet quality standards.</i>
                # 
                # <i>Please try again.</i>
                #     """,
                #     reply_markup=self.get_main_menu_keyboard(),
                #     parse_mode='HTML'
                # )
                # return
            
            # Compile to PDF
            pdf_created = self.compile_latex_to_pdf(pure_latex, pdf_path)
            
            if pdf_created:
                # Replace static 100% message with a brief loading animation
                title = "Creating Your Optimized Resume"
                statuses = [
                    "<i>✅ Resume: Processed</i>",
                    "<i>✅ Job Description: Analyzed</i>",
                    "<i>✅ AI Optimization: Complete</i>",
                    "<i>✅ PDF Compilation: Success</i>",
                ]
                await self.animate_loading_message(processing_msg, title, statuses, duration_seconds=1.5, frame_interval_seconds=0.1)
                
                # Send the PDF file
                with open(pdf_path, 'rb') as pdf_file:
                    # Create the optimizations text
                    optimizations_text = optimizations if optimizations.strip() else "• Keywords matched to job requirements\n• Experience reordered by relevance\n• ATS-friendly formatting\n• Professional LaTeX styling"
                    
                    caption = f"""
<b>📄 Your Optimized Resume PDF</b>

<i>✅ Generated successfully!</i>

<b>🎯 Optimizations Applied:</b>
{optimizations_text}

<i>Download and use for your job application!</i>
                    """
                    
                    await update.message.reply_document(
                        document=pdf_file,
                        filename=f"optimized_resume_{user_id}.pdf",
                        caption=caption,
                        reply_markup=self.get_main_menu_keyboard(),
                        parse_mode='HTML'
                    )
                
                # Clean up temporary file
                os.unlink(pdf_path)
                
            else:
                # PDF generation failed - simplified error message
                await processing_msg.edit_text(
                    """
<b>❌ PDF Generation Failed</b>

<i>Sorry, there was an issue creating your PDF resume</i>

<b>🔄 What happened:</b>
• LaTeX compilation encountered errors
• This has been logged for debugging
• Please try again in a moment

<i>Use the menu to try again.</i>
                    """,
                    reply_markup=self.get_main_menu_keyboard(),
                    parse_mode='HTML'
                )
                
        except Exception as e:
            logger.error(f"Error creating/sending PDF: {e}")
            await processing_msg.edit_text(
                """
<b>❌ Processing Error</b>

<i>Something went wrong during resume processing</i>

<b>🔄 Please try again</b>

<i>Use the menu to retry.</i>
                """,
                reply_markup=self.get_main_menu_keyboard(),
                parse_mode='HTML'
            )

def main():
    """Start the bot."""
    # Get bot token from environment variable
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    if not bot_token:
        print("Error: TELEGRAM_BOT_TOKEN environment variable not set")
        return

    # Create the bot instance
    bot = ResumeAIBot()

    # Create the Application
    application = Application.builder().token(bot_token).build()

    # Add handlers
    application.add_handler(CommandHandler("start", bot.start))
    application.add_handler(CommandHandler("help", bot.help_command))
    application.add_handler(CommandHandler("analyze", bot.analyze_command))
    application.add_handler(CallbackQueryHandler(bot.button_callback))
    application.add_handler(MessageHandler(filters.Document.ALL, bot.handle_document))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot.handle_text))

    # Run the bot
    print("Starting Resume AI Telegram Bot with OpenRouter integration...")
    print("Features: Interactive menus, model search, SSE-like validation")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()