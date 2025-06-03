# -*- coding: utf-8 -*-
"""
DekDataset - AI-Powered Thai Dataset Generator
Modern ASCII banner with comprehensive project information
"""

import sys
from datetime import datetime

def print_ascii_banner():
    """Print beautiful ASCII banner with project info"""
    
    # Color codes for terminal
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    
    # Colors
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    
    # Gradient effect
    GRAD1 = '\033[38;5;196m'  # Bright red
    GRAD2 = '\033[38;5;202m'  # Orange red
    GRAD3 = '\033[38;5;208m'  # Orange
    GRAD4 = '\033[38;5;214m'  # Yellow orange
    GRAD5 = '\033[38;5;220m'  # Yellow
    GRAD6 = '\033[38;5;226m'  # Bright yellow
    
    banner = f"""
{BOLD}{GRAD1}╔══════════════════════════════════════════════════════════════════════════════╗{RESET}
{BOLD}{GRAD1}║                                                                              ║{RESET}
{BOLD}{GRAD2}║     ██████╗ ███████╗██╗  ██╗    ██████╗  █████╗ ████████╗ █████╗          ║{RESET}
{BOLD}{GRAD2}║     ██╔══██╗██╔════╝██║ ██╔╝    ██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗         ║{RESET}
{BOLD}{GRAD3}║     ██║  ██║█████╗  █████╔╝     ██║  ██║███████║   ██║   ███████║         ║{RESET}
{BOLD}{GRAD3}║     ██║  ██║██╔══╝  ██╔═██╗     ██║  ██║██╔══██║   ██║   ██╔══██║         ║{RESET}
{BOLD}{GRAD4}║     ██████╔╝███████╗██║  ██╗    ██████╔╝██║  ██║   ██║   ██║  ██║         ║{RESET}
{BOLD}{GRAD4}║     ╚═════╝ ╚══════╝╚═╝  ╚═╝    ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝         ║{RESET}
{BOLD}{GRAD5}║                                                                              ║{RESET}
{BOLD}{GRAD5}║        ███████╗███████╗████████╗     ██████╗ ███████╗███╗   ██╗            ║{RESET}
{BOLD}{GRAD6}║        ██╔════╝██╔════╝╚══██╔══╝    ██╔════╝ ██╔════╝████╗  ██║            ║{RESET}
{BOLD}{GRAD6}║        ███████╗█████╗     ██║       ██║  ███╗█████╗  ██╔██╗ ██║            ║{RESET}
{BOLD}{CYAN}║        ╚════██║██╔══╝     ██║       ██║   ██║██╔══╝  ██║╚██╗██║            ║{RESET}
{BOLD}{BLUE}║        ███████║███████╗   ██║       ╚██████╔╝███████╗██║ ╚████║            ║{RESET}
{BOLD}{BLUE}║        ╚══════╝╚══════╝   ╚═╝        ╚═════╝ ╚══════╝╚═╝  ╚═══╝            ║{RESET}
{BOLD}{MAGENTA}║                                                                              ║{RESET}
{BOLD}{WHITE}║══════════════════════════════════════════════════════════════════════════════║{RESET}
{BOLD}{GREEN}║                     🚀 AI-Powered Thai Dataset Generator 🚀                  ║{RESET}
{BOLD}{WHITE}║══════════════════════════════════════════════════════════════════════════════║{RESET}
{BOLD}{CYAN}║                                                                              ║{RESET}
{BOLD}{CYAN}║  🎯 Version: {YELLOW}2025.05{CYAN} | Status: {GREEN}Production Ready{CYAN} | Build: {YELLOW}Web + CLI{CYAN}       ║{RESET}
{BOLD}{CYAN}║                                                                              ║{RESET}
{BOLD}{WHITE}║──────────────────────────────────────────────────────────────────────────────║{RESET}
{BOLD}{BLUE}║  ✨ FEATURES:                                                                ║{RESET}
{BOLD}{WHITE}║    🌐 Modern Web Interface       📊 Interactive Data Preview                ║{RESET}
{BOLD}{WHITE}║    🤖 Multi-Model Support        ⚡ Real-time Generation                    ║{RESET}
{BOLD}{WHITE}║    📈 Quality Control            🔧 Enterprise Ready                        ║{RESET}
{BOLD}{WHITE}║    💾 Multiple Export Formats    📋 Task Management                         ║{RESET}
{BOLD}{WHITE}║                                                                              ║{RESET}
{BOLD}{GREEN}║  🤖 AI MODELS:                                                               ║{RESET}
{BOLD}{WHITE}║    🌩️  DeepSeek API (Cloud)      🦙 Ollama (Local)                         ║{RESET}
{BOLD}{WHITE}║    🎯 Custom Model Support       ⚙️  Auto-Detection                         ║{RESET}
{BOLD}{WHITE}║                                                                              ║{RESET}
{BOLD}{YELLOW}║  📊 DATA FORMATS:                                                            ║{RESET}
{BOLD}{WHITE}║    📄 JSON Lines (JSONL)         📋 CSV Spreadsheet                         ║{RESET}
{BOLD}{WHITE}║    📦 ZIP Archives               🗃️  Parquet (Future)                       ║{RESET}
{BOLD}{WHITE}║                                                                              ║{RESET}
{BOLD}{RED}║  🌍 USE CASES:                                                                ║{RESET}
{BOLD}{WHITE}║    💬 Thai NLP Training          🏥 Medical AI Datasets                     ║{RESET}
{BOLD}{WHITE}║    📚 Educational Content        🏢 Business Intelligence                   ║{RESET}
{BOLD}{WHITE}║    🔍 Sentiment Analysis         🎯 Custom Domain Data                      ║{RESET}
{BOLD}{WHITE}║                                                                              ║{RESET}
{BOLD}{WHITE}║══════════════════════════════════════════════════════════════════════════════║{RESET}
{BOLD}{MAGENTA}║  👨‍💻 DEVELOPERS:                                                             ║{RESET}
{BOLD}{WHITE}║    🧑‍💻 ZOMBIT (zombitx64@gmail.com)  - Project Lead & Architecture         ║{RESET}
{BOLD}{WHITE}║    👨‍💻 JonusNattapong              - Core Development & AI Integration      ║{RESET}
{BOLD}{WHITE}║                                                                              ║{RESET}
{BOLD}{CYAN}║  🔗 LINKS:                                                                    ║{RESET}
{BOLD}{WHITE}║    🌐 Web Interface: {BLUE}http://localhost:8000{WHITE}                               ║{RESET}
{BOLD}{WHITE}║    📖 Documentation: {BLUE}https://github.com/zombitx64/DekDataset{WHITE}            ║{RESET}
{BOLD}{WHITE}║    🐛 Issues: {BLUE}https://github.com/zombitx64/DekDataset/issues{WHITE}            ║{RESET}
{BOLD}{WHITE}║    💬 Discussions: {BLUE}https://github.com/zombitx64/DekDataset/discussions{WHITE}  ║{RESET}
{BOLD}{WHITE}║                                                                              ║{RESET}
{BOLD}{GREEN}║  ⚡ QUICK START:                                                              ║{RESET}
{BOLD}{WHITE}║    1. Set DEEPSEEK_API_KEY in .env file                                     ║{RESET}
{BOLD}{WHITE}║    2. Run: {YELLOW}cd src/web && python app.py{WHITE}                                 ║{RESET}
{BOLD}{WHITE}║    3. Open: {BLUE}http://localhost:8000{WHITE}                                        ║{RESET}
{BOLD}{WHITE}║    4. Start generating amazing datasets! 🎉                                 ║{RESET}
{BOLD}{WHITE}║                                                                              ║{RESET}
{BOLD}{RED}║══════════════════════════════════════════════════════════════════════════════║{RESET}
{BOLD}{YELLOW}║                    🌟 Star us on GitHub! 🌟                                 ║{RESET}
{BOLD}{GREEN}║               Made with ❤️ in Thailand 🇹🇭 | MIT License                     ║{RESET}
{BOLD}{RED}║══════════════════════════════════════════════════════════════════════════════║{RESET}
"""

    # System info
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    
    system_info = f"""
{BOLD}{DIM}┌─ System Information ────────────────────────────────────────────────────────┐{RESET}
{DIM}│ 🕒 Started: {current_time}                                            │{RESET}
{DIM}│ 🐍 Python: {python_version}                                                          │{RESET}
{DIM}│ 💻 Platform: {sys.platform}                                                      │{RESET}
{DIM}│ 🎯 Mode: Web Interface + REST API                                              │{RESET}
{DIM}└─────────────────────────────────────────────────────────────────────────────┘{RESET}
"""

    tips = f"""
{BOLD}{GREEN}💡 Pro Tips:{RESET}
{WHITE}   • Use the web interface for the best experience: {BLUE}http://localhost:8000{RESET}
{WHITE}   • Try different models: DeepSeek (cloud) vs Ollama (local){RESET}
{WHITE}   • Preview your data before downloading with interactive tables{RESET}
{WHITE}   • Create custom tasks for your specific domain needs{RESET}
{WHITE}   • Monitor generation quality with built-in metrics{RESET}

{BOLD}{YELLOW}🚨 Need Help?{RESET}
{WHITE}   • Check documentation: {BLUE}README.md{RESET}
{WHITE}   • Report issues: {BLUE}https://github.com/zombitx64/DekDataset/issues{RESET}
{WHITE}   • Join discussions: {BLUE}https://github.com/zombitx64/DekDataset/discussions{RESET}

"""

    # Print everything
    print(banner)
    print(system_info)
    print(tips)

def print_loading_animation():
    """Print loading animation for startup"""
    import time
    
    CYAN = '\033[96m'
    RESET = '\033[0m'
    BOLD = '\033[1m'
    
    loading_text = f"{BOLD}{CYAN}🚀 Initializing DekDataset..."
    
    print(loading_text, end="")
    for i in range(3):
        time.sleep(0.5)
        print(".", end="", flush=True)
    print(f" Ready! 🎉{RESET}")
    print()

def print_web_server_ready():
    """Print web server ready message"""
    GREEN = '\033[92m'
    BLUE = '\033[94m'
    YELLOW = '\033[93m'
    RESET = '\033[0m'
    BOLD = '\033[1m'
    
    message = f"""
{BOLD}{GREEN}🌐 Web Server Ready!{RESET}

{YELLOW}  📡 Server URL: {BLUE}http://localhost:8000{RESET}
{YELLOW}  📖 API Docs:   {BLUE}http://localhost:8000/docs{RESET}
{YELLOW}  🎮 Dashboard:  {BLUE}http://localhost:8000{RESET}

{BOLD}{GREEN}✨ Ready to generate amazing datasets!{RESET}
"""
    print(message)

if __name__ == "__main__":
    print_ascii_banner()
    print_loading_animation()
    print_web_server_ready()

