# -*- coding: utf-8 -*-
"""
DekDataset Banner - ZOMBIT
"""

def print_ascii_banner():
    """Print a custom block ASCII art banner for 'ZOMBIT' with project info."""
    import sys
    from colorama import init, Fore, Style
    init(autoreset=True)
    # Force UTF-8 output encoding
    if sys.platform.startswith('win'):
        sys.stdout.reconfigure(encoding='utf-8')
    
    banner = f"""
{Fore.GREEN}{Style.BRIGHT}
                  ▒███████▒    ▒█████      ███▄ ▄███▓    ▄▄▄▄       ██▓   ▄▄▄█████▓
                  ▒ ▒ ▒ ▄▀░   ▒██▒  ██▒   ▓██▒▀█▀ ██▒   ▓█████▄    ▓██▒   ▓  ██▒ ▓▒
                  ░ ▒ ▄▀▒░    ▒██░  ██▒   ▓██    ▓██░   ▒██▒ ▄██   ▒██▒   ▒ ▓██░ ▒░
                    ▄▀▒   ░   ▒██   ██░   ▒██    ▒██    ▒██░█▀     ░██░   ░ ▓██▓ ░ 
                  ▒███████▒   ░ ████▓▒░   ▒██▒   ░██▒   ░▓█  ▀█▓   ░██░     ▒██▒ ░ 
                  ░▒▒ ▓░▒░▒   ░ ▒░▒░▒░    ░ ▒░   ░  ░   ░▒▓███▀▒   ░▓       ▒ ░░   
                  ░░▒ ▒ ░ ▒     ░ ▒ ▒░    ░  ░      ░   ▒░▒   ░     ▒ ░       ░    
                  ░ ░ ░ ░ ░   ░ ░ ░ ▒     ░      ░       ░    ░     ▒ ░     ░      
                    ░ ░           ░ ░            ░       ░          ░              
                  ░                                       ░

  {Fore.LIGHTMAGENTA_EX}ZOMBIT: DekDataset - Thai AI/ML Dataset Generator (DeepSeek API){Fore.GREEN}        
  {Fore.YELLOW}Project by zombit | JonusNattapong | github.com/zombitx64{Fore.GREEN}                     
  {Fore.LIGHTWHITE_EX}Version:{Fore.GREEN} 2025.05 | Python & Rust | MIT License{Fore.GREEN}                       
  {Fore.LIGHTWHITE_EX}Features:{Fore.CYAN} Batch, Parquet/Arrow, Thai-centric, Schema Validation, Metadata{Fore.GREEN} 
  {Fore.LIGHTWHITE_EX}Contact:{Fore.CYAN} zombitx64@gmail.com | github.com/JonusNattapong{Fore.GREEN}                  
  {Fore.LIGHTYELLOW_EX}Tips:{Fore.LIGHTWHITE_EX} ตั้งค่า DEEPSEEK_API_KEY ก่อนใช้งาน | รองรับ task หลากหลาย         

{Style.RESET_ALL}"""
    print(banner)

