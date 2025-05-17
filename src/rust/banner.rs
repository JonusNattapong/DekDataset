use std::io::{self, Write};

pub fn print_ascii_banner() {
    // ANSI escape codes for color
    let cyan = "\x1b[96m";
    let white = "\x1b[97m";
    let yellow = "\x1b[93m";
    let reset = "\x1b[0m";
    let bold = "\x1b[1m";
    let green = "\x1b[92m";

    let banner = format!(
"{bold}{cyan}
▒███████▒    ▒█████      ███▄ ▄███▓    ▄▄▄▄       ██▓   ▄▄▄█████▓
▒ ▒ ▒ ▄▀░   ▒██▒  ██▒   ▓██▒▀█▀ ██▒   ▓█████▄    ▓██▒   ▓  ██▒ ▓▒
░ ▒ ▄▀▒░    ▒██░  ██▒   ▓██    ▓██░   ▒██▒ ▄██   ▒██▒   ▒ ▓██░ ▒░
  ▄▀▒   ░   ▒██   ██░   ▒██    ▒██    ▒██░█▀     ░██░   ░ ▓██▓ ░ 
▒███████▒   ░ ████▓▒░   ▒██▒   ░██▒   ░▓█  ▀█▓   ░██░     ▒██▒ ░ 
░▒▒ ▓░▒░▒   ░ ▒░▒░▒░    ░ ▒░   ░  ░   ░▒▓███▀▒   ░▓       ▒ ░░   
░░▒ ▒ ░ ▒     ░ ▒ ▒░    ░  ░      ░   ▒░▒   ░     ▒ ░       ░    
░ ░ ░ ░ ░   ░ ░ ░ ▒     ░      ░       ░    ░     ▒ ░     ░      
  ░ ░           ░ ░            ░       ░          ░             
░

{yellow}
ZOMBIT: Thai AI/ML Dataset CLI Toolkit | DeepSeek API
{green}
 ◉ Project : zombit | JonusNattapong
 ◉ GitHub  : github.com/zombitx64
 ◉ Version : 2025.05 | Rust + Python | MIT License
 ◉ Tools   : Batch Translate, Thai NLP, Metadata, Schema Validate
 ◉ Format  : Parquet, Arrow, CSV, HuggingFace
 ◉ API     : DeepSeek API (ตั้งค่า DEEPSEEK_API_KEY ก่อนใช้งาน)

{white}Tips: ตั้งค่า {bold}DEEPSEEK_API_KEY{reset}{white} ก่อนใช้งาน | รองรับ task หลากหลาย NLP/ML
{reset}"
    );

    print!("{banner}");
    io::stdout().flush().unwrap();
}

/// แสดงหลอดโหลด (progress bar) แบบเท่ ๆ
pub fn print_loading_bar(percent: u8) {
    let bar_len = 30;
    let filled = (percent as usize * bar_len) / 100;
    let empty = bar_len - filled;
    let green = "\x1b[92m";
    let yellow = "\x1b[93m";
    let reset = "\x1b[0m";
    let bar = format!(
        "\r{green}[{yellow}{fill}{reset}{empty}] {percent:3}%{reset}",
        green = green,
        yellow = yellow,
        fill = "█".repeat(filled),
        empty = " ".repeat(empty),
        percent = percent,
        reset = reset
    );
    print!("{bar}");
    io::stdout().flush().unwrap();
}
