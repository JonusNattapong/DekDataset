import os
import json
import time
import re
import random
import requests
from datetime import datetime
from typing import List, Dict, Any, Optional, Union # Added Union back for completeness
from bs4 import BeautifulSoup
import asyncio

from dotenv import load_dotenv # Moved to top

# --- Playwright for complex JS websites ---
try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

# --- yt-dlp for YouTube ---
try:
    import yt_dlp
    YTDLP_AVAILABLE = True
except ImportError:
    YTDLP_AVAILABLE = False
    # Functions using yt_dlp will need to check this flag or handle ImportError

# --- Crawl4AI for general web crawling ---
try:
    from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
    from crawl4ai.models import CrawlResult
    CRAWL4AI_AVAILABLE = True
except ImportError:
    CRAWL4AI_AVAILABLE = False
    # Warning will be printed if the function is called without the library

# --- Pandas for CSV output ---
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

# Load environment variables
load_dotenv()

# --- Helper: Proxy rotation, User-Agent randomization, Delay ---
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
]
def get_random_user_agent():
    return random.choice(USER_AGENTS)

def delay(min_sec=1, max_sec=3):
    time.sleep(random.uniform(min_sec, max_sec))

def get_proxies():
    # Example: return a list of proxies, or [] if not used
    return []

def get_random_proxy():
    proxies = get_proxies()
    return random.choice(proxies) if proxies else None

# --- Playwright fetch ---
def fetch_with_playwright(url, wait_selector=None, timeout=30000, user_agent=None, headless=True, wait_until="networkidle"):
    """Enhanced Playwright fetch with better page interaction"""
    if not PLAYWRIGHT_AVAILABLE:
        raise ImportError("playwright is not installed. Please install it with: pip install playwright")
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        context = browser.new_context(
            user_agent=user_agent or get_random_user_agent(),
            viewport={'width': 1280, 'height': 1024}
        )
        
        page = context.new_page()
        try:
            page.goto(url, timeout=timeout, wait_until=wait_until)
            
            if wait_selector:
                try:
                    print(f"[DEBUG] Waiting for selector: {wait_selector}")
                    page.wait_for_selector(wait_selector, timeout=timeout)
                except Exception as e:
                    print(f"[WARN] Selector wait timeout: {e}")

            # Scroll multiple times to load more content
            for _ in range(3):
                page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                delay(1, 2)
                try:
                    # Click "Show more replies" buttons if they exist
                    more_buttons = page.query_selector_all('div[role="button"]')
                    for btn in more_buttons:
                        try:
                            btn.click()
                            delay(1)
                        except:
                            continue
                except Exception as e:
                    print(f"[DEBUG] Click interaction error: {e}")
                    
            # Final scroll back to top
            page.evaluate("window.scrollTo(0, 0)")
            delay(1)
            
            html = page.content()
            
        except Exception as e:
            print(f"[ERROR] Page interaction failed: {e}")
            html = ""
            
        finally:
            context.close()
            browser.close()
            
        return html

def extract_twitter_data(url, fetch_profile=False):
    """Extract data from Twitter/X posts using enhanced Playwright"""
    html = fetch_with_playwright(
        url, 
        wait_selector="article[data-testid='tweet']",
        timeout=45000,  # Increased timeout
        wait_until="networkidle"
    )
    
    if not html:
        print("[ERROR] Failed to fetch page content")
        return {}
        
    soup = BeautifulSoup(html, "html.parser")
    data = {}
    
    try:
        # Updated selectors for X's UI
        tweet_selectors = [
            "article[data-testid='tweet'] div[data-testid='tweetText']",
            "article div[lang]",  # Broader tweet text selector
            "div[data-testid='tweetText']"  # Direct tweet text
        ]
        
        # Extract main tweet
        main_tweet = None
        for selector in tweet_selectors:
            tweets = soup.select(selector)
            if tweets:
                main_tweet = tweets[0]
                break
                
        if main_tweet:
            # Clean and store tweet text
            text = main_tweet.get_text(strip=True)
            if text:
                data["tweet"] = text
                print(f"[DEBUG] Found main tweet: {text[:100]}...")
                
            # Get author info
            author_container = soup.select_one("div[data-testid='User-Name']")
            if author_container:
                data["profile"] = author_container.get_text(strip=True)
                print(f"[DEBUG] Found author: {data['profile']}")
        
        # Extract replies - using multiple passes with different selectors
        replies = []
        seen_texts = set()
        
        # First try specific reply selectors
        reply_containers = soup.select("article[data-testid='tweet']:not(:first-child)")
        
        if not reply_containers:
            # Fallback to broader selectors
            reply_containers = soup.select("div[data-testid='cellInnerDiv']")
        
        print(f"[DEBUG] Found {len(reply_containers)} potential replies")
        
        for container in reply_containers:
            try:
                # Try multiple text selectors
                text_element = (
                    container.select_one("div[data-testid='tweetText']") or
                    container.select_one("div[lang]") or
                    container.select_one("div[dir='auto']")
                )
                
                if not text_element:
                    continue
                    
                text = text_element.get_text(strip=True)
                
                # Skip if empty, too short, or duplicate
                if not text or len(text) < 2 or text in seen_texts:
                    continue
                    
                # Skip if it matches the main tweet
                if data.get("tweet") == text:
                    continue
                
                # Get author with fallbacks
                author = "Unknown"
                author_el = (
                    container.select_one("div[data-testid='User-Name']") or
                    container.select_one("a[role='link'] div[dir='auto']")
                )
                if author_el:
                    author = author_el.get_text(strip=True)
                
                replies.append({
                    "text": text,
                    "author": author
                })
                seen_texts.add(text)
                
            except Exception as e:
                print(f"[WARN] Error extracting reply: {e}")
                continue
        
        data["replies"] = replies
        print(f"[INFO] Successfully extracted {len(replies)} replies")
        
    except Exception as e:
        print(f"[ERROR] Failed to extract Twitter data: {e}")
    
    return data

# --- Facebook ---
def extract_facebook_data(url, fetch_profile=False):
    """Extract data from Facebook posts using Playwright"""
    html = fetch_with_playwright(
        url, 
        wait_selector="[role='article'], [role='main']",
        timeout=30000
    )
    soup = BeautifulSoup(html, "html.parser")
    data = {}
    
    try:
        # Extract main post text with better selectors
        post_selectors = [
            # Main post content selectors
            "div[data-ad-preview='message']",
            "div[data-ad-comet-preview='message']",
            "div.xdj266r", 
            "div[dir='auto']:not([aria-hidden='true'])"
        ]
        
        for selector in post_selectors:
            post = soup.select_one(selector)
            if post and post.get_text(strip=True):
                # Filter out metadata text
                text = post.get_text(strip=True)
                if not any(x in text.lower() for x in [
                    "ความรู้สึก", "แชร์", "ครั้ง", "comments",
                    "likes", "shares", "view", "ความคิดเห็น"
                ]):
                    data["post"] = text
                    break
        
        # Extract comments with better filtering
        comments = []
        seen_texts = set()  # Track unique comments
        
        comment_containers = soup.select("div.x1y1aw1k, ul[role='list'] > li")
        for container in comment_containers:
            try:
                # Find comment text
                text_elements = container.select(
                    "div[dir='auto']:not([aria-hidden='true'])"
                )
                
                for element in text_elements:
                    text = element.get_text(strip=True)
                    
                    # Skip if empty, too short, or duplicate
                    if not text or len(text) < 2 or text in seen_texts:
                        continue
                        
                    # Skip metadata/UI text
                    if any(x in text.lower() for x in [
                        "ความรู้สึก", "แชร์", "ครั้ง", "comments",
                        "likes", "shares", "view", "ความคิดเห็น",
                        "ผู้เขียน", "ตอบกลับ", "reply", "author"
                    ]):
                        continue
                    
                    # Find author (look for nearest link with role='link')
                    author = "Unknown"
                    author_el = container.find_previous(
                        lambda tag: tag.name == "a" 
                        and tag.get("role") == "link"
                        and not tag.select_one("img")  # Exclude profile picture links
                    )
                    if author_el:
                        author = author_el.get_text(strip=True)
                    
                    comments.append({
                        "text": text,
                        "author": author
                    })
                    seen_texts.add(text)
                    
            except Exception as e:
                print(f"[WARN] Error extracting comment: {e}")
                continue
        
        data["comments"] = comments
        
        # Extract profile info if requested
        if fetch_profile:
            profile_selectors = [
                "a[aria-current='page']",
                "h2 strong span",
                "a[role='link'][tabindex='0']"
            ]
            for selector in profile_selectors:
                profile = soup.select_one(selector)
                if profile:
                    data["profile"] = profile.get_text(strip=True)
                    break
    
    except Exception as e:
        print(f"[WARN] Error extracting Facebook data: {e}")
    
    return data

# Function to help fill missing data
def refill_facebook_comments(url: str, max_retries: int = 3) -> List[Dict[str, Any]]:
    """Try to get more comments with multiple retries"""
    attempts = 0
    comments = []
    
    while attempts < max_retries and len(comments) < 1:
        try:
            data = extract_facebook_data(url)
            if data.get("comments"):
                comments.extend(data["comments"])
            attempts += 1
            time.sleep(2)  # Wait between retries
        except Exception:
            attempts += 1
            
    return comments

# --- Twitter/X ---
def extract_twitter_data(url, fetch_profile=False):
    """Extract data from Twitter/X posts using enhanced Playwright"""
    html = fetch_with_playwright(
        url, 
        wait_selector="article[data-testid='tweet']",
        timeout=45000,  # Increased timeout
        wait_until="networkidle"
    )
    
    if not html:
        print("[ERROR] Failed to fetch page content")
        return {}
        
    soup = BeautifulSoup(html, "html.parser")
    data = {}
    
    try:
        # Updated selectors for X's UI
        tweet_selectors = [
            "article[data-testid='tweet'] div[data-testid='tweetText']",
            "article div[lang]",  # Broader tweet text selector
            "div[data-testid='tweetText']"  # Direct tweet text
        ]
        
        # Extract main tweet
        main_tweet = None
        for selector in tweet_selectors:
            tweets = soup.select(selector)
            if tweets:
                main_tweet = tweets[0]
                break
                
        if main_tweet:
            # Clean and store tweet text
            text = main_tweet.get_text(strip=True)
            if text:
                data["tweet"] = text
                print(f"[DEBUG] Found main tweet: {text[:100]}...")
                
            # Get author info
            author_container = soup.select_one("div[data-testid='User-Name']")
            if author_container:
                data["profile"] = author_container.get_text(strip=True)
                print(f"[DEBUG] Found author: {data['profile']}")
        
        # Extract replies - using multiple passes with different selectors
        replies = []
        seen_texts = set()
        
        # First try specific reply selectors
        reply_containers = soup.select("article[data-testid='tweet']:not(:first-child)")
        
        if not reply_containers:
            # Fallback to broader selectors
            reply_containers = soup.select("div[data-testid='cellInnerDiv']")
        
        print(f"[DEBUG] Found {len(reply_containers)} potential replies")
        
        for container in reply_containers:
            try:
                # Try multiple text selectors
                text_element = (
                    container.select_one("div[data-testid='tweetText']") or
                    container.select_one("div[lang]") or
                    container.select_one("div[dir='auto']")
                )
                
                if not text_element:
                    continue
                    
                text = text_element.get_text(strip=True)
                
                # Skip if empty, too short, or duplicate
                if not text or len(text) < 2 or text in seen_texts:
                    continue
                
                # Skip if it matches the main tweet
                if data.get("tweet") == text:
                    continue
                
                # Get author with fallbacks
                author = "Unknown"
                author_el = (
                    container.select_one("div[data-testid='User-Name']") or
                    container.select_one("a[role='link'] div[dir='auto']")
                )
                if author_el:
                    author = author_el.get_text(strip=True)
                
                replies.append({
                    "text": text,
                    "author": author
                })
                seen_texts.add(text)
                
            except Exception as e:
                print(f"[WARN] Error extracting reply: {e}")
                continue
        
        data["replies"] = replies
        print(f"[INFO] Successfully extracted {len(replies)} replies")
        
    except Exception as e:
        print(f"[ERROR] Failed to extract Twitter data: {e}")
    
    return data

async def extract_twitter_with_crawl4ai(tweet_url: str, include_sentiment: bool = False) -> Optional[List[Dict[str, Any]]]:
    """Extract Twitter/X post and replies using Crawl4AI"""
    if not CRAWL4AI_AVAILABLE:
        return None

    try:
        # Configure browser
        browser_config = BrowserConfig(
            headless=True,
            slow_mo=50,
            viewport={'width': 1280, 'height': 720}
        )
        
        # Configure extraction
        run_config = CrawlerRunConfig(
            wait_for_selectors=["article[data-testid='tweet']"],
            click_selectors=[
                "div[role='button']",  # More replies
                "span[role='button']"   # Show replies
            ],
            wait_for_scroll_bottom=True,
            scroll_timeout=10000,
            extract_rules={
                'main_tweet': {
                    'selector': "article[data-testid='tweet']:first-child",
                    'fields': {
                        'text': {'selector': "div[data-testid='tweetText']", 'type': 'text'},
                        'author': {'selector': "div[data-testid='User-Name']", 'type': 'text'},
                        'time': {'selector': "time", 'type': 'text'}
                    }
                },
                'replies': {
                    'selector': "article[data-testid='tweet']:not(:first-child)",
                    'type': 'list',
                    'fields': {
                        'text': {'selector': "div[data-testid='tweetText']", 'type': 'text'},
                        'author': {'selector': "div[data-testid='User-Name']", 'type': 'text'},
                        'time': {'selector': "time", 'type': 'text'}
                    }
                }
            }
        )

        async with AsyncWebCrawler(config=browser_config) as crawler:
            result = await crawler.arun(url=tweet_url, config=run_config)
            
            if not result or not result.extracted_content:
                return None

            content = result.extracted_content
            comments = []

            # Add main tweet
            if content.get('main_tweet'):
                main_tweet = {
                    "text": sanitize_text(content['main_tweet'].get('text', '')),
                    "created_at": datetime.now().isoformat(),
                    "platform": "twitter",
                    "post_url": tweet_url,
                    "post_type": "tweet",
                    "author": content['main_tweet'].get('author', 'Unknown'),
                    "timestamp": content['main_tweet'].get('time', ''),
                    "is_spam": False
                }
                if include_sentiment:
                    main_tweet["sentiment"] = analyze_sentiment(main_tweet["text"])
                comments.append(main_tweet)

            # Add replies
            for reply in content.get('replies', []):
                if not reply.get('text'):
                    continue
                
                entry = {
                    "text": sanitize_text(reply['text']),
                    "created_at": datetime.now().isoformat(),
                    "platform": "twitter",
                    "post_url": tweet_url,
                    "post_type": "reply",
                    "author": reply.get('author', 'Unknown'),
                    "timestamp": reply.get('time', ''),
                    "is_spam": False
                }
                
                if include_sentiment:
                    entry["sentiment"] = analyze_sentiment(entry["text"])
                
                comments.append(entry)

            return comments

    except Exception as e:
        print(f"[ERROR] Crawl4AI Twitter/X extraction failed: {e}")
        return None

def extract_twitter_comments(
    url: str,
    max_results: int = 100,
    include_sentiment: bool = False,
    **kwargs  # Add kwargs to handle additional parameters like lang
) -> List[Dict[str, Any]]:
    """Extract comments from Twitter/X posts
    
    Args:
        url: Tweet URL
        max_results: Maximum number of comments to return
        include_sentiment: Whether to perform sentiment analysis
        **kwargs: Additional parameters (e.g., lang)
    """
    try:
        # Try Crawl4AI first if available
        if CRAWL4AI_AVAILABLE:
            print("[INFO] Attempting to use Crawl4AI for Twitter/X...")
            try:
                import nest_asyncio
                nest_asyncio.apply()
            except (ImportError, RuntimeError):
                pass
            
            posts = asyncio.run(extract_twitter_with_crawl4ai(url, include_sentiment))
            if posts:
                print(f"[INFO] Successfully extracted {len(posts)} tweets/replies using Crawl4AI")
                return posts[:max_results]
            print("[INFO] Crawl4AI extraction failed, falling back to Playwright...")

        # Fall back to Playwright method
        print("[INFO] Using Playwright for Twitter/X extraction...")
        data = extract_twitter_data(url)
        comments = []

        # Add main tweet
        if "tweet" in data:
            main_tweet = {
                "text": sanitize_text(data["tweet"]),
                "created_at": datetime.now().isoformat(),
                "platform": "twitter",
                "post_url": url,
                "post_type": "tweet",
                "author": data.get("profile", "Unknown"),
                "is_spam": False
            }
            if include_sentiment:
                main_tweet["sentiment"] = analyze_sentiment(main_tweet["text"])
            comments.append(main_tweet)

        # Add replies
        for reply in data.get("replies", []):
            if not reply.get("text"):
                continue
            
            entry = {
                "text": sanitize_text(reply["text"]),
                "created_at": datetime.now().isoformat(),
                "platform": "twitter",
                "post_url": url,
                "post_type": "reply",
                "author": reply.get("author", "Unknown"),
                "is_spam": False
            }
            
            if include_sentiment:
                entry["sentiment"] = analyze_sentiment(entry["text"])
            
            comments.append(entry)

        print(f"[INFO] Extracted {len(comments)} tweets/replies using Playwright")
        return comments[:max_results]

    except Exception as e:
        print(f"[ERROR] Failed to extract Twitter/X comments: {e}")
        return []

# --- Instagram ---
def extract_instagram_data(url, fetch_profile=False):
    html = fetch_with_playwright(url, wait_selector="article")
    soup = BeautifulSoup(html, "html.parser")
    data = {}
    # Example: extract post caption
    caption = soup.select_one("div[role='button'] ~ div > span")
    if caption:
        data["caption"] = caption.get_text(strip=True)
    # Example: extract comments
    comments = []
    for c in soup.select("ul ul > div > li > div > div > div > span"):
        text = c.get_text(strip=True)
        if text:
            comments.append({"text": text})
    data["comments"] = comments
    # Example: extract profile
    if fetch_profile:
        profile = soup.select_one("header a")
        if profile:
            data["profile"] = profile.get_text(strip=True)
    return data

# --- Reddit ---
def extract_reddit_data(url, fetch_profile=False):
    html = fetch_with_playwright(url, wait_selector="div[data-test-id='post-content']")
    soup = BeautifulSoup(html, "html.parser")
    data = {}
    # Example: extract post
    post = soup.select_one("div[data-test-id='post-content']")
    if post:
        data["post"] = post.get_text(separator="\n", strip=True)
    # Example: extract comments
    comments = []
    for c in soup.select("div[data-test-id='comment']"):
        text = c.get_text(separator="\n", strip=True)
        if text:
            comments.append({"text": text})
    data["comments"] = comments
    # Example: extract profile
    if fetch_profile:
        profile = soup.select_one("a[data-click-id='user']")
        if profile:
            data["profile"] = profile.get_text(strip=True)
    return data

# --- YouTube (yt-dlp) ---
def extract_youtube_data(url, fetch_profile=False, download_video=False):
    if not YTDLP_AVAILABLE:
        print("[ERROR] yt-dlp library is not installed. Please install it: pip install yt-dlp")
        return None
    ydl_opts = {
        "quiet": True,
        "skip_download": not download_video,
        "extract_flat": False,
        "forcejson": True,
        "dump_single_json": True,
        "writesubtitles": False,
        "writeinfojson": True,
        "writeautomaticsub": False,
        "nocheckcertificate": True,
        "proxy": get_random_proxy(),
        "user_agent": get_random_user_agent(),
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=download_video)
    return info

# --- Unified interface ---
def extract_social_data(platform, url, fetch_profile=False, download_video=False):
    """
    Unified interface to extract post/comments/profile from supported platforms.
    """
    delay(1, 3)  # Respectful delay
    if platform == "facebook":
        return extract_facebook_data(url, fetch_profile=fetch_profile)
    elif platform in ["twitter", "x"]:
        return extract_twitter_data(url, fetch_profile=fetch_profile)
    elif platform == "instagram":
        return extract_instagram_data(url, fetch_profile=fetch_profile)
    elif platform == "reddit":
        return extract_reddit_data(url, fetch_profile=fetch_profile)
    elif platform == "youtube":
        return extract_youtube_data(url, fetch_profile=fetch_profile, download_video=download_video)
    else:
        raise ValueError(f"Platform '{platform}' not supported.")

# --- Legal & Ethical best practices ---
# - Always check and comply with each site's Terms of Service.
# - Use rate limiting and delays (e.g., time.sleep) between requests.
# - Respect robots.txt.
# - Use proxy rotation and random user agents for scraping.
# - Never overload or harm target servers.

# Helper function to sanitize text
def sanitize_text(text: str) -> str:
    """Clean text from unwanted characters and formatting"""
    if not text:
        return ""
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Fix spacing
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def spam_filter(comments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter out spam comments using basic heuristics"""
    filtered = []
    spam_patterns = [
        # Common spam patterns (Thai and English)
        r'ดูได้ที่|click here|คลิกเลย',
        r'line\s*id|line\s*@|ไลน์',
        r'\d{3,}.*\d{3,}',  # Multiple numbers (e.g., phone numbers)
        r'www\.|\.com|\.net|http|bit\.ly',
        r'ผลิตภัณฑ์|สนใจติดต่อ|สั่งซื้อ',
        r'ราคา.*บาท|บาท.*ราคา',
        r'promotion|โปรโมชั่น|ส่วนลด',
        r'vip|premium|member',
        r'subscribe|สมัครสมาชิก',
        r'free|ฟรี',
        r'casino|เว็บพนัน|บาคาร่า|สล็อต',
        r'viagra|ยาเพิ่ม',
        r'weight.*loss|ลดน้ำหนัก|ลดความอ้วน',
        # Add more patterns as needed
    ]
    
    # Combine patterns
    spam_regex = re.compile('|'.join(spam_patterns), re.IGNORECASE)
    
    for comment in comments:
        text = comment.get('text', '').lower()
        
        # Skip empty comments
        if not text:
            continue
            
        # Check for spam patterns
        if spam_regex.search(text):
            comment['is_spam'] = True
            continue
            
        # Check for excessive special characters
        special_char_ratio = len(re.findall(r'[!@#$%^&*()_+=\[\]{};:"/\\|,.<>?~\d]', text)) / len(text)
        if special_char_ratio > 0.3:  # More than 30% special characters
            comment['is_spam'] = True
            continue
            
        # Check for repetitive characters
        if re.search(r'(.)\1{4,}', text):  # Same character repeated 5+ times
            comment['is_spam'] = True
            continue
            
        # Check for very short or very long comments
        if len(text) < 5 or len(text) > 5000:
            comment['is_spam'] = True
            continue
        
        # Not spam
        comment['is_spam'] = False
        filtered.append(comment)
    
    return filtered

def analyze_sentiment(text: str, lang: str = "th") -> str:
    """
    Basic sentiment analysis for Thai and English text
    
    Args:
        text: Text to analyze
        lang: Language code ("th" for Thai, "en" for English)
        
    Returns:
        Sentiment label ("positive", "negative", "neutral")
    """
    # Lowercased text for matching
    text_lower = text.lower()
    
    # Thai and English positive/negative word lists
    pos_words = {
        # Thai positive words
        "ดี", "เยี่ยม", "สุดยอด", "ชอบ", "สนุก", "รัก", "ขอบคุณ", 
        "ยอดเยี่ยม", "เก่ง", "น่ารัก", "เจ๋ง", "ว้าว", "โคตรดี",
        "ประทับใจ", "พอใจ", "สวย", "เลิศ",
        # English positive words
        "good", "great", "excellent", "happy", "love", "awesome",
        "perfect", "thanks", "amazing", "wonderful", "best"
    }
    
    neg_words = {
        # Thai negative words
        "แย่", "เลว", "ไม่ดี", "เกลียด", "น่าเบื่อ", "น่ารำคาญ",
        "ผิดหวัง", "เสียใจ", "โกรธ", "ไม่พอใจ", "แย่มาก",
        # English negative words
        "bad", "awful", "terrible", "sad", "hate", "poor", "boring",
        "worst", "horrible", "disappointing"
    }
    
    # Special cases for Thai
    if "555" in text or "ฮ่าๆ" in text or "😂" in text or "🤣" in text:
        return "positive"
    
    if "❤️" in text or "❤" in text or "👍" in text:
        return "positive"
        
    if "👎" in text or "😡" in text or "🤬" in text:
        return "negative"
    
    # Count positive and negative words
    pos_count = sum(1 for word in pos_words if word in text_lower)
    neg_count = sum(1 for word in neg_words if word in text_lower)
    
    # Determine sentiment
    if pos_count > neg_count:
        return "positive"
    elif neg_count > pos_count:
        return "negative"
    else:
        return "neutral"

# --- Combined Function ---
def extract_social_media_comments(
    platform: str,
    query: str, 
    max_results: int = 100,
    include_sentiment: bool = False,
    filter_spam: bool = True,
    silent: bool = True,
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Extract comments from various social media platforms
    
    Args:
        platform: Platform name ("twitter", "x", "reddit", "youtube", "pantip", "facebook", "threads", "file")
        query: Search query or ID (depends on platform)
        max_results: Maximum number of comments to return
        include_sentiment: Whether to perform basic sentiment analysis
        filter_spam: Whether to filter out spam comments
        silent: Whether to suppress log messages
        **kwargs: Additional platform-specific parameters
        
    Returns:
        List of comments in dictionary format
    """
    # คำนวณจำนวนข้อมูลที่ต้องดึงเพิ่ม (เผื่อพบสแปม)
    initial_fetch = max_results
    if filter_spam:
        # เพิ่มจำนวนการดึงข้อมูลเริ่มต้นเป็น 150% ของที่ต้องการ
        initial_fetch = int(max_results * 1.5)
    
    platform = platform.lower()
    comments = []
    
    try:
        # Fall back to existing methods if SocialReaper fails
        if platform == "twitter" or platform == "x":
            if not silent:
                print("[INFO] Extracting data from Twitter/X...")
            comments = extract_twitter_comments(
                query, 
                max_results=initial_fetch,
                lang=kwargs.get("lang", "th"),
                include_sentiment=include_sentiment
            )
        elif platform == "threads":
            if not silent:
                print("[INFO] Extracting data from Threads...")
            comments = extract_threads_comments(
                query,  # query is thread URL
                max_results=initial_fetch,
                include_sentiment=include_sentiment,
                timeout=kwargs.get("timeout", 20) # Fixed: Use kwargs.get or a default for timeout
            )
        elif platform == "reddit":
            if not silent:
                print("[INFO] Extracting data from Reddit...")
            comments = extract_reddit_comments(
                query,  # query is subreddit name
                limit=initial_fetch,
                time_filter=kwargs.get("time_filter", "week"),
                include_sentiment=include_sentiment
            )
        elif platform == "youtube":
            if not silent:
                print("[INFO] Extracting data from YouTube...")
            comments = extract_youtube_comments(
                query,  # query is video ID or URL
                max_results=initial_fetch,
                include_sentiment=include_sentiment
            )
        elif platform == "pantip":
            if not silent:
                print("[INFO] Extracting data from Pantip...")
            comments = extract_pantip_comments(
                query,  # query is topic ID or URL
                max_results=initial_fetch,
                include_sentiment=include_sentiment
            )
        elif platform == "facebook":
            if not silent:
                print("[INFO] Extracting data from Facebook...")
            comments = extract_facebook_comments(
                query,  # query is post URL
                max_results=initial_fetch,
                include_sentiment=include_sentiment
            )
            # --- Robust spam removal and refill for Facebook ---
            if filter_spam:
                def fetch_more():
                    # Fetch more comments (with a small batch size)
                    return extract_facebook_comments(
                        query,
                        max_results=20, # This was missing in the provided snippet, assuming it's like this
                        include_sentiment=include_sentiment
                    )
                comments = remove_spam_and_fill(comments, fetch_more_func=fetch_more, max_results=max_results)
            else:
                comments = comments[:max_results]
            print(f"[INFO] Extracted {len(comments)} Facebook comments (cleaned)")
            return comments
        elif platform == "file":
            if not silent:
                print("[INFO] Loading comments from file...")
            # Load comments from a local file (JSONL format)
            try:
                with open(query, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            comment = json.loads(line)
                            comments.append(comment)
                        except json.JSONDecodeError:
                            continue
                
                print(f"[INFO] Loaded {len(comments)} comments from file")
            except Exception as e:
                print(f"[ERROR] Failed to load comments from file: {e}")
                comments = []
        
        else:
            print(f"[ERROR] Unsupported platform: {platform}")
            return []
        
        # --- Spam Filtering for other platforms ---
        if filter_spam and comments and platform != "facebook":
            print(f"[INFO] Filtering out spam comments...")
            comments = spam_filter(comments)
        
        # Limit results to max_results
        if max_results > 0 and len(comments) > max_results:
            comments = comments[:max_results]
        
        print(f"[INFO] Extracted {len(comments)} comments from {platform.capitalize()}")
        return comments
    
    except Exception as e:
        print(f"[ERROR] An error occurred: {e}")
        return []

# --- Save comments for training ---
def save_comments_for_training(
    comments: List[Dict[str, Any]],
    output_path: str,
    format: str = "jsonl",
    text_field: str = "text",
    label_field: Optional[str] = "sentiment",
    exclude_spam: bool = True,
    save_spam_separately: bool = False
) -> str:
    """
    Save extracted comments to a file in the specified format.
    
    Args:
        comments: List of comments to save
        output_path: Output file path
        format: File format ("jsonl", "csv", "txt")
        text_field: Field name for the comment text
        label_field: Field name for the label (e.g., sentiment)
        exclude_spam: Whether to exclude spam comments
        save_spam_separately: Whether to save spam comments to a separate file
        
    Returns:
        Path to the saved file
    """
    if exclude_spam:
        comments = [c for c in comments if not c.get("is_spam", False)]
    
    # Separate spam comments if needed
    spam_comments = [c for c in comments if c.get("is_spam", False)]
    if save_spam_separately and spam_comments:
        spam_path = output_path.replace(".", "_spam.")
        if spam_path == output_path:  # If there's no file extension
            spam_path = f"{output_path}_spam"
            
        if format.lower() == "jsonl":
            with open(spam_path, "w", encoding="utf-8") as f:
                for comment in spam_comments:
                    if text_field in comment:
                        f.write(json.dumps(comment, ensure_ascii=False) + "\n")
        
        elif format.lower() == "csv":
            if not PANDAS_AVAILABLE:
                print("[ERROR] pandas library is not installed. Skipping saving spam comments as CSV.")
            else:
                df = pd.DataFrame(spam_comments)
                df.to_csv(spam_path, index=False, encoding="utf-8")
        
        elif format.lower() == "txt":
            with open(spam_path, "w", encoding="utf-8") as f:
                for comment in spam_comments:
                    if text_field in comment:
                        f.write(comment[text_field] + "\n")
        
        print(f"[INFO] Saved {len(spam_comments)} spam comments to: {spam_path}")
    
    # Save main comments
    if not comments:
        print("[WARN] No comments to save")
        return output_path
    
    try:
        if format.lower() == "jsonl":
            with open(output_path, "w", encoding="utf-8") as f:
                for comment in comments:
                    if text_field in comment:
                        f.write(json.dumps(comment, ensure_ascii=False) + "\n")
    
        elif format.lower() == "csv":
            if not PANDAS_AVAILABLE:
                print("[ERROR] pandas library is not installed, but required for CSV format. Skipping CSV save.")
                print("Please install it with: pip install pandas")
                # Save as JSONL as fallback
                fallback_path = output_path.replace(".csv", "_fallback.jsonl")
                print(f"[INFO] Attempting to save as JSONL to: {fallback_path}")
                return save_comments_for_training(
                    comments, fallback_path, "jsonl", 
                    text_field, label_field, exclude_spam, False
                )

            df = pd.DataFrame(comments)
            df.to_csv(output_path, index=False, encoding="utf-8")
        
        elif format.lower() == "txt":
            with open(output_path, "w", encoding="utf-8") as f:
                for comment in comments:
                    if text_field in comment:
                        f.write(comment[text_field] + "\n")
        
        print(f"[INFO] Saved {len(comments)} comments to: {output_path}")
        return output_path
    
    except Exception as e:
        print(f"[ERROR] Failed to save comments: {e}")
        return ""