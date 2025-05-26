import sys
import os

# Add parent directory to path so we can import our module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from social_media_utils import (
    extract_social_media_comments,
    save_comments_for_training,
    analyze_sentiment
)

# 1. ตัวอย่างการดึงข้อมูลจาก YouTube
print("=== ตัวอย่างการดึงข้อมูล YouTube ===")
youtube_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # หรือใส่ video ID โดยตรง
youtube_comments = extract_social_media_comments(
    platform="youtube",
    query=youtube_url,
    max_results=10,
    include_sentiment=True
)
print(f"ดึงข้อมูลได้ {len(youtube_comments)} ความคิดเห็น")
if youtube_comments:
    print(f"ตัวอย่าง: {youtube_comments[0]['text']}")
    print(f"ความรู้สึก: {youtube_comments[0].get('sentiment', 'ไม่ระบุ')}")

# 2. ตัวอย่างการดึงข้อมูลจาก Facebook
print("\n=== ตัวอย่างการดึงข้อมูล Facebook ===")
facebook_url = "https://www.facebook.com/username/posts/123456789"  # ใส่ URL หรือ ID โพสต์จริง
facebook_comments = extract_social_media_comments(
    platform="facebook",
    query=facebook_url,
    max_results=10,
    include_sentiment=True
)
print(f"ดึงข้อมูลได้ {len(facebook_comments)} ความคิดเห็น")

# 3. ตัวอย่างการดึงข้อมูลจาก Pantip
print("\n=== ตัวอย่างการดึงข้อมูล Pantip ===")
pantip_url = "https://pantip.com/topic/12345678"  # ใส่ URL หรือ ID กระทู้จริง
pantip_comments = extract_social_media_comments(
    platform="pantip",
    query=pantip_url,
    max_results=10,
    include_sentiment=True
)
print(f"ดึงข้อมูลได้ {len(pantip_comments)} ความคิดเห็น")

# 4. ตัวอย่างการดึงข้อมูลจาก Twitter/X
print("\n=== ตัวอย่างการดึงข้อมูล Twitter/X ===")
twitter_query = "#Thailand"  # ใส่ hashtag หรือ keyword ที่ต้องการค้นหา
twitter_comments = extract_social_media_comments(
    platform="twitter",
    query=twitter_query,
    max_results=10,
    lang="th",  # ระบุภาษา (th, en, ฯลฯ)
    include_sentiment=True
)
print(f"ดึงข้อมูลได้ {len(twitter_comments)} ทวีต")

# 5. ตัวอย่างการดึงข้อมูลจาก Reddit
print("\n=== ตัวอย่างการดึงข้อมูล Reddit ===")
subreddit_name = "Thailand"  # ใส่ชื่อ subreddit
reddit_comments = extract_social_media_comments(
    platform="reddit",
    query=subreddit_name,
    max_results=10,
    time_filter="month",  # hour, day, week, month, year, all
    include_sentiment=True
)
print(f"ดึงข้อมูลได้ {len(reddit_comments)} ความคิดเห็น")

# 6. ตัวอย่างการวิเคราะห์ความรู้สึกข้อความภาษาไทยโดยตรง
print("\n=== ตัวอย่างการวิเคราะห์ความรู้สึก ===")
thai_text = "อาหารร้านนี้อร่อยมาก ชอบมากเลย"
sentiment = analyze_sentiment(thai_text, lang="th")
print(f"ข้อความ: '{thai_text}'")
print(f"ความรู้สึก: {sentiment}")

# 7. ตัวอย่างการบันทึกข้อมูลสำหรับการเทรนโมเดล
print("\n=== ตัวอย่างการบันทึกข้อมูลสำหรับเทรนโมเดล ===")

# บันทึกในรูปแบบ JSONL (แนะนำสำหรับเก็บข้อมูลที่มีโครงสร้าง)
if youtube_comments:
    jsonl_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                             "data", "youtube_comments.jsonl")
    save_comments_for_training(youtube_comments, jsonl_path, format="jsonl")
    print(f"บันทึกข้อมูล JSONL ที่: {jsonl_path}")

    # บันทึกในรูปแบบ CSV
    csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                           "data", "youtube_comments.csv")
    save_comments_for_training(youtube_comments, csv_path, format="csv")
    print(f"บันทึกข้อมูล CSV ที่: {csv_path}")
    
    # บันทึกในรูปแบบ text (เหมาะสำหรับเทรนโมเดลภาษา)
    txt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                           "data", "youtube_comments.txt")
    save_comments_for_training(youtube_comments, txt_path, format="txt", 
                              text_field="text", label_field="sentiment")
    print(f"บันทึกข้อมูล TXT ที่: {txt_path}")
