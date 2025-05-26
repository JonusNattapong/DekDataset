from social_media_utils import extract_social_media_comments, save_comments_for_training
import sys
import os
import signal
import argparse # Added

# สร้างฟังก์ชันสำหรับจัดการ timeout
def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out")

def main():
    parser = argparse.ArgumentParser(description="Extract comments from social media platforms.")
    parser.add_argument("platform", help="Platform name (e.g., youtube, pantip, twitter)")
    parser.add_argument("query", help="Search query or URL")
    parser.add_argument("--max_results", type=int, default=10, help="Maximum number of comments to return (default: 10)")
    parser.add_argument("--use-selenium", action=argparse.BooleanOptionalAction, default=True, help="For Pantip: --use-selenium to use Selenium, --no-use-selenium to use requests (default: True)")
    parser.add_argument("--timeout", type=int, default=60, help="Timeout for the entire operation in seconds (default: 60)")
    parser.add_argument("--silent", action="store_true", help="Suppress informational messages")

    args = parser.parse_args()

    platform = args.platform.lower()
    query = args.query
    max_results = args.max_results
    
    print(f"กำลังดึงข้อมูลจาก {platform}...")
    
    # ตั้งค่า timeout
    operation_timeout = args.timeout
    try:
        if hasattr(signal, 'SIGALRM'):
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(operation_timeout)
    except (AttributeError, ValueError):
        pass
    
    try:
        kwargs_for_extraction = {
            "timeout": 20,  # Default internal timeout for some functions
            "use_fallback": True
        }
        if platform == "pantip":
            kwargs_for_extraction["use_selenium"] = args.use_selenium
            print(f"[INFO] Pantip extraction will use_selenium: {args.use_selenium}")

        comments = extract_social_media_comments(
            platform=platform,
            query=query,
            max_results=max_results,
            include_sentiment=True,
            filter_spam=True,
            silent=args.silent,
            **kwargs_for_extraction
        )
        
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)
        
        print(f"ดึงข้อมูลได้ {len(comments)} ความคิดเห็น")
        
        if comments:
            # Create data directory if it doesn't exist
            data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
            os.makedirs(data_dir, exist_ok=True)
            
            output_file = os.path.join(data_dir, f"{platform}_comments.jsonl")
            save_comments_for_training(
                comments, 
                output_file,
                exclude_spam=True,
                save_spam_separately=True
            )
            print(f"บันทึกข้อมูลไปที่ {output_file}")
            
            # Display first comment as sample
            if len(comments) > 0:
                first_comment = comments[0]
                print("\nตัวอย่างความคิดเห็นแรก:")
                print(f"ข้อความ: {first_comment['text']}")
                # Remove sentiment check since we're not importing analyze_sentiment
                if 'sentiment' in first_comment:  # Keep this check since comments might still have sentiment from extraction
                    print(f"ความรู้สึก: {first_comment['sentiment']}")
        else:
            print("ไม่พบข้อมูลความคิดเห็น")
    
    except TimeoutError:
        print(f"เกิดข้อผิดพลาด: การดึงข้อมูลใช้เวลานานเกินกำหนด กรุณาลองใหม่อีกครั้ง")
    except KeyboardInterrupt:
        print("\nยกเลิกการทำงานโดยผู้ใช้")
    except Exception as e:
        print(f"เกิดข้อผิดพลาด: {e}")
    
    finally:
        # ยกเลิก timeout ในทุกกรณี
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)

if __name__ == "__main__":
    main()
