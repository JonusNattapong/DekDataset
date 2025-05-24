#!/usr/bin/env python
# coding: utf-8
# subject_prediction_gui.py - GUI อย่างง่ายสำหรับทำนายวิชาจากข้อความ

import sys
import os
import json
import torch
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class SubjectPredictionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("DekDataset - ระบบทำนายวิชาจากข้อความ")
        self.root.geometry("800x600")
        self.root.minsize(600, 400)
        
        # ตั้งค่าพาธของโมเดล
        self.model_paths = {
            "Qwen": "data/output/thai-education-qwen-model",
            "WangchanBERTa": "data/output/thai-education-model"
        }
        
        # ตัวแปรสำหรับเก็บโมเดลที่โหลดแล้ว
        self.models = {}
        self.tokenizers = {}
        self.id2labels = {}
        
        # สร้าง UI
        self.create_widgets()
        
        # โหลดโมเดลเริ่มต้น
        self.load_model_async(list(self.model_paths.keys())[0])

    def create_widgets(self):
        # สร้างเฟรมหลัก
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # เลือกโมเดล
        model_frame = ttk.LabelFrame(main_frame, text="เลือกโมเดล", padding="10")
        model_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.model_var = tk.StringVar()
        for i, model_name in enumerate(self.model_paths.keys()):
            rb = ttk.Radiobutton(model_frame, text=model_name, variable=self.model_var, value=model_name)
            rb.pack(side=tk.LEFT, padx=10)
            if i == 0:
                rb.invoke()
        
        # ช่องกรอกข้อความ
        input_frame = ttk.LabelFrame(main_frame, text="ข้อความที่ต้องการทำนาย", padding="10")
        input_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.text_input = scrolledtext.ScrolledText(input_frame, height=10, wrap=tk.WORD)
        self.text_input.pack(fill=tk.BOTH, expand=True)
        
        # ตัวอย่างข้อความ
        example_frame = ttk.LabelFrame(main_frame, text="ตัวอย่างข้อความ", padding="10")
        example_frame.pack(fill=tk.X, padx=5, pady=5)
        
        examples = [
            "ไข่ -> ตัวหนอน -> ดักแด้ -> ผีเสื้อ",  # วิทยาศาสตร์
            "การบวกเลขสองหลัก 25 + 13 = 38",  # คณิตศาสตร์
            "พระเจ้าอยู่หัวรัชกาลที่ 9 เป็นที่รักของคนไทย", # สังคมศึกษา
            "การอ่านพยัญชนะไทย ก ไก่ ข ไข่",  # ภาษาไทย
            "I can speak English very well" # ภาษาอังกฤษ
        ]
        
        for example in examples:
            btn = ttk.Button(example_frame, text=example[:30] + "..." if len(example) > 30 else example,
                            command=lambda e=example: self.fill_example(e))
            btn.pack(side=tk.LEFT, padx=5)
        
        # ปุ่มทำนาย
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=10)
        
        self.predict_btn = ttk.Button(button_frame, text="ทำนาย", command=self.predict)
        self.predict_btn.pack(side=tk.LEFT, padx=5)
        
        clear_btn = ttk.Button(button_frame, text="ล้างข้อความ", command=self.clear_text)
        clear_btn.pack(side=tk.LEFT, padx=5)
        
        # สถานะการโหลดโมเดล
        self.status_var = tk.StringVar()
        self.status_var.set("กำลังโหลดโมเดล...")
        status_label = ttk.Label(button_frame, textvariable=self.status_var)
        status_label.pack(side=tk.RIGHT, padx=5)
        
        # ผลลัพธ์การทำนาย
        result_frame = ttk.LabelFrame(main_frame, text="ผลการทำนาย", padding="10")
        result_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # สร้าง Treeview สำหรับแสดงผลการทำนาย
        columns = ('subject', 'confidence')
        self.result_tree = ttk.Treeview(result_frame, columns=columns, show='headings')
        self.result_tree.heading('subject', text='วิชา')
        self.result_tree.heading('confidence', text='ความมั่นใจ (%)')
        self.result_tree.column('subject', width=150)
        self.result_tree.column('confidence', width=100)
        self.result_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # เพิ่ม scrollbar
        scrollbar = ttk.Scrollbar(result_frame, orient=tk.VERTICAL, command=self.result_tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.result_tree.configure(yscrollcommand=scrollbar.set)

    def fill_example(self, example):
        """นำตัวอย่างมาใส่ในช่องข้อความ"""
        self.text_input.delete(1.0, tk.END)
        self.text_input.insert(tk.END, example)

    def clear_text(self):
        """ล้างข้อความในช่องข้อความ"""
        self.text_input.delete(1.0, tk.END)

    def load_model_async(self, model_name):
        """โหลดโมเดลแบบ asynchronous"""
        self.status_var.set(f"กำลังโหลดโมเดล {model_name}...")
        self.predict_btn.config(state="disabled")
        
        # สร้าง Thread ใหม่สำหรับการโหลดโมเดล
        thread = threading.Thread(target=self.load_model, args=(model_name,))
        thread.daemon = True
        thread.start()

    def load_model(self, model_name):
        """โหลดโมเดลตามที่เลือก"""
        try:
            if model_name not in self.models:
                model_path = self.model_paths.get(model_name)
                if not model_path or not os.path.exists(model_path):
                    self.show_error(f"ไม่พบโฟลเดอร์โมเดล {model_path}")
                    return
                
                # โหลด tokenizer
                tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
                if tokenizer.pad_token is None and hasattr(tokenizer, 'eos_token'):
                    tokenizer.pad_token = tokenizer.eos_token
                
                # โหลดโมเดล
                model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
                
                # ดึง labels
                if hasattr(model.config, "id2label"):
                    id2label = model.config.id2label
                else:
                    # กำหนดค่าเริ่มต้นถ้าไม่มี
                    subjects = [
                        "ภาษาไทย", "คณิตศาสตร์", "วิทยาศาสตร์", "สังคมศึกษา", 
                        "ศิลปะ", "สุขศึกษาและพลศึกษา", "การงานอาชีพและเทคโนโลยี", "ภาษาอังกฤษ"
                    ]
                    id2label = {i: subjects[i] for i in range(len(subjects))}
                
                # เก็บโมเดลไว้ใช้งาน
                self.models[model_name] = model
                self.tokenizers[model_name] = tokenizer
                self.id2labels[model_name] = id2label
            
            # อัพเดตสถานะ
            self.root.after(0, lambda: self.status_var.set(f"พร้อมใช้งานโมเดล {model_name}"))
            self.root.after(0, lambda: self.predict_btn.config(state="normal"))
            
        except Exception as e:
            self.root.after(0, lambda: self.show_error(f"เกิดข้อผิดพลาดในการโหลดโมเดล: {str(e)}"))

    def predict(self):
        """ทำนายวิชาจากข้อความที่ผู้ใช้กรอก"""
        # ดึงข้อความจาก textbox
        text = self.text_input.get(1.0, tk.END).strip()
        if not text:
            messagebox.showwarning("คำเตือน", "กรุณากรอกข้อความที่ต้องการทำนาย")
            return
        
        model_name = self.model_var.get()
        
        # ตรวจสอบว่าโมเดลโหลดเรียบร้อยหรือไม่
        if model_name not in self.models:
            messagebox.showwarning("คำเตือน", f"โมเดล {model_name} ยังไม่ได้ถูกโหลด กรุณารอสักครู่")
            return
        
        # ดึงโมเดลมาใช้งาน
        model = self.models[model_name]
        tokenizer = self.tokenizers[model_name]
        id2label = self.id2labels[model_name]
        
        try:
            # เตรียม input
            inputs = tokenizer(text, return_tensors="pt", truncation=True)
            
            # ทำนาย
            with torch.no_grad():
                outputs = model(**inputs)
            
            # แปลงผลลัพธ์เป็นความน่าจะเป็น
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
            
            # ล้างผลลัพธ์เดิม
            for item in self.result_tree.get_children():
                self.result_tree.delete(item)
            
            # เรียงลำดับผลลัพธ์ตามความมั่นใจ (จากมากไปน้อย)
            sorted_indices = torch.argsort(probs, descending=True)
            for idx in sorted_indices:
                i = idx.item()
                if i in id2label:
                    subject = id2label[i]
                    confidence = probs[i].item() * 100
                    self.result_tree.insert('', tk.END, values=(subject, f"{confidence:.2f}"))
            
            # ไฮไลท์ผลลัพธ์แรก (วิชาที่มีความน่าจะเป็นมากที่สุด)
            if self.result_tree.get_children():
                self.result_tree.selection_set(self.result_tree.get_children()[0])
        
        except Exception as e:
            self.show_error(f"เกิดข้อผิดพลาดในการทำนาย: {str(e)}")

    def show_error(self, message):
        """แสดงข้อความผิดพลาด"""
        messagebox.showerror("ข้อผิดพลาด", message)
        self.status_var.set("เกิดข้อผิดพลาด")
        self.predict_btn.config(state="normal")

def main():
    # สร้างและเริ่มต้น GUI
    root = tk.Tk()
    app = SubjectPredictionGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
