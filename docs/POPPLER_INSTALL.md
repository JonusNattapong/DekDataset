# Poppler Installation Guide for Windows

## 🚀 Quick Install Options

### Option 1: Using Conda (Recommended)
```bash
conda install -c conda-forge poppler
```

### Option 2: Direct Download (Windows)
1. Go to: https://github.com/oschwartz10612/poppler-windows/releases
2. Download the latest `Release-XX.XX.X-0.zip` 
3. Extract to `C:\poppler` or any directory
4. Add `C:\poppler\Library\bin` to your system PATH

### Option 3: Using Chocolatey
```bash
choco install poppler
```

### Option 4: Using pip with precompiled wheels
```bash
pip install pdf2image
# This might work if poppler binaries are bundled
```

## 🔧 Verify Installation

After installation, test with:
```bash
pdftoppm -h
```

## 🎯 Test OCR Again

Once Poppler is installed, retry your OCR command:
```bash
python src/python/generate_dataset.py sentiment_analysis 10 --input-file "sawasdi1.pdf"
```

## 🐛 Troubleshooting

If you still get poppler errors:

1. **Check PATH**: Make sure poppler bin directory is in your system PATH
2. **Restart terminal**: Close and reopen your terminal after installing
3. **Use absolute path**: Specify full path to poppler in environment variable:
   ```bash
   set POPPLER_PATH=C:\poppler\Library\bin
   ```

## 📱 Alternative: Try with Image First

While setting up Poppler, you can test OCR with an image file:
```bash
python src/python/generate_dataset.py sentiment_analysis 5 --input-file "image.png"
```

This will bypass PDF processing and use direct image OCR.
