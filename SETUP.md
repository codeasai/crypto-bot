# การตั้งค่า Python Environment สำหรับ Crypto Bot

## 1. สร้าง Virtual Environment

### Windows (Git Bash)
```bash
# ไปที่ root directory ของ project
cd /c/DEV/crypto-bot

# สร้าง venv
python -m venv venv

# เปิดใช้งาน venv
source venv/Scripts/activate
```

### Windows (CMD)
```bash
# ไปที่ root directory ของ project
cd C:\DEV\crypto-bot

# สร้าง venv
python -m venv venv

# เปิดใช้งาน venv
.\venv\Scripts\activate
```

## 2. ติดตั้ง Dependencies

```bash
# ติดตั้ง packages ที่จำเป็น
pip install -r requirements.txt
```

## 3. ตั้งค่า VS Code

1. กด `Ctrl + Shift + P`
2. พิมพ์ "Python: Select Interpreter"
3. เลือก interpreter จาก venv ที่สร้างไว้ (`./venv/Scripts/python.exe`)

## 4. ตั้งค่า Git Bash (ถ้าใช้)

เพิ่มใน `.bashrc` หรือ `.bash_profile`:
```bash
export PATH="/c/DEV/crypto-bot/venv/Scripts:$PATH"
```

## 5. ตรวจสอบการตั้งค่า

```bash
# ตรวจสอบ Python path
which python

# ตรวจสอบ pip path
which pip

# ตรวจสอบ packages ที่ติดตั้ง
pip list
```

## 6. สร้างไฟล์ Environment

```bash
# สร้างไฟล์ .env
echo "PYTHONPATH=/c/DEV/crypto-bot" > .env
```

## 7. Git Configuration

เพิ่มใน `.gitignore`:
```
venv/
__pycache__/
*.pyc
.env
```

## 8. การใช้งาน

### เปิดใช้งาน venv
```bash
# Git Bash
source venv/Scripts/activate

# CMD
.\venv\Scripts\activate
```

### ปิดใช้งาน venv
```bash
deactivate
```

### ติดตั้ง package ใหม่
```bash
pip install <package_name>
pip freeze > requirements.txt
```

## 9. ข้อควรระวัง

1. ตรวจสอบว่า venv ถูกเปิดใช้งานก่อนรันโปรแกรม
2. อย่าลืมบันทึก dependencies ใหม่เมื่อติดตั้ง package เพิ่มเติม
3. อย่า commit ไฟล์ใน venv/ เข้า git
4. ตรวจสอบ path ให้ถูกต้องตามระบบปฏิบัติการ 