# Python 3.11 ကို အခြေခံထားသော ပေါ့ပါးသည့် slim version ကို သုံးပါမည်
FROM python:3.11-slim

# ပတ်ဝန်းကျင် ပြင်ဆင်မှုများ (Logs များ ချက်ချင်းပေါ်ရန်နှင့် Cache မကျန်စေရန်)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Container ထဲတွင် အလုပ်လုပ်မည့် Directory ကို သတ်မှတ်ခြင်း
WORKDIR /app

# လိုအပ်သော Library စာရင်းကို Container ထဲသို့ အရင်ကူးထည့်ခြင်း
COPY requirements.txt .

# Library များကို Install လုပ်ခြင်း (Image Size သေးငယ်စေရန် cache များကို ရှင်းလင်းမည်)
RUN pip install --no-cache-dir -r requirements.txt

# ကျန်ရှိနေသော Code ဖိုင်များအားလုံးကို Container ထဲသို့ ကူးထည့်ခြင်း
COPY . .

# Bot ကို စတင် Run မည့် Command
CMD ["python", "aipro.py"]
