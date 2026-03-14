# 💡 Playwright မှ တရားဝင်ထုတ်ထားသော အဆင်သင့် Image ကို အသုံးပြုပါမည်
# (Browser နှင့် OS Dependencies အားလုံး ပါဝင်ပြီးသား ဖြစ်ပါသည်)
FROM mcr.microsoft.com/playwright/python:v1.42.0-jammy

# ပတ်ဝန်းကျင် ပြင်ဆင်မှုများ (Logs များ ချက်ချင်းပေါ်ရန်နှင့် Cache မကျန်စေရန်)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Container ထဲတွင် အလုပ်လုပ်မည့် Directory ကို သတ်မှတ်ခြင်း
WORKDIR /app

# လိုအပ်သော Library စာရင်းကို Container ထဲသို့ အရင်ကူးထည့်ခြင်း
COPY requirements.txt .

# Library များကို Install လုပ်ခြင်း
RUN pip install --no-cache-dir -r requirements.txt

# ကျန်ရှိနေသော Code ဖိုင်များအားလုံးကို Container ထဲသို့ ကူးထည့်ခြင်း
COPY . .

# Bot ကို စတင် Run မည့် Command (အစ်ကို့ရဲ့ ဖိုင်နာမည် aipro.py အတိုင်း ထားထားပါသည်)
CMD ["python", "aipro.py"]
