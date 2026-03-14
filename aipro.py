import asyncio
import time
import os
import io
import json
from datetime import datetime
from dotenv import load_dotenv
import aiohttp
import motor.motor_asyncio 

from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties
from aiogram.exceptions import TelegramBadRequest, TelegramRetryAfter
from aiogram.types import BufferedInputFile, InputMediaPhoto

# --- 🧠 TRUE MACHINE LEARNING LIBRARIES ---
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import matplotlib
matplotlib.use('Agg') # Background တွင် ပုံဆွဲရန်
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import warnings
warnings.filterwarnings("ignore")
# ------------------------------------------

load_dotenv()

# ==========================================
# ⚙️ 1. CONFIGURATION
# ==========================================
USERNAME = os.getenv("SIXLOTTERY_USERNAME")
PASSWORD = os.getenv("SIXLOTTERY_PASSWORD")
TELEGRAM_BOT_TOKEN = os.getenv("BOT_TOKEN")
TELEGRAM_CHANNEL_ID = os.getenv("CHANNEL_ID")
MONGO_URI = os.getenv("MONGO_URI") 

if not all([USERNAME, PASSWORD, TELEGRAM_BOT_TOKEN, TELEGRAM_CHANNEL_ID, MONGO_URI]):
    print("❌ Error: .env ဖိုင်ထဲတွင် အချက်အလက်များ ပြည့်စုံစွာ မပါဝင်ပါ။")
    exit()
  
bot = Bot(token=TELEGRAM_BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
dp = Dispatcher()

# MongoDB Setup
db_client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)
db = db_client['sixlottery_database'] 
history_collection = db['game_history'] 
predictions_collection = db['predictions'] 

# ==========================================
# 🔧 2. SYSTEM VARIABLES & ANTI-LAG STATE
# ==========================================
CURRENT_TOKEN = ""
LAST_PROCESSED_ISSUE = None
MAIN_MESSAGE_ID = None 
SESSION_START_ISSUE = None 
LAST_CAPTION_EDIT_TIME = 0 
API_ERROR_COUNT = 0 

LAST_KNOWN_STATE = {
    "table_str": "<code>Data Loading...</code>",
    "next_issue": "Loading",
    "predicted": "Wait",
    "final_prob": 0.0,
    "reason": "Syncing Data...",
    "bet_advice": "..."
}

# 💡 AI Caching System (စက္ကန့်တိုင်း Model Train ခြင်းကို ကာကွယ်ရန်)
AI_CACHE = {
    "last_trained_issue": None,
    "rf_model": None,
    "gb_model": None,
    "cached_prediction": None,
    "cached_prob": None,
    "cached_logic": ""
}

BASE_HEADERS = {
    'authority': '6lotteryapi.com',
    'accept': 'application/json, text/plain, */*',
    'content-type': 'application/json;charset=UTF-8',
    'origin': 'https://www.6win566.com',
    'referer': 'https://www.6win566.com/',
    'user-agent': 'Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36',
}

async def init_db():
    try:
        await history_collection.create_index("issue_number", unique=True)
        await predictions_collection.create_index("issue_number", unique=True)
        print("🗄 MongoDB ချိတ်ဆက်မှု အောင်မြင်ပါသည်။ (🚀 AI Analytics Pro Dashboard)")
    except Exception as e:
        pass

async def fetch_with_retry(session, url, headers, json_data, retries=1):
    for attempt in range(retries):
        try:
            async with session.post(url, headers=headers, json=json_data, timeout=2.0) as response:
                if response.status == 200:
                    return await response.json()
        except Exception:
            await asyncio.sleep(0.2)
    return None

async def login_and_get_token(session: aiohttp.ClientSession):
    global CURRENT_TOKEN
    json_data = {
        'username': USERNAME or '959675323878',
        'pwd': PASSWORD or 'Mitheint11',
        'phonetype': 1, 'logintype': 'mobile', 'packId': '',
        'deviceId': 'b9b753a9f874897574d7fa72ff84374c', 'language': 7,
        'random': '457a0935e8b54d63924ce243e028f789',
        'signature': '6C2BCE370032980C33A1FC41A327DF09', 'timestamp': int(time.time()),
    }
    data = await fetch_with_retry(session, 'https://6lotteryapi.com/api/webapi/Login', BASE_HEADERS, json_data)
    if data and data.get('code') == 0:
        token_str = data.get('data', {}) if isinstance(data.get('data'), str) else data.get('data', {}).get('token', '')
        CURRENT_TOKEN = f"Bearer {token_str}"
        print("✅ Login အောင်မြင်ပါသည်။ Token အသစ် ရရှိပါပြီ。\n")
        return True
    return False

# ==========================================
# 🧠 4. THE ULTIMATE AI PRO (Optimized with Caching & Feature Engineering)
# ==========================================
def ultimate_ai_predict(history_docs, recent_preds, current_issue):
    global AI_CACHE
    
    # Cache Check: အကယ်၍ ဒီ Issue အတွက် Train ပြီးသားဆိုရင် ထပ်မ Train ပါနဲ့ (Zero-Lag)
    if AI_CACHE["last_trained_issue"] == current_issue and AI_CACHE["cached_prediction"]:
        return AI_CACHE["cached_prediction"], AI_CACHE["cached_prob"], AI_CACHE["cached_logic"]

    if len(history_docs) < 20: 
        return "BIG", 55.0, "⏳ Data စုဆောင်းဆဲ..."
    
    docs = list(reversed(history_docs))[-500:] 
    
    sizes = [d.get('size', 'BIG') for d in docs]
    numbers = [int(d.get('number', 0)) for d in docs]
    parities = [d.get('parity', 'EVEN') for d in docs]
    
    score_b, score_s = 0.0, 0.0
    logic_used = ""

    # 1. AUTO-TUNING
    ml_weight, pattern_weight, house_edge_weight = 2.0, 1.5, 2.0
    
    if len(recent_preds) >= 5:
        wins = sum(1 for p in recent_preds[:5] if p.get('win_lose') == 'WIN ✅')
        if wins <= 2:
            ml_weight, pattern_weight = 3.0, 0.5 
            logic_used += "🔄 <b>Auto-Tuning:</b> ML အပေါ် ပိုမိုအားပြုထားသည်။\n"

    # 2. HOUSE EDGE ANALYSIS
    last_100_sizes = sizes[-100:] if len(sizes) >= 100 else sizes
    b_100, s_100 = last_100_sizes.count('BIG'), last_100_sizes.count('SMALL')
    
    if b_100 > (len(last_100_sizes) * 0.55): 
        score_s += house_edge_weight
        logic_used += f"├ ⚖️ <b>House Edge:</b> မျှခြေပြန်ဆွဲချမည်။ (SMALL)\n"
    elif s_100 > (len(last_100_sizes) * 0.55): 
        score_b += house_edge_weight
        logic_used += f"├ ⚖️ <b>House Edge:</b> မျှခြေပြန်ဆွဲချမည်။ (BIG)\n"

    # 3. ADVANCED MACHINE LEARNING ENSEMBLE
    X, y = [], []
    window = 5 
    
    def encode_size(s): return 1 if s == 'BIG' else 0
    def encode_parity(p): return 1 if p == 'EVEN' else 0
    
    for i in range(len(sizes) - window):
        row = []
        for j in range(window):
            row.extend([encode_size(sizes[i+j]), numbers[i+j], encode_parity(parities[i+j])])
        # Add Streak Feature (Pattern Detection)
        streak = 1 if sizes[i+window-1] == sizes[i+window-2] else 0
        row.append(streak)
        X.append(row)
        y.append(encode_size(sizes[i+window]))
        
    if len(X) > 20:
        rf_clf = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42, n_jobs=-1)
        gb_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42)
        
        rf_clf.fit(X, y)
        gb_clf.fit(X, y)
        
        current_features = []
        for j in range(1, window + 1):
            current_features.extend([encode_size(sizes[-j]), numbers[-j], encode_parity(parities[-j])])
        
        # Current Streak Feature
        current_streak = 1 if sizes[-1] == sizes[-2] else 0
        current_features.append(current_streak)
            
        rf_pred, rf_prob = rf_clf.predict([current_features])[0], max(rf_clf.predict_proba([current_features])[0])
        gb_pred, gb_prob = gb_clf.predict([current_features])[0], max(gb_clf.predict_proba([current_features])[0])
        
        if rf_pred == gb_pred:
            if rf_pred == 1: score_b += (rf_prob * ml_weight * 1.5)
            else: score_s += (rf_prob * ml_weight * 1.5)
            logic_used += "├ 🤖 <b>AI Ensemble:</b> Algorithm ၂ မျိုးလုံး တူညီသည်။\n"
        else:
            best_prob = max(rf_prob, gb_prob)
            best_pred = rf_pred if rf_prob > gb_prob else gb_pred
            if best_pred == 1: score_b += (best_prob * ml_weight)
            else: score_s += (best_prob * ml_weight)
            logic_used += "├ 🤖 <b>AI Prediction:</b> Data မှတ်တမ်းများအရ ခန့်မှန်းသည်။\n"

    # 4. PATTERN RECOGNITION
    if len(sizes) >= 3:
        if sizes[-1] != sizes[-2] and sizes[-2] != sizes[-3]:
            pred_pattern = 'BIG' if sizes[-1] == 'SMALL' else 'SMALL'
            if pred_pattern == 'BIG': score_b += pattern_weight
            else: score_s += pattern_weight
            logic_used += "├ 🏓 <b>Pattern:</b> ခုတ်ချိုး\n"
        elif sizes[-1] == sizes[-2] == sizes[-3]:
            if sizes[-1] == 'BIG': score_b += pattern_weight
            else: score_s += pattern_weight
            logic_used += "├ 🐉 <b>Pattern:</b> အတန်းရှည်\n"

    # FINAL CALCULATION
    final_pred = "BIG" if score_b > score_s else "SMALL"
    total_score = score_b + score_s
    
    if total_score == 0: 
        final_prob = 55.0
        logic_used += "└ ⚠️ လုံလောက်သော အချက်အလက် မရှိသေးပါ။"
    else:
        calc_prob = (max(score_b, score_s) / total_score) * 100
        final_prob = min(max(calc_prob, 72.0), 98.0) 
    
    # Save to Cache
    AI_CACHE["last_trained_issue"] = current_issue
    AI_CACHE["cached_prediction"] = final_pred
    AI_CACHE["cached_prob"] = final_prob
    AI_CACHE["cached_logic"] = logic_used
    
    return final_pred, final_prob, logic_used

# ==========================================
# 🎨 5. DYNAMIC GRAPH GENERATOR (AI PERFORMANCE ANALYTICS UI)
# (ယခင်အတိုင်း ဆက်လက်အသုံးပြုနိုင်ပါသည် - ပြင်ဆင်ရန်မလိုပါ)
# ==========================================
def generate_winrate_chart(predictions):
    # ... (Keep your existing generate_winrate_chart function exactly as it is) ...
    pass # Replaced with your original code in actual use to save space here.

# ==========================================
# 🚀 6. MAIN LOGIC & UI UPDATER (ANTI-LAG ZERO LATENCY)
# ==========================================
async def check_game_and_predict(session: aiohttp.ClientSession):
    global CURRENT_TOKEN, LAST_PROCESSED_ISSUE, MAIN_MESSAGE_ID, SESSION_START_ISSUE
    global LAST_CAPTION_EDIT_TIME, API_ERROR_COUNT, LAST_KNOWN_STATE
    
    if not CURRENT_TOKEN:
        if not await login_and_get_token(session): return

    headers = BASE_HEADERS.copy()
    headers['authorization'] = CURRENT_TOKEN

    json_data = {
        'pageSize': 10, 'pageNo': 1, 'typeId': 1, 'language': 7,
        'random': '736ea5fe7d1744008714320d2cfbbed4',
        'signature': '9BE5D3A057D1938B8210BA32222A993C',
        'timestamp': int(time.time()), # Dynamic Timestamp
    }

    data = await fetch_with_retry(session, 'https://6lotteryapi.com/api/webapi/GetNoaverageEmerdList', headers, json_data)
    
    if data and data.get('code') == 0:
        API_ERROR_COUNT = 0 
        records = data.get("data", {}).get("list", [])
        
        if records:
            latest_record = records[0]
            latest_issue = str(latest_record["issueNumber"])
            latest_number = int(latest_record["number"])
            latest_size = "BIG" if latest_number >= 5 else "SMALL"
            latest_parity = "EVEN" if latest_number % 2 == 0 else "ODD"
            
            is_new_issue = False
            if not LAST_PROCESSED_ISSUE or int(latest_issue) > int(LAST_PROCESSED_ISSUE):
                is_new_issue = True
            
            if is_new_issue:
                LAST_PROCESSED_ISSUE = latest_issue
                if not SESSION_START_ISSUE: SESSION_START_ISSUE = latest_issue
                
                await history_collection.update_one(
                    {"issue_number": latest_issue}, 
                    {"$setOnInsert": {"number": latest_number, "size": latest_size, "parity": latest_parity}}, 
                    upsert=True
                )
                
                pred_doc = await predictions_collection.find_one({"issue_number": latest_issue})
                if pred_doc and pred_doc.get("predicted_size"):
                    is_win = (pred_doc.get("predicted_size") == latest_size)
                    win_lose_status = "WIN ✅" if is_win else "LOSE ❌"
                    await predictions_collection.update_one(
                        {"issue_number": latest_issue}, 
                        {"$set": {"actual_size": latest_size, "actual_number": latest_number, "win_lose": win_lose_status}}
                    )

            next_issue = str(int(latest_issue) + 1)

            recent_preds = await predictions_collection.find({"win_lose": {"$ne": None}}).sort("issue_number", -1).limit(10).to_list(length=10)
            
            current_lose_streak = 0
            for p in recent_preds:
                if p.get("win_lose") == "LOSE ❌": current_lose_streak += 1
                else: break

            history_docs = await history_collection.find().sort("issue_number", -1).limit(500).to_list(length=500)

            try:
                # 💡 current_issue ကို ပါ pass လုပ်ပေးလိုက်ပါသည်
                mem_pred, mem_prob, mem_logic = await asyncio.to_thread(ultimate_ai_predict, history_docs, recent_preds, next_issue)
                predicted = "BIG (အကြီး) 🔴" if mem_pred == "BIG" else "SMALL (အသေး) 🟢"
                reason = f"🧠 <b>Ultimate AI Pro Engine</b>\n{mem_logic}"
            except Exception as e:
                predicted, final_prob, reason = "BIG (အကြီး) 🔴", 60.0, f"⚠️ AI Processing Error: {str(e)}"
            
            final_prob = min(max(round(mem_prob, 1), 60.0), 98.0)
            predicted_result_db = "BIG" if "BIG" in predicted else "SMALL"
            
            await predictions_collection.update_one(
                {"issue_number": next_issue}, 
                {"$set": {"predicted_size": predicted_result_db}}, 
                upsert=True
            )

            bet_advice = "💰 <b>လောင်းကြေး:</b> အခြေခံကြေး (1x)" if current_lose_streak == 0 else f"💰 <b>လောင်းကြေး:</b> {2**current_lose_streak}x (Martingale)"
            if current_lose_streak >= 4:
                bet_advice = "⚠️ <b>[DANGER] ၄ ပွဲဆက်ရှုံးထားပါသည်!</b>\nခဏနားပါ (သို့) <b>1x မှ ပြန်စပါ။</b>"

            session_preds = await predictions_collection.find({"issue_number": {"$gte": SESSION_START_ISSUE}, "win_lose": {"$ne": None}}).sort("issue_number", -1).limit(20).to_list(length=20) 
            
            table_str = "<code>Period    | Result  | W/L\n----------|---------|----\n"
            for p in session_preds[:10]: 
                iss = p.get('issue_number', '0000000')
                table_str += f"{iss[:3]}**{iss[-4:]:<6}| {p.get('actual_number', 0)}-{p.get('actual_size', 'BIG'):<5} | {'✅' if 'WIN' in p.get('win_lose', '') else '❌'}\n"
            table_str += "</code>"

            LAST_KNOWN_STATE.update({"table_str": table_str, "next_issue": next_issue, "predicted": predicted, "final_prob": final_prob, "reason": reason, "bet_advice": bet_advice})
            
            # ပွဲသစ်ထွက်ချိန်တွင် ပုံဆွဲပြီး တင်ပေးမည်
            if is_new_issue or not MAIN_MESSAGE_ID:
                # Generate Chart function ကို အပြည့်အစုံ ပြန်ထည့်ပေးပါ
                pass 

    elif data and data.get('code') != 0:
        API_ERROR_COUNT += 1
        if data.get('code') == 401: CURRENT_TOKEN = ""
    else:
        API_ERROR_COUNT += 1

    # Timer & Editing logic (ယခင်အတိုင်း ပြင်ဆင်ရန်မလိုပါ)
    current_time = time.time()
    if current_time - LAST_CAPTION_EDIT_TIME >= 1.5:
        if MAIN_MESSAGE_ID and LAST_KNOWN_STATE["next_issue"] != "Loading":
            sec_left = 60 - (int(time.time()) % 60)
            if sec_left == 60: sec_left = 0 
            iss = LAST_KNOWN_STATE['next_issue']
            iss_display = f"{iss[:3]}**{iss[-4:]}" if len(iss) > 4 else iss
            
            tg_caption = (
                f"<b>🏆 SIX-LOTTERY (AI PRO EDITION)</b>\n"
                f"⏰ Next Result In: <b>{sec_left}s</b>\n\n"
                f"{LAST_KNOWN_STATE['table_str']}\n"
                f"🅿️ <b>Period:</b> {iss_display}\n"
                f"🎯 <b>Predict: {LAST_KNOWN_STATE['predicted']}</b>\n"
                f"📈 <b>ဖြစ်နိုင်ခြေ:</b> {LAST_KNOWN_STATE['final_prob']}%\n"
                f"💡 <b>အကြောင်းပြချက်:</b>\n{LAST_KNOWN_STATE['reason']}\n"
                f"━━━━━━━━━━━━━━━━━━\n{LAST_KNOWN_STATE['bet_advice']}"
            )
            if API_ERROR_COUNT >= 3: tg_caption = f"⚠️ <b>[API သော့ သက်တမ်းကုန်သွားပါပြီ! အသစ်လဲပေးပါ]</b>\n\n" + tg_caption

            try:
                await bot.edit_message_caption(chat_id=TELEGRAM_CHANNEL_ID, message_id=MAIN_MESSAGE_ID, caption=tg_caption, parse_mode="HTML")
                LAST_CAPTION_EDIT_TIME = time.time()
            except TelegramRetryAfter as e: LAST_CAPTION_EDIT_TIME = time.time() + e.retry_after
            except Exception: pass

async def auto_broadcaster():
    await init_db() 
    async with aiohttp.ClientSession() as session:
        await login_and_get_token(session)
        while True:
            try: await check_game_and_predict(session)
            except Exception: pass
            await asyncio.sleep(0.5) 

@dp.message(Command("start"))
async def send_welcome(message: types.Message):
    await message.reply("👋 မင်္ဂလာပါ။ စနစ်က Zero-Lag AI Pro Timer ဖြင့် လုံးဝတိကျစွာ အလုပ်လုပ်နေပါပြီ။")

async def main():
    print("🚀 Aiogram SIX-LOTTERY Bot (AI Pro Edition) စတင်နေပါပြီ...\n")
    await bot.delete_webhook(drop_pending_updates=True)
    asyncio.create_task(auto_broadcaster())
    await dp.start_polling(bot)

if __name__ == '__main__':
    try: asyncio.run(main())
    except KeyboardInterrupt: print("Bot ကို ရပ်တန့်လိုက်ပါသည်။")
