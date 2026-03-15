import asyncio
import time
import os
from dotenv import load_dotenv
import aiohttp
import motor.motor_asyncio 

from aiogram import Bot, Dispatcher
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties

# --- 🧠 LIGHTWEIGHT MACHINE LEARNING ---
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import warnings
warnings.filterwarnings("ignore")

load_dotenv()

# ==========================================
# ⚙️ 1. CONFIGURATION
# ==========================================
TELEGRAM_BOT_TOKEN = os.getenv("BOT_TOKEN")
TELEGRAM_CHANNEL_ID = os.getenv("CHANNEL_ID")
MONGO_URI = os.getenv("MONGO_URI") 

bot = Bot(token=TELEGRAM_BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
dp = Dispatcher()

# Database Setup
db_client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)
db = db_client['sixlottery_database'] 
history_collection = db['game_history'] 
predictions_collection = db['predictions'] 

# ==========================================
# 🎨 STICKER & MULTIPLIER CONFIGURATION
# ==========================================
WIN_STICKER_ID = "CAACAgUAAxkBAAEQwp5ptvW0w1rf71LGHxi_1fzyRXThegACVR8AAkDtuFVty-3R5xnGHjoE"  
LOSE_STICKER_ID = "" 

# လောင်းကြေးအဆ (1x, 2x, 3x...)
MULTIPLIER_LIST = [1, 2, 3, 5, 8, 15, 30]

# State Variables
LAST_PROCESSED_ISSUE = None
CURRENT_PREDICTED_ISSUE = None
CURRENT_PREDICTION_SIZE = None
ACTUAL_BET_STREAK = 0 
AI_CACHE = {"last_trained_issue": None, "cached_prediction": None}

BASE_HEADERS = {
    'authority': '6lotteryapi.com',
    'accept': 'application/json, text/plain, */*',
    'content-type': 'application/json;charset=UTF-8',
    'origin': 'https://www.6win566.com',
    'referer': 'https://www.6win566.com/',
    'user-agent': 'Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36'
}

async def init_db():
    try: 
        await history_collection.create_index("issue_number", unique=True)
        await predictions_collection.create_index("issue_number", unique=True)
    except Exception: pass

async def fetch_with_retry(session, url, headers, json_data, retries=1):
    for attempt in range(retries):
        try:
            async with session.post(url, headers=headers, json=json_data, timeout=2.0) as response:
                if response.status == 200: return await response.json()
        except Exception: await asyncio.sleep(0.2)
    return None

# ==========================================
# 🧠 2. AI V3 LOGIC (ANTI-STREAK EDITION)
# ==========================================
def get_streak(sizes_list):
    if not sizes_list: return 0
    count = 1
    for i in range(len(sizes_list)-2, -1, -1):
        if sizes_list[i] == sizes_list[-1]: count += 1
        else: break
    return count

def ultimate_ai_predict(history_docs, recent_preds, current_issue):
    global AI_CACHE
    if AI_CACHE["last_trained_issue"] == current_issue and AI_CACHE["cached_prediction"]: 
        return AI_CACHE["cached_prediction"]

    if len(history_docs) < 25: return "BIG"

    docs = list(reversed(history_docs))[-500:]
    sizes = [d.get('size', 'BIG') for d in docs]
    numbers = [int(d.get('number', 0)) for d in docs]
    parities = [d.get('parity', 'EVEN') for d in docs]
    
    score_b, score_s = 0.0, 0.0
    
    # 💡 [NEW] Bot ရဲ့ ရှုံးပွဲဆက်မှုကို စစ်ဆေးခြင်း
    bot_lose_streak = 0
    if len(recent_preds) > 0:
        for p in recent_preds:
            if p.get('win_lose') == 'LOSE': bot_lose_streak += 1
            else: break
            
    # 💡 ၃ ပွဲဆက်တိုက်ရှုံးနေပါက AI အဖြေကို ပြောင်းပြန်လှန်မည် (Inverse Mode)
    inverse_mode = False
    if bot_lose_streak >= 3:
        inverse_mode = True

    # 💡 [NEW] Dragon Rider (အတန်းရှည်လိုက်ခြင်း)
    current_streak = get_streak(sizes)
    if current_streak >= 4:
        # ၄ ခါဆက်တူနေရင် ချိုးဖို့မကြိုးစားဘဲ ဆက်လိုက်မည်
        if sizes[-1] == 'BIG': score_b += 5.0
        else: score_s += 5.0
    elif current_streak == 3:
        # ၃ ခါဆိုရင်တော့ ချိုးဖို့ အားပေးမည်
        if sizes[-1] == 'BIG': score_s += 2.0
        else: score_b += 2.0

    # 💡 [NEW] Ping-Pong Pattern (B-S-B-S)
    if len(sizes) >= 4:
        if sizes[-1] != sizes[-2] and sizes[-2] != sizes[-3] and sizes[-3] != sizes[-4]:
            if sizes[-1] == 'BIG': score_s += 3.0
            else: score_b += 3.0

    # --- Machine Learning ---
    X, y, window = [], [], 5 
    def encode_size(s): return 1 if s == 'BIG' else 0
    def encode_parity(p): return 1 if p == 'EVEN' else 0
    
    for i in range(len(sizes) - window):
        row = []
        for j in range(window): row.extend([encode_size(sizes[i+j]), numbers[i+j], encode_parity(parities[i+j])])
        X.append(row); y.append(encode_size(sizes[i+window]))
        
    if len(X) > 30:
        # Overfitting မဖြစ်စေရန် Model ကို ပေါ့ပါးအောင် လုပ်ထားသည်
        rf_clf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1).fit(X, y)
        gb_clf = GradientBoostingClassifier(n_estimators=50, learning_rate=0.05, max_depth=3, random_state=42).fit(X, y)
        
        current_features = []
        for j in range(1, window + 1): current_features.extend([encode_size(sizes[-j]), numbers[-j], encode_parity(parities[-j])])
        
        rf_pred = rf_clf.predict([current_features])[0]
        gb_pred = gb_clf.predict([current_features])[0]
        
        ml_weight = 1.5
        if rf_pred == gb_pred:
            if rf_pred == 1: score_b += (ml_weight * 1.5)
            else: score_s += (ml_weight * 1.5)
        else:
            if rf_pred == 1: score_b += ml_weight
            else: score_s += ml_weight

    final_pred = "BIG" if score_b > score_s else "SMALL"
    
    # 💡 Inverse Mode ဖွင့်ထားပါက အဖြေကို ပြောင်းပြန်လှန်မည်
    if inverse_mode:
        final_pred = "SMALL" if final_pred == "BIG" else "BIG"

    AI_CACHE.update({"last_trained_issue": current_issue, "cached_prediction": final_pred})
    return final_pred

# ==========================================
# 🚀 3. CORE BOT LOGIC (MESSAGE SENDER)
# ==========================================
async def check_game_and_predict(session: aiohttp.ClientSession):
    global LAST_PROCESSED_ISSUE, CURRENT_PREDICTED_ISSUE, CURRENT_PREDICTION_SIZE, ACTUAL_BET_STREAK
    
    json_data = {'pageSize': 10, 'pageNo': 1, 'typeId': 1, 'language': 7, 'random': '736ea5fe7d1744008714320d2cfbbed4', 'signature': '9BE5D3A057D1938B8210BA32222A993C', 'timestamp': int(time.time())}
    data = await fetch_with_retry(session, 'https://6lotteryapi.com/api/webapi/GetNoaverageEmerdList', BASE_HEADERS, json_data)
    
    if data and data.get('code') == 0:
        records = data.get("data", {}).get("list", [])
        if not records: return
        
        latest_record = records[0]
        latest_issue, latest_number = str(latest_record["issueNumber"]), int(latest_record["number"])
        latest_size = "BIG" if latest_number >= 5 else "SMALL"
        latest_parity = "EVEN" if latest_number % 2 == 0 else "ODD"

        if not LAST_PROCESSED_ISSUE:
            LAST_PROCESSED_ISSUE = latest_issue
            
            recent_preds = await predictions_collection.find({"win_lose": {"$ne": None}}).sort("issue_number", -1).limit(10).to_list(length=10)
            ACTUAL_BET_STREAK = 0
            for p in recent_preds:
                if p.get("win_lose") == "LOSE": ACTUAL_BET_STREAK += 1
                else: break
            
            if ACTUAL_BET_STREAK >= len(MULTIPLIER_LIST): ACTUAL_BET_STREAK = 0

            CURRENT_PREDICTED_ISSUE = str(int(latest_issue) + 1)
            history_docs = await history_collection.find().sort("issue_number", -1).limit(500).to_list(length=500)
            CURRENT_PREDICTION_SIZE = ultimate_ai_predict(history_docs, recent_preds, CURRENT_PREDICTED_ISSUE)

            step_count = ACTUAL_BET_STREAK + 1
            pred_msg = f"⏰ Period: {CURRENT_PREDICTED_ISSUE}\n🎯 Prediction: {CURRENT_PREDICTION_SIZE} {step_count}x"
            await bot.send_message(chat_id=TELEGRAM_CHANNEL_ID, text=pred_msg)
            return

        if int(latest_issue) > int(LAST_PROCESSED_ISSUE):
            await history_collection.update_one({"issue_number": latest_issue}, {"$setOnInsert": {"number": latest_number, "size": latest_size, "parity": latest_parity}}, upsert=True)

            if CURRENT_PREDICTED_ISSUE == latest_issue and CURRENT_PREDICTION_SIZE:
                is_win = (CURRENT_PREDICTION_SIZE == latest_size)
                win_lose_db = "WIN" if is_win else "LOSE"
                
                await predictions_collection.update_one(
                    {"issue_number": latest_issue}, 
                    {"$set": {"actual_size": latest_size, "actual_number": latest_number, "win_lose": win_lose_db, "predicted_size": CURRENT_PREDICTION_SIZE}}, 
                    upsert=True
                )

                step_count = ACTUAL_BET_STREAK + 1
                icon = "🟢" if is_win else "🔴"
                result_letter = "B" if latest_size == "BIG" else "S"
                
                # Inverse Mode အလုပ်လုပ်နေကြောင်း ပြသရန်
                inverse_alert = "REV" if (not is_win and ACTUAL_BET_STREAK >= 2) else ""
                
                result_msg = (
                    f"<b>𝙎𝙄𝙓-𝙇𝙊𝙏𝙏𝙀𝙍𝙔</b>\n\n"
                    f"⏰ Period: {latest_issue}\n"
                    f"🎯 Choice: {CURRENT_PREDICTION_SIZE} {step_count}x{inverse_alert}\n"
                    f"📊 Result: {icon} {win_lose_db} | {result_letter} ({latest_number})"
                )
                
                await bot.send_message(chat_id=TELEGRAM_CHANNEL_ID, text=result_msg)
                
                try:
                    if is_win and WIN_STICKER_ID:
                        await bot.send_sticker(chat_id=TELEGRAM_CHANNEL_ID, sticker=WIN_STICKER_ID)
                    elif not is_win and LOSE_STICKER_ID:
                        await bot.send_sticker(chat_id=TELEGRAM_CHANNEL_ID, sticker=LOSE_STICKER_ID)
                except Exception as e:
                    pass

                if is_win: 
                    ACTUAL_BET_STREAK = 0
                else:
                    ACTUAL_BET_STREAK += 1
                    if ACTUAL_BET_STREAK >= len(MULTIPLIER_LIST): ACTUAL_BET_STREAK = 0

            LAST_PROCESSED_ISSUE = latest_issue

            CURRENT_PREDICTED_ISSUE = str(int(latest_issue) + 1)
            history_docs = await history_collection.find().sort("issue_number", -1).limit(500).to_list(length=500)
            recent_preds = await predictions_collection.find({"win_lose": {"$ne": None}}).sort("issue_number", -1).limit(10).to_list(length=10)
            
            CURRENT_PREDICTION_SIZE = ultimate_ai_predict(history_docs, recent_preds, CURRENT_PREDICTED_ISSUE)

            step_count = ACTUAL_BET_STREAK + 1
            inverse_alert = "REV" if ACTUAL_BET_STREAK >= 3 else ""
            pred_msg = f"⏰ Period: {CURRENT_PREDICTED_ISSUE}\n🎯 Prediction: {CURRENT_PREDICTION_SIZE} {step_count}x{inverse_alert}"
            await bot.send_message(chat_id=TELEGRAM_CHANNEL_ID, text=pred_msg)

# ==========================================
# 🔄 4. BACKGROUND TASK
# ==========================================
async def auto_broadcaster():
    await init_db() 
    async with aiohttp.ClientSession() as session:
        while True:
            try: await check_game_and_predict(session)
            except Exception: pass
            await asyncio.sleep(1.0) 

async def main():
    print("🚀 Aiogram SIX-LOTTERY Bot (AI V3 Anti-Streak) စတင်နေပါပြီ...\n")
    await bot.delete_webhook(drop_pending_updates=True)
    asyncio.create_task(auto_broadcaster())
    await dp.start_polling(bot)

if __name__ == '__main__':
    try: asyncio.run(main())
    except KeyboardInterrupt: print("Bot ကို ရပ်တန့်လိုက်ပါသည်။")
