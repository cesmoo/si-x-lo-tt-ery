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
# 🎨 STICKER CONFIGURATION (ဒီနေရာမှာ အစားထိုးပါ)
# ==========================================
# 💡 @idstickerbot မှ ရလာသော စာသားရှည်ကြီးများကို အောက်ပါ "" ထဲတွင် ထည့်ပါ
WIN_STICKER_ID = "CAACAgUAAxkBAAEQwnxptubYV3lfXdObrI2p5Hwd2wTWUAAC_hAAAhjHwVSmfHZggStHQzoE"  # ဥပမာ - "CAACAgUAAxkBAAE..."
LOSE_STICKER_ID = "CAACAgUAAxkBAAEQwn5ptuba_8vM_knhkMxJEuXk2yVEoAACNRIAAhylwVQzmQMbLqf59zoE" # ဥပမာ - "CAACAgUAAxkBAAE..."

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
# 🧠 2. PURE AI LOGIC (TEXT ONLY)
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
    ml_weight, pattern_weight, trend_weight = 2.5, 1.5, 1.0
    
    if len(recent_preds) >= 5:
        wins = sum(1 for p in recent_preds[:5] if p.get('win_lose') == 'WIN')
        if wins <= 2: 
            ml_weight, pattern_weight = 3.5, 0.5

    current_streak = get_streak(sizes)
    if current_streak >= 4:
        trend_weight = 2.0
        if sizes[-1] == 'BIG': score_s += trend_weight 
        else: score_b += trend_weight

    X, y, window = [], [], 6 
    def encode_size(s): return 1 if s == 'BIG' else 0
    def encode_parity(p): return 1 if p == 'EVEN' else 0
    
    for i in range(len(sizes) - window):
        row = []
        for j in range(window): row.extend([encode_size(sizes[i+j]), numbers[i+j], encode_parity(parities[i+j])])
        row.append(sum(numbers[i+window-3:i+window]) / 3.0)
        row.append(get_streak(sizes[i:i+window]))
        X.append(row); y.append(encode_size(sizes[i+window]))
        
    if len(X) > 30:
        rf_clf = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42, n_jobs=-1).fit(X, y)
        gb_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42).fit(X, y)
        
        current_features = []
        for j in range(1, window + 1): current_features.extend([encode_size(sizes[-j]), numbers[-j], encode_parity(parities[-j])])
        current_features.append(sum(numbers[-3:]) / 3.0)
        current_features.append(current_streak)
        
        rf_pred, rf_prob = rf_clf.predict([current_features])[0], max(rf_clf.predict_proba([current_features])[0])
        gb_pred, gb_prob = gb_clf.predict([current_features])[0], max(gb_clf.predict_proba([current_features])[0])
        
        if rf_pred == gb_pred:
            if rf_pred == 1: score_b += (rf_prob * ml_weight * 1.5)
            else: score_s += (rf_prob * ml_weight * 1.5)
        else:
            best_prob, best_pred = max(rf_prob, gb_prob), rf_pred if rf_prob > gb_prob else gb_pred
            if best_pred == 1: score_b += (best_prob * ml_weight)
            else: score_s += (best_prob * ml_weight)

    if len(sizes) >= 3:
        if sizes[-1] != sizes[-2] and sizes[-2] != sizes[-3]:
            pred_pattern = 'BIG' if sizes[-1] == 'SMALL' else 'SMALL'
            if pred_pattern == 'BIG': score_b += pattern_weight
            else: score_s += pattern_weight

    final_pred = "BIG" if score_b > score_s else "SMALL"
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
            if ACTUAL_BET_STREAK > 6: ACTUAL_BET_STREAK = 0

            CURRENT_PREDICTED_ISSUE = str(int(latest_issue) + 1)
            history_docs = await history_collection.find().sort("issue_number", -1).limit(500).to_list(length=500)
            CURRENT_PREDICTION_SIZE = ultimate_ai_predict(history_docs, recent_preds, CURRENT_PREDICTED_ISSUE)

            multiplier = 2 ** ACTUAL_BET_STREAK
            pred_msg = f"⏰ Period: {CURRENT_PREDICTED_ISSUE}\n🎯 Prediction: {CURRENT_PREDICTION_SIZE} {multiplier}x"
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

                multiplier = 2 ** ACTUAL_BET_STREAK
                icon = "🟢" if is_win else "🔴"
                result_letter = "B" if latest_size == "BIG" else "S"
                
                result_msg = (
                    f"<b>SIX-LOTTERY</b>\n\n"
                    f"⏰ Period: {latest_issue}\n"
                    f"🎯 Choice: {CURRENT_PREDICTION_SIZE} {multiplier}x\n"
                    f"📊 Result: {icon} {win_lose_db} | {result_letter} ({latest_number})"
                )
                
                # ၁။ Result စာသားကို အရင်ပို့မည်
                await bot.send_message(chat_id=TELEGRAM_CHANNEL_ID, text=result_msg)
                
                # ၂။ 💡 နိုင်/ရှုံး အလိုက် Sticker ကို ဆက်ပို့မည်
                try:
                    if is_win and WIN_STICKER_ID:
                        await bot.send_sticker(chat_id=TELEGRAM_CHANNEL_ID, sticker=WIN_STICKER_ID)
                    elif not is_win and LOSE_STICKER_ID:
                        await bot.send_sticker(chat_id=TELEGRAM_CHANNEL_ID, sticker=LOSE_STICKER_ID)
                except Exception as e:
                    print(f"Sticker ပို့ရာတွင် Error တက်ပါသည်: {e}")

                if is_win: ACTUAL_BET_STREAK = 0
                else:
                    ACTUAL_BET_STREAK += 1
                    if ACTUAL_BET_STREAK > 6: ACTUAL_BET_STREAK = 0

            LAST_PROCESSED_ISSUE = latest_issue

            CURRENT_PREDICTED_ISSUE = str(int(latest_issue) + 1)
            history_docs = await history_collection.find().sort("issue_number", -1).limit(500).to_list(length=500)
            recent_preds = await predictions_collection.find({"win_lose": {"$ne": None}}).sort("issue_number", -1).limit(10).to_list(length=10)
            
            CURRENT_PREDICTION_SIZE = ultimate_ai_predict(history_docs, recent_preds, CURRENT_PREDICTED_ISSUE)

            multiplier = 2 ** ACTUAL_BET_STREAK
            pred_msg = f"⏰ Period: {CURRENT_PREDICTED_ISSUE}\n🎯 Prediction: {CURRENT_PREDICTION_SIZE} {multiplier}x"
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
    print("🚀 Aiogram SIX-LOTTERY Bot (Text+Sticker Edition) စတင်နေပါပြီ...\n")
    await bot.delete_webhook(drop_pending_updates=True)
    asyncio.create_task(auto_broadcaster())
    await dp.start_polling(bot)

if __name__ == '__main__':
    try: asyncio.run(main())
    except KeyboardInterrupt: print("Bot ကို ရပ်တန့်လိုက်ပါသည်။")
