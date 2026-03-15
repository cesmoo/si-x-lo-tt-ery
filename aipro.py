import asyncio
import time
import os
import io
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

# --- 🧠 UPGRADED MACHINE LEARNING LIBRARIES ---
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import matplotlib
matplotlib.use('Agg') 
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

bot = Bot(token=TELEGRAM_BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
dp = Dispatcher()

db_client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)
db = db_client['trxsixlottery_database'] 
history_collection = db['game_history'] 
predictions_collection = db['predictions'] 

CURRENT_TOKEN, LAST_PROCESSED_ISSUE, MAIN_MESSAGE_ID, SESSION_START_ISSUE = "", None, None, None
LAST_CAPTION_EDIT_TIME, API_ERROR_COUNT = 0, 0 
START_OF_ROUND_BALANCE, CURRENT_BALANCE_DISPLAY = 0.0, 0.0

LAST_KNOWN_STATE = {
    "table_str": "<code>Data Loading...</code>", "next_issue": "Loading", "predicted": "Wait",
    "final_prob": 0.0, "reason": "Syncing Data...", "bet_advice": "...",
    "balance": "💳 <b>Balance:</b> Syncing...", "profit": "📊 <b>Profit:</b> 0.00 Ks"
}
AI_CACHE = {"last_trained_issue": None, "rf_model": None, "gb_model": None, "cached_prediction": None, "cached_prob": None, "cached_logic": ""}
BASE_HEADERS = {'authority': '6lotteryapi.com', 'accept': 'application/json, text/plain, */*', 'content-type': 'application/json;charset=UTF-8', 'origin': 'https://www.6win566.com', 'referer': 'https://www.6win566.com/', 'user-agent': 'Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36'}

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

async def login_and_get_token(session: aiohttp.ClientSession):
    global CURRENT_TOKEN
    json_data = {'username': USERNAME, 'pwd': PASSWORD, 'phonetype': 1, 'logintype': 'mobile', 'packId': '', 'deviceId': 'b9b753a9f874897574d7fa72ff84374c', 'language': 7, 'random': '457a0935e8b54d63924ce243e028f789', 'signature': '6C2BCE370032980C33A1FC41A327DF09', 'timestamp': int(time.time())}
    data = await fetch_with_retry(session, 'https://6lotteryapi.com/api/webapi/Login', BASE_HEADERS, json_data)
    if data and data.get('code') == 0: 
        CURRENT_TOKEN = f"Bearer {data.get('data', {}).get('token', '')}"
        print("✅ API Login အောင်မြင်ပါသည်။ Token ရရှိပါပြီ။")
        return True
    return False

async def get_user_balance(session):
    global CURRENT_TOKEN
    if not CURRENT_TOKEN: return None
    headers = BASE_HEADERS.copy()
    headers['authorization'] = CURRENT_TOKEN
    json_data = {'language': 7, 'random': '6e5c9c6f8d824252b800b40d6a0af244', 'signature': '6E635C1F332EF7D017FF2B7370160E4D', 'timestamp': int(time.time())}
    try:
        res = await fetch_with_retry(session, 'https://6lotteryapi.com/api/webapi/GetBalance', headers, json_data)
        if res and res.get('code') == 0: return float(res.get('data', {}).get('balance', 0))
    except Exception: pass
    return None

# ==========================================
# 🧠 2. THE ULTIMATE AI PRO V2 (ENHANCED)
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
        return AI_CACHE["cached_prediction"], AI_CACHE["cached_prob"], AI_CACHE["cached_logic"]

    if len(history_docs) < 25: 
        return "BIG", 55.0, "⏳ Data စုဆောင်းဆဲ..."

    docs = list(reversed(history_docs))[-500:]
    sizes = [d.get('size', 'BIG') for d in docs]
    numbers = [int(d.get('number', 0)) for d in docs]
    parities = [d.get('parity', 'EVEN') for d in docs]
    
    score_b, score_s, logic_used = 0.0, 0.0, ""
    ml_weight, pattern_weight, trend_weight = 2.5, 1.5, 1.0
    
    # --- Auto-Tuning Based on Recent Performance ---
    if len(recent_preds) >= 5:
        wins = sum(1 for p in recent_preds[:5] if p.get('win_lose') == 'WIN ✅')
        if wins <= 2: 
            ml_weight = 3.5 # AI အပေါ် ပိုမိုအားပြုမည်
            pattern_weight = 0.5
            logic_used += "🔄 <b>Auto-Tuning:</b> AI Algorithm အား ပိုမိုအသုံးပြုထားသည်။\n"

    # --- Trend & Streak Analysis (NEW) ---
    current_streak = get_streak(sizes)
    if current_streak >= 4:
        # လေးကြိမ်ဆက်တိုက်တူနေလျှင် Trend အတိုင်းလိုက်မည် သို့မဟုတ် ချိုးမည် (Martingale သဘောတရား)
        trend_weight = 2.0
        if sizes[-1] == 'BIG':
            score_s += trend_weight # ၄ ကြိမ် BIG ဖြစ်လျှင် SMALL သို့ ချိုးရန် အားပေးမည်
            logic_used += f"├ 📉 <b>Trend Force:</b> BIG အတန်းရှည်နေသဖြင့် SMALL သို့ ချိုးမည်။\n"
        else:
            score_b += trend_weight
            logic_used += f"├ 📈 <b>Trend Force:</b> SMALL အတန်းရှည်နေသဖြင့် BIG သို့ ချိုးမည်။\n"

    # --- Enhanced Machine Learning Feature Engineering ---
    X, y, window = [], [], 6 # Window size တိုးထားသည်
    
    def encode_size(s): return 1 if s == 'BIG' else 0
    def encode_parity(p): return 1 if p == 'EVEN' else 0
    
    for i in range(len(sizes) - window):
        row = []
        for j in range(window): 
            row.extend([encode_size(sizes[i+j]), numbers[i+j], encode_parity(parities[i+j])])
        
        # New Features: သုံးပွဲဆက်တိုက် ပျမ်းမျှဂဏန်း (Moving Average)
        avg_3 = sum(numbers[i+window-3:i+window]) / 3.0
        row.append(avg_3)
        row.append(get_streak(sizes[i:i+window])) # ယခင်က ဖြစ်ခဲ့သော Streak

        X.append(row)
        y.append(encode_size(sizes[i+window]))
        
    if len(X) > 30:
        # AI Models ကို အဆင့်မြှင့်တင်ထားသည်
        rf_clf = RandomForestClassifier(n_estimators=150, max_depth=10, min_samples_split=5, random_state=42, n_jobs=-1).fit(X, y)
        gb_clf = GradientBoostingClassifier(n_estimators=150, learning_rate=0.03, max_depth=5, random_state=42).fit(X, y)
        
        current_features = []
        for j in range(1, window + 1): 
            current_features.extend([encode_size(sizes[-j]), numbers[-j], encode_parity(parities[-j])])
        
        avg_3_current = sum(numbers[-3:]) / 3.0
        current_features.append(avg_3_current)
        current_features.append(current_streak)
        
        rf_pred = rf_clf.predict([current_features])[0]
        rf_prob = max(rf_clf.predict_proba([current_features])[0])
        gb_pred = gb_clf.predict([current_features])[0]
        gb_prob = max(gb_clf.predict_proba([current_features])[0])
        
        if rf_pred == gb_pred:
            if rf_pred == 1: score_b += (rf_prob * ml_weight * 1.5)
            else: score_s += (rf_prob * ml_weight * 1.5)
            logic_used += "├ 🤖 <b>AI V2 Ensemble:</b> RF နှင့် GB နှစ်မျိုးလုံး တူညီသည်။\n"
        else:
            best_prob = max(rf_prob, gb_prob)
            best_pred = rf_pred if rf_prob > gb_prob else gb_pred
            if best_pred == 1: score_b += (best_prob * ml_weight)
            else: score_s += (best_prob * ml_weight)
            logic_used += f"├ 🤖 <b>AI V2 Model:</b> Data Pattern အရ {best_prob*100:.1f}% ယုံကြည်သည်။\n"

    # --- Pattern Logic ---
    if len(sizes) >= 3:
        if sizes[-1] != sizes[-2] and sizes[-2] != sizes[-3]:
            pred_pattern = 'BIG' if sizes[-1] == 'SMALL' else 'SMALL'
            if pred_pattern == 'BIG': score_b += pattern_weight
            else: score_s += pattern_weight
            logic_used += "├ 🏓 <b>Pattern:</b> ခုတ်ချိုး အကွက်ဖြစ်သည်။\n"

    total_score = score_b + score_s
    final_pred = "BIG" if score_b > score_s else "SMALL"
    
    if total_score == 0: 
        final_prob = 55.0
        logic_used += "└ ⚠️ Data အလုံအလောက် မရှိသေးပါ။"
    else: 
        raw_prob = (max(score_b, score_s) / total_score) * 100
        # Volatility Check: ဂဏန်းတွေ အရမ်းခုန်နေရင် Probability ကို နည်းနည်းလျှော့ပြမည်
        volatility = np.std(numbers[-10:]) if len(numbers) >= 10 else 0
        if volatility > 3.0: 
            raw_prob -= 5.0 
            logic_used += "└ ⚠️ <b>Volatility:</b> ကစားကွက်မငြိမ်ပါ။ သတိထားကစားပါ။\n"
            
        final_prob = min(max(raw_prob, 65.0), 99.0) 
        
    AI_CACHE.update({
        "last_trained_issue": current_issue, 
        "cached_prediction": final_pred, 
        "cached_prob": round(final_prob, 1), 
        "cached_logic": logic_used
    })
    
    return final_pred, round(final_prob, 1), logic_used

# ==========================================
# 🎨 3. DYNAMIC GRAPH GENERATOR 
# ==========================================
def generate_winrate_chart(predictions):
    wins, losses, bar_colors, dots_list, bar_heights, history_wr = 0, 0, [], [], [], []
    latest_preds = list(reversed(predictions))[-20:]
    for i, p in enumerate(latest_preds): 
        current_played = i + 1
        if 'WIN' in p.get('win_lose', ''):
            wins += 1; bar_colors.append('#00e5ff'); dots_list.append(('G', '#1de9b6')); bar_heights.append(50 + ((wins / current_played) * 100 / 2)) 
        else:
            losses += 1; bar_colors.append('#ff4444'); dots_list.append(('R', '#ef5350')); bar_heights.append(10 + ((wins / current_played) * 100 / 3)) 
        history_wr.append((wins / current_played) * 100)
    total_played = wins + losses; win_rate = int((wins / total_played * 100)) if total_played > 0 else 0
    fig = plt.figure(figsize=(10.24, 7.68), facecolor='#1c1f26') 
    fig.text(0.05, 0.90, "AI PERFORMANCE ANALYTICS", color='#ffffff', fontsize=32, fontweight='bold', ha='left')
    ax_circle = fig.add_axes([0.08, 0.42, 0.35, 0.40]); ax_circle.set_axis_off(); ax_circle.set_xlim(0, 1); ax_circle.set_ylim(0, 1)
    theta_bg = np.linspace(-1.25*np.pi, 0.25*np.pi, 200); ax_circle.plot(0.5 + 0.45*np.cos(theta_bg), 0.5 + 0.45*np.sin(theta_bg), color='#2c313c', linewidth=12)
    if win_rate > 0:
        theta_fg = np.linspace(0.25*np.pi, 0.25*np.pi - (win_rate/100) * 1.5 * np.pi, 100); ax_circle.plot(0.5 + 0.45*np.cos(theta_fg), 0.5 + 0.45*np.sin(theta_fg), color='#00e5ff', linewidth=12)
    ax_circle.text(0.5, 0.75, f"{total_played}/20", color='#a3a8b5', fontsize=16, fontweight='bold', ha='center', va='center'); ax_circle.text(0.5, 0.65, "TOTAL WINRATE", color='#7a8294', fontsize=12, fontweight='bold', ha='center', va='center'); ax_circle.text(0.5, 0.48, f"{win_rate}%", color='#00e5ff', fontsize=65, fontweight='bold', ha='center', va='center')
    fig.text(0.74, 0.85, "SESSION PERFORMANCE TREND", color='#a3a8b5', fontsize=14, fontweight='bold', ha='center')
    ax_bar = fig.add_axes([0.55, 0.47, 0.38, 0.33]); ax_bar.set_facecolor('#1c1f26'); ax_bar.set_xlim(-0.5, 19.5); ax_bar.set_ylim(0, 105) 
    ax_bar.spines['top'].set_visible(False); ax_bar.spines['right'].set_visible(False); ax_bar.spines['left'].set_visible(False); ax_bar.spines['bottom'].set_visible(False); ax_bar.set_yticks([0, 25, 50, 75, 100]); ax_bar.set_yticklabels(['0%', '25%', '50%', '75%', '100%'], color='#7a8294', fontsize=10, fontweight='bold'); ax_bar.grid(axis='y', color='#2c313c', linestyle='-', linewidth=1.5)
    if total_played > 0:
        x_pos = np.arange(total_played); ax_bar.bar(x_pos, bar_heights, color=bar_colors, width=0.6, alpha=0.9, zorder=2); ax_bar.plot(x_pos, history_wr, color='#3b82f6', linewidth=2.5, marker='o', markersize=6, markerfacecolor='#1c1f26', markeredgecolor='#00e5ff', zorder=4); ax_bar.set_xticks(x_pos); ax_bar.set_xticklabels([str(i+1) for i in range(total_played)], color='#7a8294', fontsize=10)
    ax_win = fig.add_axes([0.05, 0.22, 0.28, 0.16]); ax_win.set_axis_off(); ax_win.set_xlim(0, 1); ax_win.set_ylim(0, 1); ax_win.add_patch(patches.FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0", fc="#1de9b6", ec="none")); ax_win.text(0.1, 0.75, "TOTAL WINS:", color='#004d40', fontsize=16, fontweight='bold', va='center'); ax_win.text(0.1, 0.35, f"{wins}", color='#000000', fontsize=48, fontweight='bold', va='center')
    ax_lose = fig.add_axes([0.35, 0.22, 0.28, 0.16]); ax_lose.set_axis_off(); ax_lose.set_xlim(0, 1); ax_lose.set_ylim(0, 1); ax_lose.add_patch(patches.FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0", fc="#ef5350", ec="none")); ax_lose.text(0.1, 0.75, "TOTAL LOSSES:", color='#4d0000', fontsize=16, fontweight='bold', va='center'); ax_lose.text(0.1, 0.35, f"{losses}", color='#ffffff', fontsize=48, fontweight='bold', va='center')
    ax_wm = fig.add_axes([0.65, 0.22, 0.30, 0.16]); ax_wm.set_axis_off(); ax_wm.text(0.5, 0.5, "DEV - WANG LIN", color='#ffffff', fontsize=26, fontweight='bold', style='italic', ha='center', va='center')
    fig.text(0.05, 0.16, "FULL PREDICTION TIMELINE (Oldest to Latest)", color='#a3a8b5', fontsize=12, fontweight='bold', ha='left')
    ax_time = fig.add_axes([0.05, 0.05, 0.9, 0.08]); ax_time.set_axis_off(); ax_time.set_xlim(-0.5, 19.5); ax_time.set_ylim(0, 1)
    if len(dots_list) > 0:
        for i, (char, color) in enumerate(dots_list): ax_time.scatter(i, 0.5, s=600, c=color, edgecolors='none', zorder=5); ax_time.text(i, 0.5, char, color='#ffffff', fontsize=14, fontweight='bold', ha='center', va='center', zorder=6)
    buf = io.BytesIO(); plt.savefig(buf, format='png', dpi=100, facecolor='#1c1f26'); buf.seek(0); plt.close(fig)
    return buf

# ==========================================
# 🚀 4. MAIN LOGIC & UI UPDATER
# ==========================================
async def check_game_and_predict(session: aiohttp.ClientSession):
    global CURRENT_TOKEN, LAST_PROCESSED_ISSUE, MAIN_MESSAGE_ID, SESSION_START_ISSUE
    global LAST_CAPTION_EDIT_TIME, API_ERROR_COUNT, LAST_KNOWN_STATE
    global START_OF_ROUND_BALANCE, CURRENT_BALANCE_DISPLAY
    
    if not CURRENT_TOKEN:
        if not await login_and_get_token(session): return

    headers = BASE_HEADERS.copy(); headers['authorization'] = CURRENT_TOKEN
    json_data = {'pageSize': 10, 'pageNo': 1, 'typeId': 13, 'language': 7, 'random': '1f379d96b6934b59978b4121e75be5e5', 'signature': '2AB571112B64A33AAEA24CC463167AD3', 'timestamp': int(time.time())}
    data = await fetch_with_retry(session, 'https://6lotteryapi.com/api/webapi/GetTRXNoaverageEmerdList', headers, json_data)
    
    if data and data.get('code') == 0:
        API_ERROR_COUNT = 0 
        
        # Initial Balance Sync
        if CURRENT_BALANCE_DISPLAY == 0.0:
            init_bal = await get_user_balance(session)
            if init_bal is not None:
                START_OF_ROUND_BALANCE = CURRENT_BALANCE_DISPLAY = init_bal
                LAST_KNOWN_STATE['balance'] = f"💳 <b>Balance:</b> {CURRENT_BALANCE_DISPLAY:.2f} Ks"

        records = data.get("data", {}).get("list", [])
        if records:
            latest_record = records[0]
            latest_issue, latest_number = str(latest_record["issueNumber"]), int(latest_record["number"])
            latest_size, latest_parity = "BIG" if latest_number >= 5 else "SMALL", "EVEN" if latest_number % 2 == 0 else "ODD"
            
            is_new_issue = False
            if not LAST_PROCESSED_ISSUE or int(latest_issue) > int(LAST_PROCESSED_ISSUE): is_new_issue = True
            
            if is_new_issue:
                await asyncio.sleep(0.5) 
                curr_bal = await get_user_balance(session)
                if curr_bal is not None:
                    if START_OF_ROUND_BALANCE > 0:
                        diff = curr_bal - START_OF_ROUND_BALANCE 
                        if diff > 0: LAST_KNOWN_STATE['profit'] = f"📈 <b>Profit:</b> +{diff:.2f} Ks ✅"
                        elif diff < 0: LAST_KNOWN_STATE['profit'] = f"📉 <b>Loss:</b> {diff:.2f} Ks ❌"
                        else: LAST_KNOWN_STATE['profit'] = f"📊 <b>Profit:</b> 0.00 Ks"
                    START_OF_ROUND_BALANCE = CURRENT_BALANCE_DISPLAY = curr_bal
                    LAST_KNOWN_STATE['balance'] = f"💳 <b>Balance:</b> {CURRENT_BALANCE_DISPLAY:.2f} Ks"

                LAST_PROCESSED_ISSUE = latest_issue
                if not SESSION_START_ISSUE: SESSION_START_ISSUE = latest_issue
                
                await history_collection.update_one({"issue_number": latest_issue}, {"$setOnInsert": {"number": latest_number, "size": latest_size, "parity": latest_parity}}, upsert=True)
                pred_doc = await predictions_collection.find_one({"issue_number": latest_issue})
                if pred_doc and pred_doc.get("predicted_size"):
                    win_lose_status = "WIN ✅" if pred_doc.get("predicted_size") == latest_size else "LOSE ❌"
                    await predictions_collection.update_one({"issue_number": latest_issue}, {"$set": {"actual_size": latest_size, "actual_number": latest_number, "win_lose": win_lose_status}})

            next_issue = str(int(latest_issue) + 1)
            recent_preds = await predictions_collection.find({"win_lose": {"$ne": None}}).sort("issue_number", -1).limit(10).to_list(length=10)
            
            current_lose_streak = 0
            for p in recent_preds:
                if p.get("win_lose") == "LOSE ❌": current_lose_streak += 1
                else: break

            history_docs = await history_collection.find().sort("issue_number", -1).limit(500).to_list(length=500)
            
            try:
                mem_pred, final_prob, mem_logic = await asyncio.to_thread(ultimate_ai_predict, history_docs, recent_preds, next_issue)
                predicted, reason = "BIG (အကြီး) 🔴" if mem_pred == "BIG" else "SMALL (အသေး) 🟢", f"🧠 <b>AI Pro V2 Engine</b>\n{mem_logic}"
            except Exception as e:
                predicted, mem_pred, final_prob, reason = "BIG (အကြီး) 🔴", "BIG", 60.0, f"⚠️ AI Error: {str(e)}"
            
            predicted_result_db = "BIG" if "BIG" in predicted else "SMALL"
            await predictions_collection.update_one({"issue_number": next_issue}, {"$set": {"predicted_size": predicted_result_db}}, upsert=True)

            # --- 💡 TEXT UI (Manual Bet Advice) ---
            display_streak = 0 if current_lose_streak > 6 else current_lose_streak
            current_bet_count = 2 ** display_streak
            
            if display_streak == 0: 
                bet_advice = f"💰 <b>အကြံပြုလောင်းကြေး:</b> အခြေခံကြေး (1x)"
            elif display_streak <= 6: 
                bet_advice = f"💰 <b>အကြံပြုလောင်းကြေး:</b> {current_bet_count}x (Martingale)"
            
            if current_lose_streak >= 6: 
                bet_advice += "\n⚠️ <b>[DANGER] ၆ ပွဲဆက်တိုက်ရှုံးထားပါသည်! နောက်ပွဲ 1x မှ ပြန်စပါ။</b>"

            session_preds = await predictions_collection.find({"issue_number": {"$gte": SESSION_START_ISSUE}, "win_lose": {"$ne": None}}).sort("issue_number", -1).limit(20).to_list(length=20) 
            
            table_str = "<code>Period    | Result  | W/L\n----------|---------|----\n"
            for p in session_preds[:10]: 
                iss = p.get('issue_number', '0000000')
                table_str += f"{iss[:3]}**{iss[-4:]:<6}| {p.get('actual_number', 0)}-{p.get('actual_size', 'BIG'):<5} | {'✅' if 'WIN' in p.get('win_lose', '') else '❌'}\n"
            table_str += "</code>"

            LAST_KNOWN_STATE.update({
                "table_str": table_str, "next_issue": next_issue, "predicted": predicted, 
                "final_prob": final_prob, "reason": reason, "bet_advice": bet_advice
            })
            
            if is_new_issue or not MAIN_MESSAGE_ID:
                try:
                    img_buf = await asyncio.to_thread(generate_winrate_chart, session_preds)
                    img_bytes = img_buf.read() 
                    
                    sec_left = 60 - (int(time.time()) % 60)
                    if sec_left == 60: sec_left = 0
                    iss_display = f"{next_issue[:3]}**{next_issue[-4:]}"
                    
                    tg_caption = (
                        f"<b>🏆 SIX-LOTTERY (AI PRO V2)</b>\n"
                        f"⏰ Next Result In: <b>{sec_left}s</b>\n\n"
                        f"{table_str}\n"
                        f"🅿️ <b>Period:</b> {iss_display}\n"
                        f"🎯 <b>Predict: {predicted}</b>\n"
                        f"📈 <b>ဖြစ်နိုင်ခြေ:</b> {final_prob}%\n"
                        f"{LAST_KNOWN_STATE['balance']}\n"
                        f"{LAST_KNOWN_STATE['profit']}\n"
                        f"💡 <b>အကြောင်းပြချက်:</b>\n{reason}\n"
                        f"━━━━━━━━━━━━━━━━━━\n{bet_advice}"
                    )
                    
                    if MAIN_MESSAGE_ID:
                        try:
                            photo = BufferedInputFile(img_bytes, filename=f"chart_{int(time.time())}.png")
                            media = InputMediaPhoto(type='photo', media=photo, caption=tg_caption, parse_mode="HTML")
                            await bot.edit_message_media(chat_id=TELEGRAM_CHANNEL_ID, message_id=MAIN_MESSAGE_ID, media=media)
                        except Exception:
                            try: await bot.delete_message(chat_id=TELEGRAM_CHANNEL_ID, message_id=MAIN_MESSAGE_ID)
                            except: pass
                            photo_fallback = BufferedInputFile(img_bytes, filename=f"fb_{int(time.time())}.png")
                            msg = await bot.send_photo(chat_id=TELEGRAM_CHANNEL_ID, photo=photo_fallback, caption=tg_caption, disable_notification=True)
                            MAIN_MESSAGE_ID = msg.message_id
                    else:
                        photo_new = BufferedInputFile(img_bytes, filename=f"new_{int(time.time())}.png")
                        msg = await bot.send_photo(chat_id=TELEGRAM_CHANNEL_ID, photo=photo_new, caption=tg_caption)
                        MAIN_MESSAGE_ID = msg.message_id
                    
                    LAST_CAPTION_EDIT_TIME = time.time()
                except Exception: pass
                return

    elif data and data.get('code') != 0:
        API_ERROR_COUNT += 1
        if data.get('code') == 401: CURRENT_TOKEN = ""
    else:
        API_ERROR_COUNT += 1

    # Timer & Editing logic 
    current_time = time.time()
    if current_time - LAST_CAPTION_EDIT_TIME >= 1.5:
        if MAIN_MESSAGE_ID and LAST_KNOWN_STATE["next_issue"] != "Loading":
            sec_left = 60 - (int(time.time()) % 60)
            if sec_left == 60: sec_left = 0 
            iss = LAST_KNOWN_STATE['next_issue']
            iss_display = f"{iss[:3]}**{iss[-4:]}" if len(iss) > 4 else iss
            
            tg_caption = (
                f"<b>🏆 SIX-LOTTERY (AI PRO V2)</b>\n"
                f"⏰ Next Result In: <b>{sec_left}s</b>\n\n"
                f"{LAST_KNOWN_STATE['table_str']}\n"
                f"🅿️ <b>Period:</b> {iss_display}\n"
                f"🎯 <b>Predict: {LAST_KNOWN_STATE['predicted']}</b>\n"
                f"📈 <b>ဖြစ်နိုင်ခြေ:</b> {LAST_KNOWN_STATE['final_prob']}%\n"
                f"{LAST_KNOWN_STATE['balance']}\n"
                f"{LAST_KNOWN_STATE['profit']}\n"
                f"💡 <b>အကြောင်းပြချက်:</b>\n{LAST_KNOWN_STATE['reason']}\n"
                f"━━━━━━━━━━━━━━━━━━\n{LAST_KNOWN_STATE['bet_advice']}"
            )
            if API_ERROR_COUNT >= 3: tg_caption = f"⚠️ <b>[API သော့ သက်တမ်းကုန်သွားပါပြီ! အသစ်လဲပေးပါ]</b>\n\n" + tg_caption

            try:
                await bot.edit_message_caption(chat_id=TELEGRAM_CHANNEL_ID, message_id=MAIN_MESSAGE_ID, caption=tg_caption, parse_mode="HTML")
                LAST_CAPTION_EDIT_TIME = time.time()
            except TelegramRetryAfter as e: LAST_CAPTION_EDIT_TIME = time.time() + e.retry_after
            except Exception: pass

# ==========================================
# 💬 5. TELEGRAM COMMAND HANDLERS
# ==========================================
@dp.message(Command("start"))
async def send_welcome(message: types.Message):
    await message.reply("👋 မင်္ဂလာပါ။ စနစ်က Zero-Lag AI Pro V2 Timer ဖြင့် အလုပ်လုပ်နေပါပြီ။\n(Auto-Bet စနစ်အား ပိတ်ထားပါသည်)")

# ==========================================
# 🔄 6. BACKGROUND TASK
# ==========================================
async def auto_broadcaster():
    await init_db() 
    async with aiohttp.ClientSession() as session:
        await login_and_get_token(session)
        while True:
            try: 
                await check_game_and_predict(session)
            except Exception: 
                pass
            await asyncio.sleep(0.5) 

async def main():
    print("🚀 Aiogram SIX-LOTTERY Bot (AI Pro V2 Edition - No Autobet) စတင်နေပါပြီ...\n")
    await bot.delete_webhook(drop_pending_updates=True)
    asyncio.create_task(auto_broadcaster())
    await dp.start_polling(bot)

if __name__ == '__main__':
    try: asyncio.run(main())
    except KeyboardInterrupt: print("Bot ကို ရပ်တန့်လိုက်ပါသည်။")
