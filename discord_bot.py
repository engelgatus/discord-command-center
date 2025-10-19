"""
Discord Personal Command Center Bot - Enhanced AI + SQLite + Slash Commands + Advanced Analytics
Created for: TikTokTechnician's Productivity Hub
Features: AI Priority, Summaries, Charts, Predictive Alerts, Context-Aware Intelligence, Smart Digest, Advanced Analytics
Python Version: 3.13.7
UI: Optimized, Aesthetic, Edgy
Host: Render.com Web Service (web_server.py)
"""

# =========================
# Imports and Dependencies
# =========================
import os
import io
import csv
import json
import math
import sqlite3
import asyncio
import random
import logging
import contextlib
import seaborn as sns
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta, timezone

import discord
from discord.ext import commands, tasks
from discord import app_commands

from dotenv import load_dotenv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'sans-serif'  # Fix font warnings
import pandas as pd

from collections import defaultdict, deque

from web_server import keep_alive
from openai import OpenAI

# =========================
# Environment & Constants
# =========================
load_dotenv()
TOKEN = os.getenv('DISCORD_BOT_TOKEN')
OPENAI_KEY = os.getenv('OPENAI_API_KEY')

client_openai = OpenAI(api_key=OPENAI_KEY) if OPENAI_KEY else None

intents = discord.Intents.default()
intents.message_content = True
intents.members = True

# Keep single-file structure but enable slash commands via a CommandTree
bot = commands.Bot(command_prefix='!', intents=intents)
tree = bot.tree

DATA_DIR = Path('bot_data')
DATA_DIR.mkdir(exist_ok=True)

# Legacy CSV/JSON paths (for migration and backward compatibility)
TASKS_FILE = DATA_DIR / 'tasks.csv'
PYTHON_FILE = DATA_DIR / 'python_log.csv'
N8N_FILE = DATA_DIR / 'n8n_log.csv'
IDEAS_FILE = DATA_DIR / 'ideas.csv'
STREAKS_FILE = DATA_DIR / 'streaks.json'
PATTERNS_FILE = DATA_DIR / 'user_patterns.json'

# SQLite DB
DB_PATH = DATA_DIR / 'command_center.db'

# Theme colors
COLOR_CYAN = 0x00FF9F
COLOR_PURPLE = 0x9D4EDD
COLOR_AMBER = 0xFFB700
COLOR_DEEP_BLUE = 0x0D1117
COLOR_MUTED = 0x64748B
COLOR_ERROR = 0xDC143C

# Logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("command-center")

# Motivation quotes (preserved)
MOTIVATION_QUOTES = [
    "Consistency breeds excellence 💎",
    "The grind never stops 🔥",
    "Another W in the books 🏆",
    "Building empire, one task at a time ⚡",
    "No days off, only progress 🚀",
    "You're different, keep going 💪",
    "Momentum is unstoppable ⚡",
    "Greatness is in the details 🎯"
]

# ======================================
# SQLite Connection Pool (Render-safe)
# ======================================
class SQLitePool:
    def __init__(self, db_path: Path, pool_size: int = 5, timeout: int = 30):
        self.db_path = str(db_path)
        self._pool = asyncio.Queue()
        self.pool_size = pool_size
        self.timeout = timeout
        self._initialized = False

    async def init(self):
        if self._initialized:
            return
        for _ in range(self.pool_size):
            conn = sqlite3.connect(self.db_path, timeout=self.timeout, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            self._pool.put_nowait(conn)
        self._initialized = True

    async def acquire(self):
        return await self._pool.get()

    async def release(self, conn):
        await self._pool.put(conn)

    async def close(self):
        while not self._pool.empty():
            conn = await self._pool.get()
            with contextlib.suppress(Exception):
                conn.close()

db_pool = SQLitePool(DB_PATH)

# ======================================
# Database Schema and Initialization
# ======================================
SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    user_name TEXT NOT NULL,
    task TEXT NOT NULL,
    category TEXT DEFAULT '',
    tags TEXT DEFAULT '',
    created_date TEXT NOT NULL,   -- YYYY-MM-DD
    created_time TEXT NOT NULL    -- HH:MM:SS
);

CREATE INDEX IF NOT EXISTS idx_tasks_user_date ON tasks(user_id, created_date);
CREATE INDEX IF NOT EXISTS idx_tasks_text ON tasks(task);

CREATE TABLE IF NOT EXISTS streaks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    streak_type TEXT NOT NULL,    -- daily_checkin | python_learning | n8n_workflows
    count INTEGER NOT NULL DEFAULT 0,
    last_date TEXT                -- YYYY-MM-DD
);

CREATE UNIQUE INDEX IF NOT EXISTS uniq_streak ON streaks(user_id, streak_type);

CREATE TABLE IF NOT EXISTS patterns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    keyword TEXT NOT NULL,
    command_used TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_patterns_user_keyword ON patterns(user_id, keyword);

CREATE TABLE IF NOT EXISTS ideas (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    user_name TEXT NOT NULL,
    idea TEXT NOT NULL,
    tags TEXT DEFAULT '',
    created_date TEXT NOT NULL,
    created_time TEXT NOT NULL
);

-- Store last 5 tasks per user quickly
CREATE TABLE IF NOT EXISTS last_tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    task TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_last_tasks_user ON last_tasks(user_id);

-- NEW: Goals tracking
CREATE TABLE IF NOT EXISTS goals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    category TEXT NOT NULL,        -- python, automation, daily
    target INTEGER NOT NULL,       -- weekly target number
    current_count INTEGER DEFAULT 0,
    week_start TEXT NOT NULL,      -- YYYY-MM-DD of Monday
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_goals_user_week ON goals(user_id, week_start);

-- NEW: Smart reminders tracking
CREATE TABLE IF NOT EXISTS reminders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    reminder_type TEXT NOT NULL,   -- pattern_detected | streak_risk | goal_behind
    message TEXT NOT NULL,
    suggested_action TEXT NOT NULL,
    sent_at TEXT NOT NULL,
    acknowledged INTEGER DEFAULT 0  -- 0=pending, 1=acknowledged
);
CREATE INDEX IF NOT EXISTS idx_reminders_user ON reminders(user_id);
"""

async def db_execute(conn, sql, params=()):
    cur = conn.cursor()
    cur.execute(sql, params)
    conn.commit()
    return cur

async def db_fetchall(conn, sql, params=()):
    cur = conn.cursor()
    cur.execute(sql, params)
    return cur.fetchall()

async def db_fetchone(conn, sql, params=()):
    cur = conn.cursor()
    cur.execute(sql, params)
    return cur.fetchone()

async def init_db():
    await db_pool.init()
    conn = await db_pool.acquire()
    try:
        # Apply schema
        conn.executescript(SCHEMA_SQL)
        conn.commit()
        # Migrate legacy data idempotently
        await migrate_from_legacy(conn)
    except Exception as e:
        log.exception("DB init failed")
        raise
    finally:
        await db_pool.release(conn)

# ======================================
# Legacy Data Migration (CSV/JSON -> DB)
# ======================================
def parse_csv(path: Path):
    if not path.exists():
        return []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)

async def upsert_streak(conn, user_id: str, streak_type: str, count: int, last_date: str | None):
    existing = await db_fetchone(conn, "SELECT id FROM streaks WHERE user_id=? AND streak_type=?", (user_id, streak_type))
    if existing:
        await db_execute(conn, "UPDATE streaks SET count=?, last_date=? WHERE user_id=? AND streak_type=?",
                         (count, last_date, user_id, streak_type))
    else:
        await db_execute(conn, "INSERT INTO streaks (user_id, streak_type, count, last_date) VALUES (?, ?, ?, ?)",
                         (user_id, streak_type, count, last_date))

async def insert_task(conn, user_id, user_name, task, category, tags, date, time_):
    await db_execute(conn, """
        INSERT INTO tasks (user_id, user_name, task, category, tags, created_date, created_time)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (str(user_id), user_name, task, category or '', tags or '', date, time_))

async def insert_idea(conn, user_id, user_name, idea, tags, date, time_):
    await db_execute(conn, """
        INSERT INTO ideas (user_id, user_name, idea, tags, created_date, created_time)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (str(user_id), user_name, idea, tags or '', date, time_))

async def push_last_task(conn, user_id: str, task: str):
    # Keep last 5
    await db_execute(conn, "INSERT INTO last_tasks (user_id, task, created_at) VALUES (?, ?, ?)",
                     (user_id, task, datetime.now().isoformat()))
    rows = await db_fetchall(conn, "SELECT id FROM last_tasks WHERE user_id=? ORDER BY id DESC", (user_id,))
    if len(rows) > 5:
        # delete older beyond last 5
        ids_to_keep = {row['id'] for row in rows[:5]}
        all_ids = await db_fetchall(conn, "SELECT id FROM last_tasks WHERE user_id=?", (user_id,))
        ids_all = [r['id'] for r in all_ids]
        ids_delete = [i for i in ids_all if i not in ids_to_keep]
        for old_id in ids_delete:
            await db_execute(conn, "DELETE FROM last_tasks WHERE id=?", (old_id,))

async def migrate_from_legacy(conn):
    # Check if tasks table has any data; if empty, migrate
    existing_count = (await db_fetchone(conn, "SELECT COUNT(*) as c FROM tasks"))['c']
    if existing_count > 0:
        return

    # [Migration code - add complete migration for all CSV files]
    # For brevity, adding basic structure
    for csv_file, category in [(TASKS_FILE, ''), (PYTHON_FILE, 'python'), (N8N_FILE, 'automation')]:
        if csv_file.exists():
            rows = parse_csv(csv_file)
            for r in rows:
                date = r.get('Date', datetime.now().strftime('%Y-%m-%d'))
                time_ = r.get('Time', datetime.now().strftime('%H:%M:%S'))
                user = r.get('User', '')
                task = r.get('Task', '')
                user_id = r.get('UserID', '')
                tags = f'#{category}' if category else ''
                await insert_task(conn, user_id, user, task, category, tags, date, time_)
                await push_last_task(conn, str(user_id), task)

# ======================================
# Streak Helpers (DB-based)
# ======================================
async def update_streak(user_id: int | str, streak_type: str):
    uid = str(user_id)
    conn = await db_pool.acquire()
    try:
        row = await db_fetchone(conn, "SELECT count, last_date FROM streaks WHERE user_id=? AND streak_type=?", (uid, streak_type))
        today = datetime.now().strftime('%Y-%m-%d')
        if not row:
            await upsert_streak(conn, uid, streak_type, 1, today)
            return 1
        count = int(row['count'] or 0)
        last_date = row['last_date']
        if last_date == today:
            return count
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        if last_date == yesterday:
            count += 1
        else:
            count = 1
        await upsert_streak(conn, uid, streak_type, count, today)
        return count
    except Exception:
        log.exception("update_streak failed")
        return 0
    finally:
        await db_pool.release(conn)

async def get_all_streaks(user_id: int | str):
    uid = str(user_id)
    conn = await db_pool.acquire()
    try:
        rows = await db_fetchall(conn, "SELECT streak_type, count FROM streaks WHERE user_id=?", (uid,))
        d = {'daily_checkin': 0, 'python_learning': 0, 'n8n_workflows': 0}
        for r in rows:
            d[r['streak_type']] = int(r['count'] or 0)
        return d
    except Exception:
        log.exception("get_all_streaks failed")
        return {'daily_checkin': 0, 'python_learning': 0, 'n8n_workflows': 0}
    finally:
        await db_pool.release(conn)

def get_streak_emoji(count: int):
    if count >= 30:
        return "👑"
    elif count >= 21:
        return "💎"
    elif count >= 14:
        return "⭐"
    elif count >= 7:
        return "🔥"
    elif count >= 3:
        return "⚡"
    else:
        return "✨"

# ======================================
# Context-Aware Intelligence (DB-backed)
# ======================================
async def learn_pattern(user_id, message_content: str, command_used: str):
    conn = await db_pool.acquire()
    try:
        keywords = [w.lower() for w in message_content.split() if len(w) > 3][:5]
        for kw in keywords:
            await db_execute(conn,
                "INSERT INTO patterns (user_id, keyword, command_used, created_at) VALUES (?, ?, ?, ?)",
                (str(user_id), kw, command_used, datetime.now().isoformat())
            )
    except Exception:
        log.exception("learn_pattern failed")
    finally:
        await db_pool.release(conn)

async def suggest_command(user_id, message_content: str):
    conn = await db_pool.acquire()
    try:
        keywords = [w.lower() for w in message_content.split() if len(w) > 3]
        if not keywords:
            return None
        scores = defaultdict(int)
        for kw in keywords:
            rows = await db_fetchall(conn, "SELECT command_used FROM patterns WHERE user_id=? AND keyword=?", (str(user_id), kw))
            for r in rows:
                scores[r['command_used']] += 1
        if not scores:
            return None
        return max(scores, key=scores.get)
    except Exception:
        log.exception("suggest_command failed")
        return None
    finally:
        await db_pool.release(conn)

async def auto_categorize(task: str):
    t = task.lower()
    tags = []
    cat = ''
    if any(k in t for k in ['python', 'code', 'script']):
        cat = 'python'
        tags.append('#python')
    if any(k in t for k in ['n8n', 'automation', 'workflow']):
        cat = cat or 'automation'
        tags.append('#automation')
    if any(k in t for k in ['video', 'edit', 'thumbnail']):
        tags.append('#video')
    if 'idea' in t:
        cat = cat or 'idea'
        tags.append('#idea')
    return cat, ' '.join(sorted(set(tags)))

async def predict_best_hour(user_id: int | str):
    uid = str(user_id)
    conn = await db_pool.acquire()
    try:
        rows = await db_fetchall(conn, "SELECT created_time FROM tasks WHERE user_id=?", (uid,))
        if not rows:
            return None
        hours = []
        for r in rows:
            try:
                h = int(r['created_time'].split(':')[0])
                hours.append(h)
            except:
                pass
        if not hours:
            return None
        counts = defaultdict(int)
        for h in hours:
            counts[h] += 1
        best = max(counts, key=counts.get)
        return best  # 0-23
    except Exception:
        log.exception("predict_best_hour failed")
        return None
    finally:
        await db_pool.release(conn)

# ======================================
# Task Logging Helpers (DB, preserving CSV compatibility)
# ======================================
async def log_task(user_name, user_id, task_description):
    now = datetime.now()
    date = now.strftime('%Y-%m-%d')
    time_ = now.strftime('%H:%M:%S')
    cat, tags = await auto_categorize(task_description)

    conn = await db_pool.acquire()
    try:
        await insert_task(conn, str(user_id), user_name, task_description, cat, tags, date, time_)
        await push_last_task(conn, str(user_id), task_description)
    except Exception:
        log.exception("log_task failed")
        raise
    finally:
        await db_pool.release(conn)

# ======================================
# Predictive Streak Alerts (DB-based)
# ======================================
async def predict_streak_break(user_id: int | str):
    uid = str(user_id)
    best_hour = await predict_best_hour(uid)
    if best_hour is None:
        return False, "Not enough data yet"
    now_hour = datetime.now().hour
    conn = await db_pool.acquire()
    try:
        today = datetime.now().strftime('%Y-%m-%d')
        row = await db_fetchone(conn, "SELECT 1 FROM tasks WHERE user_id=? AND created_date=?", (uid, today))
        has_activity_today = row is not None
    finally:
        await db_pool.release(conn)

    if not has_activity_today and now_hour > (best_hour + 2):
        return True, f"⚠️ You usually log by {int(best_hour)}:00, but haven't today!"
    return False, "You're on track!"

# ======================================
# Advanced Analytics Functions (NEW)
# ======================================
async def get_productivity_heatmap_data(user_id: str, days_back: int = 30):
    conn = await db_pool.acquire()
    try:
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days_back)
        rows = await db_fetchall(conn, """
            SELECT created_date, created_time, category 
            FROM tasks 
            WHERE user_id=? AND created_date BETWEEN ? AND ?
        """, (user_id, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')))
        data = defaultdict(lambda: defaultdict(int))
        for r in rows:
            try:
                date = datetime.strptime(r['created_date'], '%Y-%m-%d').date()
                hour = int(r['created_time'].split(':')[0])
                day_name = date.strftime('%a')
                data[hour][day_name] += 1
            except:
                pass
        return data
    finally:
        await db_pool.release(conn)

async def get_velocity_data(user_id: str, days_back: int = 30):
    conn = await db_pool.acquire()
    try:
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days_back)
        rows = await db_fetchall(conn, """
            SELECT created_date, COUNT(*) as task_count
            FROM tasks 
            WHERE user_id=? AND created_date BETWEEN ? AND ?
            GROUP BY created_date
            ORDER BY created_date
        """, (user_id, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')))
        date_counts = {r['created_date']: r['task_count'] for r in rows}
        velocity_data = []
        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.strftime('%Y-%m-%d')
            count = date_counts.get(date_str, 0)
            velocity_data.append({'date': current_date, 'count': count})
            current_date += timedelta(days=1)
        return velocity_data
    finally:
        await db_pool.release(conn)

async def get_category_breakdown(user_id: str, days_back: int = 30):
    conn = await db_pool.acquire()
    try:
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days_back)
        rows = await db_fetchall(conn, """
            SELECT category, COUNT(*) as count
            FROM tasks 
            WHERE user_id=? AND created_date BETWEEN ? AND ?
            GROUP BY category
            ORDER BY count DESC
        """, (user_id, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')))
        return [(r['category'] or 'general', r['count']) for r in rows]
    finally:
        await db_pool.release(conn)

# ======================================
# Goals System (NEW)
# ======================================
async def get_current_week_start():
    today = datetime.now().date()
    days_since_monday = today.weekday()
    monday = today - timedelta(days=days_since_monday)
    return monday

async def set_weekly_goal(user_id: str, category: str, target: int):
    week_start = await get_current_week_start()
    conn = await db_pool.acquire()
    try:
        existing = await db_fetchone(conn, """
            SELECT id FROM goals 
            WHERE user_id=? AND category=? AND week_start=?
        """, (user_id, category, week_start.strftime('%Y-%m-%d')))
        
        if existing:
            await db_execute(conn, """
                UPDATE goals SET target=? 
                WHERE user_id=? AND category=? AND week_start=?
            """, (target, user_id, category, week_start.strftime('%Y-%m-%d')))
        else:
            await db_execute(conn, """
                INSERT INTO goals (user_id, category, target, week_start, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (user_id, category, target, week_start.strftime('%Y-%m-%d'), datetime.now().isoformat()))
    finally:
        await db_pool.release(conn)

async def get_weekly_progress(user_id: str):
    week_start = await get_current_week_start()
    week_end = week_start + timedelta(days=6)
    conn = await db_pool.acquire()
    try:
        goals = await db_fetchall(conn, """
            SELECT category, target FROM goals 
            WHERE user_id=? AND week_start=?
        """, (user_id, week_start.strftime('%Y-%m-%d')))
        actual_counts = await db_fetchall(conn, """
            SELECT category, COUNT(*) as count
            FROM tasks 
            WHERE user_id=? AND created_date BETWEEN ? AND ?
            GROUP BY category
        """, (user_id, week_start.strftime('%Y-%m-%d'), week_end.strftime('%Y-%m-%d')))
        actual_dict = {r['category']: r['count'] for r in actual_counts}
        progress = []
        for goal in goals:
            category = goal['category']
            target = goal['target']
            current = actual_dict.get(category, 0)
            progress.append({
                'category': category,
                'target': target,
                'current': current,
                'percentage': min(100, (current / target) * 100) if target > 0 else 0
            })
        return progress
    finally:
        await db_pool.release(conn)

# ======================================
# Advanced Chart Generation (NEW)
# ======================================
async def generate_productivity_heatmap(user_id: str):
    data = await get_productivity_heatmap_data(user_id)
    hours = list(range(24))
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    matrix = []
    for hour in hours:
        row = [data[hour][day] for day in days]
        matrix.append(row)
    
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 8), facecolor='#0D1117')
    ax.set_facecolor('#0D1117')
    
    sns.heatmap(
        matrix, 
        xticklabels=days, 
        yticklabels=[f"{h:02d}:00" for h in hours],
        cmap='viridis',
        annot=False,
        cbar_kws={'label': 'Tasks Completed'},
        ax=ax
    )
    
    ax.set_title('PRODUCTIVITY HEATMAP - HOUR × DAY', fontsize=16, fontweight='bold', 
                 color='#FFFFFF', pad=20)
    ax.set_xlabel('DAY OF WEEK', fontsize=12, fontweight='bold', color='#64748B')
    ax.set_ylabel('HOUR OF DAY', fontsize=12, fontweight='bold', color='#64748B')
    
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight', 
                facecolor='#0D1117', edgecolor='none')
    buf.seek(0)
    plt.close()
    return buf

async def generate_chart(interaction: discord.Interaction, user_id):
    """Generate aesthetic 30-day activity heatmap"""
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=30)
    dates = [(start_date + timedelta(days=i)) for i in range(31)]
    task_counts = [0] * 31
    python_counts = [0] * 31
    n8n_counts = [0] * 31

    conn = await db_pool.acquire()
    try:
        rows = await db_fetchall(conn, """
            SELECT created_date, task, category FROM tasks
            WHERE user_id=? AND created_date BETWEEN ? AND ?
        """, (str(user_id), start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')))
        for r in rows:
            try:
                entry_date = datetime.strptime(r['created_date'], '%Y-%m-%d').date()
                idx = (entry_date - start_date).days
                task_counts[idx] += 1
                cat = (r['category'] or '').lower()
                if cat == 'python':
                    python_counts[idx] += 1
                if cat == 'automation':
                    n8n_counts[idx] += 1
            except:
                pass
    finally:
        await db_pool.release(conn)

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(14,7), facecolor='#0D1117')
    ax.set_facecolor('#0D1117')

    ax.plot(dates, task_counts, label='Tasks', marker='o', color='#00FF9F',
            linewidth=3, markersize=6, markeredgewidth=2, markeredgecolor='#0D1117')
    ax.plot(dates, python_counts, label='Python', marker='s', color='#FFB700',
            linewidth=3, markersize=6, markeredgewidth=2, markeredgecolor='#0D1117')
    ax.plot(dates, n8n_counts, label='Automation', marker='^', color='#9D4EDD',
            linewidth=3, markersize=6, markeredgewidth=2, markeredgecolor='#0D1117')

    ax.set_xlabel('DATE', fontsize=13, fontweight='bold', color='#64748B')
    ax.set_ylabel('ACTIVITY COUNT', fontsize=13, fontweight='bold', color='#64748B')
    ax.set_title('30-DAY PERFORMANCE HEATMAP', fontsize=16, fontweight='bold',
                 color='#FFFFFF', pad=20)

    legend = ax.legend(fontsize=11, framealpha=0.9, facecolor='#161B22',
                       edgecolor='#30363D', loc='upper left')
    for text in legend.get_texts():
        text.set_color('#C9D1D9')

    ax.grid(alpha=0.15, linestyle='--', color='#30363D')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#30363D')
    ax.spines['bottom'].set_color('#30363D')

    plt.xticks(rotation=45, ha='right', fontsize=9, color='#8B949E')
    plt.yticks(fontsize=10, color='#8B949E')

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='#0D1117', edgecolor='none')
    buf.seek(0)
    plt.close()

    file = discord.File(buf, filename='heatmap.png')
    embed = discord.Embed(
        description="Visual breakdown of your productivity journey",
        color=COLOR_CYAN
    )
    embed.set_image(url="attachment://heatmap.png")
    embed.set_footer(text="Last 30 days • Real-time tracking",
                     icon_url=interaction.user.display_avatar.url)

    await interaction.followup.send(embed=embed, file=file, ephemeral=True)

# ======================================
# Slash Commands - Core Features
# ======================================
@tree.command(name="ping", description="Check bot status and latency")
async def slash_ping(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=True)
    latency = round(bot.latency * 1000)
    if latency < 150:
        color = COLOR_CYAN; status_emoji = "🟢"; status = "Good"
    elif latency < 300:
        color = COLOR_AMBER; status_emoji = "🟡"; status = "Stable"
    else:
        color = COLOR_ERROR; status_emoji = "🔴"; status = "Degraded"

    embed = discord.Embed(
        title="🏓 Pong!",
        description="Bot is responsive and ready!",
        color=color,
        timestamp=datetime.now()
    )
    embed.add_field(name="⚡ Latency", value=f"`{latency}ms` - {status}", inline=True)
    embed.add_field(name=f"{status_emoji} Status", value="Online & Active", inline=True)
    if client_openai:
        embed.add_field(name="🤖 AI Engine", value="Enabled", inline=True)
    embed.set_footer(text="Command Center Bot")
    await interaction.followup.send(embed=embed, ephemeral=True)

@tree.command(name="done", description="Log a completed task")
@app_commands.describe(task="Describe what you completed")
async def slash_done(interaction: discord.Interaction, task: str):
    await interaction.response.defer(ephemeral=True)
    try:
        await log_task(interaction.user.name, interaction.user.id, task)
        await learn_pattern(interaction.user.id, task, 'done')
        streak_count = await update_streak(interaction.user.id, 'daily_checkin')
        streak_emoji = get_streak_emoji(streak_count)
        is_milestone = streak_count % 7 == 0 and streak_count > 0

        if streak_count >= 30:
            color = COLOR_PURPLE; tier = "LEGENDARY"
        elif streak_count >= 14:
            color = COLOR_AMBER; tier = "ELITE"
        elif streak_count >= 7:
            color = COLOR_CYAN; tier = "RISING"
        else:
            color = COLOR_MUTED; tier = "BUILDING"

        embed = discord.Embed(
            title=f"{streak_emoji} Task Completed!",
            description=f"**{task}**",
            color=color,
            timestamp=datetime.now()
        )
        embed.add_field(name="🔥 Daily Streak", value=f"**{streak_count}** day{'s' if streak_count != 1 else ''}", inline=True)

        conn = await db_pool.acquire()
        try:
            row = await db_fetchone(conn, "SELECT COUNT(*) as c FROM tasks WHERE user_id=?", (str(interaction.user.id),))
            total_tasks = int(row['c'])
        finally:
            await db_pool.release(conn)

        embed.add_field(name="✅ Total Tasks", value=f"**{total_tasks}** completed", inline=True)

        if is_milestone:
            embed.add_field(
                name="🎊 MILESTONE ACHIEVED!",
                value=f"**{streak_count} days** of consistency!\nYou're absolutely crushing it! 🚀",
                inline=False
            )

        embed.set_author(name=interaction.user.display_name, icon_url=interaction.user.display_avatar.url)
        embed.set_footer(text=random.choice(MOTIVATION_QUOTES))
        await interaction.followup.send(embed=embed, ephemeral=True)
    except Exception as e:
        embed = discord.Embed(description=f"Failed to log task: {e}", color=COLOR_ERROR)
        await interaction.followup.send(embed=embed, ephemeral=True)

@tree.command(name="python", description="Log a Python learning note")
@app_commands.describe(learning_note="What did you learn or build?")
async def slash_python(interaction: discord.Interaction, learning_note: str):
    await interaction.response.defer(ephemeral=True)
    note = f"[PYTHON] {learning_note}"
    try:
        await log_task(interaction.user.name, interaction.user.id, note)
        await learn_pattern(interaction.user.id, learning_note, 'python')
        streak_count = await update_streak(interaction.user.id, 'python_learning')
        streak_emoji = get_streak_emoji(streak_count)

        embed = discord.Embed(
            title="🐍 PYTHON DEVELOPMENT",
            description=f"**{learning_note}**",
            color=0x3776AB,
            timestamp=datetime.now()
        )
        embed.add_field(name=f"{streak_emoji} LEARNING STREAK", value=f"**{streak_count}** day{'s' if streak_count != 1 else ''} straight", inline=True)
        embed.add_field(name="📚 CATEGORY", value="Python Development", inline=True)
        embed.set_author(name=f"{interaction.user.display_name}'s Code Journey", icon_url=interaction.user.display_avatar.url)
        embed.set_footer(text="from progress import mastery 💻")
        embed.set_thumbnail(url=interaction.user.display_avatar.url)
        await interaction.followup.send(embed=embed, ephemeral=True)
    except Exception as e:
        embed = discord.Embed(description=f"Failed to log Python note: {e}", color=COLOR_ERROR)
        await interaction.followup.send(embed=embed, ephemeral=True)

@tree.command(name="n8n", description="Log an automation workflow note")
@app_commands.describe(workflow_note="What automation did you build or run?")
async def slash_n8n(interaction: discord.Interaction, workflow_note: str):
    await interaction.response.defer(ephemeral=True)
    note = f"[N8N] {workflow_note}"
    try:
        await log_task(interaction.user.name, interaction.user.id, note)
        await learn_pattern(interaction.user.id, workflow_note, 'n8n')
        streak_count = await update_streak(interaction.user.id, 'n8n_workflows')
        streak_emoji = get_streak_emoji(streak_count)

        embed = discord.Embed(
            title="🔗 AUTOMATION WORKFLOW",
            description=f"**{workflow_note}**",
            color=0xEA4B71,
            timestamp=datetime.now()
        )
        embed.add_field(name=f"{streak_emoji} WORKFLOW STREAK", value=f"**{streak_count}** day{'s' if streak_count != 1 else ''} running", inline=True)
        embed.add_field(name="🤖 STATUS", value="Workflow Logged", inline=True)
        embed.set_author(name=f"{interaction.user.display_name}'s Automation Lab", icon_url=interaction.user.display_avatar.url)
        embed.set_footer(text="Efficiency through automation 🚀")
        embed.set_thumbnail(url=interaction.user.display_avatar.url)
        await interaction.followup.send(embed=embed, ephemeral=True)
    except Exception as e:
        embed = discord.Embed(description=f"Failed to log workflow: {e}", color=COLOR_ERROR)
        await interaction.followup.send(embed=embed, ephemeral=True)

@tree.command(name="stats", description="View your progress dashboard")
async def slash_stats(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=True)
    user = interaction.user
    streaks = await get_all_streaks(user.id)
    total_streak_days = sum(streaks.values())
    
    if total_streak_days >= 90:
        color = COLOR_PURPLE; rank = "👑 LEGEND"; rank_icon = "👑"
    elif total_streak_days >= 50:
        color = COLOR_AMBER; rank = "💎 MASTER"; rank_icon = "💎"
    elif total_streak_days >= 20:
        color = COLOR_CYAN; rank = "⭐ ELITE"; rank_icon = "⭐"
    else:
        color = COLOR_MUTED; rank = "⚡ RISING"; rank_icon = "⚡"

    embed = discord.Embed(
        title="📊 Progress Dashboard",
        description=f"**{rank_icon} {user.display_name}**'s Command Center\n*Your journey to consistency*",
        color=color,
        timestamp=datetime.now()
    )
    
    daily_emoji = get_streak_emoji(streaks['daily_checkin'])
    python_emoji = get_streak_emoji(streaks['python_learning'])
    n8n_emoji = get_streak_emoji(streaks['n8n_workflows'])

    embed.add_field(name=f"{daily_emoji} Daily Check-In Streak", value=f"**{streaks['daily_checkin']}** day{'s' if streaks['daily_checkin'] != 1 else ''}", inline=True)
    embed.add_field(name=f"{python_emoji} Python Learning Streak", value=f"**{streaks['python_learning']}** day{'s' if streaks['python_learning'] != 1 else ''}", inline=True)
    embed.add_field(name=f"{n8n_emoji} n8n Workflow Streak", value=f"**{streaks['n8n_workflows']}** day{'s' if streaks['n8n_workflows'] != 1 else ''}", inline=True)

    conn = await db_pool.acquire()
    try:
        row = await db_fetchone(conn, "SELECT COUNT(*) as c FROM tasks WHERE user_id=?", (str(user.id),))
        total_tasks = int(row['c'])
    finally:
        await db_pool.release(conn)

    embed.add_field(name="✅ Lifetime Stats", value=f"**Total Tasks:** {total_tasks}\n**Total Streak Days:** {total_streak_days}\n**Commands Used:** {total_tasks}", inline=False)

    at_risk, message = await predict_streak_break(user.id)
    if at_risk:
        embed.add_field(name="⚠️ Streak Alert", value=message, inline=False)

    max_streak = max(streaks.values()) if streaks.values() else 0
    if max_streak > 0:
        progress_bar = "█" * min(max_streak, 20) + "░" * (20 - min(max_streak, 20))
        embed.add_field(name="📈 Progress Visualization", value=f"`{progress_bar}` {max_streak}/30 days", inline=False)

    embed.set_thumbnail(url=user.display_avatar.url)
    embed.set_author(name="Personal Command Center", icon_url=bot.user.display_avatar.url)

    if total_streak_days >= 30:
        footer_text = "👑 Legendary consistency! You're a productivity master!"
    elif total_streak_days >= 14:
        footer_text = "⭐ Outstanding work! Keep the momentum going!"
    elif total_streak_days >= 7:
        footer_text = "💎 Great progress! You're building strong habits!"
    else:
        footer_text = "✨ Every journey begins with a single step!"
    embed.set_footer(text=footer_text, icon_url=bot.user.display_avatar.url)

    await interaction.followup.send(embed=embed, ephemeral=True)
    await generate_chart(interaction, user.id)

@tree.command(name="idea", description="Capture an idea quickly")
@app_commands.describe(idea="Describe the idea")
async def slash_idea(interaction: discord.Interaction, idea: str):
    await interaction.response.defer(ephemeral=True)
    try:
        await log_task(interaction.user.name, interaction.user.id, f"[IDEA] {idea}")
        embed = discord.Embed(
            title="💡 IDEA CAPTURED",
            description=f"**{idea}**",
            color=COLOR_AMBER,
            timestamp=datetime.now()
        )
        embed.add_field(name="✨ STATUS", value="Saved to Vault", inline=True)
        embed.add_field(name="📝 TYPE", value="Quick Capture", inline=True)
        embed.set_author(name=f"{interaction.user.display_name}'s Idea Vault", icon_url=interaction.user.display_avatar.url)
        embed.set_footer(text="Innovation starts with ideas 🚀")
        await interaction.followup.send(embed=embed, ephemeral=True)
    except Exception as e:
        await interaction.followup.send(embed=discord.Embed(description=f"Failed to save idea: {e}", color=COLOR_ERROR), ephemeral=True)

@tree.command(name="priority", description="AI-powered task prioritization")
async def slash_priority(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=True)
    if not client_openai:
        await interaction.followup.send(embed=discord.Embed(description="⚠️ AI Engine Offline. Add OPENAI_API_KEY.", color=COLOR_ERROR), ephemeral=True)
        return
    
    conn = await db_pool.acquire()
    try:
        rows = await db_fetchall(conn, """
            SELECT task, created_date FROM tasks
            WHERE user_id=?
            ORDER BY id DESC LIMIT 20
        """, (str(interaction.user.id),))
        if not rows:
            await interaction.followup.send(embed=discord.Embed(description="No tasks logged yet.", color=COLOR_ERROR), ephemeral=True)
            return
        task_list = "\n".join([f"- {r['task']} ({r['created_date']})" for r in rows])
    finally:
        await db_pool.release(conn)

    prompt = f"""Analyze these completed tasks and suggest priorities:

{task_list}

Provide:
1. Task patterns you notice
2. Time patterns (if visible)  
3. 3 specific priority suggestions for next tasks
Keep it under 200 words and actionable."""

    try:
        response = client_openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300
        )
        analysis = response.choices[0].message.content
        embed = discord.Embed(
            title="Priority Insights ⚡",
            description=analysis,
            color=COLOR_AMBER,
            timestamp=datetime.now()
        )
        embed.set_footer(text=f"Based on {len(rows)} recent tasks • AI-powered insights")
        embed.set_thumbnail(url=interaction.user.display_avatar.url)
        await interaction.followup.send(embed=embed, ephemeral=True)
    except Exception as e:
        await interaction.followup.send(embed=discord.Embed(description=f"AI analysis error: {e}", color=COLOR_ERROR), ephemeral=True)

@tree.command(name="summary", description="Get an AI summary of your day or week")
@app_commands.describe(period="daily or weekly")
@app_commands.choices(period=[
    app_commands.Choice(name="daily", value="daily"),
    app_commands.Choice(name="weekly", value="weekly")
])
async def slash_summary(interaction: discord.Interaction, period: app_commands.Choice[str]):
    await interaction.response.defer(ephemeral=True)
    if not client_openai:
        await interaction.followup.send(embed=discord.Embed(title="❌ AI Engine Offline", description="Add OPENAI_API_KEY.", color=COLOR_ERROR), ephemeral=True)
        return
    
    period_value = period.value
    if period_value == "daily":
        start_date = datetime.now().date()
        title = "📅 Today's Summary"
        period_display = "today"
    else:
        start_date = datetime.now().date() - timedelta(days=7)
        title = "📅 This Week's Summary"  
        period_display = "this week"

    conn = await db_pool.acquire()
    try:
        rows = await db_fetchall(conn, """
            SELECT task, category FROM tasks
            WHERE user_id=? AND created_date>=?
            ORDER BY id DESC
        """, (str(interaction.user.id), start_date.strftime('%Y-%m-%d')))
    finally:
        await db_pool.release(conn)

    if not rows:
        await interaction.followup.send(embed=discord.Embed(title="❌ No Data Available", description=f"No activities found for {period_display} summary", color=COLOR_ERROR), ephemeral=True)
        return

    activities = []
    for r in rows[:30]:
        label = "Task"
        cat = (r['category'] or '').lower()
        if cat == 'python': label = "Python"
        elif cat == 'automation': label = "Automation"
        activities.append(f"{label}: {r['task']}")

    activity_text = "\n".join(activities)
    prompt = f"""Summarize these accomplishments in an inspirational, direct, edgy way (under 150 words):

{activity_text}

Keep it real and focused on actual progress. No corporate hype, just straight facts with personality."""

    try:
        response = client_openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=250
        )
        summary_text = response.choices[0].message.content
        embed = discord.Embed(
            title=title,
            description=summary_text,
            color=COLOR_PURPLE,
            timestamp=datetime.now()
        )
        embed.add_field(name="📊 Activity Breakdown", value=f"**{len(rows)}** activities logged {period_display}", inline=False)
        embed.set_footer(text="AI-powered insights • Command Center")
        embed.set_thumbnail(url=interaction.user.display_avatar.url)
        await interaction.followup.send(embed=embed, ephemeral=True)
    except Exception as e:
        await interaction.followup.send(embed=discord.Embed(title="❌ Summary Failed", description=f"AI analysis error: {e}", color=COLOR_ERROR), ephemeral=True)

# ======================================
# Slash Commands - Advanced Analytics (NEW)
# ======================================
@tree.command(name="productivity", description="Heat map showing your most productive hours/days")
async def slash_productivity(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=True)
    try:
        chart_buffer = await generate_productivity_heatmap(str(interaction.user.id))
        embed = discord.Embed(
            title="📈 PRODUCTIVITY HEATMAP",
            description="Your productivity patterns by hour and day of week",
            color=COLOR_CYAN,
            timestamp=datetime.now()
        )
        file = discord.File(chart_buffer, filename='productivity_heatmap.png')
        embed.set_image(url="attachment://productivity_heatmap.png")
        embed.set_footer(text="Last 30 days • Dark = Less Active, Bright = More Active")
        await interaction.followup.send(embed=embed, file=file, ephemeral=True)
    except Exception as e:
        embed = discord.Embed(description=f"Failed to generate productivity heatmap: {e}", color=COLOR_ERROR)
        await interaction.followup.send(embed=embed, ephemeral=True)

@tree.command(name="goal", description="Set a weekly goal for a category")
@app_commands.describe(
    category="python, automation, or daily",
    target="Number of tasks to complete this week"
)
@app_commands.choices(category=[
    app_commands.Choice(name="python", value="python"),
    app_commands.Choice(name="automation", value="automation"),
    app_commands.Choice(name="daily", value="daily")
])
async def slash_goal(interaction: discord.Interaction, category: app_commands.Choice[str], target: int):
    await interaction.response.defer(ephemeral=True)
    
    if target < 1 or target > 50:
        embed = discord.Embed(description="Goal target must be between 1 and 50.", color=COLOR_ERROR)
        await interaction.followup.send(embed=embed, ephemeral=True)
        return
    
    try:
        await set_weekly_goal(str(interaction.user.id), category.value, target)
        progress = await get_weekly_progress(str(interaction.user.id))
        current_goal = next((p for p in progress if p['category'] == category.value), None)
        
        embed = discord.Embed(
            title="🎯 WEEKLY GOAL SET",
            description=f"Goal set for **{category.value.title()}**: {target} tasks this week",
            color=COLOR_AMBER,
            timestamp=datetime.now()
        )
        
        if current_goal:
            current = current_goal['current']
            percentage = current_goal['percentage']
            progress_bar = "█" * int(percentage / 5) + "░" * (20 - int(percentage / 5))
            
            embed.add_field(
                name="📊 Current Progress",
                value=f"`{progress_bar}` {current}/{target} ({percentage:.1f}%)",
                inline=False
            )
            
            remaining = target - current
            if remaining > 0:
                embed.add_field(
                    name="⚡ To Go",
                    value=f"**{remaining}** more {category.value} tasks needed",
                    inline=True
                )
            else:
                embed.add_field(
                    name="🎉 Status",
                    value="**GOAL ACHIEVED!** 🚀",
                    inline=True
                )
        
        week_start = await get_current_week_start()
        week_end = week_start + timedelta(days=6)
        
        embed.add_field(
            name="📅 Week",
            value=f"{week_start.strftime('%b %d')} - {week_end.strftime('%b %d')}",
            inline=True
        )
        
        embed.set_footer(text="Use /done, /python, /n8n to make progress toward your goal")
        await interaction.followup.send(embed=embed, ephemeral=True)
        
    except Exception as e:
        embed = discord.Embed(description=f"Failed to set goal: {e}", color=COLOR_ERROR)
        await interaction.followup.send(embed=embed, ephemeral=True)

@tree.command(name="velocity", description="Track tasks per day trend (are you accelerating?)")
async def slash_velocity(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=True)
    try:
        velocity_data = await get_velocity_data(str(interaction.user.id), 30)
        recent_avg = sum(d['count'] for d in velocity_data[-7:]) / 7
        early_avg = sum(d['count'] for d in velocity_data[:7]) / 7
        
        if recent_avg > early_avg * 1.2:
            trend = "📈 **ACCELERATING** - You're picking up speed!"
        elif recent_avg < early_avg * 0.8:
            trend = "📉 **SLOWING** - Time to refocus"
        else:
            trend = "➡️ **STEADY** - Consistent pace"
        
        embed = discord.Embed(
            title="🚀 VELOCITY TRACKING",
            description=f"Your productivity velocity over time\n\n{trend}",
            color=COLOR_PURPLE,
            timestamp=datetime.now()
        )
        embed.add_field(name="📊 7-Day Average", value=f"**{recent_avg:.1f}** tasks/day", inline=True)
        embed.set_footer(text="Last 30 days • Velocity analysis")
        await interaction.followup.send(embed=embed, ephemeral=True)
    except Exception as e:
        embed = discord.Embed(description=f"Failed to generate velocity analysis: {e}", color=COLOR_ERROR)
        await interaction.followup.send(embed=embed, ephemeral=True)

@tree.command(name="categories", description="Pie chart of how you spend time (Python vs Video vs Automation)")  
async def slash_categories(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=True)
    try:
        category_data = await get_category_breakdown(str(interaction.user.id))
        if not category_data:
            embed = discord.Embed(description="No activity data found for category breakdown.", color=COLOR_MUTED)
            await interaction.followup.send(embed=embed, ephemeral=True)
            return
        
        total_tasks = sum(item[1] for item in category_data)
        embed = discord.Embed(
            title="🎯 CATEGORY BREAKDOWN",
            description="How you're spending your productive time",
            color=COLOR_AMBER,
            timestamp=datetime.now()
        )
        embed.add_field(name="📊 Total Activity", value=f"**{total_tasks}** tasks in last 30 days", inline=False)
        
        # Show top 3 categories
        top_categories = []
        for i, (cat, count) in enumerate(category_data[:3]):
            percentage = (count / total_tasks) * 100
            cat_name = cat.title() if cat else 'General'
            top_categories.append(f"**{cat_name}:** {count} ({percentage:.1f}%)")
        
        if top_categories:
            embed.add_field(name="🏆 Top Categories", value="\n".join(top_categories), inline=False)
        
        embed.set_footer(text="Last 30 days • Focus areas breakdown")
        await interaction.followup.send(embed=embed, ephemeral=True)
    except Exception as e:
        embed = discord.Embed(description=f"Failed to generate category breakdown: {e}", color=COLOR_ERROR)
        await interaction.followup.send(embed=embed, ephemeral=True)

@tree.command(name="streakmax", description="Predict when you'll hit your longest streak ever")
async def slash_streakmax(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=True)
    try:
        streaks = await get_all_streaks(interaction.user.id)
        current_max = max(streaks.values()) if streaks.values() else 0
        
        # Simple prediction based on current consistency
        conn = await db_pool.acquire()
        try:
            rows = await db_fetchall(conn, """
                SELECT created_date FROM tasks 
                WHERE user_id=? 
                ORDER BY created_date DESC LIMIT 30
            """, (str(interaction.user.id),))
            dates_with_activity = set(r['created_date'] for r in rows)
            activity_rate = len(dates_with_activity) / 30.0
        finally:
            await db_pool.release(conn)
        
        embed = discord.Embed(
            title="🎯 STREAK PREDICTION",
            color=COLOR_PURPLE,
            timestamp=datetime.now()
        )
        
        if activity_rate > 0.8:
            next_milestone = 30 if current_max < 30 else (60 if current_max < 60 else 90)
            days_to_milestone = next_milestone - current_max
            estimated_days = int(days_to_milestone / activity_rate)
            target_date = datetime.now().date() + timedelta(days=estimated_days)
            
            embed.description = f"At current pace, you'll hit **{next_milestone}** days around **{target_date.strftime('%B %d')}**!"
            embed.add_field(name="⏰ Timeline", value=f"**{estimated_days}** days to go", inline=True)
        else:
            embed.description = "Focus on consistency first - aim for 80% daily activity rate"
        
        embed.add_field(
            name="📊 Current Stats",
            value=f"**Current Max:** {current_max} days\n**Activity Rate:** {activity_rate:.1%}",
            inline=True
        )
        embed.set_footer(text="Prediction based on your activity patterns")
        await interaction.followup.send(embed=embed, ephemeral=True)
    except Exception as e:
        embed = discord.Embed(description=f"Failed to generate streak prediction: {e}", color=COLOR_ERROR)
        await interaction.followup.send(embed=embed, ephemeral=True)

# ======================================
# Bot Events
# ======================================
@bot.event
async def on_ready():
    log.info(f'🤖 {bot.user.name} is online and ready!')
    log.info(f'📊 Connected to {len(bot.guilds)} server(s)')
    log.info(f'🤖 AI Features: {"Enabled" if client_openai else "Disabled (no API key)"}')

    try:
        await tree.sync()
        log.info("Slash commands synced.")
    except Exception:
        log.exception("Slash sync failed")

    # Start background tasks (simplified for production)
    if not daily_morning_reminder.is_running():
        daily_morning_reminder.start()

@bot.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return
    if not message.content.startswith('!') and len(message.content) > 20:
        suggestion = await suggest_command(message.author.id, message.content)
        if suggestion:
            embed = discord.Embed(
                description=f"💡 Smart Tip: Looks like a `/{suggestion}` entry.\nUse the command to track it properly.",
                color=COLOR_MUTED
            )
            try:
                await message.channel.send(embed=embed, delete_after=8)
            except:
                pass
    await bot.process_commands(message)

# ======================================
# Background Tasks (Simplified for Production)
# ======================================
@tasks.loop(hours=24)
async def daily_morning_reminder():
    """Morning reminder 9 AM"""
    for guild in bot.guilds:
        channel = discord.utils.get(guild.text_channels, name='daily-tasks')
        embed = discord.Embed(
            title="☀️ GOOD MORNING COMMAND CENTER",
            description="A new day to dominate. Let's get it. ☕⚡",
            color=COLOR_AMBER,
            timestamp=datetime.now()
        )
        embed.add_field(name="🎯 TODAY'S OBJECTIVES", value="Log tasks with `/done`\nLearn with `/python`\nAutomate with `/n8n`", inline=False)
        embed.set_footer(text="Command Center • Daily Operations")
        if channel:
            with contextlib.suppress(Exception):
                await channel.send(embed=embed)

@daily_morning_reminder.before_loop
async def before_daily_morning_reminder():
    await bot.wait_until_ready()
    now = datetime.now()
    target = now.replace(hour=9, minute=0, second=0, microsecond=0)
    if now > target: target += timedelta(days=1)
    await asyncio.sleep((target - now).total_seconds())

# ======================================
# Startup
# ======================================
async def startup():
    await init_db()

if __name__ == '__main__':
    if not TOKEN:
        print("❌ ERROR: DISCORD_BOT_TOKEN not found in .env file!")
    else:
        print("🚀 Starting enhanced AI-powered bot with SQLite + Slash Commands + Advanced Analytics...")
        keep_alive()
        try:
            asyncio.run(startup())
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(startup())
        bot.run(TOKEN)
