"""
Discord Personal Command Center Bot - Enhanced AI Version
Created for: TikTokTechnician's Productivity Hub
Features: AI Priority, Summaries, Charts, Predictive Alerts, Context-Aware, Smart Digest
Python Version: 3.13.7
UI: Optimized, Aesthetic, Edgy
"""


import discord
from discord.ext import commands, tasks
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
import csv
import json
from pathlib import Path
import asyncio
import random
from web_server import keep_alive
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import io
from openai import OpenAI
import pandas as pd
from collections import defaultdict


load_dotenv()
TOKEN = os.getenv('DISCORD_BOT_TOKEN')
OPENAI_KEY = os.getenv('OPENAI_API_KEY')


# Initialize OpenAI
client_openai = OpenAI(api_key=OPENAI_KEY) if OPENAI_KEY else None


intents = discord.Intents.default()
intents.message_content = True
intents.members = True


bot = commands.Bot(command_prefix='!', intents=intents)


DATA_DIR = Path('bot_data')
DATA_DIR.mkdir(exist_ok=True)


TASKS_FILE = DATA_DIR / 'tasks.csv'
STREAKS_FILE = DATA_DIR / 'streaks.json'
PATTERNS_FILE = DATA_DIR / 'user_patterns.json'
PYTHON_FILE = DATA_DIR / 'python_log.csv'
N8N_FILE = DATA_DIR / 'n8n_log.csv'
IDEAS_FILE = DATA_DIR / 'ideas.csv'


# Edgy motivational quotes
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


# ========== PATTERN LEARNING (Context-Aware Commands) ==========


def load_patterns():
    if PATTERNS_FILE.exists():
        with open(PATTERNS_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_patterns(patterns):
    with open(PATTERNS_FILE, 'w') as f:
        json.dump(patterns, f, indent=2)


user_patterns = load_patterns()


def learn_pattern(user_id, message_content, command_used):
    """Learn user's command patterns for context-aware suggestions"""
    user_id = str(user_id)
    if user_id not in user_patterns:
        user_patterns[user_id] = {'keywords': {}, 'command_times': defaultdict(list)}
    
    keywords = [word.lower() for word in message_content.split() if len(word) > 3]
    
    for keyword in keywords[:5]:
        if keyword not in user_patterns[user_id]['keywords']:
            user_patterns[user_id]['keywords'][keyword] = []
        user_patterns[user_id]['keywords'][keyword].append(command_used)
    
    hour = datetime.now().hour
    if command_used not in user_patterns[user_id]['command_times']:
        user_patterns[user_id]['command_times'][command_used] = []
    user_patterns[user_id]['command_times'][command_used].append(hour)
    
    save_patterns(user_patterns)


def suggest_command(user_id, message_content):
    """Context-aware command suggestion"""
    user_id = str(user_id)
    if user_id not in user_patterns:
        return None
    
    keywords = [word.lower() for word in message_content.split() if len(word) > 3]
    command_scores = defaultdict(int)
    
    for keyword in keywords:
        if keyword in user_patterns[user_id].get('keywords', {}):
            commands = user_patterns[user_id]['keywords'][keyword]
            for cmd in commands:
                command_scores[cmd] += 1
    
    if command_scores:
        suggested = max(command_scores, key=command_scores.get)
        return suggested
    return None


# ========== PREDICTIVE STREAK ALERTS ==========


def predict_streak_break(user_id):
    """Predicts if user might break their streak"""
    user_id = str(user_id)
    
    if user_id not in user_patterns or 'command_times' not in user_patterns[user_id]:
        return False, "Not enough data yet"
    
    all_hours = []
    for cmd, hours in user_patterns[user_id]['command_times'].items():
        all_hours.extend(hours)
    
    if not all_hours:
        return False, "No activity pattern"
    
    avg_hour = sum(all_hours) / len(all_hours)
    current_hour = datetime.now().hour
    
    if current_hour > avg_hour + 2:
        today = datetime.now().date()
        has_activity_today = False
        
        streaks = load_streaks()
        user_streaks = streaks.get(str(user_id), {})
        
        for streak_type, data in user_streaks.items():
            if data.get('last_date') == today.strftime('%Y-%m-%d'):
                has_activity_today = True
                break
        
        if not has_activity_today:
            return True, f"⚠️ You usually log by **{int(avg_hour)}:00**, but haven't today!"
    
    return False, "You're on track!"


# ========== EXISTING FUNCTIONS ==========


def load_streaks():
    if STREAKS_FILE.exists():
        with open(STREAKS_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_streaks(streaks_data):
    with open(STREAKS_FILE, 'w') as f:
        json.dump(streaks_data, f, indent=2)


def log_task_to_csv(user_name, user_id, task_description, file_path=TASKS_FILE):
    """FIXED: Now properly logs user_id instead of placeholder"""
    file_exists = file_path.exists()
    
    with open(file_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        if not file_exists:
            writer.writerow(['Date', 'Time', 'User', 'Task', 'UserID'])
        
        now = datetime.now()
        writer.writerow([
            now.strftime('%Y-%m-%d'),
            now.strftime('%H:%M:%S'),
            user_name,
            task_description,
            str(user_id)
        ])


def update_streak(user_id, streak_type='daily_checkin'):
    streaks = load_streaks()
    user_id_str = str(user_id)
    
    if user_id_str not in streaks:
        streaks[user_id_str] = {
            'daily_checkin': {'count': 0, 'last_date': None},
            'python_learning': {'count': 0, 'last_date': None},
            'n8n_workflows': {'count': 0, 'last_date': None}
        }
    
    today = datetime.now().strftime('%Y-%m-%d')
    last_date = streaks[user_id_str][streak_type].get('last_date')
    
    if last_date == today:
        return streaks[user_id_str][streak_type]['count']
    
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    
    if last_date == yesterday:
        streaks[user_id_str][streak_type]['count'] += 1
    else:
        streaks[user_id_str][streak_type]['count'] = 1
    
    streaks[user_id_str][streak_type]['last_date'] = today
    
    save_streaks(streaks)
    return streaks[user_id_str][streak_type]['count']


def get_all_streaks(user_id):
    streaks = load_streaks()
    user_id_str = str(user_id)
    
    if user_id_str not in streaks:
        return {
            'daily_checkin': 0,
            'python_learning': 0,
            'n8n_workflows': 0
        }
    
    return {
        streak_type: data['count'] 
        for streak_type, data in streaks[user_id_str].items()
    }


def get_streak_emoji(count):
    """Returns progressively more edgy emojis"""
    if count >= 30:
        return "👑"  # Crown for legends
    elif count >= 21:
        return "💎"  # Diamond
    elif count >= 14:
        return "⭐"  # Star
    elif count >= 7:
        return "🔥"  # Fire
    elif count >= 3:
        return "⚡"  # Lightning
    else:
        return "✨"  # Sparkle


# ========== ENHANCED VISUAL PROGRESS CHARTS ==========


async def generate_chart(ctx, user_id):
    """Generate aesthetic 30-day activity heatmap"""
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=30)
    
    dates = [(start_date + timedelta(days=i)) for i in range(31)]
    task_counts = [0] * 31
    python_counts = [0] * 31
    n8n_counts = [0] * 31
    
    # Count activities per day
    for file, counts in [(TASKS_FILE, task_counts), (PYTHON_FILE, python_counts), (N8N_FILE, n8n_counts)]:
        if file.exists():
            with open(file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        entry_date = datetime.strptime(row['Date'], '%Y-%m-%d').date()
                        if start_date <= entry_date <= end_date:
                            day_index = (entry_date - start_date).days
                            counts[day_index] += 1
                    except:
                        pass
    
    # Create dark-themed aesthetic chart
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(14, 7), facecolor='#0D1117')
    ax.set_facecolor('#0D1117')
    
    # Plot with neon colors
    ax.plot(dates, task_counts, label='Tasks', marker='o', color='#00FF9F', 
            linewidth=3, markersize=6, markeredgewidth=2, markeredgecolor='#0D1117')
    ax.plot(dates, python_counts, label='Python', marker='s', color='#FFB700', 
            linewidth=3, markersize=6, markeredgewidth=2, markeredgecolor='#0D1117')
    ax.plot(dates, n8n_counts, label='Automation', marker='^', color='#9D4EDD', 
            linewidth=3, markersize=6, markeredgewidth=2, markeredgecolor='#0D1117')
    
    # Styling
    ax.set_xlabel('DATE', fontsize=13, fontweight='bold', color='#64748B')
    ax.set_ylabel('ACTIVITY COUNT', fontsize=13, fontweight='bold', color='#64748B')
    ax.set_title('30-DAY PERFORMANCE HEATMAP', fontsize=16, fontweight='bold', 
                 color='#FFFFFF', pad=20)
    
    # Legend with custom styling
    legend = ax.legend(fontsize=11, framealpha=0.9, facecolor='#161B22', 
                       edgecolor='#30363D', loc='upper left')
    for text in legend.get_texts():
        text.set_color('#C9D1D9')
    
    # Grid
    ax.grid(alpha=0.15, linestyle='--', color='#30363D')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#30363D')
    ax.spines['bottom'].set_color('#30363D')
    
    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right', fontsize=9, color='#8B949E')
    plt.yticks(fontsize=10, color='#8B949E')
    
    plt.tight_layout()
    
    # Save with transparency
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight', 
                facecolor='#0D1117', edgecolor='none')
    buf.seek(0)
    plt.close()
    
    # Send with aesthetic embed
    file = discord.File(buf, filename='heatmap.png')
    
    embed = discord.Embed(
        description="Visual breakdown of your productivity journey",
        color=0x00FF9F
    )
    embed.set_image(url="attachment://heatmap.png")
    embed.set_footer(text="Last 30 days • Real-time tracking", 
                     icon_url=ctx.author.display_avatar.url)
    
    await ctx.send(embed=embed, file=file)


# ========== BOT EVENTS ==========


@bot.event
async def on_ready():
    print(f'🤖 {bot.user.name} is online and ready!')
    print(f'📊 Connected to {len(bot.guilds)} server(s)')
    print(f'🤖 AI Features: {"Enabled" if client_openai else "Disabled (no API key)"}')
    
    if not daily_morning_reminder.is_running():
        daily_morning_reminder.start()
    
    if not daily_digest.is_running():
        daily_digest.start()
    
    if not streak_monitor.is_running():
        streak_monitor.start()


@bot.event
async def on_message(message):
    if message.author.bot:
        return
    
    # Context-aware suggestion with edgy style
    if not message.content.startswith('!') and len(message.content) > 20:
        suggested = suggest_command(message.author.id, message.content)
        if suggested:
            embed = discord.Embed(
                description=f"💡 **Smart Tip:** Looks like a `!{suggested}` entry\nUse the command to track it properly",
                color=0x64748B
            )
            await message.channel.send(embed=embed, delete_after=8)
    
    await bot.process_commands(message)


# ========== ENHANCED COMMANDS ==========


@bot.command(name='ping')
async def ping(ctx):
    latency = round(bot.latency * 1000)
    
    if latency < 150:
        color = 0x00FF9F
        status_emoji = "🟢"
        status = "Good"
        status_quality = "Excellent"
    elif latency < 300:
        color = 0xFFB700
        status_emoji = "🟡"
        status = "Good"
        status_quality = "Stable"
    else:
        color = 0xDC143C
        status_emoji = "🔴"
        status = "Slow"
        status_quality = "Degraded"
    
    embed = discord.Embed(
        title="🏓 Pong!",
        description="Bot is responsive and ready!",
        color=color,
        timestamp=datetime.now()
    )
    
    embed.add_field(
        name="⚡ Latency",
        value=f"`{latency}ms` - {status}",
        inline=True
    )
    embed.add_field(
        name=f"{status_emoji} Status",
        value=f"Online & Active",
        inline=True
    )
    
    # Show AI engine status only if we have the key
    if client_openai:
        embed.add_field(
            name="🤖 AI Engine",
            value=f"Enabled",
            inline=True
        )
    
    embed.set_footer(text="Command Center Bot")
    
    await ctx.send(embed=embed)


@bot.command(name='done')
async def done(ctx, *, task: str):
    user = ctx.author
    
    log_task_to_csv(user.name, user.id, task)
    streak_count = update_streak(user.id, 'daily_checkin')
    learn_pattern(user.id, task, 'done')
    
    streak_emoji = get_streak_emoji(streak_count)
    is_milestone = streak_count % 7 == 0 and streak_count > 0
    
    # Dynamic color based on streak tier
    if streak_count >= 30:
        color = 0x9D4EDD
        tier = "LEGENDARY"
    elif streak_count >= 14:
        color = 0xFFB700
        tier = "ELITE"
    elif streak_count >= 7:
        color = 0x00FF9F
        tier = "RISING"
    else:
        color = 0x64748B
        tier = "BUILDING"
    
    embed = discord.Embed(
        title=f"{streak_emoji} Task Completed!",
        description=f"**{task}**",
        color=color,
        timestamp=datetime.now()
    )
    
    # Streak display - clean format like old bot
    embed.add_field(
        name="🔥 Daily Streak",
        value=f"**{streak_count}** day{'s' if streak_count != 1 else ''}",
        inline=True
    )
    
    # Total tasks - clean format
    if TASKS_FILE.exists():
        with open(TASKS_FILE, 'r') as f:
            total_tasks = len(f.readlines()) - 1
            embed.add_field(
                name="✅ Total Tasks",
                value=f"**{total_tasks}** completed",
                inline=True
            )
    
    # Milestone celebration (optional field)
    if is_milestone:
        embed.add_field(
            name="🎊 MILESTONE ACHIEVED!",
            value=f"**{streak_count} days** of consistency!\nYou're absolutely crushing it! 🚀",
            inline=False
        )
    
    # Set author with user info
    embed.set_author(name=user.display_name, icon_url=user.display_avatar.url)
    
    # Random motivational footer (like old bot)
    embed.set_footer(text=random.choice(MOTIVATION_QUOTES))
    
    await ctx.send(embed=embed)


@bot.command(name='python')
async def log_python(ctx, *, learning_note: str):
    user = ctx.author
    
    log_task_to_csv(user.name, user.id, f"[PYTHON] {learning_note}", PYTHON_FILE)
    streak_count = update_streak(user.id, 'python_learning')
    learn_pattern(user.id, learning_note, 'python')
    
    streak_emoji = get_streak_emoji(streak_count)
    
    embed = discord.Embed(
        title="🐍 PYTHON DEVELOPMENT",
        description=f"**{learning_note}**",
        color=0x3776AB,
        timestamp=datetime.now()
    )
    
    embed.add_field(
        name=f"{streak_emoji} LEARNING STREAK",
        value=f"**{streak_count}** day{'s' if streak_count != 1 else ''} straight",
        inline=True
    )
    
    embed.add_field(
        name="📚 CATEGORY",
        value="Python Development",
        inline=True
    )
    
    embed.set_author(name=f"{user.display_name}'s Code Journey", icon_url=user.display_avatar.url)
    embed.set_footer(text="from progress import mastery 💻")
    embed.set_thumbnail(url=user.display_avatar.url)
    
    await ctx.send(embed=embed)


@bot.command(name='n8n')
async def log_n8n(ctx, *, workflow_note: str):
    user = ctx.author
    
    log_task_to_csv(user.name, user.id, f"[N8N] {workflow_note}", N8N_FILE)
    streak_count = update_streak(user.id, 'n8n_workflows')
    learn_pattern(user.id, workflow_note, 'n8n')
    
    streak_emoji = get_streak_emoji(streak_count)
    
    embed = discord.Embed(
        title="🔗 AUTOMATION WORKFLOW",
        description=f"**{workflow_note}**",
        color=0xEA4B71,
        timestamp=datetime.now()
    )
    
    embed.add_field(
        name=f"{streak_emoji} WORKFLOW STREAK",
        value=f"**{streak_count}** day{'s' if streak_count != 1 else ''} running",
        inline=True
    )
    
    embed.add_field(
        name="🤖 STATUS",
        value="Workflow Logged",
        inline=True
    )
    
    embed.set_author(name=f"{user.display_name}'s Automation Lab", icon_url=user.display_avatar.url)
    embed.set_footer(text="Efficiency through automation 🚀")
    embed.set_thumbnail(url=user.display_avatar.url)
    
    await ctx.send(embed=embed)


@bot.command(name='stats')
async def stats(ctx):
    user = ctx.author
    streaks = get_all_streaks(user.id)
    
    total_streak_days = sum(streaks.values())
    
    # Tier system with matching emojis
    if total_streak_days >= 90:
        color = 0x9D4EDD
        rank = "👑 LEGEND"
        rank_icon = "👑"
    elif total_streak_days >= 50:
        color = 0xFFB700
        rank = "💎 MASTER"
        rank_icon = "💎"
    elif total_streak_days >= 20:
        color = 0x00FF9F
        rank = "⭐ ELITE"
        rank_icon = "⭐"
    else:
        color = 0x64748B
        rank = "⚡ RISING"
        rank_icon = "⚡"
    
    embed = discord.Embed(
        title="📊 Progress Dashboard",
        description=f"**{rank_icon} {user.display_name}**'s Command Center\n*Your journey to consistency*",
        color=color,
        timestamp=datetime.now()
    )
    
    # Streak displays with actual values (no code blocks)
    daily_emoji = get_streak_emoji(streaks['daily_checkin'])
    python_emoji = get_streak_emoji(streaks['python_learning'])
    n8n_emoji = get_streak_emoji(streaks['n8n_workflows'])
    
    embed.add_field(
        name=f"{daily_emoji} Daily Check-In Streak",
        value=f"**{streaks['daily_checkin']}** day{'s' if streaks['daily_checkin'] != 1 else ''}",
        inline=True
    )
    
    embed.add_field(
        name=f"{python_emoji} Python Learning Streak",
        value=f"**{streaks['python_learning']}** day{'s' if streaks['python_learning'] != 1 else ''}",
        inline=True
    )
    
    embed.add_field(
        name=f"{n8n_emoji} n8n Workflow Streak",
        value=f"**{streaks['n8n_workflows']}** day{'s' if streaks['n8n_workflows'] != 1 else ''}",
        inline=True
    )
    
    # Lifetime stats - clean format
    if TASKS_FILE.exists():
        with open(TASKS_FILE, 'r') as f:
            total_tasks = len(f.readlines()) - 1
            
            embed.add_field(
                name="✅ Lifetime Stats",
                value=f"**Total Tasks:** {total_tasks}\n**Total Streak Days:** {total_streak_days}\n**Commands Used:** {total_tasks}",
                inline=False
            )
    
    # Predictive alert (only if at risk)
    at_risk, message = predict_streak_break(user.id)
    if at_risk:
        embed.add_field(
            name="⚠️ Streak Alert",
            value=message,
            inline=False
        )
    
    # Progress visualization (from old bot)
    max_streak = max(streaks.values()) if streaks.values() else 0
    if max_streak > 0:
        progress_bar = "█" * min(max_streak, 20) + "░" * (20 - min(max_streak, 20))
        embed.add_field(
            name="📈 Progress Visualization",
            value=f"`{progress_bar}` {max_streak}/30 days",
            inline=False
        )
    
    embed.set_thumbnail(url=user.display_avatar.url)
    embed.set_author(name="Personal Command Center", icon_url=bot.user.display_avatar.url)
    
    # Dynamic footer with motivational quote
    if total_streak_days >= 30:
        footer_text = "👑 Legendary consistency! You're a productivity master!"
    elif total_streak_days >= 14:
        footer_text = "⭐ Outstanding work! Keep the momentum going!"
    elif total_streak_days >= 7:
        footer_text = "💎 Great progress! You're building strong habits!"
    else:
        footer_text = "✨ Every journey begins with a single step!"
    
    embed.set_footer(text=footer_text, icon_url=bot.user.display_avatar.url)
    
    await ctx.send(embed=embed)
    
    # Generate enhanced chart
    await generate_chart(ctx, user.id)


@bot.command(name='idea')
async def quick_capture(ctx, *, idea: str):
    user = ctx.author
    log_task_to_csv(user.name, user.id, f"[IDEA] {idea}", IDEAS_FILE)
    
    embed = discord.Embed(
        title="💡 IDEA CAPTURED",
        description=f"**{idea}**",
        color=0xFFB700,
        timestamp=datetime.now()
    )
    
    embed.add_field(
        name="✨ STATUS",
        value="Saved to Vault",
        inline=True
    )
    
    embed.add_field(
        name="📝 TYPE",
        value="Quick Capture",
        inline=True
    )
    
    embed.set_author(name=f"{user.display_name}'s Idea Vault", icon_url=user.display_avatar.url)
    embed.set_footer(text="Innovation starts with ideas 🚀")
    
    await ctx.send(embed=embed)


# ========== AI COMMANDS (Enhanced UI) ==========


@bot.command(name='priority')
async def priority(ctx):
    """AI-powered task prioritization with sleek UI"""
    if not client_openai:
        embed = discord.Embed(
            description="⚠️ **AI Engine Offline**\nAdd OPENAI_API_KEY to .env file",
            color=0xDC143C
        )
        await ctx.send(embed=embed)
        return
    
    if not TASKS_FILE.exists():
        embed = discord.Embed(
            description="📝 **No Tasks Found**\nUse `!done` to start tracking",
            color=0xDC143C
        )
        await ctx.send(embed=embed)
        return
    
    with open(TASKS_FILE, 'r') as f:
        reader = list(csv.DictReader(f))
        user_tasks = [r for r in reader if 'User' in r and r['User'] == ctx.author.name]
    
    if not user_tasks:
        embed = discord.Embed(
            description="No tasks logged yet",
            color=0xDC143C
        )
        await ctx.send(embed=embed)
        return
    
    recent_tasks = user_tasks[-20:]
    task_list = "\n".join([f"- {t['Task']} ({t['Date']})" for t in recent_tasks])
    
    prompt = f"""Analyze these completed tasks and suggest priorities:


{task_list}


Provide:
1. Task patterns you notice
2. Time patterns (if visible)
3. 3 specific priority suggestions for next tasks
Keep it under 200 words and actionable."""
    
    # Loading message
    loading = await ctx.send(embed=discord.Embed(
        description="🤖 **Analyzing your tasks...**\nAI is processing your activity patterns",
        color=0x64748B
    ))
    
    try:
        response = client_openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300
        )
        
        analysis = response.choices[0].message.content
        
        await loading.delete()
        
        embed = discord.Embed(
            title="Priority Insights ⚡",
            description=analysis,
            color=0xFFB700,
            timestamp=datetime.now()
        )
        embed.set_footer(text=f"Based on {len(recent_tasks)} recent tasks • AI-powered insights")
        embed.set_thumbnail(url=ctx.author.display_avatar.url)
        
        await ctx.send(embed=embed)
        
    except Exception as e:
        await loading.delete()
        embed = discord.Embed(
            description=f"AI analysis error: {str(e)}",
            color=0xDC143C
        )
        await ctx.send(embed=embed)


@bot.command(name='summary')
async def summary(ctx, period: str = "daily"):
    """AI-generated summary with enhanced visuals"""
    if not client_openai:
        embed = discord.Embed(
            title="❌ AI Engine Offline",
            description="Add OPENAI_API_KEY to .env file",
            color=0xDC143C
        )
        await ctx.send(embed=embed)
        return
    
    if period not in ["daily", "weekly"]:
        await ctx.send("Usage: `!summary daily` or `!summary weekly`")
        return
    
    if period == "daily":
        start_date = datetime.now().date()
        title = "📅 Today's Summary"
        period_display = "today"
    else:
        start_date = datetime.now().date() - timedelta(days=7)
        title = "📅 This Week's Summary"
        period_display = "this week"
    
    activities = []
    
    for file, label in [(TASKS_FILE, "Task"), (PYTHON_FILE, "Python"), (N8N_FILE, "Automation")]:
        if file.exists():
            with open(file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        entry_date = datetime.strptime(row['Date'], '%Y-%m-%d').date()
                        if entry_date >= start_date and row.get('User') == ctx.author.name:
                            activities.append(f"{label}: {row['Task']}")
                    except:
                        pass
    
    if not activities:
        embed = discord.Embed(
            title="❌ No Data Available",
            description=f"No activities found for {period_display} summary",
            color=0xDC143C
        )
        await ctx.send(embed=embed)
        return
    
    activity_text = "\n".join(activities[:30])
    
    prompt = f"""Summarize these accomplishments in an inspirational, direct, and edgy way (under 150 words):

{activity_text}

Keep it real and focused on actual progress. No corporate hype, just straight facts with personality."""
    
    # Loading message
    loading = await ctx.send(embed=discord.Embed(
        title="⚙️ Analyzing...",
        description="AI is generating your summary",
        color=0x64748B
    ))
    
    try:
        response = client_openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=250
        )
        
        summary_text = response.choices[0].message.content
        
        await loading.delete()
        
        embed = discord.Embed(
            title=title,
            description=summary_text,
            color=0x9D4EDD,
            timestamp=datetime.now()
        )
        
        # Activity breakdown with clean formatting
        embed.add_field(
            name="📊 Activity Breakdown",
            value=f"**{len(activities)}** activities logged {period_display}",
            inline=False
        )
        
        embed.set_footer(text="AI-powered insights • Command Center")
        embed.set_thumbnail(url=ctx.author.display_avatar.url)
        
        await ctx.send(embed=embed)
        
    except Exception as e:
        await loading.delete()
        embed = discord.Embed(
            title="❌ Summary Failed",
            description=f"AI analysis error: {str(e)}",
            color=0xDC143C
        )
        await ctx.send(embed=embed)

# ========== BACKGROUND TASKS ==========

@tasks.loop(hours=24)
async def daily_digest():
    """Send daily digest at 8 PM"""
    now = datetime.now()
    
    if now.hour != 20:
        return
    
    for guild in bot.guilds:
        for member in guild.members:
            if member.bot:
                continue
            
            today = datetime.now().date()
            activities = []
            
            for file, label in [(TASKS_FILE, "Task"), (PYTHON_FILE, "Python"), (N8N_FILE, "Automation")]:
                if file.exists():
                    with open(file, 'r') as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            try:
                                entry_date = datetime.strptime(row['Date'], '%Y-%m-%d').date()
                                if entry_date == today and row.get('User') == member.name:
                                    activities.append(f"✅ {label}: {row['Task']}")
                            except:
                                pass
            
            if activities:
                streaks = get_all_streaks(member.id)
                
                embed = discord.Embed(
                    title="📬 DAILY DIGEST",
                    description="Here's what you accomplished today:",
                    color=0x00FF9F,
                    timestamp=datetime.now()
                )
                
                embed.add_field(name="TODAY'S WINS", value="\n".join(activities[:10]), inline=False)
                embed.add_field(name="TOTAL", value=f"**{len(activities)}** activities logged today", inline=False)
                embed.add_field(
                    name="STREAKS",
                    value=f"🔥 Daily: **{streaks['daily_checkin']}** | 🐍 Python: **{streaks['python_learning']}** | 🔗 n8n: **{streaks['n8n_workflows']}**",
                    inline=False
                )
                
                try:
                    await member.send(embed=embed)
                except:
                    pass


@tasks.loop(hours=1)
async def streak_monitor():
    """Monitor and alert for streak risks"""
    for guild in bot.guilds:
        for member in guild.members:
            if member.bot:
                continue
            
            at_risk, message = predict_streak_break(member.id)
            if at_risk:
                embed = discord.Embed(
                    title="⚠️ STREAK ALERT",
                    description=message,
                    color=0xDC143C
                )
                embed.add_field(
                    name="QUICK ACTIONS",
                    value="Use `!done`, `!python`, or `!n8n` to maintain your streak",
                    inline=False
                )
                
                try:
                    await member.send(embed=embed)
                except:
                    pass


@tasks.loop(hours=24)
async def daily_morning_reminder():
    """Morning motivation"""
    for guild in bot.guilds:
        channel = discord.utils.get(guild.channels, name='daily-tasks')
        if channel:
            embed = discord.Embed(
                title="☀️ GOOD MORNING COMMAND CENTER",
                description="A new day to dominate. Let's get it. ☕⚡",
                color=0xFFB700,
                timestamp=datetime.now()
            )
            
            embed.add_field(
                name="🎯 TODAY'S OBJECTIVES",
                value="Log tasks with `!done`\nLearn with `!python`\nAutomate with `!n8n`",
                inline=False
            )
            
            embed.set_footer(text="Command Center • Daily Operations")
            
            await channel.send(embed=embed)


@daily_morning_reminder.before_loop
async def before_daily_morning_reminder():
    await bot.wait_until_ready()
    
    now = datetime.now()
    target_time = now.replace(hour=9, minute=0, second=0, microsecond=0)
    
    if now > target_time:
        target_time += timedelta(days=1)
    
    wait_seconds = (target_time - now).total_seconds()
    await asyncio.sleep(wait_seconds)


@daily_digest.before_loop
async def before_daily_digest():
    await bot.wait_until_ready()
    
    now = datetime.now()
    target_time = now.replace(hour=20, minute=0, second=0, microsecond=0)
    
    if now > target_time:
        target_time += timedelta(days=1)
    
    wait_seconds = (target_time - now).total_seconds()
    await asyncio.sleep(wait_seconds)


if __name__ == '__main__':
    if not TOKEN:
        print("❌ ERROR: DISCORD_BOT_TOKEN not found in .env file!")
    else:
        print("🚀 Starting enhanced AI-powered bot...")
        keep_alive()
        bot.run(TOKEN)
