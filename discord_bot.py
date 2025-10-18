"""
Discord Personal Command Center Bot - Enhanced UI Version
Created for: TikTokTechnician's Productivity Hub
Python Version: 3.13.7
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

load_dotenv()
TOKEN = os.getenv('DISCORD_BOT_TOKEN')

intents = discord.Intents.default()
intents.message_content = True
intents.members = True

bot = commands.Bot(command_prefix='!', intents=intents)

DATA_DIR = Path('bot_data')
DATA_DIR.mkdir(exist_ok=True)

TASKS_FILE = DATA_DIR / 'tasks.csv'
STREAKS_FILE = DATA_DIR / 'streaks.json'

# Motivational quotes for variety
MOTIVATION_QUOTES = [
    "Keep up the great work! 💪",
    "Consistency is key! 🔑",
    "You're building something amazing! ✨",
    "Progress over perfection! 🎯",
    "Every streak starts with day one! 🚀",
    "Small wins lead to big victories! 🏆",
    "You're on fire! 🔥",
    "Momentum is building! ⚡"
]


def load_streaks():
    if STREAKS_FILE.exists():
        with open(STREAKS_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_streaks(streaks_data):
    with open(STREAKS_FILE, 'w') as f:
        json.dump(streaks_data, f, indent=2)


def log_task_to_csv(user_name, task_description):
    file_exists = TASKS_FILE.exists()
    
    with open(TASKS_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        if not file_exists:
            writer.writerow(['Date', 'Time', 'User', 'Task'])
        
        now = datetime.now()
        writer.writerow([
            now.strftime('%Y-%m-%d'),
            now.strftime('%H:%M:%S'),
            user_name,
            task_description
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
    """Returns progressively more exciting emojis based on streak count"""
    if count >= 30:
        return "🏆"
    elif count >= 21:
        return "⭐"
    elif count >= 14:
        return "💎"
    elif count >= 7:
        return "🎉"
    elif count >= 3:
        return "🔥"
    else:
        return "✨"


@bot.event
async def on_ready():
    print(f'🤖 {bot.user.name} is online and ready!')
    print(f'📊 Connected to {len(bot.guilds)} server(s)')
    
    if not daily_morning_reminder.is_running():
        daily_morning_reminder.start()


@bot.command(name='ping')
async def ping(ctx):
    """Health check with beautiful embed"""
    latency = round(bot.latency * 1000)
    
    # Color based on latency (green=good, yellow=ok, red=slow)
    if latency < 150:
        color = 0x00FF9F  # Cyan/green
        status = "Excellent"
    elif latency < 300:
        color = 0xFFB700  # Amber
        status = "Good"
    else:
        color = 0xDC143C  # Red
        status = "Slow"
    
    embed = discord.Embed(
        title="🏓 Pong!",
        description=f"Bot is responsive and ready!",
        color=color,
        timestamp=datetime.now()
    )
    
    embed.add_field(
        name="⚡ Latency",
        value=f"`{latency}ms` - {status}",
        inline=True
    )
    
    embed.add_field(
        name="🟢 Status",
        value="Online & Active",
        inline=True
    )
    
    embed.set_footer(text="Command Center Bot")
    
    await ctx.send(embed=embed)


@bot.command(name='done')
async def done(ctx, *, task: str):
    """Log completed task with enhanced visual feedback"""
    user = ctx.author
    
    log_task_to_csv(user.name, task)
    streak_count = update_streak(user.id, 'daily_checkin')
    
    # Get appropriate emoji for streak level
    streak_emoji = get_streak_emoji(streak_count)
    
    # Milestone celebration
    is_milestone = streak_count % 7 == 0 and streak_count > 0
    
    # Create embed with dynamic color based on streak
    if streak_count >= 30:
        color = 0x9D4EDD  # Purple for legendary streaks
    elif streak_count >= 7:
        color = 0xFFB700  # Gold for week+ streaks
    else:
        color = 0x00FF9F  # Cyan for regular streaks
    
    embed = discord.Embed(
        title=f"{streak_emoji} Task Completed!",
        description=f"**{task}**",
        color=color,
        timestamp=datetime.now()
    )
    
    # Main streak display
    embed.add_field(
        name="🔥 Daily Streak",
        value=f"**{streak_count}** day{'s' if streak_count != 1 else ''}",
        inline=True
    )
    
    # Total tasks count
    if TASKS_FILE.exists():
        with open(TASKS_FILE, 'r') as f:
            total_tasks = len(f.readlines()) - 1
            embed.add_field(
                name="✅ Total Tasks",
                value=f"**{total_tasks}** completed",
                inline=True
            )
    
    # Add milestone celebration
    if is_milestone:
        embed.add_field(
            name="🎊 MILESTONE ACHIEVED!",
            value=f"**{streak_count} days** of consistency!\nYou're absolutely crushing it! 🚀",
            inline=False
        )
    
    # Set author with user's profile pic
    embed.set_author(
        name=user.display_name,
        icon_url=user.display_avatar.url
    )
    
    # Random motivational footer
    embed.set_footer(
        text=random.choice(MOTIVATION_QUOTES),
        icon_url=bot.user.display_avatar.url
    )
    
    await ctx.send(embed=embed)


@bot.command(name='python')
async def log_python(ctx, *, learning_note: str):
    """Log Python learning with coding-themed design"""
    user = ctx.author
    
    log_task_to_csv(user.name, f"[PYTHON] {learning_note}")
    streak_count = update_streak(user.id, 'python_learning')
    
    streak_emoji = get_streak_emoji(streak_count)
    
    embed = discord.Embed(
        title="🐍 Python Learning Progress",
        description=f"``````",
        color=0x3776AB,  # Python official blue
        timestamp=datetime.now()
    )
    
    embed.add_field(
        name=f"{streak_emoji} Learning Streak",
        value=f"**{streak_count}** day{'s' if streak_count != 1 else ''}",
        inline=True
    )
    
    embed.add_field(
        name="📚 Category",
        value="Python Development",
        inline=True
    )
    
    embed.set_author(
        name=f"{user.display_name}'s Code Journey",
        icon_url=user.display_avatar.url
    )
    
    embed.set_footer(
        text="from learning import growth 💻",
        icon_url=bot.user.display_avatar.url
    )
    
    await ctx.send(embed=embed)


@bot.command(name='n8n')
async def log_n8n(ctx, *, workflow_note: str):
    """Log n8n workflow with automation-themed design"""
    user = ctx.author
    
    log_task_to_csv(user.name, f"[N8N] {workflow_note}")
    streak_count = update_streak(user.id, 'n8n_workflows')
    
    streak_emoji = get_streak_emoji(streak_count)
    
    embed = discord.Embed(
        title="🔗 Automation Workflow Logged",
        description=f"⚙️ **{workflow_note}**",
        color=0xEA4B71,  # n8n brand color (pinkish-red)
        timestamp=datetime.now()
    )
    
    embed.add_field(
        name=f"{streak_emoji} Workflow Streak",
        value=f"**{streak_count}** day{'s' if streak_count != 1 else ''}",
        inline=True
    )
    
    embed.add_field(
        name="🤖 Type",
        value="Automation Win",
        inline=True
    )
    
    embed.set_author(
        name=f"{user.display_name}'s Automation Lab",
        icon_url=user.display_avatar.url
    )
    
    embed.set_footer(
        text="Automate everything! 🚀",
        icon_url=bot.user.display_avatar.url
    )
    
    await ctx.send(embed=embed)


@bot.command(name='stats')
async def stats(ctx):
    """Enhanced stats dashboard with beautiful layout"""
    user = ctx.author
    streaks = get_all_streaks(user.id)
    
    # Calculate total streak days
    total_streak_days = sum(streaks.values())
    
    # Dynamic color based on total progress
    if total_streak_days >= 50:
        color = 0x9D4EDD  # Purple - legendary
    elif total_streak_days >= 20:
        color = 0xFFB700  # Gold - impressive
    else:
        color = 0x00FF9F  # Cyan - building
    
    embed = discord.Embed(
        title="📊 Progress Dashboard",
        description=f"**{user.display_name}'s Command Center**\n*Your journey to consistency*",
        color=color,
        timestamp=datetime.now()
    )
    
    # Main streak fields with better formatting
    daily_emoji = get_streak_emoji(streaks['daily_checkin'])
    python_emoji = get_streak_emoji(streaks['python_learning'])
    n8n_emoji = get_streak_emoji(streaks['n8n_workflows'])
    
    embed.add_field(
        name=f"{daily_emoji} Daily Check-In Streak",
        value=f"``````",
        inline=True
    )
    
    embed.add_field(
        name=f"{python_emoji} Python Learning Streak",
        value=f"``````",
        inline=True
    )
    
    embed.add_field(
        name=f"{n8n_emoji} n8n Workflow Streak",
        value=f"``````",
        inline=True
    )
    
    # Total tasks completed
    if TASKS_FILE.exists():
        with open(TASKS_FILE, 'r') as f:
            total_tasks = len(f.readlines()) - 1
            
            embed.add_field(
                name="✅ Lifetime Stats",
                value=f"**Total Tasks:** {total_tasks}\n**Total Streak Days:** {total_streak_days}\n**Commands Used:** {total_tasks}",
                inline=False
            )
    
    # Progress bar visualization
    max_streak = max(streaks.values()) if streaks.values() else 0
    if max_streak > 0:
        progress_bar = "█" * min(max_streak, 20) + "░" * (20 - min(max_streak, 20))
        embed.add_field(
            name="📈 Progress Visualization",
            value=f"`{progress_bar}` {max_streak}/30 days",
            inline=False
        )
    
    # Set user thumbnail
    embed.set_thumbnail(url=user.display_avatar.url)
    
    # Set author
    embed.set_author(
        name="Personal Command Center",
        icon_url=bot.user.display_avatar.url
    )
    
    # Dynamic footer based on progress
    if total_streak_days >= 30:
        footer_text = "🏆 Legendary consistency! You're a productivity master!"
    elif total_streak_days >= 14:
        footer_text = "⭐ Outstanding work! Keep the momentum going!"
    elif total_streak_days >= 7:
        footer_text = "💎 Great progress! You're building strong habits!"
    else:
        footer_text = "✨ Every journey begins with a single step!"
    
    embed.set_footer(
        text=footer_text,
        icon_url=bot.user.display_avatar.url
    )
    
    await ctx.send(embed=embed)


@bot.command(name='idea')
async def quick_capture(ctx, *, idea: str):
    """Quick idea capture with creative design"""
    user = ctx.author
    log_task_to_csv(user.name, f"[IDEA] {idea}")
    
    embed = discord.Embed(
        title="💡 Idea Captured!",
        description=f"_{idea}_",
        color=0xFFB700,  # Gold for ideas
        timestamp=datetime.now()
    )
    
    embed.add_field(
        name="✨ Status",
        value="Safely stored for later!",
        inline=True
    )
    
    embed.add_field(
        name="📝 Type",
        value="Quick Capture",
        inline=True
    )
    
    embed.set_author(
        name=f"{user.display_name}'s Idea Box",
        icon_url=user.display_avatar.url
    )
    
    embed.set_footer(
        text="Great ideas start here! 🚀",
        icon_url=bot.user.display_avatar.url
    )
    
    await ctx.send(embed=embed)


@tasks.loop(hours=24)
async def daily_morning_reminder():
    """Beautiful morning motivation message"""
    for guild in bot.guilds:
        channel = discord.utils.get(guild.channels, name='daily-tasks')
        if channel:
            embed = discord.Embed(
                title="☀️ Good Morning, Champion!",
                description="A new day, a new opportunity to build your streaks! ☕✨",
                color=0xFFB700,
                timestamp=datetime.now()
            )
            
            embed.add_field(
                name="🎯 Today's Mission",
                value="• Log one Python win with `!python`\n• Complete a task with `!done`\n• Capture any ideas with `!idea`",
                inline=False
            )
            
            embed.add_field(
                name="💪 Quick Tip",
                value="Start with the smallest win. Momentum builds from there!",
                inline=False
            )
            
            embed.set_thumbnail(url=guild.icon.url if guild.icon else None)
            
            embed.set_footer(
                text="You've got this! 🚀",
                icon_url=bot.user.display_avatar.url
            )
            
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


if __name__ == '__main__':
    if not TOKEN:
        print("❌ ERROR: DISCORD_BOT_TOKEN not found in .env file!")
    else:
        print("🚀 Starting enhanced bot...")
	keep_alive()
        bot.run(TOKEN)
