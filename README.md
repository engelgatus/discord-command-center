# Discord Command Center Bot 🤖

A powerful Discord productivity bot designed for personal task tracking, streak management, and habit building. Built with Python, SQLite, and AI integration for smart productivity insights.

## ✨ Features

### 🎯 Core Productivity Tracking
- **Daily Task Logging**: Track completed tasks with `/done`
- **Python Learning**: Log coding progress with `/python`
- **Automation Workflows**: Track n8n automations with `/n8n`
- **Quick Idea Capture**: Save ideas instantly with `/idea`

### 📊 Advanced Analytics
- **Streak Tracking**: Daily check-ins, Python learning, and automation streaks
- **Productivity Heatmaps**: Visual charts showing your most productive hours/days
- **Velocity Analysis**: Track tasks-per-day trends over time
- **Category Breakdown**: See how you spend your time across different areas

### 🤖 AI-Powered Features
- **Smart Prioritization**: AI suggests next tasks based on your patterns
- **Daily/Weekly Summaries**: AI-generated progress reports
- **Pattern Learning**: Bot learns your habits and suggests relevant commands
- **Predictive Alerts**: Get notified when you might break a streak

### 🏆 Goal System
- Set weekly goals for Python, automation, or daily tasks
- Track progress with visual indicators
- Get motivational milestone celebrations

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Discord Bot Token
- OpenAI API Key (optional, for AI features)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/engelgatus/discord-command-center.git
   cd discord-command-center
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   - Copy `.env.example` to `.env`
   - Fill in your Discord bot token and OpenAI API key

4. **Run the bot**
   ```bash
   python discord_bot.py
   ```

### Discord Server Setup

Create these channels for optimal organization:

**⚙️ COMMAND CENTER**
- `📋-daily-tasks` - Main productivity hub
- `🎯-habit-tracker` - Recurring habits
- `📊-progress-dashboard` - Stats and milestones
- `💭-quick-capture` - Rapid idea logging

**📚 LEARNING LAB**
- `🐍-python-log` - Python learning journey
- `🔗-n8n-workflows` - Automation tracking

## 📱 Commands

### Slash Commands (Recommended)
| Command | Description | Usage |
|---------|-------------|-------|
| `/ping` | Check bot status | Health check and latency |
| `/done <task>` | Log completed task | Updates daily streak |
| `/python <note>` | Log Python learning | Updates Python streak |
| `/n8n <workflow>` | Log automation | Updates automation streak |
| `/idea <idea>` | Capture idea quickly | Saves for later review |
| `/stats` | View dashboard | Complete progress overview |
| `/priority` | AI task suggestions | Smart next-step recommendations |
| `/summary daily/weekly` | AI progress report | Motivational summaries |
| `/productivity` | Heatmap visualization | Hour/day activity patterns |
| `/goal <category> <target>` | Set weekly goals | Python, automation, daily targets |
| `/velocity` | Track productivity trends | Acceleration analysis |
| `/categories` | Time breakdown | See where you spend effort |
| `/streakmax` | Predict longest streak | Future milestone estimates |

## 🎨 Streak System

The bot uses a progressive streak system with visual rewards:

- ✨ **Starting Out** (1-2 days)
- ⚡ **Building Momentum** (3-6 days)  
- 🔥 **On Fire** (7-13 days)
- ⭐ **Elite Level** (14-20 days)
- 💎 **Master Tier** (21-29 days)
- 👑 **Legendary** (30+ days)

### Streak Types
- **Daily Check-in**: Any task logged with `/done`
- **Python Learning**: Progress logged with `/python`
- **Automation**: Workflows logged with `/n8n`

## 📈 Data & Privacy

- **Local Storage**: All data stored in SQLite database
- **Privacy First**: User IDs stored securely, no personal data exposed
- **Export Ready**: Data can be exported to CSV for analysis
- **Backup Friendly**: Simple file-based storage system

## 🔧 Configuration

### Environment Variables
```bash
DISCORD_BOT_TOKEN=your_discord_bot_token
OPENAI_API_KEY=your_openai_key  # Optional for AI features
```

### Bot Permissions Required
- Send Messages
- Use Slash Commands  
- Embed Links
- Attach Files
- Read Message History

## 🚀 Deployment

### Render.com (Recommended)
1. Fork this repository
2. Connect to Render.com
3. Set environment variables in Render dashboard
4. Deploy as Web Service (uses `web_server.py` for keep-alive)

### Heroku
1. Create Heroku app
2. Set config vars for tokens
3. Deploy via Git or GitHub integration

### Self-Hosted
1. Run on any server with Python 3.8+
2. Use screen/tmux for persistence
3. Set up systemd service for auto-restart

## 🤝 Contributing

This is a personal productivity bot, but feel free to:
- Fork and adapt for your needs
- Submit bug reports
- Suggest new features
- Share your customizations

## 📄 License

MIT License - see LICENSE file for details

## 🎯 Why This Bot?

Built for content creators, developers, and productivity enthusiasts who want:
- **Consistent Habit Tracking** without complex apps
- **Visual Progress Motivation** with streaks and milestones  
- **AI-Powered Insights** to optimize productivity
- **Community Accountability** in Discord servers
- **Simple but Powerful** logging system

Perfect for tracking coding journeys, content creation workflows, and daily productivity habits.

---

**Made with ⚡ by [engelgatus](https://github.com/engelgatus)**

*Part of the TikTokTechnician productivity ecosystem*