import threading
import subprocess
import sys

def run_telegram():
    subprocess.run([sys.executable, "telegram_bot.py"])

def run_whatsapp():
    subprocess.run([sys.executable, "whatsapp_bot.py"])

if __name__ == "__main__":
    print("🚀 Starting ALL Dzukku Bots...")

    t1 = threading.Thread(target=run_telegram)
    t2 = threading.Thread(target=run_whatsapp)

    t1.start()
    t2.start()

    t1.join()
    t2.join()