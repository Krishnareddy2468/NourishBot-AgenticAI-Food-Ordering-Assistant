import qrcode
from PIL import Image, ImageDraw, ImageFont
import os

def generate_qr(url, filename, color):
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=10,
        border=4,
    )
    qr.add_data(url)
    qr.make(fit=True)

    img = qr.make_image(fill_color=color, back_color="white").convert("RGB")

    # Add padding for text
    width, height = img.size
    new_height = height + 80
    final_img = Image.new("RGB", (width, new_height), "white")
    final_img.paste(img, (0, 0))

    # Add restaurant name text below QR
    draw = ImageDraw.Draw(final_img)
    text = "Dzukku Restaurant"
    draw.text((width // 2, height + 15), text, fill=color, anchor="mt")
    draw.text((width // 2, height + 40), "Scan to Chat with Us!", fill="gray", anchor="mt")

    final_img.save(filename)
    print(f"✅ QR Code saved: {filename}")

if __name__ == "__main__":
    # ── Telegram QR ─────────────────────────────
    # Replace with your actual bot username
    telegram_url = "https://t.me/Dzukkuuu_111bot"
    generate_qr(telegram_url, "dzukku_telegram_qr.png", "#2C3E50")
    print(f"   Link: {telegram_url}")

    # ── WhatsApp QR ──────────────────────────────
    # Replace with Dzukku's actual WhatsApp number (with country code, no +)
    whatsapp_number = "9014919983"
    whatsapp_url = f"https://wa.me/{whatsapp_number}?text=Hi%20Dzukku!"
    generate_qr(whatsapp_url, "dzukku_whatsapp_qr.png", "#25D366")
    print(f"   Link: {whatsapp_url}")

    print("\n🎉 Both QR codes generated!")
    print("📁 Check your Dzukku folder for the PNG files")