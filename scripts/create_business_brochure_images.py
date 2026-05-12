from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "assets" / "business-brochure"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


F = {
    "xs": font(22),
    "s": font(28),
    "m": font(34),
    "l": font(46, True),
    "xl": font(64, True),
    "bold": font(30, True),
}


BG = "#090a12"
PANEL = "#1b1b2e"
PANEL_2 = "#101421"
TEXT = "#f1eefb"
MUTED = "#aaa5b8"
ORANGE = "#ff7141"
GOLD = "#f3a313"
GREEN = "#18a375"
BLUE = "#3b82f6"
PURPLE = "#8b5cf6"
RED = "#ef4444"


def gradient(size: tuple[int, int]) -> Image.Image:
    w, h = size
    img = Image.new("RGB", size, BG)
    pix = img.load()
    for y in range(h):
        for x in range(w):
            rx = int(28 + 42 * max(0, 1 - ((x - 120) ** 2 + (y - 90) ** 2) ** 0.5 / 760))
            gx = int(11 + 16 * max(0, 1 - ((x - 120) ** 2 + (y - 90) ** 2) ** 0.5 / 760))
            bx = int(18 + 34 * max(0, 1 - ((x - w + 240) ** 2 + (y - h + 190) ** 2) ** 0.5 / 760))
            pix[x, y] = (rx, gx, bx)
    return img


def draw_pill(draw: ImageDraw.ImageDraw, box, text: str, fill: str, outline: str | None = None):
    draw.rounded_rectangle(box, radius=28, fill=fill, outline=outline or fill, width=2)
    draw.text((box[0] + 26, box[1] + 13), text, fill=TEXT, font=F["bold"])


def draw_card(draw: ImageDraw.ImageDraw, box, title: str, subtitle: str = "", color: str = ORANGE):
    draw.rounded_rectangle(box, radius=28, fill=PANEL, outline="#38364f", width=2)
    draw.rounded_rectangle((box[0] + 38, box[1] + 38, box[0] + 110, box[1] + 110), radius=20, fill="#30283f")
    draw.text((box[0] + 60, box[1] + 52), "•", fill=color, font=F["xl"])
    draw.text((box[0] + 38, box[1] + 154), title, fill=TEXT, font=F["l"])
    if subtitle:
        draw.text((box[0] + 38, box[1] + 222), subtitle, fill=MUTED, font=F["m"])


def role_selector():
    img = gradient((1600, 900))
    d = ImageDraw.Draw(img)
    draw_pill(d, (160, 138, 472, 198), "Live restaurant control", "#54271f", "#9a4a34")
    d.text((160, 236), "Pick the control room for this shift", fill=TEXT, font=F["xl"])
    d.text((160, 318), "Admin, waiter, and kitchen teams work from role-specific spaces.", fill=MUTED, font=F["m"])
    draw_card(d, (160, 395, 490, 670), "Admin / POS", "Orders, menu, staff, payments")
    draw_card(d, (520, 395, 850, 670), "Waiter", "Table map, sessions, bills", GOLD)
    draw_card(d, (880, 395, 1210, 670), "Kitchen", "Live KDS and item status", GREEN)
    img.save(OUT / "01-role-selector.png")


def login():
    img = gradient((1600, 900))
    d = ImageDraw.Draw(img)
    box = (520, 120, 1080, 780)
    d.rounded_rectangle(box, radius=34, fill=PANEL, outline="#38364f", width=2)
    d.text((735, 230), "Dzukku POS", fill=TEXT, font=F["xl"])
    d.text((640, 304), "Restaurant operations, floor service,", fill=MUTED, font=F["m"])
    d.text((690, 348), "and kitchen execution.", fill=MUTED, font=F["m"])
    labels = [("Admin control", 590), ("Waiter workflow", 740), ("Kitchen realtime", 890)]
    for label, x in labels:
        d.rounded_rectangle((x, 420, x + 140, 490), radius=18, fill="#25243a", outline="#3d3b55")
        d.text((x + 18, 438), label, fill=MUTED, font=F["xs"])
    d.text((590, 526), "Email", fill=MUTED, font=F["bold"])
    d.rounded_rectangle((590, 565, 1010, 625), radius=16, fill="#2a2945", outline="#4d4b69")
    d.text((615, 580), "staff@dzukku.com", fill="#807b91", font=F["s"])
    d.text((590, 660), "Password", fill=MUTED, font=F["bold"])
    d.rounded_rectangle((590, 700, 1010, 760), radius=16, fill="#2a2945", outline="#4d4b69")
    d.rounded_rectangle((590, 790, 1010, 850), radius=18, fill=ORANGE)
    d.text((762, 806), "Sign In", fill="#fff8f2", font=F["bold"])
    img.save(OUT / "02-login.png")


def dashboard():
    img = gradient((1600, 900))
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, 230, 900), fill="#14111c")
    d.text((42, 32), "Dzukku", fill=TEXT, font=F["l"])
    nav = ["Dashboard", "Orders", "Deliveries", "KDS", "Tables", "Reservations", "Menu", "Employees", "Invoices", "Analytics"]
    y = 112
    for i, item in enumerate(nav):
        fill = ORANGE if i == 0 else MUTED
        if i == 0:
            d.rounded_rectangle((24, y - 12, 204, y + 36), radius=12, fill=ORANGE)
            fill = "#ffffff"
        d.text((50, y), item, fill=fill, font=F["s"])
        y += 52
    d.text((270, 34), "Dashboard", fill=TEXT, font=F["l"])
    d.text((270, 82), "Live orders, revenue, kitchen queue, and service status", fill=MUTED, font=F["s"])
    draw_pill(d, (820, 32, 1040, 86), "Live sync on", "#0e3a32", "#1c765f")
    draw_pill(d, (1060, 32, 1260, 86), "40 queued", "#4b241f", "#8e3a31")
    draw_pill(d, (1280, 32, 1480, 86), "8 ready", "#12392d", "#1f765f")
    d.rounded_rectangle((270, 130, 1530, 300), radius=28, fill="#151927", outline="#313a55", width=2)
    d.text((305, 168), "OPERATIONS PULSE", fill="#f3c8a2", font=F["bold"])
    d.text((305, 210), "One live queue for admin, waiter, and kitchen.", fill=TEXT, font=F["l"])
    d.text((305, 258), "Orders flow from acceptance to prep to service with real-time updates.", fill=MUTED, font=F["m"])
    stats = [("77", "Total orders", ORANGE), ("33K", "Revenue", GREEN), ("40", "Pending", GOLD), ("10", "Delivered", GREEN)]
    x = 290
    for value, label, color in stats:
        d.rounded_rectangle((x, 340, x + 210, 455), radius=16, fill=PANEL, outline="#3e3c55")
        d.text((x + 22, 374), value, fill=color, font=F["xl"])
        d.text((x + 22, 426), label, fill=MUTED, font=F["s"])
        x += 235
    d.rounded_rectangle((290, 500, 870, 790), radius=16, fill=PANEL, outline="#3e3c55")
    d.text((320, 535), "Orders by Hour", fill=TEXT, font=F["bold"])
    points = [(330, 740), (390, 550), (470, 720), (560, 700), (650, 730), (760, 650), (840, 620)]
    d.line(points, fill=ORANGE, width=5, joint="curve")
    d.rounded_rectangle((910, 500, 1490, 790), radius=16, fill=PANEL, outline="#3e3c55")
    d.text((940, 535), "Orders by Status", fill=TEXT, font=F["bold"])
    bars = [(960, 720, 1040, 585, GOLD), (1090, 720, 1170, 665, BLUE), (1220, 720, 1300, 630, PURPLE), (1350, 720, 1430, 610, GREEN)]
    for x1, y1, x2, y2, c in bars:
        d.rounded_rectangle((x1, y2, x2, y1), radius=8, fill=c)
    img.save(OUT / "03-dashboard.png")


def waiter():
    img = gradient((1600, 900))
    d = ImageDraw.Draw(img)
    d.text((45, 52), "FLOOR SERVICE", fill="#f3c8a2", font=F["bold"])
    d.text((45, 95), "Waiter Portal", fill=TEXT, font=F["xl"])
    d.text((45, 168), "Run the dining floor from table opening to kitchen fire to checkout.", fill=MUTED, font=F["m"])
    draw_pill(d, (1170, 45, 1430, 100), "Kitchen sync live", "#0e3a32", "#1c765f")
    d.rounded_rectangle((45, 230, 1530, 390), radius=28, fill="#151927", outline="#313a55", width=2)
    d.text((85, 278), "SERVICE FLOW", fill="#f3c8a2", font=F["bold"])
    d.text((85, 320), "Pick a table to start a guest session", fill=TEXT, font=F["l"])
    d.text((85, 365), "Available tables open instantly; occupied ones keep live order readiness.", fill=MUTED, font=F["s"])
    d.rounded_rectangle((45, 430, 1530, 820), radius=24, fill="#15161f", outline="#33333d", width=2)
    d.text((75, 465), "Table Map", fill=TEXT, font=F["bold"])
    x0, y0 = 75, 520
    for i in range(20):
        row, col = divmod(i, 9)
        x, y = x0 + col * 160, y0 + row * 105
        occ = i == 0
        outline = GOLD if occ else GREEN
        status = "OCCUPIED" if occ else "AVAILABLE"
        d.rounded_rectangle((x, y, x + 130, y + 78), radius=16, fill="#20202b", outline=outline, width=3)
        d.text((x + 38, y + 18), f"T{i+1:02d}", fill=TEXT, font=F["bold"])
        d.text((x + 30, y + 48), status, fill=outline, font=F["xs"])
    img.save(OUT / "04-waiter-portal.png")


def kitchen():
    img = gradient((1600, 900))
    d = ImageDraw.Draw(img)
    d.text((45, 52), "BACK OF HOUSE", fill="#f3c8a2", font=F["bold"])
    d.text((45, 95), "Kitchen Display", fill=TEXT, font=F["xl"])
    d.text((45, 168), "Track active tickets by item and release orders only when every line item is done.", fill=MUTED, font=F["m"])
    draw_pill(d, (1180, 45, 1415, 100), "Realtime linked", "#0e3a32", "#1c765f")
    d.rounded_rectangle((45, 230, 1530, 390), radius=28, fill="#151927", outline="#313a55", width=2)
    d.text((85, 278), "KITCHEN PULSE", fill="#f3c8a2", font=F["bold"])
    d.text((85, 320), "44 live line items on the board", fill=TEXT, font=F["l"])
    d.text((85, 365), "Each action updates waiter and admin views, so service stays aligned.", fill=MUTED, font=F["s"])
    columns = [("Pending", "36", GOLD), ("Cooking", "1", BLUE), ("Done", "7", GREEN)]
    x = 45
    for title, count, color in columns:
        d.rounded_rectangle((x, 500, x + 455, 850), radius=18, fill="#15161f", outline="#33333d", width=2)
        d.text((x + 26, 530), f"{title}  {count}", fill=TEXT, font=F["bold"])
        d.line((x, 570, x + 455, 570), fill=color, width=3)
        for n in range(3):
            y = 595 + n * 78
            d.rounded_rectangle((x + 22, y, x + 433, y + 60), radius=14, fill="#0e0f17", outline="#272733")
            d.text((x + 42, y + 16), f"#DZK-ORD{24+n:03d}", fill=ORANGE, font=F["xs"])
            item = ["Veg Spring Rolls", "Veg Biryani", "Butter Chicken"][n]
            d.text((x + 190, y + 16), f"1x {item}", fill=TEXT, font=F["s"])
        x += 500
    img.save(OUT / "05-kitchen-display.png")


def chat(name: str, channel: str):
    img = Image.new("RGB", (760, 1600), "#101015")
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((35, 50, 725, 1520), radius=44, fill="#15151d", outline="#32323b", width=3)
    d.text((265, 90), "Foodbot", fill=TEXT, font=F["l"])
    d.text((335, 140), "bot", fill=MUTED, font=F["s"])
    y = 220
    messages = [
        ("Hello Krishna! Welcome to Dzukku Restaurant", 560),
        ("Where would you like to order from?\n\n• Dzukku Bot — chat & order right here\n• Zomato / Swiggy — order via delivery app", 660),
    ]
    for text, w in messages:
        d.rounded_rectangle((55, y, w, y + 150), radius=22, fill="#302838")
        d.text((80, y + 25), text, fill=TEXT, font=F["s"], spacing=8)
        y += 175
    buttons = ["Order via Dzukku Bot", "Zomato", "Swiggy"]
    for label in buttons:
        d.rounded_rectangle((60, y, 700, y + 75), radius=16, fill="#3b3742")
        d.text((120, y + 20), label, fill=TEXT, font=F["s"])
        y += 88
    if channel:
        d.rounded_rectangle((55, y + 25, 680, y + 185), radius=22, fill="#302838")
        d.text((80, y + 50), f"Connecting you to {channel}", fill=TEXT, font=F["s"])
        d.text((80, y + 98), f"Hello! I can help you find your next meal.", fill=TEXT, font=F["s"])
        y += 215
    quick = [("Menu", "Specials"), ("Order", "Reserve a Table"), ("My Cart", "Info")]
    y = 1240
    for left, right in quick:
        d.rounded_rectangle((65, y, 365, y + 72), radius=15, fill="#44424a")
        d.rounded_rectangle((395, y, 695, y + 72), radius=15, fill="#44424a")
        d.text((125, y + 18), left, fill=TEXT, font=F["s"])
        d.text((455, y + 18), right, fill=TEXT, font=F["s"])
        y += 90
    img.save(OUT / name)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    role_selector()
    login()
    dashboard()
    waiter()
    kitchen()
    chat("06-telegram-dzukku.png", "")
    chat("07-swiggy-assistant.png", "Swiggy")
    chat("08-zomato-assistant.png", "Zomato")
    print(f"Generated brochure images in {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
