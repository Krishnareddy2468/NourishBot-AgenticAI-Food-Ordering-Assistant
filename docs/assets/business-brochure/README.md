# Dzukku Business Brochure Images

Save the supplied screenshots in this folder using these exact filenames:

1. `01-role-selector.png` - role/workspace selection screen
2. `02-login.png` - Dzukku POS login screen
3. `03-dashboard.png` - admin/owner dashboard screen
4. `04-waiter-portal.png` - waiter portal table map screen
5. `05-kitchen-display.png` - kitchen display/KDS screen
6. `06-telegram-dzukku.png` - Telegram Dzukku Bot direct ordering screen
7. `07-swiggy-assistant.png` - Telegram Swiggy assistant screen
8. `08-zomato-assistant.png` - Telegram Zomato assistant screen

The business brochure already references these paths:

`docs/assets/business-brochure/*.png`

After adding the images, regenerate the PDF and Word-compatible file:

```bash
python3 scripts/export_brochure_assets.py docs/DZUKKU_BUSINESS_BROCHURE.html
```
