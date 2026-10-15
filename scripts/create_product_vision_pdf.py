#!/usr/bin/env python3
"""Generate the NourishBot present-state and future-vision product PDF."""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "NOURISHBOT_PRODUCT_VISION.pdf"

INK = colors.HexColor("#19231D")
MUTED = colors.HexColor("#667169")
ORANGE = colors.HexColor("#E75D32")
ORANGE_LIGHT = colors.HexColor("#FFF0E8")
GREEN = colors.HexColor("#25765B")
GREEN_LIGHT = colors.HexColor("#EAF5EF")
CREAM = colors.HexColor("#FBF7F0")
LINE = colors.HexColor("#E3D9CD")
WHITE = colors.white


def register_fonts() -> tuple[str, str]:
    regular_candidates = [
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/Library/Fonts/Arial.ttf"),
    ]
    bold_candidates = [
        Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
        Path("/Library/Fonts/Arial Bold.ttf"),
    ]
    regular = next((path for path in regular_candidates if path.exists()), None)
    bold = next((path for path in bold_candidates if path.exists()), None)
    if regular and bold:
        pdfmetrics.registerFont(TTFont("NBRegular", str(regular)))
        pdfmetrics.registerFont(TTFont("NBBold", str(bold)))
        return "NBRegular", "NBBold"
    return "Helvetica", "Helvetica-Bold"


FONT, FONT_BOLD = register_fonts()
BASE = getSampleStyleSheet()

STYLES = {
    "title": ParagraphStyle(
        "Title",
        parent=BASE["Title"],
        fontName=FONT_BOLD,
        fontSize=34,
        leading=39,
        textColor=INK,
        alignment=TA_LEFT,
        spaceAfter=10,
    ),
    "subtitle": ParagraphStyle(
        "Subtitle",
        parent=BASE["Normal"],
        fontName=FONT,
        fontSize=15,
        leading=22,
        textColor=MUTED,
        spaceAfter=18,
    ),
    "h1": ParagraphStyle(
        "H1",
        parent=BASE["Heading1"],
        fontName=FONT_BOLD,
        fontSize=22,
        leading=27,
        textColor=INK,
        spaceBefore=5,
        spaceAfter=12,
    ),
    "h2": ParagraphStyle(
        "H2",
        parent=BASE["Heading2"],
        fontName=FONT_BOLD,
        fontSize=14,
        leading=18,
        textColor=GREEN,
        spaceBefore=12,
        spaceAfter=7,
    ),
    "h3": ParagraphStyle(
        "H3",
        parent=BASE["Heading3"],
        fontName=FONT_BOLD,
        fontSize=11,
        leading=15,
        textColor=INK,
        spaceBefore=8,
        spaceAfter=4,
    ),
    "body": ParagraphStyle(
        "Body",
        parent=BASE["BodyText"],
        fontName=FONT,
        fontSize=9.5,
        leading=14.2,
        textColor=INK,
        spaceAfter=7,
    ),
    "small": ParagraphStyle(
        "Small",
        parent=BASE["BodyText"],
        fontName=FONT,
        fontSize=7.7,
        leading=10.5,
        textColor=MUTED,
    ),
    "label": ParagraphStyle(
        "Label",
        parent=BASE["Normal"],
        fontName=FONT_BOLD,
        fontSize=7.5,
        leading=9,
        textColor=ORANGE,
        spaceAfter=6,
    ),
    "quote": ParagraphStyle(
        "Quote",
        parent=BASE["BodyText"],
        fontName=FONT_BOLD,
        fontSize=13,
        leading=19,
        textColor=INK,
        leftIndent=10,
        rightIndent=10,
        alignment=TA_CENTER,
    ),
    "table_head": ParagraphStyle(
        "TableHead",
        parent=BASE["Normal"],
        fontName=FONT_BOLD,
        fontSize=8,
        leading=10,
        textColor=WHITE,
    ),
    "table": ParagraphStyle(
        "Table",
        parent=BASE["Normal"],
        fontName=FONT,
        fontSize=7.8,
        leading=10.6,
        textColor=INK,
    ),
}


def p(text: str, style: str = "body") -> Paragraph:
    return Paragraph(text, STYLES[style])


def bullet_list(items: list[str], level: int = 0) -> ListFlowable:
    return ListFlowable(
        [ListItem(p(item), leftIndent=4 * mm) for item in items],
        bulletType="bullet",
        bulletFontName=FONT_BOLD,
        bulletFontSize=6,
        bulletColor=ORANGE,
        leftIndent=(6 + level * 4) * mm,
        bulletOffsetY=1,
        spaceAfter=5,
    )


def section_label(text: str) -> list:
    return [p(text.upper(), "label"), HRFlowable(width="100%", thickness=0.7, color=LINE, spaceAfter=10)]


def info_box(title: str, body: str, color: colors.Color = GREEN_LIGHT) -> Table:
    box = Table(
        [[p(title, "h3"), p(body, "body")]],
        colWidths=[42 * mm, 118 * mm],
        hAlign="LEFT",
    )
    box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), color),
        ("BOX", (0, 0), (-1, -1), 0.6, LINE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    return box


def data_table(headers: list[str], rows: list[list[str]], widths: list[float]) -> Table:
    data = [[p(header, "table_head") for header in headers]]
    data.extend([[p(str(cell), "table") for cell in row] for row in rows])
    table = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), INK),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, CREAM]),
        ("GRID", (0, 0), (-1, -1), 0.45, LINE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    return table


def flow_table() -> Table:
    steps = [
        ("01", "Chat", "Telegram or WhatsApp"),
        ("02", "Choose", "NourishBot, Swiggy, or Zomato"),
        ("03", "Discover", "Nearby or along a driving route"),
        ("04", "Verify", "Menu, price, fulfillment, future slot"),
        ("05", "Confirm", "Explicit approval before checkout"),
        ("06", "Fulfil", "Home delivery or drive-through pickup"),
    ]
    cells = []
    for number, title, detail in steps:
        cells.append([
            p(f"<font color='#E75D32'><b>{number}</b></font><br/><b>{title}</b><br/>{detail}", "table")
        ])
    table = Table([cell for cell in cells], colWidths=[160 * mm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), CREAM),
        ("BOX", (0, 0), (-1, -1), 0.6, LINE),
        ("LINEBELOW", (0, 0), (-1, -2), 0.4, LINE),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    return table


def cover(canvas, doc) -> None:
    canvas.saveState()
    width, height = A4
    canvas.setFillColor(CREAM)
    canvas.rect(0, 0, width, height, fill=1, stroke=0)
    canvas.setFillColor(ORANGE)
    canvas.rect(0, height - 7 * mm, width, 7 * mm, fill=1, stroke=0)
    canvas.setFillColor(GREEN)
    canvas.circle(width - 26 * mm, height - 30 * mm, 14 * mm, fill=1, stroke=0)
    canvas.setFillColor(ORANGE_LIGHT)
    canvas.circle(width - 11 * mm, height - 54 * mm, 22 * mm, fill=1, stroke=0)
    canvas.setFillColor(MUTED)
    canvas.setFont(FONT, 7.5)
    canvas.drawRightString(width - 20 * mm, 14 * mm, "Product vision • 15 October 2026")
    canvas.restoreState()


def later_pages(canvas, doc) -> None:
    canvas.saveState()
    width, height = A4
    canvas.setStrokeColor(LINE)
    canvas.setLineWidth(0.5)
    canvas.line(20 * mm, height - 15 * mm, width - 20 * mm, height - 15 * mm)
    canvas.setFont(FONT_BOLD, 7)
    canvas.setFillColor(GREEN)
    canvas.drawString(20 * mm, height - 11.5 * mm, "NOURISHBOT • PRODUCT VISION")
    canvas.setFont(FONT, 7)
    canvas.setFillColor(MUTED)
    canvas.drawRightString(width - 20 * mm, 11 * mm, f"{doc.page}")
    canvas.restoreState()


def build_story() -> list:
    story = [
        Spacer(1, 40 * mm),
        p("NOURISHBOT", "label"),
        p("The scheduled meal<br/>ordering assistant", "title"),
        p(
            "Present product definition, operating model, technical boundaries, launch plan, "
            "and future enhancements.",
            "subtitle",
        ),
        Spacer(1, 8 * mm),
        info_box(
            "One-line idea",
            "NourishBot helps a user discover the right restaurant, schedule a future meal through "
            "NourishBot, Swiggy, or Zomato, and choose home delivery or a route-aware drive-through pickup.",
            ORANGE_LIGHT,
        ),
        Spacer(1, 24 * mm),
        p(
            "“Plan the meal before hunger becomes a last-minute decision.”",
            "quote",
        ),
        PageBreak(),
    ]

    story += section_label("01 • Executive summary")
    story += [
        p("A focused product, not another food marketplace", "h1"),
        p(
            "NourishBot is a conversational coordination layer for scheduled meals. Users speak to it "
            "through Telegram or WhatsApp, describe what they want to eat and when, and select one of "
            "three ordering choices: NourishBot, Swiggy, or Zomato. It is deliberately not an instant "
            "ordering product. Every order needs a future date, local time, timezone, fulfillment mode, "
            "and explicit confirmation.",
        ),
        p(
            "Google Maps supports discovery. It can find strong restaurant candidates near a location "
            "or identify low-detour drive-through options along an origin-to-destination driving route. "
            "Maps never places an order. The selected ordering source must still verify the restaurant, "
            "menu, price, future availability, and fulfillment support.",
        ),
        Spacer(1, 3 * mm),
        data_table(
            ["Decision", "Product position"],
            [
                ["Primary job", "Schedule a meal reliably before it is needed."],
                ["Ordering choices", "NourishBot, Swiggy, and Zomato only."],
                ["Channels", "Telegram and WhatsApp conversations."],
                ["Fulfillment", "Scheduled home delivery or drive-through/pickup."],
                ["Discovery", "Google Maps nearby and along-route restaurant search."],
                ["Hard boundary", "No instant checkout and no invented provider capability."],
            ],
            [42 * mm, 118 * mm],
        ),
        Spacer(1, 6 * mm),
        info_box(
            "Important operating truth",
            "For NourishBot in-house orders, NourishBot can coordinate its own delivery workflow. For "
            "Swiggy or Zomato orders, fulfillment remains subject to that provider's authenticated tools, "
            "commercial permissions, serviceability, and native scheduling support.",
        ),
        PageBreak(),
    ]

    story += section_label("02 • Customer problem and promise")
    story += [
        p("Why this product should exist", "h1"),
        p(
            "Food apps are optimized for ordering now. People who already know tomorrow's routine—office "
            "workers, commuters, students, families, and diet-conscious customers—still repeat the same "
            "search, comparison, address, timing, and confirmation steps at the busiest moment of the day.",
        ),
        p("NourishBot changes the unit of value from an order to a reliable meal plan.", "quote"),
        p("Customer pains", "h2"),
        bullet_list([
            "Last-minute meal decisions collide with meetings, commutes, queues, and restaurant rush hours.",
            "The highest-rated restaurant may be inconvenient when the customer is already driving somewhere.",
            "Diet preferences, budget, provider, location, and time must be re-entered repeatedly.",
            "A user cannot safely assume that a provider supports scheduled checkout or drive-through pickup.",
            "Chat-based ordering can become risky if an AI is allowed to guess prices, IDs, or availability.",
        ]),
        p("Product promise", "h2"),
        bullet_list([
            "Plan breakfast, lunch, snacks, or dinner in one conversation.",
            "Find restaurants that fit the user's location, route, preferences, and budget.",
            "Use only NourishBot, Swiggy, or Zomato for the actual order path.",
            "Keep every state-changing action behind deterministic checks and explicit confirmation.",
            "Make home delivery and drive-through pickup equally understandable, without mixing them together.",
        ]),
        PageBreak(),
    ]

    story += section_label("03 • Present product experience")
    story += [
        p("The end-to-end customer journey", "h1"),
        flow_table(),
        Spacer(1, 7 * mm),
        p("Home-delivery flow", "h2"),
        bullet_list([
            "User selects NourishBot, Swiggy, or Zomato and describes the meal and dietary preference.",
            "NourishBot collects a complete delivery destination and checks the relevant serviceability rules.",
            "Maps may help discover nearby restaurants; the ordering source verifies the exact restaurant and menu.",
            "The user chooses a future date, time, and timezone, reviews the final cart, and confirms explicitly.",
            "The order is scheduled only if the selected source exposes a native scheduled transaction path.",
        ]),
        p("Drive-through / on-route pickup flow", "h2"),
        bullet_list([
            "User shares the route origin and destination, optionally by sending a current location.",
            "Google Routes computes the driving route; Places searches for drive-through candidates near that route.",
            "Results are ranked using operational status, rating, rating volume, and routing summary where available.",
            "The user chooses a candidate; NourishBot searches that restaurant on Swiggy or Zomato and verifies pickup support.",
            "Only a provider tool that supports the chosen fulfillment and a future time may complete checkout.",
        ]),
        PageBreak(),
    ]

    story += section_label("04 • What exists today")
    story += [
        p("Current implemented capabilities", "h1"),
        data_table(
            ["Area", "Present capability", "Current boundary"],
            [
                ["Conversation", "Telegram bot and signed WhatsApp Cloud API webhook.", "Production credentials and account approval are environment-dependent."],
                ["Meal planning", "Breakfast, lunch, snacks, dinner, budget, quantity, and supported diet preferences.", "No medical, allergy-safe, or invented nutrition claims."],
                ["Scheduling", "Mandatory future time, minimum lead time, horizon, timezone, and operating-hour checks.", "ASAP and delayed imitation of instant checkout are blocked."],
                ["Providers", "NourishBot in-house flow plus Swiggy and Zomato MCP adapters.", "Provider names do not imply partnership or guaranteed endpoint availability."],
                ["Maps", "Nearby best-restaurant search and along-route drive-through discovery.", "Requires Places API (New), Routes API, and a restricted API key."],
                ["Fulfillment", "Scheduled home delivery, pickup, and drive-through state in the in-house workflow.", "External fulfillment depends on each provider's native tool schema."],
                ["Operations", "Scheduled activation worker, kitchen queue, dispatch/tracking views, and WebSocket events.", "Workers, PostgreSQL, and Redis must be continuously available."],
                ["Safety", "Planner → deterministic guard → executor → verifier → response.", "Generated text cannot directly persist an order."],
            ],
            [28 * mm, 67 * mm, 65 * mm],
        ),
        Spacer(1, 6 * mm),
        p("Customer-facing surfaces", "h2"),
        bullet_list([
            "Landing page explaining scheduling, providers, meal periods, Maps discovery, and fulfillment choices.",
            "Unified staff login and workspace for admin, kitchen, waiter/collection, delivery, and tracking roles.",
            "Customer order tracking route with scheduled time, provider, status, and delivery information.",
        ]),
        PageBreak(),
    ]

    story += section_label("05 • Agent and system design")
    story += [
        p("How NourishBot stays agentic without becoming unsafe", "h1"),
        data_table(
            ["Stage", "Responsibility", "Trust model"],
            [
                ["Context", "Reads customer, session, cart, history, menu, and operational state.", "Database-backed facts only."],
                ["Planner", "Proposes goals, missing information, constraints, and tool calls.", "May propose; cannot commit."],
                ["Guard", "Normalizes fulfillment and blocks missing schedule, identity, address, or provider capability.", "Deterministic application code."],
                ["Executor", "Runs approved read/write tools and re-reads prices and availability.", "Validated arguments and transactional writes."],
                ["Verifier", "Confirms committed state, totals, item availability, status, and IDs.", "Database is the source of truth."],
                ["Responder", "Turns verified facts into a concise Telegram or WhatsApp reply.", "Cannot add unverified facts."],
            ],
            [28 * mm, 68 * mm, 64 * mm],
        ),
        Spacer(1, 7 * mm),
        p("Integration boundaries", "h2"),
        bullet_list([
            "Google Maps tools are read-only discovery tools and have no cart, payment, or order authority.",
            "Google place IDs are never reused as Swiggy or Zomato restaurant IDs.",
            "Provider transaction tools are hidden when their schemas do not expose a native schedule field.",
            "Drive-through checkout must expose compatible pickup, curbside, or drive-through fulfillment support.",
            "Every checkout and cancellation requires clear user confirmation and an audit trail.",
        ]),
        Spacer(1, 4 * mm),
        info_box(
            "Failure behavior",
            "If Maps is unavailable, discovery stops cleanly. If a provider supports browsing but not scheduled "
            "checkout, NourishBot may show verified options but must refuse to place an instant order.",
            ORANGE_LIGHT,
        ),
        PageBreak(),
    ]

    story += section_label("06 • Users and core scenarios")
    story += [
        p("Who receives the most value", "h1"),
        data_table(
            ["User", "Scenario", "Value"],
            [
                ["Daily commuter", "Find a low-detour dinner stop between office and home and schedule drive-through pickup.", "Less route disruption and no queue-time ordering."],
                ["Office worker", "Schedule a healthy lunch before meetings begin.", "Avoids rush-time decisions and missed meals."],
                ["Student", "Plan affordable snacks or dinner around class hours.", "Budget and timing stay visible before confirmation."],
                ["Family planner", "Arrange tomorrow's dinner for home delivery.", "One conversation preserves preferences, address, and schedule."],
                ["Diet-focused user", "Ask for vegetarian, vegan, balanced, or high-protein options.", "Suggestions stay tied to verified menu metadata."],
            ],
            [31 * mm, 82 * mm, 47 * mm],
        ),
        p("Example conversation", "h2"),
        info_box(
            "User",
            "“Tomorrow I am driving from Madhapur to Hyderabad Airport. Find a vegetarian drive-through "
            "option on my route and schedule lunch for 1:00 PM IST through Swiggy.”",
            CREAM,
        ),
        Spacer(1, 3 * mm),
        info_box(
            "NourishBot behavior",
            "Compute route → search Maps along route → show transparent candidates and detour context → let the "
            "user choose → search the exact restaurant on Swiggy → verify menu, pickup support, future slot, cart, "
            "and total → ask for confirmation → return only a provider-verified order ID.",
        ),
        PageBreak(),
    ]

    story += section_label("07 • Launch definition")
    story += [
        p("What 'public-ready' must mean", "h1"),
        p(
            "A polished interface is not sufficient evidence that the product can safely process public orders. "
            "Launch readiness requires verified infrastructure, provider behavior, security controls, operational "
            "ownership, and rollback procedures.",
        ),
        data_table(
            ["Gate", "Evidence required"],
            [
                ["Provider", "Authenticated Swiggy and Zomato tools tested end to end; supported schedule and fulfillment fields recorded."],
                ["Maps", "Restricted key, enabled APIs, billing/quota alerts, staging route searches, and failure-path tests."],
                ["Channels", "Telegram token and verified WhatsApp webhook, signature validation, retries, and duplicate suppression."],
                ["Data", "Managed PostgreSQL and Redis, migrations, backups, restore test, retention, and PII handling."],
                ["Workers", "Celery worker/beat health checks and alerts; due schedules activate once and at the correct time."],
                ["Security", "Strong secrets, exact CORS origins, HTTPS, least privilege, rate limits, and log redaction."],
                ["Quality", "Unit, integration, end-to-end, scheduling-boundary, concurrency, load, and rollback tests."],
                ["Operations", "Owner for failed orders, refunds, cancellations, customer support, provider downtime, and Maps quota exhaustion."],
            ],
            [37 * mm, 123 * mm],
        ),
        Spacer(1, 6 * mm),
        info_box(
            "Go-live rule",
            "A provider is live only after authenticated staging tests prove the exact scheduled order and fulfillment "
            "path. A visible button, configured URL, or successful restaurant search is not enough.",
            ORANGE_LIGHT,
        ),
        PageBreak(),
    ]

    story += section_label("08 • Future enhancements")
    story += [
        p("A phased roadmap that protects the core promise", "h1"),
        data_table(
            ["Phase", "Enhancements", "Outcome"],
            [
                ["1 • Launch hardening\n0–2 months", "Provider capability registry; E2E contract tests; Maps quota monitoring; stronger observability; support console; cancellation/refund drills.", "A measurable, supportable public beta."],
                ["2 • Personal meal memory\n2–4 months", "Consent-based preference memory; saved routes; recurring meal templates; spend limits; household profiles; smarter dietary filters.", "Fewer repeated choices without reducing control."],
                ["3 • Route intelligence\n4–8 months", "Live traffic-aware departure guidance; detour thresholds; stop sequencing; geofenced arrival; pickup-ready timing; EV charging + meal stops.", "Drive-through becomes a true commute assistant."],
                ["4 • Proactive scheduling\n6–10 months", "Calendar-aware suggestions; recurring weekly plans; reorder approval windows; delay/re-route alerts; weather-aware alternatives.", "NourishBot anticipates routine while keeping confirmation explicit."],
                ["5 • Partner platform\n9–15 months", "Restaurant onboarding; verified nutrition fields; prep capacity APIs; white-label assistants; corporate meal accounts; partner analytics.", "A scalable scheduled-meal coordination network."],
            ],
            [35 * mm, 86 * mm, 39 * mm],
        ),
        p("Future home-delivery model", "h2"),
        bullet_list([
            "Near term: NourishBot coordinates its in-house delivery workflow; Swiggy/Zomato retain fulfillment responsibility for their orders.",
            "Later: add an explicit courier orchestration layer only after contracts, delivery liability, dispatch APIs, proof-of-delivery, refunds, and customer support are defined.",
            "Never represent NourishBot as physically delivering a third-party order unless that operational chain is real, authorized, and observable.",
        ]),
        PageBreak(),
    ]

    story += section_label("09 • Prioritization and non-goals")
    story += [
        p("What to build next—and what to resist", "h1"),
        data_table(
            ["Priority", "Build", "Reason"],
            [
                ["P0", "Real provider contract tests and capability registry.", "The product promise depends on verified scheduling and fulfillment."],
                ["P0", "Public-launch observability, alerting, support, backup, and rollback.", "Operational failure is more damaging than a missing feature."],
                ["P1", "Saved routes and repeat meal templates.", "High recurring value with low cognitive load."],
                ["P1", "Provider-neutral comparison using verified totals and detour time.", "Supports better decisions without becoming another marketplace."],
                ["P2", "Calendar and recurring meal suggestions.", "Useful after consent, reliability, and confirmation behavior are proven."],
                ["P3", "Partner and enterprise tooling.", "Scale only after consumer workflows are stable."],
            ],
            [22 * mm, 70 * mm, 68 * mm],
        ),
        p("Explicit non-goals", "h2"),
        bullet_list([
            "No instant or ASAP ordering.",
            "No generic list of unverified MCP providers.",
            "No claim of official Swiggy, Zomato, Google Maps, or restaurant partnership without written authorization.",
            "No fabricated nutrition, allergy safety, price, ETA, availability, provider ID, or order confirmation.",
            "No autonomous recurring purchase without a clear approval model, spending controls, and revocation path.",
            "No attempt to replace a navigation app; NourishBot uses route context to improve meal discovery.",
        ]),
        PageBreak(),
    ]

    story += section_label("10 • Metrics and business model")
    story += [
        p("How to know the product is working", "h1"),
        data_table(
            ["Metric group", "Measures"],
            [
                ["Reliability", "Scheduled-order success rate; duplicate-order rate; provider failure rate; worker activation accuracy."],
                ["Customer value", "Time from intent to confirmed schedule; repeat scheduling rate; saved-route reuse; cancellation rate."],
                ["Route quality", "Median added detour; candidate-to-selection rate; pickup-ready timing accuracy; route-search failure rate."],
                ["Safety", "Instant-checkout blocks; confirmation-gate violations; hallucinated-fact incidents; signature/idempotency failures."],
                ["Operations", "Support contacts per 100 orders; refund resolution time; delivery exceptions; alert time-to-acknowledge."],
                ["Economics", "Contribution margin per completed order; infrastructure/provider cost; partner revenue; customer acquisition payback."],
            ],
            [38 * mm, 122 * mm],
        ),
        p("Potential revenue paths", "h2"),
        bullet_list([
            "Transparent restaurant or provider referral revenue, disclosed where required.",
            "Premium personal planning: saved routines, family profiles, advanced budget controls, and proactive scheduling.",
            "Corporate meal scheduling for teams, offices, campuses, and commuting workforces.",
            "Restaurant software subscription for scheduled demand, capacity planning, pickup timing, and customer communication.",
        ]),
        p(
            "Revenue should never influence the ranking without disclosure. Route convenience, verified availability, "
            "user preference, price, and service reliability must remain explainable.",
        ),
        PageBreak(),
    ]

    story += section_label("11 • Risks and controls")
    story += [
        p("Known risks before scaling", "h1"),
        data_table(
            ["Risk", "Control"],
            [
                ["Provider MCP changes or is unavailable", "Capability registry, startup checks, contract tests, circuit breaker, and honest degraded mode."],
                ["Maps candidate is not actually drive-through", "Label as candidate; verify exact fulfillment with provider/restaurant before checkout."],
                ["Future schedule is unsupported", "Schema gate blocks transaction; never place an immediate order as a workaround."],
                ["Wrong route or location", "Echo origin/destination, let user correct them, and avoid silently reusing stale route context."],
                ["Sensitive location history", "Consent, minimization, retention limits, encryption, access controls, and route-history deletion."],
                ["Order duplication", "Channel message idempotency, explicit confirmation, provider idempotency keys where supported."],
                ["Delivery responsibility is unclear", "Show which party fulfils the order and who handles support before confirmation."],
                ["NDA/IP exposure", "Use original product language, synthetic demo data, clean assets, and legal review of history and integrations."],
            ],
            [50 * mm, 110 * mm],
        ),
        PageBreak(),
    ]

    story += section_label("12 • Final product definition")
    story += [
        p("The concise version", "h1"),
        p(
            "NourishBot is an AI scheduled-meal assistant for Telegram and WhatsApp. It helps a user "
            "choose breakfast, lunch, snacks, or dinner; discover restaurants near a place or along a "
            "driving route with Google Maps; and schedule through NourishBot, Swiggy, or Zomato for "
            "home delivery or drive-through pickup.",
        ),
        p(
            "Its differentiation is not chat alone. The product combines future scheduling, route-aware "
            "restaurant discovery, explicit fulfillment choices, provider-aware execution, and deterministic "
            "safety controls. It refuses instant checkout and refuses to claim a capability that the selected "
            "provider has not proven.",
        ),
        Spacer(1, 7 * mm),
        info_box(
            "Present",
            "A working scheduled-order architecture with Telegram, WhatsApp, NourishBot in-house ordering, "
            "Swiggy/Zomato MCP boundaries, Maps discovery tools, home delivery, drive-through, operations, "
            "tracking, and focused automated tests.",
            GREEN_LIGHT,
        ),
        Spacer(1, 4 * mm),
        info_box(
            "Future",
            "A trusted meal-planning layer that remembers routines with consent, understands commutes and "
            "traffic, coordinates restaurants and fulfillment partners, and supports households, workplaces, "
            "and restaurant operators without removing human confirmation.",
            ORANGE_LIGHT,
        ),
        Spacer(1, 9 * mm),
        p("North-star principle", "h2"),
        p(
            "The user should feel that tomorrow's meal is already handled—without losing visibility, choice, "
            "or control.",
            "quote",
        ),
        Spacer(1, 12 * mm),
        HRFlowable(width="100%", thickness=0.7, color=LINE, spaceAfter=6),
        p(
            "Status note: This document describes the repository's current product direction and a proposed "
            "future roadmap. Swiggy, Zomato, and Google Maps names describe intended technical integrations; "
            "they do not assert partnership, endorsement, or production availability.",
            "small",
        ),
    ]
    return story


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        rightMargin=25 * mm,
        leftMargin=25 * mm,
        topMargin=22 * mm,
        bottomMargin=18 * mm,
        title="NourishBot Product Vision",
        author="NourishBot",
        subject="Present product definition and future enhancements",
    )
    doc.build(build_story(), onFirstPage=cover, onLaterPages=later_pages)
    print(f"Generated {OUTPUT}")


if __name__ == "__main__":
    main()
