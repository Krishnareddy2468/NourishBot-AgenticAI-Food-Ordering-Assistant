"use client";

import { useState, useRef, useEffect, useCallback } from "react";

const ZOMATO_PROXY_BASE = "/api/zomato";
const CHAT_REQUEST_TIMEOUT_MS = 30000;

// parse bot messages - handles bold, line breaks, code
function renderBotText(text) {
  if (!text) return "";
  const qrPattern = /\[QR Code Image Saved to\s+([^\]]+)\]/i;
  const qrMatch = text.match(qrPattern);
  let rendered = text;

  if (qrMatch?.[1]) {
    const rawPath = qrMatch[1].trim();
    const fileName = rawPath.split("/").pop();
    if (fileName) {
      const qrHtml = [
        '<div class="qr-preview">',
        "<div class=\"qr-preview-title\">Scan this QR to pay</div>",
        `<img class=\"qr-preview-image\" src=\"${ZOMATO_PROXY_BASE}/qr/${encodeURIComponent(fileName)}\" alt=\"UPI QR code\" />`,
        "</div>",
      ].join("");
      rendered = rendered.replace(qrPattern, qrHtml);
    }
  }

  return rendered
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/\*([^*]+)\*/g, "<strong>$1</strong>")
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/━+/g, '<hr class="msg-divider"/>')
    .replace(/\n/g, "<br/>");
}

function timeNow() {
  return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

// detect if bot message is asking for location
function needsLocation(text) {
  if (!text) return false;
  const t = text.toLowerCase();
  return (
    t.includes("share your delivery address") ||
    t.includes("share location") ||
    t.includes("need your location") ||
    t.includes("city name") ||
    t.includes("i need your location") ||
    t.includes("please share your") ||
    t.includes("share your **city")
  );
}

// detect if bot message is an error
function isErrorMessage(text) {
  if (!text) return false;
  return (
    text.includes("went wrong") ||
    text.includes("API Key Error") ||
    text.includes("Rate limit") ||
    text.includes("Connection issue") ||
    text.includes("Setup required") ||
    text.includes("Please try again")
  );
}

// quick suggestions that show up on the welcome screen
const SUGGESTIONS = [
  { emoji: "🍕", title: "Pizza nearby", query: "Show me pizza places near me", desc: "Find the best pizzerias" },
  { emoji: "🍗", title: "Biryani", query: "Show me biryani restaurants", desc: "Discover top biryani spots" },
  { emoji: "🍔", title: "Burgers", query: "Show me burger places", desc: "Quick bites & burgers" },
  { emoji: "🥘", title: "South Indian", query: "Find South Indian restaurants", desc: "Dosas, idlis, and more" },
  { emoji: "🌮", title: "Chinese", query: "Show me Chinese restaurants", desc: "Noodles, dim sum & more" },
  { emoji: "🧁", title: "Desserts & Cafes", query: "Show me dessert cafes", desc: "Cakes, ice cream & more" },
  { emoji: "🥗", title: "Healthy & Veg", query: "Show me healthy vegetarian restaurants", desc: "Salads, bowls & clean food" },
  { emoji: "⭐", title: "Top Rated", query: "Show me top rated restaurants nearby", desc: "Highest rated near you" },
];

// sidebar quick-action buttons
const QUICK_ACTIONS = [
  { emoji: "🍕", label: "Pizza",        query: "Show me pizza places near me" },
  { emoji: "🍗", label: "Biryani",      query: "Show me biryani restaurants" },
  { emoji: "🍔", label: "Burgers",      query: "Show me burger restaurants" },
  { emoji: "🌮", label: "Chinese",      query: "Show me Chinese restaurants" },
  { emoji: "🥘", label: "South Indian", query: "Show me South Indian restaurants" },
  { emoji: "☕", label: "Cafes",        query: "Show me cafes and coffee shops" },
  { emoji: "🥗", label: "Healthy",      query: "Show me healthy vegetarian restaurants" },
  { emoji: "🍜", label: "Noodles",      query: "Show me noodle and pasta restaurants" },
  { emoji: "⭐", label: "Top Rated",    query: "Show me top rated restaurants nearby" },
  { emoji: "🚀", label: "Fast Delivery",query: "Show restaurants with fastest delivery" },
  { emoji: "💸", label: "Budget Eats",  query: "Show cheap and budget-friendly restaurants" },
  { emoji: "📦", label: "Track Order",  query: "Track my current order" },
];

// smart quick replies based on what the bot just said
function getQuickReplies(botMsg, state) {
  if (!botMsg) return [];
  const msg = botMsg.toLowerCase();

  if (
    msg.includes("reply with a restaurant number") ||
    msg.includes("pick a restaurant") ||
    msg.includes("which one would you like to order from")
  ) {
    return [
      { label: "1️⃣ First", query: "1" },
      { label: "2️⃣ Second", query: "2" },
      { label: "3️⃣ Third", query: "3" },
      { label: "4️⃣ Fourth", query: "4" },
      { label: "5️⃣ Fifth", query: "5" },
    ];
  }
  if (msg.includes("to order, just tell me") || msg.includes("browse the menu") || msg.includes("what would you like to order")) {
    return [
      { label: "� Add to Cart", query: "1 of each" },
      { label: "📋 Show Full Menu", query: "show menu of first restaurant" },
      { label: "🟢 Veg Only", query: "show veg items only" },
      { label: "🔴 Non-Veg", query: "show non-veg items" },
    ];
  }
  if (msg.includes("add more items") || msg.includes("say \"checkout\"") || msg.includes("anything else")) {
    return [
      { label: "✅ Checkout", query: "checkout" },
      { label: "📋 Show Menu", query: "show menu" },
      { label: "🛒 View Cart", query: "show my cart" },
      { label: "🟢 Add Veg Item", query: "show veg items only" },
    ];
  }
  if (msg.includes("share your delivery address") || msg.includes("delivery address")) {
    return [
      { label: "📍 Use My GPS", query: "__gps__" },
      { label: "🏠 Madhapur, Hyderabad", query: "Madhapur, Hyderabad" },
      { label: "🏢 Koramangala, Bengaluru", query: "Koramangala, Bengaluru" },
      { label: "🏙️ Vijayawada", query: "Vijayawada" },
    ];
  }
  if (msg.includes("reply *yes* to confirm") || msg.includes("confirm your order") || msg.includes("shall i place")) {
    return [
      { label: "✅ Yes, Place Order", query: "yes" },
      { label: "✏️ Edit Cart First", query: "show my cart" },
      { label: "❌ Cancel", query: "no" },
    ];
  }
  if (msg.includes("order placed successfully") || msg.includes("order placed") || msg.includes("order has been placed")) {
    return [
      { label: "📦 Track Order", query: "track my order" },
      { label: "🍽️ Order More", query: "start over" },
    ];
  }
  if (msg.includes("what are you craving") || msg.includes("what would you like") || msg.includes("how can i help")) {
    return [
      { label: "🍕 Pizza", query: "show me pizza places" },
      { label: "🍗 Biryani", query: "show me biryani restaurants" },
      { label: "🍔 Burgers", query: "show me burger places" },
      { label: "🥘 South Indian", query: "show me South Indian restaurants" },
      { label: "⭐ Top Rated", query: "show me top rated restaurants nearby" },
    ];
  }
  if (msg.includes("order status") || msg.includes("← current") || msg.includes("order confirmed") || msg.includes("being prepared") || msg.includes("out for delivery")) {
    const replies = [{ label: "🔄 Refresh Status", query: "track my order" }];
    if (msg.includes("delivered")) {
      replies.push({ label: "⭐ Rate Order", query: "I want to rate my order" });
      replies.push({ label: "🍽️ Order Again", query: "start over" });
    }
    return replies;
  }
  if (msg.includes("reply with item number") || msg.includes("item number to add")) {
    return [
      { label: "🟢 Veg Only", query: "show veg items only" },
      { label: "⭐ Bestsellers", query: "show bestsellers" },
      { label: "💰 Budget Items", query: "show items under 200 rupees" },
    ];
  }
  if (msg.includes("show me") && (msg.includes("restaurant") || msg.includes("places")) && !msg.includes("menu")) {
    return [
      { label: "📋 Show Menu", query: "show menu of first restaurant" },
      { label: "⭐ Filter 4★+", query: "show only restaurants rated 4 stars and above" },
      { label: "🟢 Veg Restaurants", query: "show only vegetarian restaurants" },
      { label: "💰 Budget Options", query: "show budget friendly restaurants" },
    ];
  }
  // Bot listed restaurants ("Here are the top ... restaurants")
  if (msg.includes("here are the top") && (msg.includes("restaurant") || msg.includes("pizza") || msg.includes("biryani") || msg.includes("burger"))) {
    return [
      { label: "1️⃣ First", query: "1" },
      { label: "2️⃣ Second", query: "2" },
      { label: "3️⃣ Third", query: "3" },
      { label: "4️⃣ Fourth", query: "4" },
    ];
  }
  if (msg.includes("city name") || msg.includes("need your location") || msg.includes("share location") || msg.includes("i need your location") || msg.includes("please share your")) {
    return [
      { label: "📍 Use My GPS", query: "__gps__" },
      { label: "📌 Hyderabad", query: "Hyderabad" },
      { label: "📌 Bengaluru", query: "Bengaluru" },
      { label: "📌 Mumbai", query: "Mumbai" },
      { label: "📌 Chennai", query: "Chennai" },
    ];
  }
  return [];
}

export default function Home() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [recordingSupported, setRecordingSupported] = useState(true);
  const [loadingStatus, setLoadingStatus] = useState("Thinking...");
  const [cartItems, setCartItems] = useState([]);
  const [cartOpen, setCartOpen] = useState(false);
  const [currentRestaurant, setCurrentRestaurant] = useState(null);
  const [quickReplies, setQuickReplies] = useState([]);
  const [showScrollBtn, setShowScrollBtn] = useState(false);
  const [userLocation, setUserLocation] = useState(null);
  const [locationLoading, setLocationLoading] = useState(false);
  const [botState, setBotState] = useState("idle");
  const [currentOrderId, setCurrentOrderId] = useState(null);
  const [lastOrderStatus, setLastOrderStatus] = useState(null);
  const messagesEndRef = useRef(null);
  const messagesContainerRef = useRef(null);
  const inputRef = useRef(null);
  const lastSentMessageRef = useRef(null);  // for retry
  const mediaRecorderRef = useRef(null);
  const mediaStreamRef = useRef(null);
  const audioChunksRef = useRef([]);

  // keep user id across page reloads
  const userId = useRef(
    typeof window !== "undefined"
      ? sessionStorage.getItem("bot_user_id") || (() => {
        const id = `web_${Date.now()}`;
        sessionStorage.setItem("bot_user_id", id);
        return id;
      })()
      : `web_${Date.now()}`
  );

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading, scrollToBottom]);

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  useEffect(() => {
    if (typeof window !== "undefined") {
      const canRecord = Boolean(navigator?.mediaDevices?.getUserMedia) && typeof window.MediaRecorder !== "undefined";
      setRecordingSupported(canRecord);
    }
    return () => {
      if (mediaStreamRef.current) {
        for (const track of mediaStreamRef.current.getTracks()) {
          track.stop();
        }
      }
    };
  }, []);

  // show "scroll to bottom" if user scrolls up
  const handleScroll = useCallback(() => {
    const container = messagesContainerRef.current;
    if (!container) return;
    const distFromBottom = container.scrollHeight - container.scrollTop - container.clientHeight;
    setShowScrollBtn(distFromBottom > 150);
  }, []);

  const sendMessage = useCallback(async (text, locationOverride) => {
    if (!text.trim() || isLoading) return;
    lastSentMessageRef.current = text.trim();

    const userMsg = { role: "user", content: text.trim(), time: timeNow() };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setQuickReplies([]);
    setIsLoading(true);
    setLoadingStatus("Thinking...");
    let timeoutId;

    try {
      const controller = new AbortController();
      timeoutId = setTimeout(() => controller.abort(), CHAT_REQUEST_TIMEOUT_MS);
      const res = await fetch(`${ZOMATO_PROXY_BASE}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: controller.signal,
        body: JSON.stringify({
          message: text.trim(),
          user_id: userId.current,
          user_name: "Guest",
          user_location: locationOverride || userLocation,
        }),
      });

      if (!res.ok) {
        if (res.status === 504) {
          throw new Error("REQUEST_TIMEOUT");
        }
        throw new Error(`Server error (${res.status})`);
      }
      const data = await res.json();

      // Show thinking steps as the loading status before adding the final message
      if (data.thinking_steps && data.thinking_steps.length > 0) {
        for (const step of data.thinking_steps) {
          setLoadingStatus(step);
          await new Promise((r) => setTimeout(r, 220));
        }
      }

      const botMsg = {
        role: "bot",
        content: data.response,
        time: timeNow(),
        thinkingSteps: data.thinking_steps || [],
        isError: isErrorMessage(data.response),
      };
      setMessages((prev) => [...prev, botMsg]);

      if (data.cart_items) setCartItems(data.cart_items);
      if (data.restaurant) setCurrentRestaurant(data.restaurant);
      if (data.state) setBotState(data.state);
      if (data.order?.order_id) {
        setCurrentOrderId(data.order.order_id);
        setLastOrderStatus(data.order.status || "confirmed");
      }

      // generate contextual quick replies
      const replies = getQuickReplies(data.response, data.state);
      // append location share button only if not already included
      const alreadyHasGps = replies.some((r) => r.query === "__gps__");
      if (!alreadyHasGps && needsLocation(data.response)) {
        replies.push({ label: "📍 Share My Location", query: "__gps__" });
      }
      setQuickReplies(replies);
    } catch (error) {
      const isTimeoutError = error?.name === "AbortError";
      const isBackendTimeout = error?.message === "REQUEST_TIMEOUT";
      if (isTimeoutError || isBackendTimeout) {
        console.warn("Chat request timed out");
      } else {
        console.error("Error:", error);
      }
      setMessages((prev) => [
        ...prev,
        {
          role: "bot",
          content: (isTimeoutError || isBackendTimeout)
            ? "⏱️ **Request timed out** — This took too long.\n\nTry a more specific query like `biryani in Madhapur`."
            : "🌐 **Connection issue** — Couldn't reach Zomatobot backend.\n\nMake sure backend is running on port 8000 and web chat is enabled.",
          time: timeNow(),
          isError: true,
          thinkingSteps: [],
        },
      ]);
      setQuickReplies([]);
    } finally {
      if (timeoutId) clearTimeout(timeoutId);
      setIsLoading(false);
      setLoadingStatus("Thinking...");
      // refocus input after sending
      setTimeout(() => inputRef.current?.focus(), 100);
    }
  }, [isLoading, userLocation]);

  const handleSubmit = (e) => {
    e.preventDefault();
    sendMessage(input);
  };

  const handleRetry = () => {
    if (lastSentMessageRef.current) {
      sendMessage(lastSentMessageRef.current);
    }
  };

  const handleShareLocation = useCallback(async (onSuccess) => {
    if (!navigator.geolocation) {
      alert("Geolocation is not supported by your browser. Please type your city name.");
      return;
    }
    setLocationLoading(true);
    navigator.geolocation.getCurrentPosition(
      async (pos) => {
        const { latitude, longitude } = pos.coords;
        // reverse-geocode using a free public API
        try {
          const r = await fetch(
            `https://nominatim.openstreetmap.org/reverse?lat=${latitude}&lon=${longitude}&format=json`
          );
          if (r.ok) {
            const geo = await r.json();
            const address = geo.display_name || `${latitude.toFixed(4)}, ${longitude.toFixed(4)}`;
            const short = geo.address
              ? [geo.address.suburb, geo.address.city || geo.address.town || geo.address.village, geo.address.state]
                  .filter(Boolean)
                  .join(", ")
              : address;
            setUserLocation(short);
            setLocationLoading(false);
            if (onSuccess) onSuccess(short);
            else sendMessage(`My location is: ${short}`, short);
          }
        } catch {
          const coord = `${latitude.toFixed(4)}, ${longitude.toFixed(4)}`;
          setUserLocation(coord);
          setLocationLoading(false);
          sendMessage(`My location coordinates are: ${coord}`, coord);
        }
      },
      () => {
        setLocationLoading(false);
        alert("Location permission denied. Please type your city or area name.");
      },
      { timeout: 8000 }
    );
  }, [sendMessage]);

  const sendVoiceMessage = useCallback(async (audioBlob) => {
    if (!audioBlob || isLoading) return;

    setMessages((prev) => [...prev, { role: "user", content: "🎤 Voice message", time: timeNow() }]);
    setQuickReplies([]);
    setIsLoading(true);
    setLoadingStatus("Transcribing your voice...");

    try {
      const formData = new FormData();
      formData.append("audio", audioBlob, "voice-message.webm");
      formData.append("user_id", userId.current);
      formData.append("user_name", "Guest");
      if (userLocation) {
        formData.append("user_location", userLocation);
      }

      const res = await fetch(`${ZOMATO_PROXY_BASE}/voice-upload`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        throw new Error(`Voice upload failed (${res.status})`);
      }

      const data = await res.json();

      if (data.thinking_steps && data.thinking_steps.length > 0) {
        for (const step of data.thinking_steps) {
          setLoadingStatus(step);
          await new Promise((r) => setTimeout(r, 220));
        }
      }

      const botMsg = {
        role: "bot",
        content: data.response,
        time: timeNow(),
        thinkingSteps: data.thinking_steps || [],
        isError: isErrorMessage(data.response),
      };
      setMessages((prev) => [...prev, botMsg]);

      if (data.cart_items) setCartItems(data.cart_items);
      if (data.restaurant) setCurrentRestaurant(data.restaurant);
      if (data.state) setBotState(data.state);
      if (data.order?.order_id) {
        setCurrentOrderId(data.order.order_id);
        setLastOrderStatus(data.order.status || "confirmed");
      }

      const replies = getQuickReplies(data.response, data.state);
      const alreadyHasGps = replies.some((r) => r.query === "__gps__");
      if (!alreadyHasGps && needsLocation(data.response)) {
        replies.push({ label: "📍 Share My Location", query: "__gps__" });
      }
      setQuickReplies(replies);
    } catch (error) {
      console.error("Voice upload error:", error);
      setMessages((prev) => [
        ...prev,
        {
          role: "bot",
          content: "🎙️ **Voice input failed** — I couldn't process that recording.\n\nPlease try again or send text.",
          time: timeNow(),
          isError: true,
          thinkingSteps: [],
        },
      ]);
      setQuickReplies([]);
    } finally {
      setIsLoading(false);
      setLoadingStatus("Thinking...");
      setTimeout(() => inputRef.current?.focus(), 100);
    }
  }, [isLoading, userLocation]);

  const handleVoiceInput = useCallback(async () => {
    if (isLoading || !recordingSupported) return;

    if (isRecording && mediaRecorderRef.current) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      setLoadingStatus("Preparing recording...");
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);

      mediaStreamRef.current = stream;
      mediaRecorderRef.current = recorder;
      audioChunksRef.current = [];

      recorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      recorder.onstop = async () => {
        const blob = new Blob(audioChunksRef.current, { type: recorder.mimeType || "audio/webm" });
        if (mediaStreamRef.current) {
          for (const track of mediaStreamRef.current.getTracks()) {
            track.stop();
          }
        }
        mediaStreamRef.current = null;
        mediaRecorderRef.current = null;
        audioChunksRef.current = [];
        if (blob.size > 0) {
          await sendVoiceMessage(blob);
        }
      };

      recorder.start();
      setIsRecording(true);
      setLoadingStatus("Listening... Tap mic again to stop");
    } catch (error) {
      console.error("Microphone access denied/unavailable:", error);
      setMessages((prev) => [
        ...prev,
        {
          role: "bot",
          content: "🎤 **Microphone access blocked** — Please allow mic permission and try again.",
          time: timeNow(),
          isError: true,
          thinkingSteps: [],
        },
      ]);
      setIsRecording(false);
    }
  }, [isLoading, isRecording, recordingSupported, sendVoiceMessage]);

  const handleQuickReply = (qr) => {
    if (qr.query === "__gps__") {
      handleShareLocation();
    } else {
      sendMessage(qr.query);
    }
  };

  // ── Order status polling ────────────────────────────────────────────────────
  useEffect(() => {
    const isOrderActive = ["order_placed", "tracking"].includes(botState);
    if (!currentOrderId || !isOrderActive) return;

    let knownStatus = lastOrderStatus;

    const poll = async () => {
      try {
        const res = await fetch(`${ZOMATO_PROXY_BASE}/order-status/${userId.current}`);
        if (!res.ok) return;
        const data = await res.json();
        if (data.status && data.status !== knownStatus) {
          knownStatus = data.status;
          setLastOrderStatus(data.status);
          if (data.message) {
            setMessages((prev) => [
              ...prev,
              { role: "bot", content: data.message, time: timeNow(), isError: false, thinkingSteps: [] },
            ]);
          }
          if (data.status === "delivered" || data.status === "cancelled") {
            clearInterval(intervalId);
            setQuickReplies([
              { label: "⭐ Rate Order", query: "I want to rate my order" },
              { label: "🍽️ Order Again", query: "start over" },
            ]);
          }
        }
      } catch (_) {
        // silently ignore network errors during polling
      }
    };

    const intervalId = setInterval(poll, 5000);
    return () => clearInterval(intervalId);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentOrderId, botState]);

  const handleNewChat = () => {
    setMessages([]);
    setCartItems([]);
    setCurrentRestaurant(null);
    setQuickReplies([]);
    setBotState("idle");
    setCurrentOrderId(null);
    setLastOrderStatus(null);
    // clear the stored session so we get a fresh one
    const newId = `web_${Date.now()}`;
    userId.current = newId;
    if (typeof window !== "undefined") sessionStorage.setItem("bot_user_id", newId);
  };

  const cartSubtotal = cartItems.reduce((sum, item) => sum + item.price * item.quantity, 0);
  const cartTax = Math.round(cartSubtotal * 0.05);
  const cartDelivery = currentRestaurant ? 35 : 0;
  const cartTotal = cartSubtotal + cartTax + cartDelivery;

  return (
    <div className="app-container">
      {/* sidebar */}
      <aside className="sidebar">
        <div className="sidebar-header">
          <div className="sidebar-logo">
            <div className="sidebar-logo-icon">🍕</div>
            <div>
              <h1>Zomato Bot</h1>
              <p>AI Food Assistant</p>
            </div>
          </div>
        </div>

        <div className="sidebar-features">
          <div className="sidebar-section-title">Quick Order</div>
          <div className="sidebar-quick-grid">
            {QUICK_ACTIONS.map((a, i) => (
              <button
                key={i}
                className="sidebar-grid-btn"
                onClick={() => sendMessage(a.query)}
                disabled={isLoading}
                title={a.query}
              >
                <span className="sidebar-grid-emoji">{a.emoji}</span>
                <span className="sidebar-grid-label">{a.label}</span>
              </button>
            ))}
          </div>
        </div>

        <div className="sidebar-quick-actions">
          <button className="quick-action-btn" onClick={() => sendMessage("help")}>
            ❓ Show Help
          </button>
          <button className="quick-action-btn" onClick={() => handleShareLocation()}>
            📍 {locationLoading ? "Getting location..." : userLocation ? "📍 " + userLocation.split(",")[0] : "Share My Location"}
          </button>
          <button className="quick-action-btn" onClick={handleNewChat}>
            🔄 New Conversation
          </button>
        </div>
      </aside>

      {/* main chat */}
      <main className="main-content">
        <header className="chat-header">
          <div className="chat-header-left">
            <h2>🤖 Zomato AI Assistant</h2>
            <div className="header-status">
              <div className="status-dot" />
              <span>Online</span>
            </div>
          </div>
          <div className="header-actions">
            {cartItems.length > 0 && (
              <button className="header-btn" onClick={() => setCartOpen(!cartOpen)} id="cart-toggle-btn">
                🛒 Cart ({cartItems.length})
              </button>
            )}
            <button className="header-btn" onClick={handleNewChat} id="new-chat-btn">
              🔄 New Chat
            </button>
          </div>
        </header>

        <div className="messages-container" ref={messagesContainerRef} onScroll={handleScroll}>
          {messages.length === 0 ? (
            <div className="welcome-screen">
              <div className="welcome-icon">🍽️</div>
              <h2>What are you craving today?</h2>
              <p>
                I&apos;m your AI food ordering assistant. Search restaurants,
                browse menus, and place orders — all through natural conversation!
              </p>
              <div className="welcome-suggestions">
                {SUGGESTIONS.map((s, i) => (
                  <div className="suggestion-card" key={i} onClick={() => sendMessage(s.query)} id={`suggestion-${i}`}>
                    <div className="suggestion-card-emoji">{s.emoji}</div>
                    <h4>{s.title}</h4>
                    <p>{s.desc}</p>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <div className="messages-inner">
              {messages.map((msg, i) => (
                <div className={`message ${msg.role}`} key={i}>
                  <div className="message-avatar">
                    {msg.role === "bot" ? "🤖" : "👤"}
                  </div>
                  <div style={{ maxWidth: "100%" }}>
                    <div
                      className={`message-bubble ${msg.isError ? "error-bubble" : ""}`}
                      dangerouslySetInnerHTML={{ __html: renderBotText(msg.content) }}
                    />
                    <div className="message-time">{msg.time}</div>
                    {/* Retry button on error messages */}
                    {msg.isError && msg.role === "bot" && i === messages.length - 1 && (
                      <button className="retry-btn" onClick={handleRetry} disabled={isLoading}>
                        🔄 Retry
                      </button>
                    )}
                  </div>
                </div>
              ))}

              {isLoading && (
                <div className="typing-indicator">
                  <div className="message-avatar" style={{ background: "var(--brand-soft)" }}>🤖</div>
                  <div className="typing-bubble">
                    <div className="typing-dots">
                      <div className="typing-dot" />
                      <div className="typing-dot" />
                      <div className="typing-dot" />
                    </div>
                    <div className="typing-status">{loadingStatus}</div>
                  </div>
                </div>
              )}

              {/* quick reply chips */}
              {!isLoading && quickReplies.length > 0 && (
                <div className="quick-replies">
                  {quickReplies.map((qr, i) => (
                    <button
                      key={i}
                      className={`quick-reply-btn ${qr.query === "__gps__" ? "location-btn" : ""}`}
                      onClick={() => handleQuickReply(qr)}
                      disabled={locationLoading}
                    >
                      {locationLoading && qr.query === "__gps__" ? "📡 Getting location..." : qr.label}
                    </button>
                  ))}
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>
          )}

          {showScrollBtn && messages.length > 0 && (
            <button className="scroll-bottom-btn" onClick={scrollToBottom}>
              ↓
            </button>
          )}
        </div>

        {/* input */}
        <div className="input-container">
          <form className="input-wrapper" onSubmit={handleSubmit}>
            <input
              ref={inputRef}
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder={
                isLoading
                  ? loadingStatus
                  : "Type your message... (e.g. 'Show me pizza places')"
              }
              disabled={isLoading}
              id="chat-input"
            />
            <button
              type="button"
              className={`voice-btn ${isRecording ? "recording" : ""}`}
              onClick={handleVoiceInput}
              disabled={isLoading || !recordingSupported}
              title={recordingSupported ? (isRecording ? "Stop recording" : "Start voice input") : "Voice input not supported in this browser"}
              id="voice-btn"
            >
              {isRecording ? "⏹" : "🎤"}
            </button>
            <button type="submit" className="send-btn" disabled={!input.trim() || isLoading} id="send-btn">
              ↑
            </button>
          </form>
          <div className="input-hint">
            AI-powered food ordering • Voice input available {recordingSupported ? "🎤" : "(browser not supported)"}
          </div>
        </div>
      </main>

      {/* cart panel */}
      <div className={`cart-panel ${cartOpen ? "open" : ""}`}>
        <div className="cart-panel-inner">
          <div className="cart-header">
            <h3>🛒 Your Cart</h3>
            <button className="cart-close-btn" onClick={() => setCartOpen(false)} id="cart-close-btn">✕</button>
          </div>

          {currentRestaurant && (
            <div style={{ marginBottom: "16px", fontSize: "13px", color: "var(--text-secondary)" }}>
              {currentRestaurant.image} {currentRestaurant.name}
              <br />
              <span style={{ fontSize: "11px", color: "var(--text-muted)" }}>
                ⭐ {currentRestaurant.rating} • {currentRestaurant.delivery_time}
              </span>
            </div>
          )}

          <div className="cart-items">
            {cartItems.length === 0 ? (
              <p style={{ color: "var(--text-muted)", fontSize: "13px", textAlign: "center", padding: "40px 0" }}>
                Your cart is empty
              </p>
            ) : (
              cartItems.map((item, i) => (
                <div className="cart-item" key={i}>
                  <div className="cart-item-info">
                    <h4>{item.is_veg ? "🟢" : "🔴"} {item.name}</h4>
                    <p>{item.size ? `${item.size} • ` : ""}Qty: {item.quantity}</p>
                  </div>
                  <div className="cart-item-price">₹{item.price * item.quantity}</div>
                </div>
              ))
            )}
          </div>

          {cartItems.length > 0 && (
            <div className="cart-total">
              <div className="cart-total-row"><span>Subtotal</span><span>₹{cartSubtotal}</span></div>
              <div className="cart-total-row"><span>GST (5%)</span><span>₹{cartTax}</span></div>
              <div className="cart-total-row"><span>Delivery</span><span>₹{cartDelivery}</span></div>
              <div className="cart-total-row total"><span>Total</span><span>₹{cartTotal}</span></div>
              <button className="checkout-btn" onClick={() => { setCartOpen(false); sendMessage("checkout"); }}>
                Proceed to Checkout →
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
