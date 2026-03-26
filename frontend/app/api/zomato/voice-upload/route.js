import { NextResponse } from "next/server";

const BACKEND_API_URL = process.env.BACKEND_API_URL || process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export async function POST(request) {
  try {
    const incomingForm = await request.formData();

    const forwardForm = new FormData();
    const audio = incomingForm.get("audio");
    if (!audio) {
      return NextResponse.json({ detail: "Missing audio file" }, { status: 400 });
    }

    forwardForm.append("audio", audio);
    forwardForm.append("user_id", String(incomingForm.get("user_id") || "web_user"));
    forwardForm.append("user_name", String(incomingForm.get("user_name") || "Guest"));

    const userLocation = incomingForm.get("user_location");
    if (userLocation) {
      forwardForm.append("user_location", String(userLocation));
    }

    const backendResponse = await fetch(`${BACKEND_API_URL}/api/chat/voice-upload`, {
      method: "POST",
      body: forwardForm,
      cache: "no-store",
    });

    const text = await backendResponse.text();

    return new NextResponse(text, {
      status: backendResponse.status,
      headers: {
        "Content-Type": backendResponse.headers.get("content-type") || "application/json",
      },
    });
  } catch (error) {
    return NextResponse.json(
      {
        detail: `Frontend proxy could not reach backend voice endpoint: ${error instanceof Error ? error.message : "Unknown error"}`,
      },
      { status: 502 }
    );
  }
}
