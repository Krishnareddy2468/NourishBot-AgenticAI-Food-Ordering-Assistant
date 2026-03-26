import { NextResponse } from "next/server";

const BACKEND_API_URL = process.env.BACKEND_API_URL || process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export async function GET(_request, { params }) {
  try {
    const { userId } = await params;

    if (!userId) {
      return NextResponse.json({ detail: "Missing userId" }, { status: 400 });
    }

    const backendResponse = await fetch(`${BACKEND_API_URL}/api/chat/order-status/${userId}`, {
      method: "GET",
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
        detail: `Frontend proxy could not reach backend: ${error instanceof Error ? error.message : "Unknown error"}`,
      },
      { status: 502 }
    );
  }
}
