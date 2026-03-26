import { NextResponse } from "next/server";

const BACKEND_API_URL = process.env.BACKEND_API_URL || process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export async function GET(_request, { params }) {
  try {
    const { fileName } = await params;
    if (!fileName) {
      return NextResponse.json({ detail: "Missing QR file name" }, { status: 400 });
    }

    const backendResponse = await fetch(`${BACKEND_API_URL}/api/chat/qr/${encodeURIComponent(fileName)}`, {
      method: "GET",
      cache: "no-store",
    });

    if (!backendResponse.ok) {
      const text = await backendResponse.text();
      return new NextResponse(text, {
        status: backendResponse.status,
        headers: {
          "Content-Type": backendResponse.headers.get("content-type") || "application/json",
        },
      });
    }

    const imageBytes = await backendResponse.arrayBuffer();
    return new NextResponse(imageBytes, {
      status: 200,
      headers: {
        "Content-Type": backendResponse.headers.get("content-type") || "image/png",
        "Cache-Control": "no-store",
      },
    });
  } catch (error) {
    return NextResponse.json(
      {
        detail: `Frontend proxy could not fetch QR image: ${error instanceof Error ? error.message : "Unknown error"}`,
      },
      { status: 502 }
    );
  }
}
