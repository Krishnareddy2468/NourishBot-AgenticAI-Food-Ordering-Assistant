import Link from 'next/link'

export function NourishMark({ className = '' }) {
  return (
    <svg
      className={className}
      viewBox="0 0 52 48"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      aria-hidden="true"
    >
      {/* A single bowl silhouette whose open rim doubles as a clock face. */}
      <path
        d="M5 18c1 14.8 9.3 24 21 24s20-9.2 21-24c-3.4 5.2-11.3 8-21 8S8.4 23.2 5 18Z"
        fill="currentColor"
      />
      <path
        d="M5 18c0-5.3 9.4-9.5 21-9.5S47 12.7 47 18s-9.4 9.5-21 9.5S5 23.3 5 18Z"
        stroke="currentColor"
        strokeWidth="3.5"
      />
      <path
        d="M26 4.5V18l7 4"
        stroke="#DD7048"
        strokeWidth="4.8"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path
        d="M18 44.5h16"
        stroke="currentColor"
        strokeWidth="3"
        strokeLinecap="round"
      />
    </svg>
  )
}

export default function NourishLogo({ className = '', tagline = '' }) {
  return (
    <Link className={`nourish-logo ${className}`.trim()} href="/" aria-label="NourishBot home">
      <span className="nourish-logo-mark"><NourishMark /></span>
      <span className="nourish-logo-copy">
        <span className="nourish-logo-wordmark"><strong>Nourish</strong><em>Bot</em></span>
        {tagline && <small>{tagline}</small>}
      </span>
    </Link>
  )
}
