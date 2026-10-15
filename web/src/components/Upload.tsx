import { useRef, useState } from 'react'

export function Upload({ onFile, error }: { onFile: (file: File) => void; error: string | null }) {
  const input = useRef<HTMLInputElement>(null)
  const [over, setOver] = useState(false)
  return (
    <section className="hero">
      <h1>
        What to watch <em>next</em>, from your Letterboxd ratings.
      </h1>
      <p>
        Drop your Letterboxd export: each film you have not seen gets a score from its description, from the
        taste of people who rate like you, and from public opinion. Everything runs in your browser; your
        ratings are never sent anywhere.
      </p>
      <button
        className={`drop ${over ? 'over' : ''}`}
        onClick={() => input.current?.click()}
        onDragOver={(e) => {
          e.preventDefault()
          setOver(true)
        }}
        onDragLeave={() => setOver(false)}
        onDrop={(e) => {
          e.preventDefault()
          setOver(false)
          const file = e.dataTransfer.files[0]
          if (file) onFile(file)
        }}
      >
        <strong>Drop your Letterboxd export here</strong>
        <span>or click to choose the .zip file</span>
      </button>
      <input
        ref={input}
        type="file"
        accept=".zip,application/zip"
        hidden
        onChange={(e) => e.target.files?.[0] && onFile(e.target.files[0])}
      />
      {error && <p className="error">{error}</p>}
      <ol className="steps">
        <li>On letterboxd.com, open Settings, then the Data tab.</li>
        <li>Click Export your data: a .zip file downloads.</li>
        <li>Drop it above. Rate at least a few dozen films for good results.</li>
      </ol>
    </section>
  )
}
