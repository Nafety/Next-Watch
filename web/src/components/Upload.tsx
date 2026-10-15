import { useRef, useState } from 'react'

export function Upload({ onFile, error }: { onFile: (file: File) => void; error: string | null }) {
  const input = useRef<HTMLInputElement>(null)
  const [over, setOver] = useState(false)
  return (
    <section className="hero">
      <h1>Find what to watch next, from your Letterboxd ratings.</h1>
      <p>
        Each film you have not seen is scored on its description, on the taste of people who rate like you, and
        on public opinion. It all runs in your browser: your ratings are never sent anywhere.
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
        <span className="button primary">Choose your export</span>
        <span>or drop the .zip file here</span>
      </button>
      <input
        ref={input}
        type="file"
        accept=".zip,application/zip"
        hidden
        onChange={(e) => e.target.files?.[0] && onFile(e.target.files[0])}
      />
      <ol className="steps">
        <li>
          <b>1</b>On letterboxd.com, open Settings, then the Data tab.
        </li>
        <li>
          <b>2</b>Click Export your data. A .zip file downloads.
        </li>
        <li>
          <b>3</b>Choose it above. The more films you have rated, the better.
        </li>
        <li>
          <b>4</b>Heart the films you want to see, then export your favourites and import the file on Letterboxd:
          they are all added to your watchlist at once.
        </li>
      </ol>
      {error && <p className="error">{error}</p>}
    </section>
  )
}
