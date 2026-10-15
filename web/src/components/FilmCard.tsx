import type { Movie } from '../lib/tmdb'

const POSTER = 'https://image.tmdb.org/t/p/w342'

export interface Signals {
  quality: number
  content: number
  collaborative: number
}

function Heart({ filled }: { filled: boolean }) {
  return (
    <svg width="17" height="17" viewBox="0 0 24 24" aria-hidden="true">
      <path
        d="M12 20.5s-7.5-4.6-7.5-10.1A4.4 4.4 0 0 1 12 7.6a4.4 4.4 0 0 1 7.5 2.8c0 5.5-7.5 10.1-7.5 10.1z"
        fill={filled ? 'currentColor' : 'none'}
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinejoin="round"
      />
    </svg>
  )
}

/** A standardized signal as a bar: -2 standard deviations or less is empty, +4 or more is full. */
function Signal({ label, value }: { label: string; value: number }) {
  const width = Math.min(100, Math.max(4, ((value + 2) / 6) * 100))
  return (
    <div className="signal" title={`${label}: ${value >= 0 ? '+' : ''}${value.toFixed(1)} standard deviations`}>
      <span>{label}</span>
      <i>
        <b style={{ width: `${width}%` }} />
      </i>
    </div>
  )
}

export function FilmCard(props: {
  movie: Movie | undefined
  id: number
  signals?: Signals
  inWatchlist: boolean
  liked: boolean
  onLike: () => void
}) {
  const { movie, id, signals, inWatchlist, liked, onLike } = props
  const year = movie?.release_date?.slice(0, 4)
  const directors = movie?.credits.crew.filter((c) => c.job === 'Director').map((c) => c.name).slice(0, 2)
  const link = `https://letterboxd.com/tmdb/${id}/`
  return (
    <article className="card">
      <div className="poster">
        <a href={link} target="_blank" rel="noreferrer" tabIndex={-1}>
          {movie?.poster_path ? (
            <img src={POSTER + movie.poster_path} alt="" loading="lazy" />
          ) : (
            <div className="blank">{movie?.title ?? ''}</div>
          )}
        </a>
        {inWatchlist && <span className="badge">Watchlist</span>}
        <button
          className="like"
          aria-pressed={liked}
          aria-label={liked ? 'Remove from favourites' : 'Add to favourites'}
          onClick={onLike}
        >
          <Heart filled={liked} />
        </button>
      </div>
      <h3>
        <a href={link} target="_blank" rel="noreferrer">
          {movie?.title ?? 'Loading...'}
        </a>
      </h3>
      <div className="meta">{[year, directors?.join(', ')].filter(Boolean).join(' · ')}</div>
      {movie && (
        <div className="rating">
          TMDB <b>{movie.vote_average.toFixed(1)}</b>
          <span className="meta"> · {movie.vote_count.toLocaleString('en')} votes</span>
        </div>
      )}
      {signals && (
        <div className="signals">
          <Signal label="Quality" value={signals.quality} />
          <Signal label="Content" value={signals.content} />
          <Signal label="Taste" value={signals.collaborative} />
        </div>
      )}
    </article>
  )
}
