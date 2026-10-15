import { useEffect, useMemo, useState } from 'react'
import type { Analysis } from '../lib/analyse'
import type { AppData } from '../lib/data'
import { letterboxdCsv, type Favorite } from '../lib/favorites'
import { rank, type Mode } from '../lib/scoring'
import { movie, pool, type Movie } from '../lib/tmdb'
import { FilmCard, type Signals } from './FilmCard'

const PAGE = 20

const MODE_LABELS: Record<Mode, { label: string; help: string }> = {
  mix: { label: 'Balanced', help: 'Your taste and public opinion, in equal parts.' },
  taste: {
    label: 'Close to my ratings',
    help: 'Only films like the ones you rated well, whatever the public thinks.',
  },
  public: { label: 'Public favourites', help: 'The best rated films you have not seen, to look beyond your habits.' },
}

/** Fetches the TMDB details (title, poster, director) of the films on screen (tmdb.ts caches them). */
function useMovies(ids: number[]): Map<number, Movie> {
  const [movies, setMovies] = useState(new Map<number, Movie>())
  const key = ids.join(',')
  useEffect(() => {
    let current = true
    const wanted = key ? key.split(',').map(Number) : []
    pool(wanted, movie).then((found) => {
      if (current) setMovies((m) => new Map([...m, ...found.map((f) => [f.id, f] as const)]))
    })
    return () => {
      current = false
    }
  }, [key])
  return movies
}

export function Results(props: {
  analysis: Analysis
  data: AppData
  tab: 'films' | 'favourites'
  favorites: Favorite[]
  isFavorite: (id: number) => boolean
  toggle: (film: Favorite) => void
}) {
  const { analysis, data, tab, favorites, isFavorite, toggle } = props
  const row = useMemo(() => new Map([...data.catalog.ids].map((id, i) => [id, i])), [data])
  const decades = useMemo(() => {
    const years = data.catalog.years.filter((y): y is number => y !== null)
    const set = new Set(years.map((y) => Math.floor(y / 10) * 10))
    return [...set].sort((a, b) => b - a)
  }, [data])

  const [mode, setMode] = useState<Mode>('mix')
  const [decade, setDecade] = useState<number | null>(null)
  const [genre, setGenre] = useState<string | null>(null)
  const [hideWatchlist, setHideWatchlist] = useState(false)
  // The page belongs to one view: a new mode or filter starts again from the best films
  const view = JSON.stringify([mode, decade, genre, hideWatchlist])
  const [paging, setPaging] = useState({ view, page: 0 })
  const page = paging.view === view ? paging.page : 0
  const setPage = (next: number) => setPaging({ view, page: next })

  const ranked = useMemo(() => rank(analysis.scored, mode), [analysis, mode])
  const filtered = useMemo(
    () =>
      ranked.filter((film) => {
        const i = row.get(film.id)!
        const year = data.catalog.years[i]
        if (decade !== null && (year === null || Math.floor(year / 10) * 10 !== decade)) return false
        if (genre !== null && !data.catalog.genres[i].includes(genre)) return false
        return !(hideWatchlist && analysis.watchlist.has(film.id))
      }),
    [ranked, row, data, decade, genre, hideWatchlist, analysis],
  )

  const shown = filtered.slice(page * PAGE, page * PAGE + PAGE)
  const signals = useMemo(() => new Map<number, Signals>(ranked.map((f) => [f.id, f])), [ranked])
  const movies = useMovies(tab === 'films' ? shown.map((f) => f.id) : favorites.map((f) => f.id))

  const card = (id: number) => {
    const m = movies.get(id)
    return (
      <FilmCard
        key={id}
        id={id}
        movie={m}
        signals={signals.get(id)}
        inWatchlist={analysis.watchlist.has(id)}
        liked={isFavorite(id)}
        onLike={() => toggle({ id, title: m?.title ?? String(id), year: Number(m?.release_date?.slice(0, 4)) || null })}
      />
    )
  }

  if (tab === 'favourites') {
    const download = () => {
      const url = URL.createObjectURL(new Blob([letterboxdCsv(favorites)], { type: 'text/csv' }))
      Object.assign(document.createElement('a'), { href: url, download: 'favourites.csv' }).click()
      URL.revokeObjectURL(url)
    }
    return (
      <>
        <div className="section">
          <span className="label">
            Favourites{favorites.length ? ` · ${favorites.length}` : ''}
          </span>
          {favorites.length > 0 && (
            <button className="button" onClick={download}>
              Export for Letterboxd
            </button>
          )}
        </div>
        {favorites.length ? (
          <>
            <p className="mode-help">Import the exported file into your watchlist or a list on Letterboxd.</p>
            <div className="grid">{favorites.map((f) => card(f.id))}</div>
          </>
        ) : (
          <div className="empty">
            <h2>No favourites yet</h2>
            <p>Use the heart on a poster to keep a film here. Favourites are saved in this browser.</p>
          </div>
        )}
      </>
    )
  }

  return (
    <>
      <div className="section">
        <span className="label">Recommended for you</span>
        <span className="note">
          Based on {analysis.found} of your {analysis.rated} rated films
          {analysis.miniseries ? `, ${analysis.miniseries} miniseries left out` : ''}
        </span>
      </div>
      <div className="filters">
        <div className="modes" role="group" aria-label="Recommend">
          {(Object.keys(MODE_LABELS) as Mode[]).map((m) => (
            <button key={m} aria-pressed={mode === m} onClick={() => setMode(m)}>
              {MODE_LABELS[m].label}
            </button>
          ))}
        </div>
        <select
          className="select"
          aria-label="Decade"
          value={decade ?? ''}
          onChange={(e) => setDecade(e.target.value ? Number(e.target.value) : null)}
        >
          <option value="">Any decade</option>
          {decades.map((d) => (
            <option key={d} value={d}>
              {d}s
            </option>
          ))}
        </select>
        <select
          className="select"
          aria-label="Genre"
          value={genre ?? ''}
          onChange={(e) => setGenre(e.target.value || null)}
        >
          <option value="">Any genre</option>
          {data.genreNames.map((g) => (
            <option key={g} value={g}>
              {g}
            </option>
          ))}
        </select>
        <label className="toggle">
          <input type="checkbox" checked={hideWatchlist} onChange={(e) => setHideWatchlist(e.target.checked)} />
          Hide watchlist
        </label>
      </div>
      <p className="mode-help">{MODE_LABELS[mode].help}</p>
      <dl className="legend">
        <div>
          <dt>Quality</dt>
          <dd>Its rating on TMDB, trusted more when many people voted.</dd>
        </div>
        <div>
          <dt>Content</dt>
          <dd>How close its story, genres, director and cast are to the films you rated well.</dd>
        </div>
        <div>
          <dt>Taste</dt>
          <dd>Whether people who rate films like you liked it (MovieLens, 200,000 people).</dd>
        </div>
      </dl>
      {shown.length ? (
        <div className="grid">{shown.map((f) => card(f.id))}</div>
      ) : (
        <div className="empty">
          <h2>No films with these filters</h2>
          <p>Try another decade or genre.</p>
        </div>
      )}
      {filtered.length > PAGE && (
        <div className="pager">
          <button className="button" disabled={page === 0} onClick={() => setPage(page - 1)}>
            Previous
          </button>
          <span>
            {page * PAGE + 1}–{Math.min(filtered.length, (page + 1) * PAGE)} of {filtered.length.toLocaleString('en')}
          </span>
          <button className="button" disabled={(page + 1) * PAGE >= filtered.length} onClick={() => setPage(page + 1)}>
            More films
          </button>
        </div>
      )}
    </>
  )
}
