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
  const years = useMemo(() => {
    const all = data.catalog.years.filter((y): y is number => y !== null)
    return [Math.min(...all), Math.max(...all)]
  }, [data])

  const [mode, setMode] = useState<Mode>('mix')
  const [from, setFrom] = useState(years[0])
  const [to, setTo] = useState(years[1])
  const [genres, setGenres] = useState<string[]>([])
  const [hideWatchlist, setHideWatchlist] = useState(false)
  // The page belongs to one view: a new mode or filter starts again from the best films
  const view = JSON.stringify([mode, from, to, genres, hideWatchlist])
  const [paging, setPaging] = useState({ view, page: 0 })
  const page = paging.view === view ? paging.page : 0
  const setPage = (next: number) => setPaging({ view, page: next })

  const ranked = useMemo(() => rank(analysis.scored, mode), [analysis, mode])
  const filtered = useMemo(
    () =>
      ranked.filter((film) => {
        const i = row.get(film.id)!
        const year = data.catalog.years[i]
        if (year === null || year < from || year > to) return false
        if (genres.length && !data.catalog.genres[i].some((g) => genres.includes(g))) return false
        return !(hideWatchlist && analysis.watchlist.has(film.id))
      }),
    [ranked, row, data, from, to, genres, hideWatchlist, analysis],
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
    if (!favorites.length) {
      return (
        <div className="empty">
          <h2>No favourites yet</h2>
          <p>Use the heart on a poster to keep a film here. Favourites are saved in this browser.</p>
        </div>
      )
    }
    const download = () => {
      const url = URL.createObjectURL(new Blob([letterboxdCsv(favorites)], { type: 'text/csv' }))
      Object.assign(document.createElement('a'), { href: url, download: 'favourites.csv' }).click()
      URL.revokeObjectURL(url)
    }
    return (
      <>
        <div className="controls">
          <span className="field">
            {favorites.length} film{favorites.length > 1 ? 's' : ''}
          </span>
          <button className="button" onClick={download}>
            Download for Letterboxd
          </button>
          <span className="field">Import the file into your watchlist or a list on Letterboxd.</span>
        </div>
        <div className="grid">{favorites.map((f) => card(f.id))}</div>
      </>
    )
  }

  return (
    <>
      <p className="summary">
        {analysis.found} of your {analysis.rated} rated films found on TMDB
        {analysis.miniseries ? ` (${analysis.miniseries} miniseries left out)` : ''}.
      </p>
      <div className="controls">
        <div className="segmented" role="group" aria-label="Recommend">
          {(Object.keys(MODE_LABELS) as Mode[]).map((m) => (
            <button key={m} aria-pressed={mode === m} onClick={() => setMode(m)}>
              {MODE_LABELS[m].label}
            </button>
          ))}
        </div>
        <label className="field">
          From
          <input type="number" value={from} min={years[0]} max={to} onChange={(e) => setFrom(Number(e.target.value))} />
          to
          <input type="number" value={to} min={from} max={years[1]} onChange={(e) => setTo(Number(e.target.value))} />
        </label>
        <label className="field">
          <input type="checkbox" checked={hideWatchlist} onChange={(e) => setHideWatchlist(e.target.checked)} />
          Hide my watchlist
        </label>
      </div>
      <p className="mode-help">{MODE_LABELS[mode].help}</p>
      <div className="chips" role="group" aria-label="Genres">
        {data.genreNames.map((g) => (
          <button
            key={g}
            className="chip"
            aria-pressed={genres.includes(g)}
            onClick={() => setGenres((gs) => (gs.includes(g) ? gs.filter((x) => x !== g) : [...gs, g]))}
          >
            {g}
          </button>
        ))}
      </div>
      {shown.length ? (
        <div className="grid">{shown.map((f) => card(f.id))}</div>
      ) : (
        <div className="empty">
          <h2>No films with these filters</h2>
          <p>Widen the years or remove a genre.</p>
        </div>
      )}
      <div className="pager">
        <button className="button" disabled={page === 0} onClick={() => setPage(page - 1)}>
          Previous
        </button>
        <span>
          {filtered.length ? `${page * PAGE + 1}–${Math.min(filtered.length, (page + 1) * PAGE)}` : 0} of{' '}
          {filtered.length.toLocaleString('en')}
        </span>
        <button className="button primary" disabled={(page + 1) * PAGE >= filtered.length} onClick={() => setPage(page + 1)}>
          More films
        </button>
      </div>
    </>
  )
}
