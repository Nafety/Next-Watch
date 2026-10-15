// Favorites, kept in the browser (localStorage) so they survive reloads; exportable for Letterboxd.

import { useState } from 'react'

export interface Favorite {
  id: number
  title: string
  year: number | null
}

const key = (username: string) => `next-watch:favorites:${username}`

function load(username: string): Favorite[] {
  try {
    return JSON.parse(localStorage.getItem(key(username)) ?? '[]') as Favorite[]
  } catch {
    return [] // private window, blocked storage...
  }
}

/** Favourites of one Letterboxd account, kept in this browser. */
export function useFavorites(username: string) {
  const [state, setState] = useState(() => ({ username, favorites: load(username) }))
  // Another account: its own favourites
  const favorites = state.username === username ? state.favorites : load(username)

  const save = (next: Favorite[]) => {
    setState({ username, favorites: next })
    try {
      localStorage.setItem(key(username), JSON.stringify(next))
    } catch {
      // storage unavailable: favourites stay for this visit only
    }
  }
  const toggle = (film: Favorite) =>
    save(favorites.some((f) => f.id === film.id) ? favorites.filter((f) => f.id !== film.id) : [film, ...favorites])
  const clear = () => save([])
  return { favorites, isFavorite: (id: number) => favorites.some((f) => f.id === id), toggle, clear }
}

/** Letterboxd's import format (tmdbID gives an exact match): import into a list or your watchlist. */
export function letterboxdCsv(favorites: Favorite[]): string {
  const quote = (value: string) => `"${value.replaceAll('"', '""')}"`
  const lines = favorites.map((f) => [f.id, quote(f.title), f.year ?? ''].join(','))
  return ['tmdbID,Title,Year', ...lines].join('\n')
}
