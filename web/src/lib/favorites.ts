// Favorites, kept in the browser (localStorage) so they survive reloads; exportable for Letterboxd.

import { useEffect, useState } from 'react'

export interface Favorite {
  id: number
  title: string
  year: number | null
}

const KEY = 'next-watch:favorites'

function load(): Favorite[] {
  try {
    return JSON.parse(localStorage.getItem(KEY) ?? '[]') as Favorite[]
  } catch {
    return [] // private window, blocked storage...
  }
}

export function useFavorites() {
  const [favorites, setFavorites] = useState<Favorite[]>(load)
  useEffect(() => {
    try {
      localStorage.setItem(KEY, JSON.stringify(favorites))
    } catch {
      // storage unavailable: favorites stay for this visit only
    }
  }, [favorites])

  const toggle = (film: Favorite) =>
    setFavorites((current) =>
      current.some((f) => f.id === film.id) ? current.filter((f) => f.id !== film.id) : [film, ...current],
    )
  return { favorites, isFavorite: (id: number) => favorites.some((f) => f.id === id), toggle }
}

/** Letterboxd's import format (tmdbID gives an exact match): import into a list or your watchlist. */
export function letterboxdCsv(favorites: Favorite[]): string {
  const quote = (value: string) => `"${value.replaceAll('"', '""')}"`
  const lines = favorites.map((f) => [f.id, quote(f.title), f.year ?? ''].join(','))
  return ['tmdbID,Title,Year', ...lines].join('\n')
}
