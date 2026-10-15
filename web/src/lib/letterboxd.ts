// Reading a Letterboxd data export (the ZIP from Settings > Data > Export your data), in the browser.

import JSZip from 'jszip'

export interface LetterboxdFilm {
  uri: string
  title: string
  year: number | null
}

export interface RatedFilm extends LetterboxdFilm {
  rating: number
}

/** A CSV file as objects keyed by the header (quoted fields, commas and line breaks inside quotes). */
export function parseCsv(text: string): Record<string, string>[] {
  const rows: string[][] = []
  let row: string[] = []
  let field = ''
  let quoted = false
  for (let i = 0; i < text.length; i++) {
    const c = text[i]
    if (quoted) {
      if (c === '"' && text[i + 1] === '"') {
        field += '"'
        i++
      } else if (c === '"') quoted = false
      else field += c
    } else if (c === '"') quoted = true
    else if (c === ',') {
      row.push(field)
      field = ''
    } else if (c === '\n' || c === '\r') {
      if (c === '\r' && text[i + 1] === '\n') i++
      row.push(field)
      rows.push(row)
      row = []
      field = ''
    } else field += c
  }
  if (field || row.length) rows.push([...row, field])
  const [header, ...body] = rows.filter((r) => r.some((cell) => cell !== ''))
  return body.map((cells) => Object.fromEntries(header.map((name, i) => [name, cells[i] ?? ''])))
}

function film(row: Record<string, string>): LetterboxdFilm {
  const year = Number(row.Year)
  return { uri: row['Letterboxd URI'], title: row.Name, year: Number.isInteger(year) && year > 0 ? year : null }
}

export async function readExport(file: Blob): Promise<{ rated: RatedFilm[]; watchlist: LetterboxdFilm[] }> {
  const zip = await JSZip.loadAsync(file)
  const read = async (name: string) => {
    const entry = zip.file(name)
    return entry ? parseCsv(await entry.async('string')) : []
  }
  const ratings = await read('ratings.csv')
  if (!zip.file('ratings.csv')) throw new Error('This ZIP has no ratings.csv: is it a Letterboxd export?')
  return {
    rated: ratings.map((row) => ({ ...film(row), rating: Number(row.Rating) })).filter((f) => f.rating > 0),
    watchlist: (await read('watchlist.csv')).map(film),
  }
}
