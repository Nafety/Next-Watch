// Picking the TMDB search result that corresponds to a Letterboxd film (title + year).
// Same rules as the Python version (film_reco/matching.py), with rapidfuzz's token_sort_ratio.

export interface SearchResult {
  id: number
  title: string
  original_title?: string
  release_date?: string
  popularity?: number
}

export const MIN_SCORE = 80

/** rapidfuzz's default_process: lower case, anything but letters and digits becomes a space, trimmed. */
function process(text: string): string {
  return text.toLowerCase().replace(/[^\p{L}\p{N}]+/gu, ' ').trim()
}

/** Length of the longest common subsequence of two strings. */
function lcs(a: string, b: string): number {
  let previous = new Array<number>(b.length + 1).fill(0)
  for (const ca of a) {
    const current = [0]
    for (let j = 0; j < b.length; j++) {
      current.push(ca === b[j] ? previous[j] + 1 : Math.max(previous[j + 1], current[j]))
    }
    previous = current
  }
  return previous[b.length]
}

/** rapidfuzz.fuzz.token_sort_ratio with default_process: 0-100. */
export function tokenSortRatio(a: string, b: string): number {
  const sorted = (s: string) => process(s).split(/\s+/).filter(Boolean).sort().join(' ')
  const x = sorted(a)
  const y = sorted(b)
  if (!x.length && !y.length) return 100
  return (200 * lcs([...x].join(''), [...y].join(''))) / ([...x].length + [...y].length)
}

export function yearOf(result: SearchResult): number | null {
  const year = Number((result.release_date ?? '').slice(0, 4))
  return Number.isInteger(year) && year > 0 ? year : null
}

/** 0-100: title similarity (English or original title), minus a penalty when the years differ. */
export function matchScore(title: string, year: number | null, result: SearchResult): number {
  const similarity = Math.max(
    tokenSortRatio(title, result.title),
    tokenSortRatio(title, result.original_title || result.title),
  )
  const resultYear = yearOf(result)
  if (year === null || resultYear === null) return similarity - 10
  const gap = Math.abs(year - resultYear)
  return similarity - (gap === 0 ? 0 : gap === 1 ? 5 : 40)
}

/** Best result if convincing enough, else null. Ties go to the most popular film. */
export function bestMatch(
  title: string,
  year: number | null,
  results: SearchResult[],
): { result: SearchResult; score: number } | null {
  let best: { result: SearchResult; score: number } | null = null
  for (const result of results) {
    const score = matchScore(title, year, result)
    if (
      !best ||
      score > best.score ||
      (score === best.score && (result.popularity ?? 0) > (best.result.popularity ?? 0))
    ) {
      best = { result, score }
    }
  }
  return best && best.score >= MIN_SCORE ? best : null
}
