// The browser must recommend exactly what the Python package recommends (fixture made in Python).

import { readFile } from 'node:fs/promises'
import { describe, expect, it } from 'vitest'
import { loadData } from './data'
import fixture from './fixtures/parity.json'
import { tokenSortRatio } from './matching'
import { rank, score, type Mode } from './scoring'

const data = await loadData(async (file) => {
  const buffer = await readFile(new URL(`../../public/data/${file}`, import.meta.url))
  return buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength)
})

describe('same results as Python', () => {
  const scored = score(data.catalog, fixture.rated, [])

  for (const mode of ['mix', 'taste', 'public'] as Mode[]) {
    it(`ranks the films like Python in ${mode} mode`, () => {
      const ranked = rank(scored, mode).slice(0, 30)
      expect(ranked.map((r) => r.id)).toEqual(fixture.expected[mode].ids)
      ranked.forEach((r, i) => expect(r.score).toBeCloseTo(fixture.expected[mode].scores[i], 3))
    })
  }

  it('matches titles like rapidfuzz', () => {
    for (const [a, b, expected] of fixture.tokenSortRatio as [string, string, number][]) {
      expect(tokenSortRatio(a, b)).toBeCloseTo(expected, 1)
    }
  })
})
