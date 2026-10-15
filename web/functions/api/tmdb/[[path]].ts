// Cloudflare Pages Function: relays a few read-only TMDB requests, adding the API key server-side.
// The key (TMDB_API_KEY) is a secret of the Pages project and never reaches the browser.

interface Env {
  TMDB_API_KEY: string
}

// Only what the app needs: film search, TV search (to spot miniseries), and one film's details
const ALLOWED = [/^search\/movie$/, /^search\/tv$/, /^movie\/\d+$/]
const DAY = 24 * 60 * 60

export const onRequestGet: PagesFunction<Env> = async ({ request, env, params }) => {
  const path = ([] as string[]).concat(params.path ?? []).join('/')
  if (!ALLOWED.some((pattern) => pattern.test(path))) {
    return new Response('Not found', { status: 404 })
  }
  const query = new URL(request.url).searchParams
  query.set('api_key', env.TMDB_API_KEY)
  const response = await fetch(`https://api.themoviedb.org/3/${path}?${query}`, {
    // TMDB data changes slowly: Cloudflare caches each answer for a week
    cf: { cacheTtl: 7 * DAY, cacheEverything: true },
  })
  return new Response(response.body, {
    status: response.status,
    headers: {
      'Content-Type': 'application/json',
      'Cache-Control': response.ok ? `public, max-age=${DAY}` : 'no-store',
    },
  })
}
