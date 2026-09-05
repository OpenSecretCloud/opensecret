const HEALTH_PATH = "/health";
const TUF_TIMESTAMP_PATH = "/tuf/metadata/timestamp.json";
const VERSIONED_METADATA_PATH =
  /^\/tuf\/metadata\/[1-9][0-9]*\.(?:root|snapshot|targets)\.json$/;
const TARGET_PREFIX = "/tuf/targets/";
const TARGET_DIRECTORY_SEGMENT = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/;
const HASHED_TARGET_FILENAME =
  /^[0-9a-f]{64}\.[A-Za-z0-9][A-Za-z0-9._-]{0,191}\.json$/;
const MAX_PATH_LENGTH = 2048;
const MAX_TARGET_SEGMENTS = 16;

type CachePolicy = "immutable" | "mutable";

interface AssetFetcher {
  fetch(request: Request): Promise<Response>;
}

export interface Env {
  ASSETS: AssetFetcher;
}

export interface AssetRoute {
  cachePolicy: CachePolicy;
  pathname: string;
}

const SECURITY_HEADERS: Readonly<Record<string, string>> = {
  "access-control-allow-origin": "*",
  "content-security-policy":
    "default-src 'none'; base-uri 'none'; frame-ancestors 'none'; sandbox",
  "cross-origin-resource-policy": "cross-origin",
  "permissions-policy": "camera=(), geolocation=(), microphone=()",
  "referrer-policy": "no-referrer",
  "strict-transport-security": "max-age=31536000; includeSubDomains",
  "x-content-type-options": "nosniff",
  "x-frame-options": "DENY",
  "x-robots-tag": "noindex, nofollow, nosnippet",
};

const MUTABLE_CACHE_HEADERS: Readonly<Record<string, string>> = {
  "cache-control":
    "no-cache, no-store, max-age=0, must-revalidate, no-transform",
  "cdn-cache-control": "no-store",
};

const IMMUTABLE_CACHE_HEADERS: Readonly<Record<string, string>> = {
  "cache-control": "public, max-age=31536000, immutable, no-transform",
  "cdn-cache-control": "public, max-age=31536000, immutable, no-transform",
};

function responseHeaders(
  cachePolicy: CachePolicy,
  contentType: string,
): Headers {
  return new Headers({
    ...SECURITY_HEADERS,
    ...(cachePolicy === "immutable"
      ? IMMUTABLE_CACHE_HEADERS
      : MUTABLE_CACHE_HEADERS),
    "content-type": contentType,
  });
}

function textResponse(
  body: string,
  status: number,
  method: string,
  extraHeaders?: HeadersInit,
): Response {
  const headers = responseHeaders("mutable", "text/plain; charset=utf-8");
  if (extraHeaders !== undefined) {
    new Headers(extraHeaders).forEach((value, name) =>
      headers.set(name, value),
    );
  }

  return new Response(method === "HEAD" ? null : body, { status, headers });
}

function isHashedTargetPath(pathname: string): boolean {
  if (!pathname.startsWith(TARGET_PREFIX)) {
    return false;
  }

  const segments = pathname.slice(TARGET_PREFIX.length).split("/");
  if (
    segments.length === 0 ||
    segments.length > MAX_TARGET_SEGMENTS ||
    !HASHED_TARGET_FILENAME.test(segments.at(-1) ?? "")
  ) {
    return false;
  }

  return segments
    .slice(0, -1)
    .every((segment) => TARGET_DIRECTORY_SEGMENT.test(segment));
}

export function classifyAssetPath(pathname: string): AssetRoute | null {
  if (
    pathname.length === 0 ||
    pathname.length > MAX_PATH_LENGTH ||
    pathname.includes("%") ||
    pathname.includes("//") ||
    pathname.endsWith("/")
  ) {
    return null;
  }

  if (pathname === TUF_TIMESTAMP_PATH) {
    return { cachePolicy: "mutable", pathname };
  }

  if (VERSIONED_METADATA_PATH.test(pathname) || isHashedTargetPath(pathname)) {
    return { cachePolicy: "immutable", pathname };
  }

  return null;
}

function healthResponse(method: string): Response {
  const headers = responseHeaders("mutable", "text/plain; charset=utf-8");
  return new Response(method === "HEAD" ? null : "ok\n", {
    status: 200,
    headers,
  });
}

function copyRepresentationHeaders(
  source: Headers,
  destination: Headers,
): void {
  for (const name of ["content-length", "etag", "last-modified"] as const) {
    const value = source.get(name);
    if (value !== null) {
      destination.set(name, value);
    }
  }
}

export async function handleRequest(
  request: Request,
  env: Env,
): Promise<Response> {
  const url = new URL(request.url);
  // URL.search is empty for a bare trailing `?`, so inspect the serialized URL
  // as well: query aliases must never select a separately cached TUF object.
  const hasUrlSuffix = request.url.includes("?") || request.url.includes("#");
  const isHealth = url.pathname === HEALTH_PATH && !hasUrlSuffix;
  const assetRoute = !hasUrlSuffix ? classifyAssetPath(url.pathname) : null;

  if (!isHealth && assetRoute === null) {
    return textResponse("Not found\n", 404, request.method);
  }

  if (request.method !== "GET" && request.method !== "HEAD") {
    return textResponse("Method not allowed\n", 405, request.method, {
      allow: "GET, HEAD",
    });
  }

  if (isHealth) {
    return healthResponse(request.method);
  }

  // The guard above proves this for the non-health path.
  const route = assetRoute as AssetRoute;
  const assetRequest = new Request(new URL(route.pathname, url.origin), {
    headers: { accept: "application/json" },
    method: "GET",
  });

  let asset: Response;
  try {
    asset = await env.ASSETS.fetch(assetRequest);
  } catch {
    return textResponse("Repository unavailable\n", 503, request.method);
  }

  if (asset.status === 404) {
    return textResponse("Not found\n", 404, request.method);
  }
  if (asset.status !== 200 || asset.redirected) {
    return textResponse("Repository unavailable\n", 503, request.method);
  }

  const headers = responseHeaders(
    route.cachePolicy,
    "application/json; charset=utf-8",
  );
  copyRepresentationHeaders(asset.headers, headers);

  return new Response(request.method === "HEAD" ? null : asset.body, {
    status: 200,
    headers,
  });
}

export default {
  fetch: handleRequest,
};
