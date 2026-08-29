import { describe, expect, test } from "bun:test";

import { classifyAssetPath, handleRequest, type Env } from "./worker";

const SHA256 = "a".repeat(64);

function envReturning(response: Response, requests: Request[] = []): Env {
  return {
    ASSETS: {
      async fetch(request) {
        requests.push(request);
        return response.clone();
      },
    },
  };
}

describe("TUF path contract", () => {
  test("classifies only the mutable timestamp as non-immutable", () => {
    expect(classifyAssetPath("/tuf/metadata/timestamp.json")).toEqual({
      cachePolicy: "mutable",
      pathname: "/tuf/metadata/timestamp.json",
    });

    for (const pathname of [
      "/tuf/metadata/1.root.json",
      "/tuf/metadata/12.snapshot.json",
      "/tuf/metadata/999.targets.json",
      `/tuf/targets/policy/${SHA256}.builders.json`,
      `/tuf/targets/sigstore/${SHA256}.trusted_root.json`,
      `/tuf/targets/channels/${SHA256}.prod.json`,
      `/tuf/targets/releases/1.2.3/prod/${SHA256}.manifest.sigstore.json`,
    ]) {
      expect(classifyAssetPath(pathname)).toEqual({
        cachePolicy: "immutable",
        pathname,
      });
    }
  });

  test("rejects aliases, unversioned metadata, and unhashed targets", () => {
    for (const pathname of [
      "/",
      "/tuf/",
      "/tuf//metadata/timestamp.json",
      "/tuf/metadata/timestamp.json/",
      "/tuf/metadata/%74imestamp.json",
      "/tuf/metadata/root.json",
      "/tuf/metadata/snapshot.json",
      "/tuf/metadata/targets.json",
      "/tuf/metadata/0.root.json",
      "/tuf/metadata/01.root.json",
      "/tuf/metadata/1.delegated.json",
      "/tuf/targets/builders.json",
      `/tuf/targets/${"A".repeat(64)}.builders.json`,
      `/tuf/targets/policy//${SHA256}.builders.json`,
      `/tuf/targets/.hidden/${SHA256}.builders.json`,
      `/tuf/targets/policy/${SHA256}.builders.txt`,
    ]) {
      expect(classifyAssetPath(pathname)).toBeNull();
    }
  });
});

describe("attestations Worker", () => {
  test("serves GET and HEAD health checks without consulting assets", async () => {
    let assetFetches = 0;
    const env: Env = {
      ASSETS: {
        async fetch() {
          assetFetches += 1;
          return new Response(null, { status: 404 });
        },
      },
    };

    const get = await handleRequest(
      new Request("https://attestations.trymaple.ai/health"),
      env,
    );
    expect(get.status).toBe(200);
    expect(await get.text()).toBe("ok\n");
    expect(get.headers.get("cache-control")).toContain("no-store");

    const head = await handleRequest(
      new Request("https://attestations.trymaple.ai/health", {
        method: "HEAD",
      }),
      env,
    );
    expect(head.status).toBe(200);
    expect(await head.text()).toBe("");
    expect(assetFetches).toBe(0);
  });

  test("allows only GET and HEAD on recognized paths", async () => {
    const response = await handleRequest(
      new Request(
        "https://attestations.trymaple.ai/tuf/metadata/timestamp.json",
        {
          method: "POST",
        },
      ),
      envReturning(new Response(null, { status: 404 })),
    );

    expect(response.status).toBe(405);
    expect(response.headers.get("allow")).toBe("GET, HEAD");
    expect(response.headers.get("cache-control")).toContain("no-store");
  });

  test("rejects query aliases and unknown paths before reading assets", async () => {
    let assetFetches = 0;
    const env: Env = {
      ASSETS: {
        async fetch() {
          assetFetches += 1;
          return new Response("unexpected");
        },
      },
    };

    for (const url of [
      "https://attestations.trymaple.ai/tuf/metadata/timestamp.json?cache=bust",
      "https://attestations.trymaple.ai/tuf/metadata/timestamp.json?",
      "https://attestations.trymaple.ai/tuf/metadata/root.json",
      "https://attestations.trymaple.ai/tuf/targets/builders.json",
    ]) {
      const response = await handleRequest(new Request(url), env);
      expect(response.status).toBe(404);
    }
    expect(assetFetches).toBe(0);
  });

  test("serves timestamp bytes with no-cache semantics", async () => {
    const requests: Request[] = [];
    const bytes = new TextEncoder().encode('{\n  "signed": true\n}\n');
    const response = await handleRequest(
      new Request(
        "https://attestations.trymaple.ai/tuf/metadata/timestamp.json",
        {
          headers: {
            authorization: "Bearer must-not-be-forwarded",
            cookie: "session=must-not-be-forwarded",
            "if-none-match": '"caller-etag"',
            range: "bytes=0-1",
          },
        },
      ),
      envReturning(
        new Response(bytes, {
          headers: {
            etag: '"timestamp-etag"',
            "last-modified": "Sat, 29 Aug 2026 12:00:00 GMT",
          },
        }),
        requests,
      ),
    );

    expect(response.status).toBe(200);
    expect(new Uint8Array(await response.arrayBuffer())).toEqual(bytes);
    expect(response.headers.get("cache-control")).toContain("no-cache");
    expect(response.headers.get("cache-control")).toContain("no-store");
    expect(response.headers.get("cdn-cache-control")).toBe("no-store");
    expect(response.headers.get("content-type")).toBe(
      "application/json; charset=utf-8",
    );
    expect(response.headers.get("access-control-allow-origin")).toBe("*");
    expect(response.headers.get("x-content-type-options")).toBe("nosniff");
    expect(response.headers.get("etag")).toBe('"timestamp-etag"');
    expect(requests).toHaveLength(1);
    expect(requests[0].method).toBe("GET");
    expect(requests[0].url).toBe(
      "https://attestations.trymaple.ai/tuf/metadata/timestamp.json",
    );
    expect(requests[0].headers.get("accept")).toBe("application/json");
    expect(requests[0].headers.has("authorization")).toBe(false);
    expect(requests[0].headers.has("cookie")).toBe(false);
    expect(requests[0].headers.has("if-none-match")).toBe(false);
    expect(requests[0].headers.has("range")).toBe(false);
  });

  test("serves versioned metadata and hashed targets as immutable", async () => {
    for (const pathname of [
      "/tuf/metadata/3.targets.json",
      `/tuf/targets/releases/1.2.3/prod/${SHA256}.manifest.json`,
    ]) {
      const response = await handleRequest(
        new Request(`https://attestations.trymaple.ai${pathname}`),
        envReturning(new Response('{"ok":true}\n')),
      );

      expect(response.status).toBe(200);
      expect(response.headers.get("cache-control")).toBe(
        "public, max-age=31536000, immutable, no-transform",
      );
      expect(response.headers.get("cdn-cache-control")).toBe(
        "public, max-age=31536000, immutable, no-transform",
      );
    }
  });

  test("serves HEAD using GET asset headers and no body", async () => {
    const requests: Request[] = [];
    const response = await handleRequest(
      new Request("https://attestations.trymaple.ai/tuf/metadata/2.root.json", {
        method: "HEAD",
      }),
      envReturning(
        new Response("root bytes", { headers: { etag: '"root-etag"' } }),
        requests,
      ),
    );

    expect(response.status).toBe(200);
    expect(await response.text()).toBe("");
    expect(response.headers.get("etag")).toBe('"root-etag"');
    expect(requests[0].method).toBe("GET");
  });

  test("maps absent assets to 404 and failures or redirects to 503", async () => {
    const url = "https://attestations.trymaple.ai/tuf/metadata/1.root.json";

    const absent = await handleRequest(
      new Request(url),
      envReturning(new Response(null, { status: 404 })),
    );
    expect(absent.status).toBe(404);
    expect(absent.headers.get("cache-control")).toContain("no-store");
    expect(absent.headers.get("cdn-cache-control")).toBe("no-store");

    const failed = await handleRequest(
      new Request(url),
      envReturning(new Response(null, { status: 500 })),
    );
    expect(failed.status).toBe(503);
    expect(failed.headers.get("cache-control")).toContain("no-store");
    expect(failed.headers.get("cdn-cache-control")).toBe("no-store");

    const redirect = await handleRequest(
      new Request(url),
      envReturning(Response.redirect("https://example.invalid/root.json")),
    );
    expect(redirect.status).toBe(503);

    const unavailable = await handleRequest(new Request(url), {
      ASSETS: {
        async fetch() {
          throw new Error("asset binding unavailable");
        },
      },
    });
    expect(unavailable.status).toBe(503);
  });
});
